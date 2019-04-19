from __future__ import print_function,division

import numpy as np
import os
from collections import OrderedDict
from scipy import interpolate as interp
import numpy.core.numeric as NX
import sys

from lsstools.MeasuredPower import MeasuredPower1D, MeasuredPower2D

def interp1d_manual_k_binning(kin, Pin, kind='manual_Pk_k_bins', fill_value=None, bounds_error=False,
                              Ngrid=None, L=None, k_bin_width=1.0, verbose=False,
                              Pk=None
    ):
    """
    Interpolate following a fixed k binning scheme that's also used to measure power spectra
    in cy_power_estimator.pyx.

    Parameters
    ----------
    kind : string
        Use 'manual_Pk_k_bins' for 1d power, or 'manual_Pk_k_mu_bins' for 2d power.

    L : float
        boxsize in Mpc/h

    kin, Pin: numpy.ndarray
        These are interpolated

    Pk : Dict containing MeasuredPower1D or MeasuredPower2D entries
        Only used to read power spectrum measurement options, e.g. Nmu, los, etc.
    """
    # check args
    if (fill_value is None) and (not bounds_error):
        raise Exception("Must provide fill_value if bounds_error=False")
    if Ngrid is None:
        raise Exception("Must provide Ngrid")
    if L is None:
        raise Exception("Must provide L")
    
    
    if kind == 'manual_Pk_k_bins':
    
        dk = 2.0*np.pi/float(L)

        # check that kin has all k bins
        if k_bin_width == 1.:
            # 18 Jan 2019: somehow need 0.99 factor for nbodykit 0.3 to get last k bin right.
            #kin_expected = np.arange(1,np.max(kin)*0.99/dk+1)*dk
            # 16 Mar 2019: Fix expected k bins to match nbodykit for larger Ngrid
            kin_expected = np.arange(1, kin.shape[0]+1)*dk

            if verbose:
                print("kin:", kin)
                print("kin_expected:", kin_expected)
                print("kin/kin_expected (should be between 0.5 and 1.5):\n", kin/kin_expected)

            # bin center is computed by averaging k within bin, so it's not exactly dk*i.
            if not np.allclose(kin, kin_expected, rtol=0.35):
                print("kin:", kin)
                print("kin_expected:", kin_expected)
                print("kin/kin_expected (should be between 0.5 and 1.5):\n", kin/kin_expected)
                raise Exception('Found issue with k bins when interpolating')

        else:
            raise Exception("k_bin_width=%s not implemented yet" % str(k_bin_width))


        def interpolator(karg):
            """
            Function that interpolates Pin from kin to karg.
            """
            ibin = round_float2int_arr(karg/(dk*k_bin_width))
            # first bin is dropped
            ibin -= 1

            # k's between kmin and max
            max_ibin = Pin.shape[0]-1
            Pout = np.where((ibin>=0) & (ibin<=max_ibin), Pin[ibin%(max_ibin+1)], np.zeros(ibin.shape)+np.nan)

            # k<kmin
            if np.where(ibin<0)[0].shape[0] > 0:
                if bounds_error:
                    raise Exception("Bounds error: k<kmin in interpolation, k=%s" % str(karg))
                else:
                    Pout = np.where(ibin<0, np.zeros(Pout.shape)+fill_value[0], Pout)

            # k>kmax
            if np.where(ibin>max_ibin)[0].shape[0] > 0:
                if bounds_error:
                    raise Exception("Bounds error: k>kmax in interpolation, k=%s" % str(karg))
                else:
                    Pout = np.where(ibin>max_ibin, np.zeros(Pout.shape)+fill_value[1], Pout)

            if verbose:
                print("kin:\n", kin)
                print("Pin:\n", Pin)
                print("karg:\n", karg)
                print("Pout:\n", Pout)
            return Pout
        
        if verbose:
            print("Test manual_Pk_k_bins interpolator")
            print("Pin-interpolator(kin):\n", Pin-interpolator(kin))
            print("isclose:\n", 
                  np.isclose(Pin, interpolator(kin), 
                             rtol=0.05, 
                             atol=0.05*np.mean(Pin[np.where(~np.isnan(Pin))[0]]**2)**0.5,
                             equal_nan=True))
        if False:
            # ok on 64^3 but sometimes crashes 512^3 runs b/c of nan differences at high k
            assert np.allclose(Pin, interpolator(kin), 
                               rtol=0.05, 
                               atol=0.05*np.mean(Pin[np.where(~np.isnan(Pin))[0]]**2)**0.5,
                               equal_nan=True)
        if verbose:
            print("OK")
            print("test interpolator:", interpolator(kin))

        return interpolator


    elif kind == 'manual_Pk_k_mu_bins':

        check_Pk_is_2d(Pk)

        # get los and other attrs
        Pk0 = Pk[Pk.keys()[0]]
        los0 = Pk0.bstat.power.attrs['los']
        Nmu0 = Pk0.Nmu
        Nk0 = Pk0.Nk

        # Check kin and Pin have right shape
        assert kin.shape == (Nk0*Nmu0,)
        assert Pin.shape == (Nk0*Nmu0,)

        edges = Pk0.bstat.power.edges
        print('edges:', edges)

        # setup edges
        # see project_to_basis in https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/algorithms/fftpower.html#FFTPower
        kedges = edges['k']
        muedges = edges['mu']
        Nk = len(kedges) - 1
        Nmu = len(muedges) - 1

        assert Nk == Nk0
        assert Nmu == Nmu0

        # for indexing to be correct, first mu bin has to start at 0. Since       
        assert muedges[0] == 0.0
        assert muedges[-1] == 1.0
        assert kedges[0] > 0.0
        assert kedges[0] < 2.0*np.pi/L

        assert Pk0.k.shape == (Nk*Nmu,)

        def interpolator(karg, muarg):
            """
            Function that interpolates Pin(kin) to karg, muarg.
            Use same binning as what is used to get P(k,mu) in 2d FFTPower code.

            Parameters
            ----------
            karg : np.ndarray, (N,)
            muarg : np.ndarray, (N,)
            """
            k_indices = np.digitize(karg, kedges)
            mu_indices = np.digitize(np.abs(muarg), muedges)

            # Have mu_indices[0]=0.0. Since bins are inclusive on left edge, will
            # get mu_indices=1..Nmu+1. But we want 0..Nmu, so subtract 1
            mu_indices -= 1

            # When mu==1, assign to last bin, ie. this is right-inclusive.
            # Then we get mu_indices=0..Nmu-1 which is what we want.
            ww = np.where(np.abs(muarg)>=muedges[-1])[0]
            mu_indices[ww] = Nmu-1

            # Also, first k edge is < 2pi/L so never get k_index=0, so subtract 1.
            k_indices -= 1

            print('k_indices:', k_indices)
            print('mu_indices:', mu_indices)

            # Want to get Pin at indices k_indices, mu_indices.
            # Problem: Pin is (Nk*Nmu,) array so need to convert 2d to 1d index.
            #multi_index = np.ravel_multi_index([k_indices, mu_indices], (Nk,Nmu))
            #multi_index = mu_indices*Nk + k_indices
            multi_index = k_indices*Nmu + mu_indices
            print('multi_index:', multi_index)

            # TEST: interp Pk0 instead of Pin
            return Pk0.k[multi_index]
            #return Pin[multi_index]

        # some quick tests of interpolator
        print('Nk, Nmu, Nk*Nmu=', Nk, Nmu, Nk*Nmu)

        karr = np.array([0.003, 0.01, 0.02,0.03])
        muarr = np.array([0.0,0.99, 1.0, -1.0])
        print('karr=', karr)
        print('muarr=', muarr)
        interpolator(karr, muarr)


        print('k2d:', Pk0.k2d[:3,:3])
        print('mu2d:', Pk0.mu2d[:3,:3])
        print('P2d:', Pk0.P2d[:3,:3])

        ik,imu = 1,2
        print('TEST ik, imu=', ik, imu)
        print('k2d=%g' % Pk0.k2d[ik,imu])
        print('mu2d=%g' % Pk0.mu2d[ik,imu])
        print('P2d=%g' % Pk0.P2d[ik,imu])
        xindex = np.ravel_multi_index([np.array([ik]), np.array([imu])], (Nk,Nmu))
        print('xindex=%d' % xindex)
        print('k=%g' % Pk0.k[xindex])
        print('mu=%g' % Pk0.mu[xindex])
        print('P=%g' % Pk0.P[xindex].real)
        print('interp(k)=%g' % interpolator(np.array([Pk0.k2d[ik,imu]]), np.array([Pk0.mu2d[ik,imu]])))

        raise Exception('todo: check interpolator works')

    else:
        raise Exception("invalid kind %s" % str(kind))


def check_Pk_is_2d(Pk):
    # check Pk is 2d and always has some los etc
    Pk0 = Pk[Pk.keys()[0]]
    los = Pk0.bstat.power.attrs['los']
    Nmu = Pk0.Nmu
    Nk = Pk0.Nk

    for key, Pki in Pk.items():
        assert type(Pki) == MeasuredPower2D
        assert Pki.bstat.power.attrs['mode'] == '2d'
        assert Pki.bstat.power.attrs['los'] == los
        assert Pki.bstat.power.shape == (Nk, Nmu)




def round_float2int_arr(x):
    """round float to nearest int"""
    return np.where(x>=0.0, (x+0.5).astype('int'), (x-0.5).astype('int'))

    
def main():
    pass
        
if __name__ == '__main__':
    sys.exit(main())

