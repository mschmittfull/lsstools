from __future__ import print_function, division

import numpy as np
import os
from collections import OrderedDict
from scipy import interpolate as interp
import numpy.core.numeric as NX
import sys

from lsstools.MeasuredPower import MeasuredPower1D, MeasuredPower2D


def interp1d_manual_k_binning(kin,
                              Pin,
                              kind='manual_Pk_k_bins',
                              fill_value=None,
                              bounds_error=False,
                              Ngrid=None,
                              L=None,
                              k_bin_width=1.0,
                              verbose=False,
                              Pkref=None):
    """
    Interpolate following a fixed k binning scheme that's also used to measure power spectra
    in cy_power_estimator.pyx.

    Parameters
    ----------
    kind : string
        Use 'manual_Pk_k_bins' for 1d power, or 'manual_Pk_k_mu_bins' for 2d power.

    L : float
        boxsize in Mpc/h

    kin, Pin: numpy.ndarray, (Nk*Nmu,)
        These are interpolated. Defined at k,mu bin central values.

    Pkref : MeasuredPower1D or MeasuredPower2D.
        This is used to get options of the measured power spectrum corresponding to
        Pin, e.g. Nk, Nmu, los, etc. (Note that Pin is ndarray so can't infer from that.)
        Does not use Pkref.power.k, Pkref.power.power etc.
    """
    # check args
    if (fill_value is None) and (not bounds_error):
        raise Exception("Must provide fill_value if bounds_error=False")
    if Ngrid is None:
        raise Exception("Must provide Ngrid")
    if L is None:
        raise Exception("Must provide L")

    if kind == 'manual_Pk_k_bins':

        check_Pk_is_1d(Pkref)

        dk = 2.0 * np.pi / float(L)

        # check that kin has all k bins
        if k_bin_width == 1.:
            # 18 Jan 2019: somehow need 0.99 factor for nbodykit 0.3 to get last k bin right.
            #kin_expected = np.arange(1,np.max(kin)*0.99/dk+1)*dk
            # 16 Mar 2019: Fix expected k bins to match nbodykit for larger Ngrid
            kin_expected = np.arange(1, kin.shape[0] + 1) * dk

            if verbose:
                print("kin:", kin)
                print("kin_expected:", kin_expected)
                print("kin/kin_expected (should be between 0.5 and 1.5):\n",
                      kin / kin_expected)

            # bin center is computed by averaging k within bin, so it's not exactly dk*i.
            if not np.allclose(kin, kin_expected, rtol=0.35):
                print("kin:", kin)
                print("kin_expected:", kin_expected)
                print("kin/kin_expected (should be between 0.5 and 1.5):\n",
                      kin / kin_expected)
                raise Exception('Found issue with k bins when interpolating')

        else:
            raise Exception("k_bin_width=%s not implemented yet" %
                            str(k_bin_width))

        def interpolator(karg):
            """
            Function that interpolates Pin from kin to karg.
            """
            ibin = round_float2int_arr(karg / (dk * k_bin_width))
            # first bin is dropped
            ibin -= 1

            # k's between kmin and max
            max_ibin = Pin.shape[0] - 1
            Pout = np.where((ibin >= 0) & (ibin <= max_ibin),
                            Pin[ibin % (max_ibin + 1)],
                            np.zeros(ibin.shape) + np.nan)

            # k<kmin
            if np.where(ibin < 0)[0].shape[0] > 0:
                if bounds_error:
                    raise Exception(
                        "Bounds error: k<kmin in interpolation, k=%s" %
                        str(karg))
                else:
                    Pout = np.where(ibin < 0,
                                    np.zeros(Pout.shape) + fill_value[0], Pout)

            # k>kmax
            if np.where(ibin > max_ibin)[0].shape[0] > 0:
                if bounds_error:
                    raise Exception(
                        "Bounds error: k>kmax in interpolation, k=%s" %
                        str(karg))
                else:
                    Pout = np.where(ibin > max_ibin,
                                    np.zeros(Pout.shape) + fill_value[1], Pout)

            if verbose:
                print("kin:\n", kin)
                print("Pin:\n", Pin)
                print("karg:\n", karg)
                print("Pout:\n", Pout)
            return Pout

        if verbose:
            print("Test manual_Pk_k_bins interpolator")
            print("Pin-interpolator(kin):\n", Pin - interpolator(kin))
            print("isclose:\n",
                  np.isclose(Pin,
                             interpolator(kin),
                             rtol=0.05,
                             atol=0.05 *
                             np.mean(Pin[np.where(~np.isnan(Pin))[0]]**2)**0.5,
                             equal_nan=True))
        if False:
            # ok on 64^3 but sometimes crashes 512^3 runs b/c of nan differences at high k
            assert np.allclose(
                Pin,
                interpolator(kin),
                rtol=0.05,
                atol=0.05 * np.mean(Pin[np.where(~np.isnan(Pin))[0]]**2)**0.5,
                equal_nan=True)
        if verbose:
            print("OK")
            print("test interpolator:", interpolator(kin))

    elif kind == 'manual_Pk_k_mu_bins':

        check_Pk_is_2d(Pkref)

        # get los and other attrs
        los0 = Pkref.bstat.power.attrs['los']
        Nmu0 = Pkref.Nmu
        Nk0 = Pkref.Nk

        edges = Pkref.bstat.power.edges
        print('edges:', edges)

        # setup edges
        # see project_to_basis in https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/algorithms/fftpower.html#FFTPower
        kedges = edges['k']
        muedges = edges['mu']
        Nk = len(kedges) - 1
        Nmu = len(muedges) - 1

        assert Nk == Nk0
        assert Nmu == Nmu0

        # For indexing to be correct, first mu bin has to start at 0.
        assert muedges[0] == 0.0
        assert muedges[-1] == 1.0
        assert kedges[0] > 0.0
        assert kedges[0] < 2.0 * np.pi / L  # will drop first bin b/c of this

        assert Pkref.k.shape == (Nk * Nmu,)

        # Check kin and Pin have right shape and indexing
        assert kin.shape == (Nk * Nmu,)
        assert Pin.shape == (Nk * Nmu,)
        ww = np.where(~np.isnan(kin))
        assert np.allclose(kin[ww], Pkref.k[ww])

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

            # nbodykit uses power[1:-1] at the end to drop stuff <edges[0]
            # and >=edges[-1]. Similarly, digitize returns 0 if mu<edges[0] (never occurs)
            # and Nmu if mu>=edges[-1]. Subtract one so we get we get mu_indices=0..Nmu,
            # and assign mu=1 to mu_index=Nmu-1
            mu_indices -= 1
            # When mu==1, assign to last bin, so it is right-inclusive.
            mu_indices[np.isclose(np.abs(muarg), 1.0)] = Nmu - 1
            #mu_indices[mu_indices>Nmu-1] = Nmu-1

            # Same applies to k:
            k_indices -= 1

            # mu>=mumin=0
            assert np.all(mu_indices[~np.isnan(muarg)] >= 0)
            # mu<=mumax=1
            if not np.all(mu_indices[~np.isnan(muarg)] < Nmu):
                print("Found mu>1: ", muarg[mu_indices > Nmu - 1])
                raise Exception('Too large mu')

            # take lowest k bin when karg=0
            #k_indices[karg==0] = 0

            ##print('k_indices:', k_indices)
            #print('mu_indices:', mu_indices)

            #print('edges:', edges)
            #raise Exception('tmp')

            # Want to get Pin at indices k_indices, mu_indices.
            # Problem: Pin is (Nk*Nmu,) array so need to convert 2d to 1d index.
            # Use numpy ravel
            #multi_index = np.ravel_multi_index([k_indices, mu_indices], (Nk,Nmu))
            # Do manually (same result as ravel when 0<=k_indices<=Nk-1 and 0<=mu_indices<=Nmu-1.)
            # Also take modulo max_multi_index to avoid errror when k_indices or mu_indices out of bounds,
            # will handle those cases explicitly later.
            max_multi_index = (Nk - 1) * Nmu + (Nmu - 1)
            multi_index = (k_indices * Nmu + mu_indices) % (max_multi_index + 1)
            #print('multi_index:', multi_index)

            if False:
                # Just for testing: interp Pkref instead of Pin
                Pout = Pkref.P[multi_index]
            else:
                # interp Pin. Note this is wrong when k or mu are out of bounds
                # (will handle below).
                Pout = Pin[multi_index]

            # Handle out of bounds cases

            # k>kmax
            if not np.all(k_indices < Nk):
                if bounds_error:
                    print('too large k: ', karg[k_indices >= Nk])
                    raise Exception(
                        "Bounds error: k>kmax in interpolation, k=%s" %
                        str(karg))
                else:
                    Pout = np.where(k_indices < Nk, Pout,
                                    np.zeros(Pout.shape) + fill_value[1])

            # k<kmin
            if not np.all(k_indices >= 0):
                if bounds_error:
                    print('too small k: ', karg[k_indices < 0])
                    raise Exception(
                        "Bounds error: k<kmin in interpolation, k=%s" %
                        str(karg))
                else:
                    Pout = np.where(k_indices >= 0, Pout,
                                    np.zeros(Pout.shape) + fill_value[0])

            # handle nan input
            Pout = np.where(np.isnan(karg), np.zeros(Pout.shape) + np.nan, Pout)
            Pout = np.where(np.isnan(muarg),
                            np.zeros(Pout.shape) + np.nan, Pout)

            return Pout

        if False:
            # some quick tests of interpolator (interpolate Pkref instead of Pin above)
            print('Nk, Nmu, Nk*Nmu=', Nk, Nmu, Nk * Nmu)

            karr = np.array([0.0188, 0.0189, 0.0314, 0.0315])
            muarr = np.array([0.0, 0.1999, 0.2001, -1.0])
            print('karr=', karr)
            print('muarr=', muarr)
            print('interpolator=', interpolator(karr, muarr))

            print('k2d:', Pkref.k2d[:3, :3])
            print('mu2d:', Pkref.mu2d[:3, :3])
            print('P2d:', Pkref.P2d[:3, :3])

            ik, imu = 17, 3
            print('TEST ik, imu=', ik, imu)
            print('k2d=%g' % Pkref.k2d[ik, imu])
            print('mu2d=%g' % Pkref.mu2d[ik, imu])
            print('P2d=%g' % Pkref.P2d[ik, imu])
            xindex = np.ravel_multi_index(
                [np.array([ik]), np.array([imu])], (Nk, Nmu))
            print('xindex=%d' % xindex)
            print('k=%g' % Pkref.k[xindex])
            print('mu=%g' % Pkref.mu[xindex])
            print('P=%g' % Pkref.P[xindex].real)
            print('P=%g' % Pin[xindex].real)

            print('interp(P)=%g' % interpolator(np.array(
                [Pkref.k2d[ik, imu]]), np.array([Pkref.mu2d[ik, imu]])))

            raise Exception('just checking interpolator')

    else:
        raise Exception("invalid kind %s" % str(kind))

    return interpolator


def check_Pk_is_1d(Pkref):
    # check Pkref is 2d
    assert type(Pkref) == MeasuredPower1D
    assert Pkref.bstat.power.attrs['mode'] == '1d'
    assert Pkref.bstat.power.shape == (Pkref.Nk,)


def check_Pk_is_2d(Pkref):
    # check Pkref is 2d
    assert type(Pkref) == MeasuredPower2D
    assert Pkref.bstat.power.attrs['mode'] == '2d'
    assert Pkref.bstat.power.shape == (Pkref.Nk, Pkref.Nmu)


def round_float2int_arr(x):
    """round float to nearest int"""
    return np.where(x >= 0.0, (x + 0.5).astype('int'), (x - 0.5).astype('int'))


def main():
    pass


if __name__ == '__main__':
    sys.exit(main())
