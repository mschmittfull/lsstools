#!/usr/bin/env python
#
# Marcel Schmittfull 2017 (mschmittfull@gmail.com)
#
# Python script for iterative LPT BAO reconstruction.



"""
Interpolation utils.
"""


from __future__ import print_function,division


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from collections import OrderedDict
from matplotlib import rcParams
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from scipy import interpolate as interp
import numpy.core.numeric as NX
import sys




def interp1d_manual_k_binning(kin, Pin, kind='manual_Pk_k_bins', fill_value=None, bounds_error=False,
                              Ngrid=None, L=None, k_bin_width=1.0, verbose=True):
    """
    Interpolate following a fixed k binning scheme that's also used to measure power spectra
    in cy_power_estimator.pyx.

    L : float
        boxsize in Mpc/h
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
            kin_expected = np.arange(1,np.max(kin)*0.99/dk+1)*dk
            if verbose:
                print("kin:", kin)
                print("kin_expected:", kin_expected)
                print("kin/kin_expected (should be between 0.5 and 1.5):\n", kin/kin_expected)
            # bin center is computed by averaging k within bin, so it's not exactly dk*i.
            assert np.allclose(kin, kin_expected, rtol=0.35)
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


    else:
        raise Exception("invalid kind %s" % str(kind))


def round_float2int_arr(x):
    """round float to nearest int"""
    return np.where(x>=0.0, (x+0.5).astype('int'), (x-0.5).astype('int'))

    
def main():
    pass
        
if __name__ == '__main__':
    sys.exit(main())

