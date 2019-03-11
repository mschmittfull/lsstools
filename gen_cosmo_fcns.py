#!/usr/bin/env python

from __future__ import print_function,division

import numpy as np
import cPickle
from scipy import interpolate
from collections import OrderedDict


def generate_calc_Da(test_plot=False, N_integration=10000, cosmo=None, verbose=True):
    """
    Return the function calc_Da, which takes a as argument
    and returns D(a).
    """
    if verbose:
        print("Compute D(a)")
    a = np.linspace(1.0e-4, 1.0, N_integration)
    H_over_H0 = np.sqrt( cosmo.Om_r/a**4 + cosmo.Om_m/a**3 
                         + cosmo.Om_L + cosmo.Om_K/a**2 )
    Da = np.zeros(a.shape)
    # compute the integral
    for imax, aa in enumerate(a):
        Da[imax] = np.trapz( 
            1.0/(a[:imax+1]*H_over_H0[:imax+1])**3, a[:imax+1] )
    # Prefactors
    Da = Da * 5./2.*cosmo.Om_m * H_over_H0
    if verbose:
        print("Got D(a)")
    
    def calc_Da(aeval):
        return np.interp(aeval,a,Da)

    # Plot
    if test_plot:
        import matplotlib.pyplot as plt
        print("Making growth.pdf")
        fig, ax = plt.subplots(1,1)
        ax.plot(a,calc_Da(a),'b-')
        ax.plot(a,a,'b:')
        ax.set_xlabel('$a$')
        ax.set_ylabel('$D(a)$')
        plt.tight_layout()
        plt.savefig("growth.pdf")
        #plt.show()

    return calc_Da


def generate_calc_chi(test_plot=False, N_integration=10000, cosmo=None):
    """
    Return the function calc_chiMpc_a, which takes a as argument
    and returns chi(a) in Mpc.
    """
    print("Compute chi(a)")
    # This must be linear in a (assume this for gradient below!)
    a = np.linspace(1.0e-4, 1.0, N_integration)
    H_over_H0 = np.sqrt( cosmo.Om_r/a**4 + cosmo.Om_m/a**3 
                         + cosmo.Om_L + cosmo.Om_K/a**2 )
    chi_a = np.zeros(a.shape)
    # compute the integral to get H0*chi(a) = int_a^1 da/(a^2 H(a)/H0)
    for imin, aa in enumerate(a):
        chi_a[imin] = np.trapz( 
            1.0/(a[imin:]**2*H_over_H0[imin:]), a[imin:] )
    # Prefactors: chi = [H0*chi]/H0
    chi_a /= (100.*cosmo.h0)  # this is chi in units of 1/(km/s/Mpc)=Mpc/(km/s)
    # Multiply by c in km/s to get chi in Mpc
    chi_a *= 2.99792458e5
    print("Got chi(a)")
    
    def calc_chiMpc_a(aeval):
        """
        Calculate chi(a) in Mpc.
        """
        #if type(aeval)==list:
        #    aeval = np.array(aeval)
        return np.interp(aeval,a,chi_a)

    # Also compute derivative dchi/da
    # TODO: could also get this easier from 1/(a**2 H(a))
    grad_a = np.gradient(chi_a, a[1]-a[0]) 
    def calc_delchiMpc_dela_a(aeval):
        return np.interp(aeval,a,grad_a)

    from scipy.optimize import root
    def calc_a_chiMpc(chiMpc_eval):
        """
        Invert chi(a) to get a(chi). Assumes chi in Mpc.
        """
        if type(chiMpc_eval)==np.ndarray:
            aout = np.zeros(chiMpc_eval.shape) + np.nan
            #print("Chi eval:", chiMpc_eval)
            for cc, this_chiMpc in enumerate(chiMpc_eval):
                #print("this chi", cc, this_chiMpc, calc_a_chiMpc(this_chiMpc))
                aout[cc] = calc_a_chiMpc(this_chiMpc)
            return aout
        else:
            def rootfcn(a):
                return calc_chiMpc_a(a)-chiMpc_eval
            soln = root(rootfcn, 0.5)
            return soln.x[0]

    
    # Plot
    if test_plot:
        import matplotlib.pyplot as plt
        print("Making chi.pdf")
        fig, axlst = plt.subplots(1,2,figsize=(16,8),sharey=True)
        for i,zmax in enumerate([20,1100]):
            z = 1./a-1
            ax = axlst[i]
            ax.plot(z, calc_chiMpc_a(a), 'b-', label="$\chi$")
            # matter domination in flat universe (Dodelson Eq. 2.43)
            ax.plot(z, 2./(100.*cosmo.h0/2.9979e5)*(1.-a**0.5), 'b:', 
                    label="$\\frac{2}{H_0} (1-a^{1/2})$, matter dom.")
            ax.plot(z, 0.*z + 2./(100.*cosmo.h0/2.9979e5), 'r:', 
                    label="$\\frac{2}{H_0}$")
            ax.set_xlabel('$z$')
            ax.set_xlim((0,zmax))
        axlst[0].set_ylabel('$\chi(z)$ [Mpc]')
        axlst[-1].legend(loc="best")
        plt.tight_layout()
        plt.savefig("chi_a.pdf")
        #plt.show()

        # plot d chi/ d a
        fig, ax = plt.subplots(1,1)
        atmp = np.linspace(1.0e-4,1.0,333)
        ax.plot(atmp, calc_delchiMpc_dela_a(atmp))
        ax.set_ylabel('$d\chi/d a$')
        ax.set_xlabel('$a$')
        plt.tight_layout()
        plt.savefig("dchi_da.pdf")


    return calc_chiMpc_a, calc_delchiMpc_dela_a, calc_a_chiMpc

