#!/usr/bin/env python
#
# Marcel Schmittfull 2017 (mschmittfull@gmail.com)
#
# Python script for initial condition reconstruction transfer functions.
#

from __future__ import print_function,division

import numpy as np
from scipy import interpolate as interp

# MS packages
from lsstools import interpolation_utils

def safe_divide(numerator, denomi):
    return np.where(denomi==0, 0*numerator, numerator/denomi)



def highlevel_get_interp_filters_minimizing_sqerror(
        sources=None, target=None, Pk=None,
        interp_kind=None, bounds_error=False,
        Pkinfo=None):

    # construct matrices needed to get trf fcns
    (kvec, sources_X_target, sources_rms, sources_correl) = construct_matrices_needed_to_get_min_sqerror_trf_fcns(
        sources=sources, target=target, Pk=Pk)

    # actually compute the transfer functions
    interp_trf_fcn_tuple = get_interp_filters_minimizing_sqerror(
        kvec, sources_X_target=sources_X_target, sources_rms=sources_rms, 
        sources_correl=sources_correl,
        interp_kind=interp_kind, bounds_error=bounds_error,
        Pkinfo=Pkinfo)

    return interp_trf_fcn_tuple


def construct_matrices_needed_to_get_min_sqerror_trf_fcns(
        sources=None, target=None, Pk=None, verbose=True):
    """
    Input
    -----
    sources : list of strings, representing ids of source fields.
    target : string, representing id of target field.
    Pk : Measured power spectra between all fields.
    """
    kvec = Pk[(target,target)].k
    Nk = kvec.shape[0]
    assert type(sources) == list
    Nsources = len(sources)

    # construct rms and correl matrix etc 
    sources_X_target = np.zeros((Nsources,Nk)) + np.nan
    sources_rms = np.zeros((Nsources,Nk)) + np.nan
    for isource, source in enumerate(sources):
        # <s_i,f>
        sources_X_target[isource,:] = Pk[(source,target)].P
        # sqrt(<s_i^2>)
        sources_rms[isource,:] = np.sqrt(Pk[(source,source)].P)

    # <s_is_j>/sqrt{<s_i^2><s_j^2>}
    sources_correl = np.zeros((Nsources,Nsources,Nk)) + np.nan
    for isource, source in enumerate(sources):
        for isource2, source2 in enumerate(sources):
            if isource2 == isource:
                # diagonal correl is 1
                sources_correl[isource,isource2,:] = np.ones(Nk,dtype='float')
            else:
                # off-diagonal correl
                sources_correl[isource,isource2,:] = np.where( 
                    Pk[(source,source2)].P == 0.,
                    np.zeros(Pk[(source,source2)].P.shape),
                    Pk[(source,source2)].P / (sources_rms[isource,:]*sources_rms[isource2,:]) )

    # enforce exact symmetry of correl mat
    for isource in range(Nsources):
        for isource2 in range(isource+1,Nsources):
            sources_correl[isource,isource2,:] = sources_correl[isource2,isource,:]

    # print first few correl entries
    if verbose:
        if sources_correl.shape[2] >= 3:
            for ik in range(3):
                print("k, correl_mat:")
                print(kvec[ik])
                print(sources_correl[:,:,ik])
                
            
    return kvec, sources_X_target, sources_rms, sources_correl



def get_filters_minimizing_sqerror_at_fixed_k(
        sources_X_target_vec=None, sources_rms_vec=None, sources_correl_mat=None):
    """
    For n source fields s_i, get n transfer functions t_i such that the
    squared error relative to a target field f is minimized, i.e. get t_i
    such that

      < |\sum_i t_i s_i - f|^2 >

    is minimized. These transfer functions are

      t_i = S_ii^{-1/2} (C^{-1})_ij S_jj^{-1/2} <s_j f>

    where 

      sources_X_target_vec = <s_j f>,
      sources_rms_vec = S_ii^{1/2} = sqrt{<s_i^2>},
      sources_correl_mat = C_ij = <s_is_j>/sqrt{<s_i^2><s_j^2>}.
    """
    # check input data has correct shape
    assert type(sources_X_target_vec) == np.ndarray
    assert type(sources_rms_vec) == np.ndarray
    assert type(sources_correl_mat) == np.ndarray
    Nsources = sources_X_target_vec.shape[0]
    #print("Get transfer functions for %d source fields matched to target field" % Nsources)
    assert sources_X_target_vec.shape == (Nsources,)
    assert sources_rms_vec.shape == (Nsources,)
    assert sources_correl_mat.shape == (Nsources, Nsources)

    # compute transfer functions t_i, return as (Nsources,) vector
    inv_sources_rms_vec = safe_divide(np.ones(Nsources), sources_rms_vec)
    # np.where(sources_rms_vec==0., np.zeros(Nsources)+np.nan, 1.0/sources_rms_vec)
    return ( inv_sources_rms_vec 
             * np.linalg.solve(sources_correl_mat, sources_X_target_vec*inv_sources_rms_vec) )


def get_filters_minimizing_sqerror(
        sources_X_target=None, sources_rms=None, sources_correl=None):
    """
    For Nsources source fields s_i(k), get Nsources transfer functions t_i(k) such that the
    squared error relative to a target field f(k) is minimized, i.e. get t_i(k)
    such that

      < |\sum_i t_i(k) s_i(k) - f(k)|^2 >

    is minimized. See get_filters_minimizing_sqerror_at_fixed_k for details.

    Input
    -----
    sources_X_target : (Nsources,Nk) array representing <s_j(k) f(k)>
    sources_rms : (Nsources,Nk) array representing sqrt{<s_i^2(k)>}
    sources_correl : (Nsources,Nsources,Nk) array C_ij(k) = <s_is_j>/sqrt{<s_i^2><s_j^2>}.
    
    Returns
    -------
    filters_isource_jk : (Nsources,Nk) array representing filter_{i_source}(j_k)
    """
    Nsources = sources_X_target.shape[0]
    Nk = sources_X_target.shape[1]
    print("Compute transfer functions for %d sources for %d k-values" % (
        Nsources, Nk))
    filters_isource_ik = np.zeros((Nsources,Nk)) + np.nan
    for ik in xrange(Nk):
        filters_isource_ik[:,ik] = get_filters_minimizing_sqerror_at_fixed_k(
            sources_X_target_vec=sources_X_target[:,ik], 
            sources_rms_vec=sources_rms[:,ik],
            sources_correl_mat=sources_correl[:,:,ik])
    return filters_isource_ik


def get_interp_filters_minimizing_sqerror(
        kvec=None, sources_X_target=None, sources_rms=None, sources_correl=None,
        interp_kind=None, bounds_error=None, Pkinfo=None):
    """
    For arbitrary numbe of sources, compute transfer functions using
    get_interp_filters_minimizing_sqerror.

    Input
    -----
    kvec : (Nk,) array representing discrete k values
    sources_X_target : (Nsources,Nk) array representing <s_j(k) f(k)>
    sources_rms : (Nsources,Nk) array representing sqrt{<s_i^2(k)>}
    sources_correl : (Nsources,Nsources,Nk) array C_ij(k) = <s_is_j>/sqrt{<s_i^2><s_j^2>}.
    
    Returns
    -------
    Nsources-tuple containing (interp_t1(k), interp_t2(k), ..., interp_t_Nsources(k))
    """
    print("get_interp_filters_minimizing_sqerror...")
    # check input
    Nsources = sources_X_target.shape[0]
    Nk = kvec.shape[0]
    assert sources_X_target.shape == (Nsources,Nk)
    assert sources_rms.shape == (Nsources,Nk)
    assert sources_correl.shape == (Nsources,Nsources,Nk)
    
    # compute filters
    filters_isource_ik = get_filters_minimizing_sqerror(
        sources_X_target=sources_X_target, sources_rms=sources_rms,
        sources_correl=sources_correl)

    # create Nsources-tuple, filled with None
    interp_t_list = [None for i in range(Nsources)]

    # create interpolators
    for isource in range(Nsources):
        this_tk = filters_isource_ik[isource,:]
        if interp_kind == 'manual_Pk_k_bins':
            interp_t_list[isource] = interpolation_utils.interp1d_manual_k_binning(
                kvec, this_tk,
                kind=interp_kind,
                fill_value=(this_tk[0], this_tk[-1]),
                bounds_error=bounds_error,
                Ngrid=Pkinfo['Ngrid'], L=Pkinfo['boxsize'],
                k_bin_width=Pkinfo['k_bin_width']
                )
        else:
            print("TODO: better use interp_kind=manual_Pk_k_bins")
            interp_t_list[isource] = interp.interp1d(
                kvec, this_tk,
                kind=interp_kind,
                fill_value=(this_tk[0], this_tk[-1]),
                bounds_error=bounds_error)
    print("DONE: get_interp_filters_minimizing_sqerror")
    return tuple(interp_t_list)
    
    
def get_a1_a2_filters(Pks, a2_from_deltalin_minus_a1_noO2_delta_chi_div=False):
    """
    Get a1(k) and a2(k), using correct cross-terms <delta_chi,delta_chi^[2]>.
    """
    r12sq = safe_divide(
        (Pks[('delta_chi_div','delta_chi_div_2ndorder')][1])**2,
        (Pks[('delta_chi_div','delta_chi_div')][1]
         *Pks[('delta_chi_div_2ndorder','delta_chi_div_2ndorder')][1]))

    ratio_01_11 = safe_divide(
        Pks[('deltalin_unsmoothed','delta_chi_div')][1],
        Pks[('delta_chi_div','delta_chi_div')][1])
    ratio_02_22 = safe_divide(
        Pks[('deltalin_unsmoothed','delta_chi_div_2ndorder')][1],
        Pks[('delta_chi_div_2ndorder','delta_chi_div_2ndorder')][1])
    ratio_12_11 = safe_divide(
        Pks[('delta_chi_div','delta_chi_div_2ndorder')][1],
        Pks[('delta_chi_div','delta_chi_div')][1])
    ratio_12_22 = safe_divide(
        Pks[('delta_chi_div','delta_chi_div_2ndorder')][1],
        Pks[('delta_chi_div_2ndorder','delta_chi_div_2ndorder')][1])

    # a1
    a1tilde = ( ratio_01_11 - ratio_02_22*ratio_12_11) / (1.0-r12sq)

    # a2
    if not a2_from_deltalin_minus_a1_noO2_delta_chi_div:
        a2tilde = ( ratio_02_22 - ratio_01_11*ratio_12_22) / (1.0-r12sq)
    else:
        # compute a2 from deltalin_minus_a1_noO2_delta_chi_div
        a2tilde = (safe_divide(Pks[('deltalin_minus_a1_noO2_delta_chi_div','delta_chi_div_2ndorder')][1],
                               Pks[('delta_chi_div_2ndorder','delta_chi_div_2ndorder')][1])
                    / (1.0-r12sq))

    return a1tilde, a2tilde


def get_general_a1_a2_filters(
        Pks, 
        field1=None, field2=None, target_field=None,
        a2_from_target_field_minus_a1_noO2_field1=False,
        target_field_minus_a1_noO2_field1=None):
    """
    Compute transfer functions a1 and a2 such that 
    
      combined_field = a1*field1 + a2*field2

    is as close as possible to target_field, i.e. compute a1 and a2 such that
    <(combined_field - target_field)^2> is minimized.
    """
    r12sq = safe_divide(
        (Pks[(field1,field2)][1])**2,
        (Pks[(field1,field1)][1]
         *Pks[(field2,field2)][1]))

    # '0' stands for target field
    ratio_01_11 = safe_divide(
        Pks[(target_field,field1)][1],
        Pks[(field1,field1)][1])
    ratio_02_22 = safe_divide(
        Pks[(target_field,field2)][1],
        Pks[(field2,field2)][1])
    ratio_12_11 = safe_divide(
        Pks[(field1,field2)][1],
        Pks[(field1,field1)][1])
    ratio_12_22 = safe_divide(
        Pks[(field1,field2)][1],
        Pks[(field2,field2)][1])

    # a1
    a1tilde = ( ratio_01_11 - ratio_02_22*ratio_12_11) / (1.0-r12sq)

    # a2
    if not a2_from_target_field_minus_a1_noO2_field1:
        a2tilde = ( ratio_02_22 - ratio_01_11*ratio_12_22) / (1.0-r12sq)
    else:
        # compute a2 from target_field - a1_noO2*field1
        a2tilde = (safe_divide(
            Pks[(target_field_minus_a1_noO2_field1,field2)][1],
            Pks[(field2,field2)][1])
            / (1.0-r12sq))

    return a1tilde, a2tilde

def get_interp_general_a1_a2(
        Pks,
        field1=None, field2=None, target_field=None,
        a2_from_target_field_minus_a1_noO2_field1=False,
        target_field_minus_a1_noO2_field1=None,
        interp_kind=None,bounds_error=False):
    """
    Interpolator for general a1 and a2 filters.
    """
    a1,a2 = get_general_a1_a2_filters(
        Pks, field1=field1, field2=field2, target_field=target_field,
        a2_from_target_field_minus_a1_noO2_field1=a2_from_target_field_minus_a1_noO2_field1,
        target_field_minus_a1_noO2_field1=target_field_minus_a1_noO2_field1)

    # interpolate
    kest = Pks[(field1,field1)][0]
    print("TODO: better implement trf fcns using manual_Pk_k_bins interpolation")
    interp_a1 = interp.interp1d(
        kest, a1,
        kind=interp_kind,
        fill_value=(a1[0],a1[-1]), 
        bounds_error=bounds_error)
    interp_a2 = interp.interp1d(
        kest, a2,
        kind=interp_kind,
        fill_value=(a2[0],a2[-1]),
        bounds_error=bounds_error)

    return interp_a1, interp_a2



def get_interp_a1_a2(Pks, a2_from_deltalin_minus_a1_noO2_delta_chi_div=False,
                     interp_kind=None,bounds_error=False):
    # get a1, a2
    a1,a2 = get_a1_a2_filters(Pks, 
        a2_from_deltalin_minus_a1_noO2_delta_chi_div=a2_from_deltalin_minus_a1_noO2_delta_chi_div)

    # interpolate
    kest = Pks[('deltalin_unsmoothed','deltalin_unsmoothed')][0]
    print("TODO: better implement trf fcns using manual_Pk_k_bins interpolation")
    interp_a1 = interp.interp1d(
        kest, a1,
        kind=interp_kind,
        fill_value=(a1[0],a1[-1]), 
        bounds_error=bounds_error)
    interp_a2 = interp.interp1d(
        kest, a2,
        kind=interp_kind,
        fill_value=(a2[0],a2[-1]),
        bounds_error=bounds_error)

    return interp_a1, interp_a2



def get_a1bar_noO2(Pks, target_field='deltalin_unsmoothed', in_field='delta_chi_div'):
    """
    Compute bar t_1(k) = <delta_0 delta_chi> / <delta_chi delta_chi>,
    which is the weight in absense of any 2nd order correction.
    """
    a1bar_noO2 = safe_divide(Pks[(target_field,in_field)][1],
                             Pks[(in_field,in_field)][1])
    return a1bar_noO2


def get_interp_a1bar_noO2(Pks, target_field='deltalin_unsmoothed', in_field='delta_chi_div',
                          interp_kind=None, bounds_error=False):
    """
    Compute interpolator for a1bar_noO2.
    """
    a1bar_noO2 = get_a1bar_noO2(Pks, target_field=target_field, in_field=in_field)
    kest = Pks[(in_field,in_field)][0]
    print("TODO: implement using manual_Pk_k_bins interpolation")
    interp_a1bar_noO2 = interp.interp1d(kest, a1bar_noO2,
                                     fill_value=(a1bar_noO2[0],a1bar_noO2[-1]), 
                                     kind=interp_kind, bounds_error=bounds_error)
    return interp_a1bar_noO2
