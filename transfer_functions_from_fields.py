from __future__ import print_function,division

import numpy as np
from scipy import interpolate as interp
from collections import OrderedDict
import copy
import sys

from nbodykit.source.mesh.field import FieldMesh

# MS packages
import transfer_functions
import interpolation_utils

def generate_sources_and_get_interp_filters_minimizing_sqerror(
    gridx=None, gridk=None,
    linear_sources=None,
    fixed_linear_sources=[],
    non_orth_linear_sources=[],
    quadratic_sources=None, field_to_smoothen_and_square=None, 
    Rsmooth_for_quadratic_sources=None,
    quadratic_sources2=None, field_to_smoothen_and_square2=None, 
    Rsmooth_for_quadratic_sources2=None,
    sources_for_trf_fcn=None,
    target=None,
    target_spec=None,
    save_bestfit_field=None,
    N_ortho_iter=0,
    orth_method='CholeskyDecomp',
    interp_kind='linear', bounds_error=False,
    Pk_ptcle2grid_deconvolution=None, 
    k_bin_width=1.0, 
    Pk_1d_2d_mode='1d', RSD_poles=None, RSD_Nmu=None, RSD_los=None,
    kmax=None,
    save_target_contris=False, save_cholesky_internals=False):

    """
    - Given target and source fields in k space (in gridk), compute transfer functions.

    - As an intermediate step, we compute power spectra and orthogonalize
    the fields to improve numerical stability.

    - Also compute best combination and store in gridk.G[save_bestfit_field].

    - If sources_for_trf_fcn is None, use sources_for_trf_fcn = combi of all fields in 
      linear_sources and quadratic_sources.
    
    """

    if interp_kind not in ['manual_Pk_k_bins', 'manual_Pk_k_mu_bins']:
        raise Exception("Please use interp_kind=manual_Pk_k_bins or manual_Pk_k_mu_bins")
    
    Pkmeas = None

    # column info
    input_column_infos = gridk.column_infos.copy()
    
    # linear sources
    if False:
        for s in linear_sources+non_orth_linear_sources+fixed_linear_sources:
            if not gridk.has_column(s):
                raise Exception("Linear source %s not on grid" % s)
        
    # list with 'squared' (quadratic or cubic) columns
    sqcols = []

    # smoothen with some R=XX
    # BUG until 31 dec 2017: always smoothed delta_h rather than field_to_smoothen_and_square
    if field_to_smoothen_and_square is not None:
        gridk.append_column('%s_smoothed'%field_to_smoothen_and_square, gridk.G[field_to_smoothen_and_square])
        gridk.apply_smoothing('%s_smoothed'%field_to_smoothen_and_square, mode='Gaussian', 
                              R=Rsmooth_for_quadratic_sources,
                              kmax=kmax)

    # compute quadratic sources
    for source in quadratic_sources:
        # compute delta^2 or tidal_s2 or tidal_G2
        sqcol = '%s_%s'%(field_to_smoothen_and_square,source)
        sqcols += [sqcol]
        gridk.append_column(
            sqcol,
            gridk.calc_quadratic_field(
                basefield='%s_smoothed'%field_to_smoothen_and_square,
                quadfield=source))
        
    # don't need smoothed density any more
    if gridk.has_column('%s_smoothed'%field_to_smoothen_and_square):
        gridk.drop_column('%s_smoothed'%field_to_smoothen_and_square)


    ## 2nd set of quadratic sources
    if field_to_smoothen_and_square2 is not None:
        gridk.append_column('%s_smoothed'%field_to_smoothen_and_square2, gridk.G[field_to_smoothen_and_square2])
        gridk.apply_smoothing('%s_smoothed'%field_to_smoothen_and_square2, mode='Gaussian', 
                              R=Rsmooth_for_quadratic_sources2,
                              kmax=kmax)
    
    for source in quadratic_sources2:
        if source not in ['growth','tidal_s2', 'tidal_G2', 'F2']:
            raise Exception("Invalid quadratic source %s" % str(source))
        # compute delta^2 or tidal_s2 or tidal_G2
        sqcol = 'source2_%s_%s'%(field_to_smoothen_and_square2,source)
        sqcols += [sqcol]
        gridk.append_column(
            sqcol,
            gridk.calc_quadratic_field(
                basefield='%s_smoothed'%field_to_smoothen_and_square2,
                quadfield=source))
        
    # don't need smoothed density any more
    if gridk.has_column('%s_smoothed'%field_to_smoothen_and_square2):
        gridk.drop_column('%s_smoothed'%field_to_smoothen_and_square2)

        
    # construct all original sources for which to compute trf fcns: 
    print("sources_for_trf_fcn, save_bestfit_field:", sources_for_trf_fcn, save_bestfit_field)
    if sources_for_trf_fcn is None:
        # Default: linear + quadratic sources + non_orth_linear_sources
        sources = linear_sources[:]
        for s in sqcols:
            sources.append(s)
        for s in non_orth_linear_sources:
            # it's important to append this as last sources so other fields get not orthogonalized w.r.t to non_orth fields.
            sources.append(s)
    else:
        # use sources supplied by arg
        sources = sources_for_trf_fcn
        for s in sources:
            if not gridk.has_column(s):
                # source is not on grid, so have to compute it.
                if s == 'deltalin+deltalin_F2':
                    gridk.append_column(
                        'deltalin+deltalin_F2', 
                        FieldMesh(gridk.G['deltalin'].compute(mode='complex')
                                  + gridk.G['deltalin_F2'].compute(mode='complex')))
                    if 'deltalin_F2' not in sources:
                        gridk.drop_column('deltalin_F2')

                elif s == '[delta_h_WEIGHT_M1]_MINUS_[delta_h]':
                    gridk.append_column(
                        '[delta_h_WEIGHT_M1]_MINUS_[delta_h]',
                        FieldMesh(
                            gridk.G['delta_h_WEIGHT_M1'].compute(mode='complex')
                            - gridk.G['delta_h'].compute(mode='complex')))
                    if ('delta_h_WEIGHT_M1' not in sources) and ('delta_h_WEIGHT_M1'!=target):
                        gridk.drop_column('delta_h_WEIGHT_M1')
                
                else:
                    raise Exception("Do not know how to generate source %s" % s)

        # ensure that non_orthogonal sources are at the end of the list (important for 
        # orthgonalization procedure)
        N_non_orth_linear_sources = len(non_orth_linear_sources)
        if N_non_orth_linear_sources > 0:
            if not (non_orth_linear_sources == sources[-N_non_orth_linear_sources:]):
                raise Exception("Non_orth_linear_sources must be listed last in sources_for_trf_fcn")
        
        #print("sources:", sources)
        #raise Exception("todo")

    print("sources:", sources)

    # Also collect fixed sources that have no trf fcns.
    # todo: also allow quadratic fixed sources
    fixed_sources = fixed_linear_sources[:]
    
    
    # re-label trf fcn sources so we can orthogonalize them easier
    osources = []
    iortho = 0
    initial_source_of_osource = OrderedDict()
    for isource, source in enumerate(sources):
        if source in non_orth_linear_sources:
            osource = 'NON_ORTH s^%d_%d'%(iortho,isource)
        else:
            osource = 'ORTH s^%d_%d'%(iortho,isource)
        initial_source_of_osource[osource] = source
        gridk.append_column(osource, gridk.G[source])
        osources.append(osource)
        # drop from memory if not needed any more
        if source in fixed_linear_sources:
            pass
        elif (target_spec is not None) and (source in target_spec.linear_target_contris):
            pass
        else:
            gridk.drop_column(source)

    print("initial_source_of_osource:", initial_source_of_osource)
    Nsources = len(osources)

    # #####################################################################################################
    # Compute orthogonalized sources
    # #####################################################################################################
    # modifies gridk, Pkmeas
    osources, Pkmeas, ortho_rot_matrix_sources, orth_internals_sources = gridk.compute_orthogonalized_fields(
        N_ortho_iter=N_ortho_iter, 
        orth_method=orth_method,
        all_in_fields=osources,
        orth_prefix='ORTH s', 
        non_orth_prefix='NON_ORTH s',
        Pkmeas=Pkmeas, 
        Pk_ptcle2grid_deconvolution=Pk_ptcle2grid_deconvolution,
        k_bin_width=k_bin_width,
        Pk_1d_2d_mode=Pk_1d_2d_mode, RSD_poles=RSD_poles, RSD_Nmu=RSD_Nmu,
        RSD_los=RSD_los,
        interp_kind=interp_kind,
        delete_original_fields=True)

    

    # #####################################################################################################
    # If target_spec is given, compute composite target field as specified by target_spec
    # #####################################################################################################

    if target_spec is not None:
        # compute composite target field. 
        # check target_spec is ok

        if Pk_1d_2d_mode != '1d':
            raise Exception('target_spec only implemented for 1d power so far')

        if target_spec.minimization_objective not in [
                '(T*target-T*sources)^2/(T*target)^2', '(target0+T*other_targets-T*sources)^2']:
            raise Exception("Invalid minimization_objective %s" % 
                            str(target_spec.minimization_objective))

        if target_spec.minimization_objective in [
                '(T*target-T*sources)^2/(T*target)^2',
                '(target0+T*other_targets-T*sources)^2']:
            # Find optimal combination of target_contris that minimizes (T*target-T*sources)^2/(T*target)^2
            # or (T*target-T*sources)^2
            # See notes around 12 May 2018.
            if target_spec.minimization_objective == '(T*target-T*sources)^2/(T*target)^2':
                if N_ortho_iter == 0:
                    raise Exception("Must orthogonalize all sources (maths assumes this). Please set N_ortho_iter=1.")
                if non_orth_linear_sources not in [None,[]]:
                    raise Exception("Must orthogonalize all sources (maths assumes this).")
                if fixed_linear_sources not in [None,[]]:
                    #raise Exception("Must not have fixed_linear_sources here (maths assumes that all sources have trf fcn)")
                    print("Warning: Target weights are not optimal if model has fixed_linear sources, but proceed anyways.")


            # power spectra needed for normalization
            if target_spec.target_norm['type'].startswith('MatchPower'):
                tmp_cols = [target_spec.target_norm['Pk_to_match_id1']]
                if target_spec.target_norm['Pk_to_match_id1'] != target_spec.target_norm['Pk_to_match_id2']:
                    tmp_cols.append(target_spec.target_norm['Pk_to_match_id2'])
                if target_spec.target_norm['type'] == 'MatchPowerAndLowKLimit':
                    for mycol in [target_spec.target_norm['LowK_Pnorm_to_match_id1'],
                                  target_spec.target_norm['LowK_Pnorm_to_match_id1']]:
                        if mycol not in tmp_cols:
                            tmp_cols.append(mycol)
                Pkmeas = gridk.calc_all_power_spectra(
                    columns=tmp_cols,
                    Pk_ptcle2grid_deconvolution=Pk_ptcle2grid_deconvolution,
                    k_bin_width=k_bin_width,
                    mode=Pk_1d_2d_mode, poles=RSD_poles, Nmu=RSD_Nmu,
                    line_of_sight=RSD_los,
                    Pkmeas=Pkmeas)

                    
            # First orthogonalize the target_contris among themselves using Gram-Schmidt/Cholesky (assumed in maths).
            # 'tc' stands for target_contribution, 'otc' stands for orthogonalized target contribution.
            # Init 0th iteration of target orthogonalization.
            iortho = 0
            orth_target_contris = []
            initial_field_of_otc = OrderedDict()
            for itc, tc in enumerate(target_spec.linear_target_contris):
                otc = 'ORTH tc^%d_%d' % (iortho,itc)
                initial_field_of_otc[otc] = tc
                orth_target_contris.append(otc)
                gridk.append_column(otc, gridk.G[tc])
                gridk.drop_column(tc)
            print('initial_field_of_otc:', initial_field_of_otc)

            # Actually do the orthogonalization of target contris
            # modifies gridk, Pkmeas
            orth_target_contris, Pkmeas, ortho_rot_matrix_targets, orth_internals_targets = (
                gridk.compute_orthogonalized_fields(
                    N_ortho_iter=1,
                    orth_method='CholeskyDecomp',
                    all_in_fields=orth_target_contris,
                    orth_prefix='ORTH tc',
                    non_orth_prefix='NON_ORTH tc',
                    Pkmeas=Pkmeas, 
                    Pk_ptcle2grid_deconvolution=Pk_ptcle2grid_deconvolution,
                    delete_original_fields=True))
           
            N_target_contris = len(orth_target_contris)
            kvec = Pkmeas[Pkmeas.keys()[0]][0]
            Nk = kvec.shape[0]

            Amat = None
            Amat_lambdas = None
            Amat_Qmat = None

            # Get alpha_i weights for target field
            
            if target_spec.minimization_objective in [
                    '(target0+T*other_targets-T*sources)^2', '(T*target-T*sources)^2/(T*sources)^2']:
                # Minimize mean squared model error MSE.
                # Use transfer_functions.py to get alpha_i. 
                # Dividing my sources^2 doesn't affect alphas, so use same code here.

                # Target=target0-fixed_linear_sources.
                TMP_target_minus_fixed_sources = orth_target_contris[0]
                if len(fixed_sources)>0:
                    TMP_target_minus_fixed_sources = '[%s]' % TMP_target_minus_fixed_sources
                    for fs in fixed_sources:
                        TMP_target_minus_fixed_sources = '%s_MINUS_[%s]' % (TMP_target_minus_fixed_sources,fs)
                    print("TMP_target_minus_fixed_sources=%s" % TMP_target_minus_fixed_sources)
                    # compute target-fixed sources on grid
                    gridk.append_column(TMP_target_minus_fixed_sources, gridk.G[orth_target_contris[0]])
                    for fs in fixed_sources:
                        #gridk.G[TMP_target_minus_fixed_sources] -= gridk.G[fs]
                        gridk.G[TMP_target_minus_fixed_sources] = FieldMesh(
                            gridk.G[TMP_target_minus_fixed_sources].compute(mode='complex')
                            - gridk.G[fs].compute(mode='complex'))
                        
                # Sources=target_contri1, target_contri2, ... and non-fixed model sources.
                TMP_osources = orth_target_contris[1:] + osources
                print("TMP_osources:", TMP_osources)
                #raise Exception("mydbg")

                # calc power spectra needed to get alpha_i for composite target.
                # include orth_target_contris
                Pkmeas = gridk.calc_all_power_spectra(
                    columns=[TMP_target_minus_fixed_sources]+TMP_osources,
                    Pk_ptcle2grid_deconvolution=Pk_ptcle2grid_deconvolution,
                    k_bin_width=k_bin_width,
                    mode=Pk_1d_2d_mode, poles=RSD_poles, Nmu=RSD_Nmu,
                    line_of_sight=RSD_los,
                    Pkmeas=Pkmeas)

                # get trf fcns
                if Pk_1d_2d_mode == '1d':
                    TMP_interp_tuple_osources = transfer_functions.highlevel_get_interp_filters_minimizing_sqerror(
                        sources=TMP_osources, target=TMP_target_minus_fixed_sources, 
                        Pk=Pkmeas,
                        interp_kind=interp_kind, bounds_error=bounds_error,
                        Pkinfo={'Ngrid': gridk.Ngrid, 'boxsize': gridk.boxsize,
                                'k_bin_width': k_bin_width,
                                'Pk_1d_2d_mode': Pk_1d_2d_mode, 'RSD_poles': RSD_poles,
                                'RSD_Nmu': RSD_Nmu, 'RSD_los': RSD_los,
                                'Pk_ptcle2grid_deconvolution': Pk_ptcle2grid_deconvolution})
                else:
                    raise Exception('not implemented')

                if target_spec.target_norm['type'] in ['alpha0=1','MatchPower','MatchPowerAndLowKLimit']:
                    if Pk_1d_2d_mode != '1d':
                        raise Exception('Not implemented')
                    # Start with alpha0=1, alpha1=T_0(k), alpha2=T_1(k), etc
                    my_interp_alpha_opt = (np.zeros( (N_target_contris,) )).tolist()
                    my_interp_alpha_opt[0] = (lambda myk: 0*myk+1.0)
                    for itc in range(1,N_target_contris):
                        # Had bug here: when minimizing, minimize |target-sources|^2, but treating target
                        # contri as source. So need to multiply trf fcn of target contri by -1 if we want
                        # to add it to target instead of subtracting it.
                        my_interp_alpha_opt[itc] = (lambda myk: -TMP_interp_tuple_osources[itc-1](myk))

                    # normalization
                    interp_alpha_opt = (np.zeros( (N_target_contris,) )).tolist()                    
                    if target_spec.target_norm['type'] == 'alpha0=1':
                        # normalization is already ok, so just copy
                        for itc in range(N_target_contris):
                            interp_alpha_opt[itc] = my_interp_alpha_opt[itc]

                    elif target_spec.target_norm['type'] in ['MatchPower','MatchPowerAndLowKLimit']:
                        # normalize alpha's such that P_target=c*P_MatchPower, where c=1 for MatchPower
                        # and c is some number if MatchPowerAndLowKLimit.
                        if False:
                            # Compute P_target(k) by constructing target field for current alpha's.
                            # Then change normalization of alphas later. (On 64^3 get same result as
                            # sum_i alpha_i^2 P_ii below.)
                            gridk.append_column('TMP_FIELD_FOR_NORM',
                                                0.0*gridk.G[orth_target_contris[0]].compute(mode='complex'))
                            for itc, tc_itc in enumerate(orth_target_contris):
                                #gridk.G['TMP_FIELD_FOR_NORM'] += (
                                #    (my_interp_alpha_opt[itc](gridk.G['ABSK'].data)) * gridk.G[tc_itc])
                                def mult_by_alpha(k3vec, val, itc=itc):
                                    absk = np.sqrt(sum(ki ** 2 for ki in k3vec)) # absk on the mesh
                                    return my_interp_alpha_opt[itc](absk) * val
                                to_add = gridk.G[tc_itc].apply(
                                    mult_by_alpha, mode='complex', kind='wavenumber')
                                gridk.G['TMP_FIELD_FOR_NORM'] = FieldMesh(
                                    gridk.G['TMP_FIELD_FOR_NORM'].compute(mode='complex')
                                    + to_add.compute(mode='complex'))
                                del mult_by_alpha, to_add
                                
                            tmp_Pkmeas = gridk.calc_all_power_spectra(
                                columns=['TMP_FIELD_FOR_NORM'],
                                Pk_ptcle2grid_deconvolution=Pk_ptcle2grid_deconvolution,
                                k_bin_width=k_bin_width,
                                mode=Pk_1d_2d_mode, poles=RSD_poles, Nmu=RSD_Nmu,
                                line_of_sight=RSD_los,
                                Pkmeas=None)
                            tmp_Ptarget = tmp_Pkmeas[('TMP_FIELD_FOR_NORM','TMP_FIELD_FOR_NORM')][1]
                            gridk.drop_column('TMP_FIELD_FOR_NORM')
                        else:
                            # Calc P_target(k)=sum_i alpha_i^2(k) P_ii(k) (assume orth_target_contris are orthogonal).
                            # (On 64^3 get same result as when constructing target at the field level as above).
                            Pkmeas = gridk.calc_all_power_spectra(
                                columns=orth_target_contris,
                                Pk_ptcle2grid_deconvolution=Pk_ptcle2grid_deconvolution,
                                k_bin_width=k_bin_width,
                                mode=Pk_1d_2d_mode, poles=RSD_poles, Nmu=RSD_Nmu,
                                line_of_sight=RSD_los,
                                Pkmeas=Pkmeas)
                            tmp_Ptarget = np.zeros(kvec.shape)
                            for itc, tc in enumerate(orth_target_contris):
                                tmp_Ptarget += Pkmeas[(tc,tc)][1] * (my_interp_alpha_opt[itc](kvec))**2
                        # normfac = sqrt(P_wanted / P_target)
                        tmp_normfac = np.sqrt(
                            Pkmeas[(target_spec.target_norm['Pk_to_match_id1'],target_spec.target_norm['Pk_to_match_id2'])][1]
                            / tmp_Ptarget)

                        if target_spec.target_norm['type'] == 'MatchPowerAndLowKLimit':
                            # Multiply by k-independent constant to fix low-k limit.
                            # Currently, power is Pk_to_match. Multiply by c=LowK_Pnorm_to_match(low k)/Pk_to_match(low k)
                            # so k dependence is same as Pk_to_match, but norm is given by low k limit.
                            my_Pratio = (
                                Pkmeas[(target_spec.target_norm['LowK_Pnorm_to_match_id1'],
                                        target_spec.target_norm['LowK_Pnorm_to_match_id2'])][1]
                                /
                                Pkmeas[(target_spec.target_norm['Pk_to_match_id1'],
                                        target_spec.target_norm['Pk_to_match_id2'])][1] )
                            print("MatchPowerAndLowKLimit Pratio:", my_Pratio)
                            k_indices = target_spec.target_norm['LowK_k_bin_indices']
                            lowk_const = np.mean(my_Pratio[k_indices])
                            print("MatchPowerAndLowKLimit lowk_const (expect 1/b1):", lowk_const)
                            tmp_normfac *= lowk_const
                            print("tmp_normfac:", tmp_normfac)
                            #raise Exception("bla")
                            

                        if Pk_1d_2d_mode != '1d':
                            raise Exception('Not implemented')
                        else:                           
                            interp_tmp_normfac = interpolation_utils.interp1d_manual_k_binning(
                                kvec, tmp_normfac,
                                #kind='manual_Pk_k_bins',
                                kind=interp_kind,
                                fill_value=(tmp_normfac[0], tmp_normfac[-1]),
                                bounds_error=False,
                                Ngrid=gridk.Ngrid, L=gridk.boxsize, k_bin_width=k_bin_width,
                                Pkref=Pkmeas[Pkmeas.keys()[0]])
                            # multiply by tmp_normfac
                            for itc in range(N_target_contris):
                                interp_alpha_opt[itc] = (
                                    lambda myk: interp_tmp_normfac(myk) * my_interp_alpha_opt[itc](myk))

                else:
                    raise Exception("Invalid target_norm %s" % str(target_spec.target_norm['type']))

                
                # also save alpha_opt at kvec so we can easier export it
                alpha_opt = np.zeros( (N_target_contris, Nk) ) + np.nan
                for itc in range(N_target_contris):
                    alpha_opt[itc,:] = interp_alpha_opt[itc](kvec)
                    
                    
            elif target_spec.minimization_objective == '(T*target-T*sources)^2/(T*target)^2':
                # Minimize (T*target-T*sources)^2/(T*target)^2.
                # Maths implicitly assumes that target contris are orthogonal among themselves,
                # and that sources are also orthogonal among themselves.

                # Optional todo: re-factor this so we can easier use it elsewhere if needed
                
                # compute cross-spectra between sources and target_contris
                Pkmeas = gridk.calc_all_power_spectra(
                    columns=osources + orth_target_contris + fixed_linear_sources,
                    Pk_ptcle2grid_deconvolution=Pk_ptcle2grid_deconvolution,
                    k_bin_width=k_bin_width,
                    mode=Pk_1d_2d_mode, poles=RSD_poles, Nmu=RSD_Nmu,
                    line_of_sight=RSD_los,
                    Pkmeas=Pkmeas)
                
                # Compute A matrix: A_ij = delta_ij - sum_\mu^Nsources r_{i\mu} r_{j\mu},
                # where i,j labels target_contris. 'tc' stands for target_contri.
                Amat = np.zeros( (N_target_contris,N_target_contris,Nk) )
                for itc, tc_itc in enumerate(orth_target_contris):
                    for jtc, tc_jtc in enumerate(orth_target_contris):
                        mysum = 0.0
                        for mu in range(Nsources):
                            r_itc_mu = Pkmeas[(tc_itc,osources[mu])][1] / np.sqrt(
                                Pkmeas[(tc_itc,tc_itc)][1] * Pkmeas[(osources[mu],osources[mu])][1] )
                            r_jtc_mu = Pkmeas[(tc_jtc,osources[mu])][1] / np.sqrt(
                                Pkmeas[(tc_jtc,tc_jtc)][1] * Pkmeas[(osources[mu],osources[mu])][1] )
                            mysum += r_itc_mu*r_jtc_mu
                        Amat[itc,jtc,:] = -mysum
                        if itc == jtc:
                            Amat[itc,jtc,:] += 1.0

                # enforce exact symmetry
                for itc in range(N_target_contris):
                    for jtc in range(itc+1,N_target_contris):
                        Amat[itc,jtc,:] = Amat[jtc,itc,:]

                # do eigendecomposition of A at every k
                Amat_lambdas = np.zeros( (N_target_contris, Nk) ) + np.nan
                Amat_Qmat = np.zeros ( (N_target_contris, N_target_contris, Nk) ) + np.nan
                alpha_opt = np.zeros( (N_target_contris, Nk) ) + np.nan
                for ik in range(Nk):
                    tmp_lambdas, tmp_Q = np.linalg.eigh(Amat[:,:,ik])
                    Amat_lambdas[:,ik] = tmp_lambdas
                    Amat_Qmat[:,:,ik] = tmp_Q
                    
                    # get normalization factor for alphas
                    if target_spec.target_norm['type'] == 'alpha0=1':
                        # normalize such that first field is untouched, i.e. alpha0=1 (used always until 30.5.2018)
                        target_normfac = 1.0 / Amat_Qmat[0,0,ik]
                        
                    elif target_spec.target_norm['type'] == 'MatchPower':
                        # normalize such that power spectrum of target agrees with a given fixed power spectrum
                        Pkfix = Pkmeas[(target_spec.target_norm['Pk_to_match_id1'],target_spec.target_norm['Pk_to_match_id2'])][1][ik]
                        sumQsqPii = 0.0
                        for tmp_itarget in range(N_target_contris):
                            sumQsqPii += (Amat_Qmat[tmp_itarget,0,ik]**2 
                                          * Pkmeas[(orth_target_contris[tmp_itarget],orth_target_contris[tmp_itarget])][1][ik])
                        # multiply by sign of Q_{00}(k=0) to get alpha_0(k=0)>0, which is useful (otherwise
                        # may get completely negative field, which is ok, but not easy to explain).
                        target_normfac = np.sqrt(Pkfix/sumQsqPii) * np.sign(Amat_Qmat[0,0,0])
                    else:
                        raise Exception("Invalid target_norm %s" % str(target_spec.target_norm))
                    # compute alpha
                    alpha_opt[:,ik] = Amat_Qmat[:,0,ik] * target_normfac

                # print A, eigenvalues and eigenvectors
                for ik in range(Nk):
                    print("ik=%d (k=%g):" % (ik,kvec[ik]))
                    print("Amat:\n", Amat[:,:,ik])
                    print("lambdas:\n", Amat_lambdas[:,ik])
                    print("Q:\n", Amat_Qmat[:,:,ik])
                    print("alpha^opt:\n", alpha_opt[:,ik])
                for ik in range(Nk):
                    print("ik=%d (k=%g):" % (ik,kvec[ik]))
                    print("alpha^opt:\n", alpha_opt[:,ik])

                # interpolate alpha_opt in k so we can rotate at the field level
                # Not sure if kind=nearest is best.
                interp_alpha_opt = (np.zeros( (N_target_contris,) )).tolist()
                for itc in range(N_target_contris):
                    #     # use nearest interp; used until 6 June 2018
                    #     raise Exception("please use manual_Pk_k_bins interpolation")
                    #     interp_alpha_opt[itc] = interp.interp1d(
                    #         kvec, alpha_opt[itc,:],
                    #         kind='nearest',
                    #         fill_value=(alpha_opt[itc,0], alpha_opt[itc,-1]),
                    #         bounds_error=False)

                    if Pk_1d_2d_mode != '1d':
                        raise Exception('Not implemented')
                    else:
                        # use manual k binning interp which gives good orthogonalization
                        interp_alpha_opt[itc] = interpolation_utils.interp1d_manual_k_binning(
                            kvec, alpha_opt[itc,:],
                            #kind='manual_Pk_k_bins',
                            kind=interp_kind,
                            fill_value=(alpha_opt[itc,0], alpha_opt[itc,-1]),
                            bounds_error=False,
                            Ngrid=gridk.Ngrid, L=gridk.boxsize, k_bin_width=k_bin_width,
                            Pkref=Pkmeas[Pkmeas.keys()[0]])

            # get target field by summing up target contributions weighted by alpha_opt
            # copy column info from input
            gridk.append_column(target_spec.save_bestfit_target_field,
                                FieldMesh(0.0*gridk.G[orth_target_contris[0]].compute(mode='complex')),
                                column_info={'input_column_infos': input_column_infos})
            for itc, tc_itc in enumerate(orth_target_contris):
                print("Add target contri %s to target" % tc_itc)
                #gridk.G[target_spec.save_bestfit_target_field] += (
                #    (interp_alpha_opt[itc](gridk.G['ABSK'].data)) * gridk.G[tc_itc])
                if interp_kind == 'manual_Pk_k_bins':
                    def mult_by_alpha(k3vec, val, itc=itc):
                        absk = np.sqrt(sum(ki ** 2 for ki in k3vec)) # absk on the mesh
                        return interp_alpha_opt[itc](absk) * val
                elif interp_kind == 'manual_Pk_k_mu_bins':
                    # use k, mu
                    raise Exception('todo')
                else:
                    raise Exception('invalid interp_kind')
                to_add = gridk.G[tc_itc].apply(
                    mult_by_alpha, mode='complex', kind='wavenumber')
                gridk.G[target_spec.save_bestfit_target_field] = FieldMesh(
                    gridk.G[target_spec.save_bestfit_target_field].compute(mode='complex')
                    + to_add.compute(mode='complex'))
                del mult_by_alpha, to_add

            # free memory
            for itc, tc_itc in enumerate(orth_target_contris):
                gridk.drop_column(tc_itc)

        else:
            raise Exception("Invalid minimization_objective %s" % 
                            str(target_spec.minimization_objective))
            
                
    # #####################################################################################################
    # If there are fixed_sources, 
    # compute target-fixed_sources b/c want to minimize [(target-fixed_sources)-sum_i T_i source_i]^2
    # #####################################################################################################
    target_minus_fixed_sources = target
    if len(fixed_sources)>0:
        if target_spec is not None:
            print("Warning: Target weights are not optimal if model has fixed_linear sources, but proceed anyways.")
            #raise Exception("Must not use fixed_sources if target_spec is not None")
            if target_spec.minimization_objective == '(T*target-T*sources)^2/(T*sources)^2':
                raise Exception("Fixed sources not implemented if minimizing (T*target-T*sources)^2/(T*sources)^2")
        # get id of target-fixed_sources
        target_minus_fixed_sources = '[%s]' % target
        for fs in fixed_sources:
            target_minus_fixed_sources = '%s_MINUS_[%s]' % (target_minus_fixed_sources, fs)
        print("target_minus_fixed_sources=%s" % target_minus_fixed_sources)
        # compute grid of target-fixed_sources
        gridk.append_column(target_minus_fixed_sources, gridk.G[target])
        for fs in fixed_sources:
            #gridk.G[target_minus_fixed_sources] -= gridk.G[fs]
            gridk.G[target_minus_fixed_sources] = FieldMesh(
                gridk.G[target_minus_fixed_sources].compute(mode='complex')
                - gridk.G[fs].compute(mode='complex'))
    
            
    # ################################################################################
    # Compute transfer functions of orthogonalized source fields. 
    # ################################################################################

    # calc power spectra needed for trf fcns
    if not gridk.has_column('ABSK'):
        gridk.compute_helper_grid('ABSK')
    tmpcols = [target_minus_fixed_sources]+osources+fixed_sources
    # for completeness include power spectra with target (useful for plotting later)
    if target != target_minus_fixed_sources:
        tmpcols.append(target)
    # calc power spectra
    Pkmeas = gridk.calc_all_power_spectra(
        columns=tmpcols,
        Pk_ptcle2grid_deconvolution=Pk_ptcle2grid_deconvolution,
        k_bin_width=k_bin_width,
        mode=Pk_1d_2d_mode, poles=RSD_poles, Nmu=RSD_Nmu,
        line_of_sight=RSD_los,
        Pkmeas=Pkmeas)

    if (target_spec is None) or (target_spec.minimization_objective in [
            '(target0+T*other_targets-T*sources)^2','(T*target-T*sources)^2/(T*target)^2']):
        # get betas of sources by minimizing (target-sources)^2. dividing by target^2 doesn't change that.
        # get trf fcns of orthogonalized fields (and non_orth_linear_sources which are included in osources)
        # Note: if 1d, interp_tuple_osources will be functions of k, otherwise of k, mu.
        interp_tuple_osources = transfer_functions.highlevel_get_interp_filters_minimizing_sqerror(
            sources=osources, target=target_minus_fixed_sources, 
            Pk=Pkmeas,
            interp_kind=interp_kind, bounds_error=bounds_error,
            Pkinfo={'Ngrid': gridk.Ngrid, 'boxsize': gridk.boxsize,
                    'k_bin_width': k_bin_width,
                    'Pk_1d_2d_mode': Pk_1d_2d_mode, 'RSD_poles': 'RSD_poles',
                    'RSD_Nmu': RSD_Nmu, 'RSD_los': RSD_los,
                    'Pk_ptcle2grid_deconvolution': Pk_ptcle2grid_deconvolution})

    elif target_spec.minimization_objective == '(T*target-T*sources)^2/(T*model)^2':
        raise Exception("Implement this using Amat...")
    

    if save_bestfit_field is not None:
        # ################################################################################
        # Compute best estimate of \hat delta_m = fixed_sources + t_1 delta_h + t_2 delta_h^2 + ...,
        # and \hat delta_h = t_1 delta_m + t_2 delta_m^2 + ....
        # Actually sum up orthogonalized fields with correpsonding transfer fcns.
        # ################################################################################

        
        # compute best linear combination of osource fields matching target
        gridk.append_column(save_bestfit_field,
                            FieldMesh(0.0*gridk.G[osources[0]].compute(mode='complex')),
                            column_info={'input_column_infos': input_column_infos})

        # add sources with trf fcns
        for counter, s in enumerate(osources):
            #gridk.G[save_bestfit_field] += (
            #    interp_tuple_osources[counter](gridk.G['ABSK'].data) * gridk.G[s])
            if Pk_1d_2d_mode == '1d':
                def multiply_me(k3vec, val, counter=counter):
                    absk = np.sqrt(sum(ki ** 2 for ki in k3vec)) # absk on the mesh
                    return interp_tuple_osources[counter](absk) * val
            elif Pk_1d_2d_mode == '2d':
                def multiply_me(k3vec, val, counter=counter):
                    absk = (sum(ki ** 2 for ki in k3vec))**0.5 # absk on the mesh
                    # Dont use absk[absk==0]=1 b/c interp does not allow k=1.
                    with np.errstate(invalid='ignore', divide='ignore'):
                        mu = sum(k3vec[i]*RSD_los[i] for i in range(3)) / absk
                    return interp_tuple_osources[counter](absk,mu) * val
            else:
                raise Exception('Invalid Pk_1d_2d_mode %s' % Pk_1d_2d_mode)
            to_add = gridk.G[s].apply(
                multiply_me, mode='complex', kind='wavenumber')
            gridk.G[save_bestfit_field] = FieldMesh(
                gridk.G[save_bestfit_field].compute(mode='complex')
                + to_add.compute(mode='complex'))
            del multiply_me, to_add

        # add fixed sources
        for fs in fixed_sources:
            # gridk.G[save_bestfit_field] += gridk.G[fs]
            gridk.G[save_bestfit_field] = FieldMesh(
                gridk.G[save_bestfit_field].compute(mode='complex')
                + gridk.G[fs].compute(mode='complex'))

            
        
    # ################################################################################
    # Free memory
    # ################################################################################      

    # drop quadratic source fields and orthogonalized source fields
    for tmpcol in sqcols+osources:
        if gridk.has_column(tmpcol):
            gridk.drop_column(tmpcol)

    # drop all fields
    for tmpcol in gridk.G.keys():
        #if tmpcol.startswith('hat') or tmpcol=='ABSK' or tmpcol==save_bestfit_field:
        if tmpcol=='ABSK' or tmpcol==save_bestfit_field:
            continue
        elif (target_spec is not None) and (tmpcol == target_spec.save_bestfit_target_field):
            continue
        else:
            if gridk.has_column(tmpcol):
                gridk.drop_column(tmpcol)




    # ################################################################################
    # Save trf fncs and other info and return them so we can write them to pickle later.
    # ################################################################################
    
    if N_ortho_iter == 0:
        # have not rotated anything, so orthogonalized sources are same as original ones.
        # copy over power spectra in that case so we can easier plot them later.
        for osource1, source1 in initial_source_of_osource.items():
            Pkmeas[(source1,target)] = Pkmeas[(osource1,target)]
            Pkmeas[(target,source1)] = Pkmeas[(target,osource1)]
            for osource2, source2 in initial_source_of_osource.items():
                Pkmeas[(source1,source2)] = Pkmeas[(osource1,osource2)]

    trf_results = OrderedDict()
    trf_results['info'] = {
        'linear_sources': linear_sources, 
        'fixed_linear_sources': fixed_linear_sources,
        'quadratic_sources': quadratic_sources, 
        'field_to_smoothen_and_square': field_to_smoothen_and_square,
        'Rsmooth_for_quadratic_sources': Rsmooth_for_quadratic_sources,
        'quadratic_sources2': quadratic_sources2, 
        'field_to_smoothen_and_square2': field_to_smoothen_and_square2,
        'Rsmooth_for_quadratic_sources2': Rsmooth_for_quadratic_sources2,
        'target': target,
        'save_bestfit_field': save_bestfit_field,
        'N_ortho_iter': N_ortho_iter,
        'orth_method': orth_method,
        'sources': sources, 'osources': osources,
        'initial_source_of_osource': initial_source_of_osource,
        'interp_kind': interp_kind, 'bounds_error': bounds_error,
        'Pk_ptcle2grid_deconvolution': Pk_ptcle2grid_deconvolution, 
        'kmax': kmax
        }

    # eval orth trf fcns at kvec
    kvec = Pkmeas[Pkmeas.keys()[0]].k
    Nk = kvec.shape[0]
    if Pk_1d_2d_mode == '2d':
        muvec = Pkmeas[Pkmeas.keys()[0]].mu
    else:
        muvec = None
    Tk_osources = np.zeros( (Nsources, Nk) )
    for isource in range(Nsources):
        if Pk_1d_2d_mode == '1d':
            Tk_osources[isource,:] = interp_tuple_osources[isource](kvec)
        elif Pk_1d_2d_mode == '2d':
            Tk_osources[isource,:] = interp_tuple_osources[isource](kvec, muvec)
        else:
            raise Exception('Invalid Pk_1d_2d_mode %s' % Pk_1d_2d_mode)
        
    trf_fcns_orth_fields = OrderedDict()
    trf_fcns_orth_fields[target] = OrderedDict()
    for osource, interp_otf in zip(osources,interp_tuple_osources):
        if Pk_1d_2d_mode == '1d':
            trf_fcns_orth_fields[target][osource] = [kvec, interp_otf(kvec)]
        elif Pk_1d_2d_mode == '2d':
            trf_fcns_orth_fields[target][osource] = [(kvec, muvec), interp_otf(kvec,muvec)]

    #print("kvec:", kvec.shape, kvec)
    #raise Exception("dbg kvec")

        
    trf_results['kvec'] = kvec
    trf_results['muvec'] = muvec
    trf_results['Pkmeas'] = Pkmeas
    trf_results['trf_fcns_orth_fields'] = trf_fcns_orth_fields
    
    
    if orth_method == 'CholeskyDecomp':
        if N_ortho_iter in [0,None]:
            # orig and orth fields are the same
            trf_results['trf_fcns_orig_fields'] = {target: OrderedDict()}
            for osource in trf_fcns_orth_fields[target].keys():
                trf_results['trf_fcns_orig_fields'][target][initial_source_of_osource[osource]] = (
                    trf_fcns_orth_fields[target][osource])
        
        elif N_ortho_iter == 1:
            # not getting trf fcns of orig fields in this case
            trf_results['trf_fcns_orig_fields'] = {}
            if save_cholesky_internals:
                trf_results['CholeskyInternals'] = orth_internals_sources
            # {
            #     'Smat': Smat, 'inv_Lmat': inv_Lmat, 'Cmat': Cmat,
            #     'Mrotmat': Mrotmat,
            #     'inv_sqrt_Sii_vec': inv_sqrt_Sii_vec}

        else:
            raise Exception("Invalid N_ortho_iter %s" % str(N_ortho_iter))


    if orth_method == 'EigenDecomp':
        # If using EigenDecomp, rotate trf fcns back to original (non-orthogonalized) fields.
        if ortho_rot_matrix_sources is None:
            assert N_ortho_iter in [None, 0]
            interp_tuple = copy.deepcopy(interp_tuple_osources)
            #raise Exception("ok")
        else:
            # rotate trf fcns back so they refer to non-orthogonalized fields
            Tk_sources = np.zeros( (Nsources, Nk) )
            for isource in range(Nsources):
                # multiply by rotation matrix from left
                for jsource in range(Nsources):
                    Tk_sources[isource,:] += ortho_rot_matrix_sources[isource,jsource,:] * Tk_osources[jsource,:]

            # convert Tk_sources to interpolation tuple
            interp_tuple = []
            for isource in range(Nsources):
                raise Exception("please use manual_Pk_k_bins interpolation")
                interp_tuple.append(
                    interp.interp1d(kvec, Tk_sources[isource,:],
                                    kind=interp_kind, bounds_error=bounds_error,
                                    fill_value=(Tk_sources[isource,0],Tk_sources[isource,-1])) )
            interp_tuple = tuple(interp_tuple)    

        # save rotated trf fcns of original non-orth in dict so we can plot them later using pickle
        trf_fcns_orig_fields = OrderedDict()
        trf_fcns_orig_fields[target] = OrderedDict()
        kmeas = Pkmeas[(target, target)][0]
        for source, interp_tf in zip(sources,interp_tuple):
            trf_fcns_orig_fields[target][source] = [kmeas, interp_tf(kmeas)]
        trf_results['trf_fcns_orig_fields'] = trf_fcns_orig_fields
            
        #print("internal trf fcns:", trf_fcns)
                

    # save info about target contris
    if target_spec is not None:
        if save_target_contris:
            trf_results['target_contris'] = {'target_spec': target_spec,
                                             'orth_target_contris': orth_target_contris,
                                             'Amat': Amat,
                                             'Amat_lambdas': Amat_lambdas,
                                             'Amat_Qmat': Amat_Qmat,
                                             'alpha_opt': alpha_opt}
        

    return trf_results



