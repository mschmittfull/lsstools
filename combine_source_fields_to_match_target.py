#!/usr/bin/env python

# Uses nbodykit 0.3.


from __future__ import print_function,division
import numpy as np
import os
from collections import OrderedDict, namedtuple
from scipy import interpolate as interp
import random
import glob
import sys
import json
from shutil import rmtree


from nbodykit.source.mesh.field import FieldMesh


# MS packages
from Grid import RealGrid, ComplexGrid
from cosmo_model import CosmoModel
from gen_cosmo_fcns import generate_calc_Da
import paint_utils
import transfer_functions_from_fields



   
    
def actually_calc_Pks(opts, paths, delete_cache=True,
                      only_exec_trf_specs_subset=None):
    """
    Parameters
    ----------
    only_exec_trf_specs_subset : None or list.
        None: Execute all trf specs.
        List: Execute only trf specs in the list

    TODO: should only use kwargs, not opts dict...
    """

    
    # init some things
    cache_fnames = []

    gridx = None
    gridk = None

    gridk_cache_fname = os.path.join(paths['cache_path'], 'gridk_qPk_my_cache.bigfile')
    cache_fnames.append(gridk_cache_fname)
    cached_columns = []
    if os.path.exists(gridk_cache_fname):
        rmtree(gridk_cache_fname)
    
    
    
    # ################################################################################
    # Compute density of all input catalogs needed for trf fcns
    # ################################################################################

    cat_infos = OrderedDict()
    for cat_id, cat_opts in opts['cats'].items():

        # TODO: could update densities_needed_for_trf_fcns if only_exec_trf_specs_subset is not None.
        
        if cat_id not in opts['densities_needed_for_trf_fcns']:
            print("Warning: Not reading %s b/c not needed for trf fcns" % cat_id)
            continue
        
        # default args for painting
        default_paint_kwargs = {
            'gridx': gridx, 'gridk': gridk, 'cache_path': paths['cache_path'],
            'Ngrid': opts['Ngrid'], 'boxsize': opts['boxsize'],
            'grid_ptcle2grid_deconvolution': opts['grid_ptcle2grid_deconvolution'],
            'f_log_growth': opts.get('f_log_growth', None),
            'kmax': opts['kmax']}

        # nbodykit config
        # TODO: Specify "Painter" dict in cat_opts so we don't need to copy by hand here
        config_dict = {
            'Nmesh': opts['Ngrid'],
            'output': os.path.join(paths['cache_path'], 'test_paint_baoshift'),
            'DataSource': {
                'plugin': 'Subsample',
                'path': get_full_fname(paths['in_path'], cat_opts['in_fname'], opts['ssseed'])},
            'Painter': {
                'plugin': 'DefaultPainter',
                'normalize': True, # not used when paint_mode='momentum_divergence' 
                'setMean': 0.0,
                'paint_mode': cat_opts.get('paint_mode', 'overdensity'),
                'velocity_column': cat_opts.get('velocity_column', None),
                'fill_empty_cells': cat_opts.get('fill_empty_cells', None),
                'randseed_for_fill_empty_cells': cat_opts.get('randseed_for_fill_empty_cells', None),
                'raise_exception_if_too_many_empty_cells': cat_opts.get(
                    'raise_exception_if_too_many_empty_cells', True)
                }
            }

        if cat_opts['weight_ptcles_by'] is not None:
            config_dict['Painter']['weight'] = cat_opts['weight_ptcles_by']


        # paint delta and save it in gridk.G[cat_id] 
        if gridx is None and gridk is None:
            # return gridx and gridk
            gridx, gridk = paint_utils.paint_cat_to_gridk(config_dict, column=cat_id,
                                                        **default_paint_kwargs)
        else:
            # modify existing gridx and gridk
            paint_utils.paint_cat_to_gridk(config_dict, column=cat_id,
                                                        **default_paint_kwargs)
        
        print("\n\nINFO %s:\n"%cat_id)
        print(gridk.column_infos[cat_id])

        # save info in a more accessible way
        cat_infos[cat_id] = {
            'simple': simplify_cat_info(
                gridk.column_infos[cat_id], weight_ptcles_by=cat_opts['weight_ptcles_by']),
            'full': gridk.column_infos[cat_id]}
        print("\n\nsimple cat info:")
        print(cat_infos[cat_id]['simple'])
        print("")

        
        # ################################################################################
        # Cache grid to disk, drop from memory, and reload below
        # ################################################################################
        cached_columns += gridk.G.keys() #dtype.names
        gridk.save_to_bigfile(gridk_cache_fname, gridk.G.keys(), 
                           new_dataset_for_each_column=True,
                           overwrite_file_if_exists=False)
        for c in gridk.G.keys():
            if c != 'ABSK':
                gridk.drop_column(c)





    for ext_grid_id, ext_grid_spec in opts['ext_grids_to_load'].items():
    
        # ######################################################################################
        # Get linear density or other density saved in grid on disk.
        # ######################################################################################
        
        #ext_grid_spec['scale_factor']
        print("Attempt reading ext_grid %s:\n%s" % (ext_grid_id, str(ext_grid_spec)))

        
        # Factor to linearly rescale deltalin to redshift of the snapshot
        if ext_grid_spec['scale_factor'] != opts['sim_scale_factor']:
            print("Linearly rescale %s from a=%g to a=%g" % (
                ext_grid_id, ext_grid_spec['scale_factor'],opts['sim_scale_factor']))
            cosmo = CosmoModel(**opts['cosmo_params'])
            calc_Da = generate_calc_Da(cosmo=cosmo)
            rescalefac = (calc_Da(opts['sim_scale_factor']) 
                          / calc_Da(ext_grid_spec['scale_factor']))
            del cosmo
        else:
            rescalefac = 1.0

        if ext_grid_spec.has_key('additional_rescale_factor'):
            rescalefac *= ext_grid_spec['additional_rescale_factor']

        # default args for painting
        default_paint_kwargs = {
            'gridx': gridx, 'gridk': gridk, 'cache_path': paths['cache_path'],
            'Ngrid': opts['Ngrid'], 'boxsize': opts['boxsize'],
            'grid_ptcle2grid_deconvolution': opts['grid_ptcle2grid_deconvolution'],
            'kmax': opts['kmax']}
            
        if ext_grid_spec['file_format'] == 'nbkit_BigFileGrid':
            # nbodykit kit config
            config_dict = {
                'Nmesh': opts['Ngrid'],
                'output': os.path.join(paths['cache_path'],'test_paint_baoshift'),
                'DataSource': {
                    'plugin': 'BigFileGrid',
                    'path': os.path.join(paths['in_path'],ext_grid_spec['dir']),
                    'dataset': ext_grid_spec['dataset_name']},
                'Painter': {
                    'plugin': 'DefaultPainter',
                    'normalize': ext_grid_spec['nbkit_normalize'],
                    'setMean': ext_grid_spec['nbkit_setMean']}
                }

            # get density from external file and rescale it
            assert 'rescalefac' not in default_paint_kwargs
            if gridx is None and gridk is None:
                gridx, gridk = paint_utils.paint_cat_to_gridk(
                    config_dict, column=ext_grid_id, rescalefac=rescalefac,
                    **default_paint_kwargs)
            else:
                # modify existing gridx and gridk
                paint_utils.paint_cat_to_gridk(
                    config_dict, column=ext_grid_id, rescalefac=rescalefac,
                    **default_paint_kwargs)
                
        elif ext_grid_spec['file_format'] == 'rhox_ms_binary':
            # get rho(x)
            gridx.append_column_from_rhox_ms_binary(
                fname=os.path.join(os.path.join(paths['in_path'],ext_grid_spec['dir']),ext_grid_spec['fname']),
                column=ext_grid_id)
            # convert to delta(x) = (rho(x)-rhobar)/rhobar
            rhobar = np.mean(gridx.G[ext_grid_id])
            print("rhobar:", rhobar)
            gridx.G[ext_grid_id] = (gridx.G[ext_grid_id] - rhobar) / rhobar
            gridx.print_summary_stats(ext_grid_id)

            # scale from redshift of ICs file (usually z=0) to redshift of snapshot
            print("Rescale %s by rescalefac=%g" % (ext_grid_id, rescalefac))
            gridx.G[ext_grid_id] *= rescalefac
            gridx.print_summary_stats(ext_grid_id)

            gridk.append_column(ext_grid_id, gridx.fft_x2k(ext_grid_id, drop_column=True))


        else:
            raise Exception("Iinvalid file_format: %s" % str(ext_grid_id['file_format']))

        # ################################################################################
        # Cache grid to disk, drop from memory, and reload below
        # ################################################################################
        cached_columns += gridk.G.keys()
        print("try caching these columns:", gridk.G.keys())
        gridk.save_to_bigfile(gridk_cache_fname, gridk.G.keys(), 
                           new_dataset_for_each_column=True,
                           overwrite_file_if_exists=False)
        #raise Exception("mytmp")
        for c in gridk.G.keys():
            if c != 'ABSK':
                gridk.drop_column(c)

    
    for shifted_ext_grid_id, shifted_ext_grid_spec in opts.get('shifted_ext_grids',{}).items():
    
        # ######################################################################################
        # Shift densities if needed
        # ######################################################################################

        raise Exception("Not implemented yet")
        
        print("Try to get shifted grid %s" % shifted_ext_grid_id)
        
        # load grid_to_shift and displacement_source from cache
        grid_to_shift_id = shifted_ext_grid_spec['grid_to_shift']
        displacement_source_id = shifted_ext_grid_spec['displacement_source']
        gridk.append_columns_from_bigfile(gridk_cache_fname, 
                                       [grid_to_shift_id, displacement_source_id], 
                                       replace_existing_col=True)

        # relabel for convenience, so we can apply smoothing etc
        # (important in case grid_to_shift_id==displacement_source_id but only one shall be smoothed)
        gridk.append_column('tmp_grid_to_shift', gridk.G[grid_to_shift_id])
        gridk.append_column('tmp_displacement_source', gridk.G[displacement_source_id])
        gridk.drop_column(grid_to_shift_id)
        if gridk.has_column(displacement_source_id):
            gridk.drop_column(displacement_source_id)
        del grid_to_shift_id,displacement_source_id

        # apply smoothing if desired
        if shifted_ext_grid_spec['grid_to_shift_smoothing'] is not None:
            gridk.apply_smoothing('tmp_grid_to_shift',
                                  mode=shifted_ext_grid_spec['grid_to_shift_smoothing']['mode'],
                                  R=shifted_ext_grid_spec['grid_to_shift_smoothing']['R'],
                                  kmax=opts['kmax'])
        if shifted_ext_grid_spec['displacement_source_smoothing'] is not None:
            gridk.apply_smoothing('tmp_displacement_source',
                                  mode=shifted_ext_grid_spec['displacement_source_smoothing']['mode'],
                                  R=shifted_ext_grid_spec['displacement_source_smoothing']['R'],
                                  kmax=opts['kmax'])

        # to shift a density, create uniform catalog with mass weights 1+delta and then shift that
            
        raise Exception("continue shifting field")

            
                


    # init dicts where to save power specra of best-fit fields and trf fcn results
    Pkmeas = OrderedDict()        
    trf_results = OrderedDict()

    
    # ################################################################################
    # For each trf fcn spec... 
    # ################################################################################

    if only_exec_trf_specs_subset is None:
        exec_trf_specs = opts['trf_specs']
    else:
        exec_trf_specs = only_exec_trf_specs_subset
    
    for trf_spec in exec_trf_specs:

        # dbg: empty all grids
        if False:
            if gridx.G is not None:
                for tmpcol in gridx.G.keys():
                    gridx.drop_column(tmpcol)
            if gridk.G is not None:
                for tmpcol in gridk.G.keys():
                    gridk.drop_column(tmpcol)

        for c in cached_columns:
            if c != 'ABSK':
                gridk.drop_column(c)

        
        
        # ################################################################################
        # Compute quadratic fields, trf fcn matched to target field, and best-fit-field,
        # stored in gridk.G[trf_spec.save_bestfit_field]
        # ################################################################################

        # load linear sources, fixed_linear_sources, field_to_square and target_field
        # TODO: include MatchPower field
        tmp_cols = trf_spec.linear_sources[:]
        for col in getattr(trf_spec, 'fixed_linear_sources', []):
            if (col not in tmp_cols) and (col in cached_columns):
                tmp_cols.append(col)
        for col in getattr(trf_spec, 'non_orth_linear_sources', []):
            if (col not in tmp_cols) and (col in cached_columns):
                tmp_cols.append(col)
        for col in [trf_spec.field_to_smoothen_and_square, trf_spec.target_field,
                    trf_spec.field_to_smoothen_and_square2]:
            if (col is not None) and (col not in tmp_cols) and (col in cached_columns):
                tmp_cols.append(col)
        if hasattr(trf_spec, 'target_spec'):
            for col in getattr(trf_spec.target_spec, 'linear_target_contris', []):
                if (col not in tmp_cols) and (col in cached_columns):
                    tmp_cols.append(col)
            if trf_spec.target_spec is not None:
                if trf_spec.target_spec.target_norm['type'] == 'MatchPower':
                    for mykey in ['Pk_to_match_id1', 'Pk_to_match_id2']:
                        col = trf_spec.target_spec.target_norm[mykey]
                        if (col not in tmp_cols) and (col in cached_columns):
                            tmp_cols.append(col)
                elif trf_spec.target_spec.target_norm['type'] == 'MatchPowerAndLowKLimit':
                    for mykey in ['Pk_to_match_id1', 'Pk_to_match_id2',
                                  'LowK_Pnorm_to_match_id1', 'LowK_Pnorm_to_match_id2']:
                        col = trf_spec.target_spec.target_norm[mykey]
                        if (col not in tmp_cols) and (col in cached_columns):
                            tmp_cols.append(col)
        print("tmp_cols:", tmp_cols)
        #raise Exception("mytmp")
                
        
        gridk.append_columns_from_bigfile(gridk_cache_fname, tmp_cols,
                                       replace_existing_col=True)

        # Generate all sources and compute transfer functions at the field level.
        # Also compute best combination and store in gridk.G[trf_spec.save_bestfit_field].
        # TODO: too many options, should refactor...
        trf_results_here = (
            transfer_functions_from_fields.generate_sources_and_get_interp_filters_minimizing_sqerror(
            gridx=gridx, gridk=gridk,
            linear_sources=trf_spec.linear_sources,
            fixed_linear_sources=getattr(trf_spec, 'fixed_linear_sources', []),
            non_orth_linear_sources=getattr(trf_spec, 'non_orth_linear_sources', []),
            quadratic_sources=trf_spec.quadratic_sources,
            field_to_smoothen_and_square=trf_spec.field_to_smoothen_and_square,
            quadratic_sources2=trf_spec.quadratic_sources2,
            field_to_smoothen_and_square2=trf_spec.field_to_smoothen_and_square2,
            Rsmooth_for_quadratic_sources=opts['Rsmooth_for_quadratic_sources'],
            Rsmooth_for_quadratic_sources2=opts['Rsmooth_for_quadratic_sources2'],
            sources_for_trf_fcn=trf_spec.sources_for_trf_fcn,
            target=trf_spec.target_field,
            target_spec=getattr(trf_spec, 'target_spec', None),
            save_bestfit_field=trf_spec.save_bestfit_field,
            N_ortho_iter=opts['N_ortho_iter_for_trf_fcns'],
            orth_method=opts['orth_method_for_trf_fcns'],
            interp_kind=opts['interp_kind_for_trf_fcns'],
            bounds_error=False,
            Pk_ptcle2grid_deconvolution=opts['Pk_ptcle2grid_deconvolution'], 
            k_bin_width=opts['k_bin_width'],
            Pk_1d_2d_mode=opts.get('Pk_1d_2d_mode','1d'), RSD_poles=opts.get('RSD_poles', None), 
            RSD_Nmu=opts.get('RSD_Nmu', None), RSD_los=opts.get('RSD_los',None),
            kmax=opts['kmax']))
      
        # save all trf fcns in dict
        trf_results[str(trf_spec)] = trf_results_here
        del trf_results_here

        # save bestfit field in cache (needed when modeling mass-weighted halos)
        gridk.save_to_bigfile(gridk_cache_fname, [trf_spec.save_bestfit_field],
                           new_dataset_for_each_column=True,
                           overwrite_file_if_exists=False)

        # also save target if needed
        if getattr(trf_spec, 'target_spec', None) is not None:
            gridk.save_to_bigfile(gridk_cache_fname, [trf_spec.target_spec.save_bestfit_target_field],
                               new_dataset_for_each_column=True,
                               overwrite_file_if_exists=False)
        
        
        # ############################################################
        # Calculate auto- and cross spectra of target and best-fit-field.
        # ############################################################
        cols_Pk = [trf_spec.save_bestfit_field,trf_spec.target_field]
        #cols_Pk += opts['cats'].keys()  # not needed
        #cols_Pk += opts['densities_needed_for_trf_fcns']  # not needed?
        for ext_grid_id in opts['ext_grids_to_load'].keys():
            cols_Pk.append(ext_grid_id)
        gridk.append_columns_from_bigfile(gridk_cache_fname, cols_Pk,
                                       replace_existing_col=False)

        # compute power spectra and include them in Pkmeas dict
        Pkmeas = gridk.calc_all_power_spectra(
            columns=cols_Pk,
            Pk_ptcle2grid_deconvolution=opts['Pk_ptcle2grid_deconvolution'],
            k_bin_width=opts['k_bin_width'],
            mode=opts.get('Pk_1d_2d_mode','1d'), poles=opts.get('RSD_poles',None), 
            Nmu=opts.get('RSD_Nmu',None),
            line_of_sight=opts.get('RSD_los',None),
            Pkmeas=Pkmeas)
        print("cols_Pk:\n", cols_Pk)
        
        
        # ############################################################
        # Calculate residual (target-best_fit_field) on grid and get its power spectrum
        # ############################################################
        residual_key = '[%s]_MINUS_[%s]' % (trf_spec.save_bestfit_field,trf_spec.target_field)
        gridk.append_column(
            residual_key,
            FieldMesh(
                gridk.G[trf_spec.save_bestfit_field].compute(mode='complex')
                - gridk.G[trf_spec.target_field].compute(mode='complex') ))

        Pkmeas = gridk.calc_all_power_spectra(
            columns=[residual_key],
            Pk_ptcle2grid_deconvolution=opts['Pk_ptcle2grid_deconvolution'],
            k_bin_width=opts['k_bin_width'],
            mode=opts.get('Pk_1d_2d_mode','1d'), poles=opts.get('RSD_poles',None), 
            Nmu=opts.get('RSD_Nmu',None),
            line_of_sight=opts.get('RSD_los',None),
            Pkmeas=Pkmeas)


        #raise Exception('compare codes here (this is new code)')

        
        print("Pkmeas keys:\n", Pkmeas.keys())


        # ############################################################
        # Export bestfit field to disk
        # ############################################################
        if trf_spec.export_bestfit_field:
            out_rho_fname = os.path.join(
                opts['out_rho_path'],
                '%s_TARGET_%s_GRIDK.bigfile' % (trf_spec.save_bestfit_field,trf_spec.target_field))
            print("Export bestfit field to %s" % out_rho_path)
            gridk.save_to_bigfile(out_rho_fname, columns=[trf_spec.save_bestfit_field],
                               overwrite_file_if_exists=True)
            print("Exported bestfit field to %s" % out_rho_path)

        # save all fields to disk for slice and scatter plotting
        if opts['save_grids4plots']:
            for col in gridk.G.keys():
                if col not in ['ABSK']:
                    gridk.store_smoothed_gridx(
                        col, paths['grids4plots_path'], col,
                        helper_gridx=gridx, R=opts['grids4plots_R'], 
                        plot=False, replace_nan=0.0)

        
    for c in cached_columns:
        if c != 'ABSK':
            gridk.drop_column(c)                   

    # ############################################################
    # Calculate helper power spectra that are useful for plotting
    # ############################################################
    cols_Pk = opts['Pkmeas_helper_columns']
    gridk.append_columns_from_bigfile(gridk_cache_fname, cols_Pk, replace_existing_col=True,
                                   raise_exception_if_column_does_not_exist=False)

    # compute power spectra and include them in Pkmeas dict
    Pkmeas = gridk.calc_all_power_spectra(
        columns=cols_Pk,
        Pk_ptcle2grid_deconvolution=opts['Pk_ptcle2grid_deconvolution'],
        k_bin_width=opts['k_bin_width'],
        mode=opts.get('Pk_1d_2d_mode','1d'), poles=opts.get('RSD_poles',None), 
        Nmu=opts.get('RSD_Nmu',None),
        line_of_sight=opts.get('RSD_los',None),
        Pkmeas=Pkmeas)
    print("Computed helper Pkmeas for cols_Pk:\n", cols_Pk)

                    
    # # ############################################################
    # # Calculate some special power spectra of external grids
    # # ############################################################
    # special_cols = ['deltalin','deltaZA']
    # for c in cached_columns:
    #     if c != 'ABSK':
    #         gridk.drop_column(c)
    
        
        
    # # ############################################################
    # # Calculate power spectra of delta's and deltalin and all other external grids.
    # # ############################################################
    # if False:
    #     # takes too much memory
    #     cols_Pk = [t.save_bestfit_field for t in opts['trf_specs']]
    #     cols_Pk += opts['cats'].keys()
    #     for ext_grid_id in opts['ext_grids_to_load'].keys():
    #         cols_Pk.append(ext_grid_id)
    #     gridk.append_columns_from_bigfile(gridk_cache_fname, cols_Pk,
    #                                    replace_existing_col=False)

    #     print("cols_Pk:", cols_Pk)
    #     print("gridk:", gridk.G.keys())

    #     # Calc a few power spectra

    #     Pkmeas = gridk.calc_all_power_spectra(
    #         columns=cols_Pk,
    #         Pk_ptcle2grid_deconvolution=opts['Pk_ptcle2grid_deconvolution'],
    #         k_bin_width=opts['k_bin_width'],
    #        mode=opts.get('Pk_1d_2d_mode','1d'), poles=opts.get('RSD_poles',None), 
    #       Nmu=opts.get('RSD_Nmu',None),
    #        line_of_sight=opts.get('RSD_los',None),
    #         Pkmeas=Pkmeas)
    #     print("Pkmeas keys:", Pkmeas.keys())
   

    # ################################################################################
    # Delete temporary files
    # ################################################################################
    # not working in parallel
    # if delete_cache:
    #     for fname in cache_fnames:
    #         if os.path.exists(fname):
    #             rmtree(fname)
        #os.system('rm -f %s' % fname)
    #os.system('rm -r %s' % paths['cache_path'])

    

    # #################################################################################
    # create pickle dict
    # #################################################################################
    pickle_dict = {'Pkmeas': Pkmeas, 
                   'trf_results': trf_results,
                   'cat_infos': cat_infos,
                   'opts': opts,
                   'gridk_cache_fname': gridk_cache_fname,
                   'exec_trf_specs': exec_trf_specs}


    return pickle_dict
        



def simplify_cat_info(full_dict, weight_ptcles_by='NOT_DEFINED'):
    if ((weight_ptcles_by in [None,'Mass[1e10Msun/h]','Mass[1e10Msun/h]^2','Mass[1e10Msun/h]^3'])
        or weight_ptcles_by.startswith('MassWith')):
        # assume we deal with nonuniform catalog
        out_dict = OrderedDict()
        try:
            Ntot = full_dict['Nbkit_infodict']['Ntot']
            #tmp_infodict = json.loads(full_dict['Nbkit_infodict']['MS_infodict'])
            tmp_infodict = full_dict['Nbkit_infodict']['MS_infodict']
            L = float(tmp_infodict['cat_attrs']['BoxSize'][0])

            out_dict['Ntot'] = Ntot
            out_dict['boxsize'] = L
            out_dict['nbar'] = float(Ntot) / L**3
            out_dict['1_over_nbar'] = L**3/float(Ntot)
            out_dict['weight_ptcles_by'] = weight_ptcles_by
        except:
            raise Exception("Could not construct simplified catalog info")

    elif weight_ptcles_by == 'Mass':
        # uniform cat
        out_dict = OrderedDict()

        try:
            tmp_infodict = json.loads(full_dict['Nbkit_infodict']['MS_infodict'])
            #nonuniform_cat_infodict = json.loads(tmp_infodict['cat_attrs']['nonuniform_cat_infodict'])
            nonuniform_cat_infodict = tmp_infodict['cat_attrs']['nonuniform_cat_infodict']
            Ntot_nonuni_cat = nonuniform_cat_infodict['Nbkit_infodict']['Ntot']
            L_nonuni_cat = float(tmp_infodict['cat_attrs']['BoxSize'][0])

            out_dict['Ntot_nonuni_cat'] = Ntot_nonuni_cat
            out_dict['boxsize'] = L_nonuni_cat
            out_dict['nbar_nonuni_cat'] = float(Ntot_nonuni_cat) / L_nonuni_cat**3
            out_dict['1_over_nbar_nonuni_cat'] = L_nonuni_cat**3 / float(Ntot_nonuni_cat)
            out_dict['weight_ptcles_by'] = weight_ptcles_by
        except:
            raise Exception("Could not construct simplified catalog info")


    else:
        raise Exception("invalid weight_ptcles_by %s" % str(weight_ptcles_by))

    return out_dict

def get_full_fname(base_path, fname, ssseed=None):
    if 'ssseed' not in fname:
        return os.path.join(base_path,fname)
    else:
        assert ssseed is not None
        return os.path.join(base_path, 'SSSEED%d/'%ssseed, fname)

