#!/usr/bin/env python
#
# Marcel Schmittfull 2018 (mschmittfull@gmail.com)




from __future__ import print_function,division

import cPickle
import numpy as np
import os
from collections import OrderedDict, namedtuple, Counter
import random
import glob
import sys
import matplotlib.pyplot as plt


# MS packages
from psirec import constants
from psirec import path_utils
from lsstools.combine_source_fields_to_match_target import get_full_fname, simplify_cat_info
from lsstools import paint_utils


def main(argv):
    """
    Calc cumulants of simulated density.

    Run using 
      ./run_nbk03.sh python test_linear_term_fisher_nbk03.py
    or 
      ./run_nbk03.sh mpiexec -n 2 python test_linear_term_fisher_nbk03.py
    """

    
    #####################################
    # OPTIONS
    #####################################

    opts = OrderedDict()
    
    ## ANALYSIS
    opts['Ngrid'] = 64

    # k bin width for power spectra, in units of k_f=2pi/L. Must be >=1.0. Choose 1.,2.,3. usually.
    opts['k_bin_width'] = 1.0


    if True:
        # L=500 ms_gadget sims produced with MP-Gadget, 1536^3 particles
        opts['sim_name'] = 'ms_gadget'
        opts['sim_irun'] = 4
        # use value from cmd line b/c later options depend on this
        opts['sim_seed'] = 403
        opts['ssseed'] = 40000+opts['sim_seed']      # seed used to draw subsample
        opts['sim_Ntimesteps'] = None  # Nbody, so used thousands of time steps
        opts['sim_Nptcles'] = 1536
        opts['sim_boxsize'] = 500.0
        opts['boxsize'] = opts['sim_boxsize']
        opts['sim_wig_now_string'] = 'wig'
        # scale factor of simulation snapshot (only used to rescale deltalin -- do not change via arg!)
        opts['sim_scale_factor'] = 0.6250
        # halo mass
        opts['halo_mass_string'] = '13.8_15.1'

        # linear density (ICs of the sims)
        opts['ext_grids_to_load'] = OrderedDict()
        if False:
            # deltanonl painted from all 1536^3 DM particles (includes CIC artifacts, small shot noise)
            opts['ext_grids_to_load']['delta_m'] = {
                'dir': 'snap_%.4f_PtcleDensity_Ng%d' % (opts['sim_scale_factor'], opts['Ngrid']),
                'file_format': 'nbkit_BigFileGrid',
                'dataset_name': 'Field',
                'scale_factor': opts['sim_scale_factor'], 'nbkit_normalize': True, 'nbkit_setMean': 0.0}

            

    # ######################################################################
    # Catalogs to read
    # ######################################################################
    opts['cats'] = OrderedDict()

    if opts['sim_name'] == 'ms_gadget':    

        tmp_halo_dir = 'nbkit_fof_%.4f/ll_0.200_nmin25' % opts['sim_scale_factor']
        ## nonuniform catalogs without ptcle masses
        if True:
            # halos without mass weight, narrow mass cuts: 10.8..11.8..12.8..13.8..15.1
            opts['cats']['delta_h'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_BOUNDS_log10M_%s.hdf5" % (
                    tmp_halo_dir, opts['halo_mass_string']),
                'weight_ptcles_by': None}
        if False:
            # halos not weighted by mass but including mass info in file, broad mass cut
            opts['cats']['delta_h'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols.hdf5_BOUNDS_log10M_%s.hdf5" % (
                    tmp_halo_dir, opts['halo_mass_string']),
                'weight_ptcles_by': None}
        if False:
            # halos in narrow mass bins, no mass weights
            opts['cats']['delta_h_M10.8-11.8'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_10.8_11.8.hdf5" % tmp_halo_dir,
                'weight_ptcles_by': None}
            opts['cats']['delta_h_M11.8-12.8'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_11.8_12.8.hdf5" % tmp_halo_dir,
                'weight_ptcles_by': None}
            opts['cats']['delta_h_M12.8-13.8'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_12.8_13.8.hdf5" % tmp_halo_dir,
                'weight_ptcles_by': None}
            opts['cats']['delta_h_M13.8-15.1'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_13.8_15.1.hdf5" % tmp_halo_dir,
                'weight_ptcles_by': None}

        if False:
            # halos in broad mass bins, no mass weights
            opts['cats']['delta_h_M12.8-16.0'] = {
                'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols.hdf5_BOUNDS_log10M_12.8_16.0.hdf5" % tmp_halo_dir,
                'weight_ptcles_by': None}
            
            
    else:
        raise Exception("Invalid sim_name %s" % opts['sim_name'])
        

        

    if False:
        # halos weighted by exact mass
        opts['cats']['delta_h_WEIGHT_M1'] = {
            'in_fname': "%s/fof_nbkfmt.hdf5_WithMassCols.hdf5_BOUNDS_log10M_%s.hdf5" % (
                    tmp_halo_dir, opts['halo_mass_string']),
            'weight_ptcles_by': 'Mass[1e10Msun/h]'}
    if False:
        # weighted by exact mass^2
        opts['cats']['delta_h_WEIGHT_M2'] = {
            'in_fname': opts['cats']['delta_h']['in_fname'], 
            'weight_ptcles_by': 'Mass[1e10Msun/h]^2'}
    if False:
        # halos weighted by noisy mass
        #for myscatter in ['0.04dex','0.1dex','0.3dex','0.6dex']:
        for myscatter in ['0.1dex','0.2dex','0.4dex']:
            opts['cats']['delta_h_WEIGHT_M%s'%myscatter] = {
                'in_fname': opts['cats']['delta_h']['in_fname'], 
                'weight_ptcles_by': 'MassWith%sScatter[1e10Msun/h]' % myscatter}



    
    ## Smoothing types and smoothing scales in Mpc/h
    smoothing_lst = []
    if True:
        # apply Gaussian smoothing
        #for R in [160.,80.,40.,20.,10.,5.,2.5]:
        for R in [20.]:
            #for R in [160.,40.,10.]:
            smoothing_lst.append( 
                dict(type='Gaussian', R=R))
        print("smoothing_lst:", smoothing_lst)

    
    #### Output pickles and cache
    opts['pickle_path'] = '$SCRATCH/lssbisp2013/BenLT/pickle/'
    opts['cache_base_path'] = '$SCRATCH/lssbisp2013/BenLT/cache/'


    ## ANTI-ALIASING OPTIONS (do not change unless you know what you do)
    # Kmax above which to 0-pad. Should use kmax<=2pi/L*N/2 to avoid
    # unwanted Dirac delta images/foldings when multipling fields.
    opts['kmax'] = 2.0*np.pi/opts['sim_boxsize'] * float(opts['Ngrid'])/2.0
    #opts['kmax'] = None
    # CIC deconvolution of grid: None or 'grid_non_isotropic'
    opts['grid_ptcle2grid_deconvolution'] = None # 'grid_non_isotropic' #'grid_non_isotropic'
    # CIC deconvolution of power: None or 'power_isotropic_and_aliasing' 
    opts['Pk_ptcle2grid_deconvolution'] = None

    


    
    #####################################
    # START PROGRAM
    #####################################
    

    ### derived options (do not move above b/c command line args might
    ### overwrite some options!)
    opts['in_path'] = path_utils.get_in_path(opts)
    # for output densities
    opts['out_rho_path'] = os.path.join(
        opts['in_path'], 
        'out_rho_Ng%d' % opts['Ngrid'])
        
    # expand environment names in paths
    paths = {}
    for key in ['in_path', 'in_fname', 'in_fname_PTsim_psi_calibration',
                'in_fname_halos_to_displace_by_mchi',
                'pickle_path', 'cache_base_path', 'grids4plots_base_path',
                'out_rho_path']:
        if opts.has_key(key):
            if opts[key] is None:
                paths[key] = None
            else:
                paths[key] = os.path.expandvars(opts[key])
        
                #mslogging.setup_logging(use_mpi=opts['use_mpi'])
    
    # unique id for cached files so we can run multiple instances at the same time
    file_exists = True
    while file_exists:
        cacheid = ('CACHE%06x' % random.randrange(16**6)).upper()
        paths['cache_path'] = os.path.join(paths['cache_base_path'], cacheid)
        file_exists = (len(glob.glob(paths['cache_path']))>0)
    # create cache path
    if not os.path.exists(paths['cache_path']):
        #os.system('mkdir -p %s' % paths['cache_path'])
        os.makedirs(paths['cache_path'])
        
    # Check some params
    if ((opts['grid_ptcle2grid_deconvolution'] is not None) and
        (opts['Pk_ptcle2grid_deconvolution'] is not None)):
        raise Exception("Must not simultaneously apply ptcle2grid deconvolution to grid and Pk.")

 

    
    # #################################################################################
    # Init some thnigs
    # #################################################################################

    Nsmoothings = len(smoothing_lst)
    fig_hist, axlst_hist = plt.subplots(Nsmoothings,1, sharex=False,
                                        figsize=(6,4*Nsmoothings))
    
    pickle_dicts = OrderedDict()
    pickle_dicts['opts'] = opts.copy()


    # init some things
    cache_fnames = []

    gridx = None
    gridk = None

    gridk_cache_fname = os.path.join(paths['cache_path'], 'gridk_qPk_my_cache.hdf5')
    cache_fnames.append(gridk_cache_fname)
    cached_columns = []
    if os.path.exists(gridk_cache_fname):
        os.remove(gridk_cache_fname)

    from nbodykit import setup_logging
    setup_logging()


    # ################################################################################
    # Compute density of all input catalogs 
    # ################################################################################

    cat_infos = OrderedDict()
    for cat_id, cat_opts in opts['cats'].items():

        # default args for painting
        default_paint_kwargs = {
            'gridx': gridx, 'gridk': gridk, 'cache_path': paths['cache_path'],
            'Ngrid': opts['Ngrid'], 'boxsize': opts['boxsize'],
            'grid_ptcle2grid_deconvolution': opts['grid_ptcle2grid_deconvolution'],
            'kmax': opts['kmax']}

        # nbodykit config
        config_dict = {
            'Nmesh': opts['Ngrid'],
            'output': os.path.join(paths['cache_path'], 'test_paint_baoshift'),
            'DataSource': {
                'plugin': 'Subsample',
                'path': get_full_fname(paths['in_path'], cat_opts['in_fname'], opts['ssseed'])},
            'Painter': {
                'plugin': 'DefaultPainter',
                'normalize': True,
                'setMean': 0.0
                }
            }

        if cat_opts['weight_ptcles_by'] is not None:
            config_dict['Painter']['weight'] = cat_opts['weight_ptcles_by']

        # paint deltanonl and save it in gridk.G[cat_id] 
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

        # apply smoothing
        #gridk.apply_smoothing(cat_id, mode='Gaussian', R=20.0)

        # fft to x space
        gridx.append_column(cat_id, gridk.fft_k2x(cat_id, drop_column=False))

        if False:
            # test kappa2
            gridk.append_column('kappa2', gridk.calc_kappa2(cat_id, gridx=gridx))
            gridx.append_column('kappa2', gridk.fft_k2x('kappa2', drop_column=True))

        # test quadratic fields
        for quadfield in ['shift']:
            gridx.append_column(
                quadfield,
                gridk.calc_quadratic_field(basefield=cat_id, quadfield=quadfield,
                    gridx=gridx, return_in_k_space=False))
            gridk.append_column(quadfield, gridx.fft_x2k(quadfield, drop_column=True))
            
    # test compute_orthogonalized_fields
    # modifies gridk, Pkmeas
    gridk.rename_column('delta_h', 'ORTH s^0_0')
    gridk.rename_column('shift', 'ORTH s^0_1')
    osources, Pkmeas, ortho_rot_matrix_sources, orth_internals_sources = gridk.compute_orthogonalized_fields(
        N_ortho_iter=1, 
        orth_method='CholeskyDecomp',
        all_in_fields=['ORTH s^0_0','ORTH s^0_1'],
        orth_prefix='ORTH s', 
        non_orth_prefix='NON_ORTH s',
        Pkmeas=None, 
        Pk_ptcle2grid_deconvolution=None,
        k_bin_width=1.0,
        delete_original_fields=True)

    for osource in osources:
        gridx.append_column(osource, gridk.fft_k2x(osource, drop_column=True))




    # ################################################################################
    # Empty cache
    # ################################################################################
    from shutil import rmtree
    rmtree(paths['cache_path'])



    raise Exception("continue here")
        
if __name__ == '__main__':
    sys.exit(main(sys.argv))


