#!/usr/bin/env python
#
# Marcel Schmittfull 2018 (mschmittfull@gmail.com)
#
# Python script for BAO reconstruction.
#
# Uses nbodykit 0.3.0



from __future__ import print_function,division

import cPickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from collections import OrderedDict
from matplotlib import rcParams
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from scipy import interpolate as interp
#from scipy.interpolate import RectBivariateSpline
#import time
#import re
#import h5py
import random
import glob
import sys



# MS packages
from nbodykit.source.catalog import HDFCatalog
from nbodykit.source.mesh.bigfile import BigFileMesh
# import constants
# import Catalog
from Grid import RealGrid, ComplexGrid
from psirec.dict_utils import *
from psirec.nbkit03_utils import get_cstat
#convert_np_arrays_to_lists
# from NbkitRunner import NbkitRunner
# import fft_ms_v2
# import mslogging
# import Pickler
# from cosmo_model import CosmoModel
# from gen_cosmo_fcns import generate_calc_Da
# from path_utils import get_in_path
# import transfer_functions
from nbodykit import logging


def paint_cat_to_gridk(
        PaintGrid_config, gridx=None, gridk=None,
        column=None,
        drop_gridx_column=True, rescalefac=1.0, 
        skip_fft=False,
        Ngrid=None, boxsize=None,
        grid_ptcle2grid_deconvolution=None,
        kmax=None,
        cache_path=None
        ):

    from nbodykit import CurrentMPIComm
    comm = CurrentMPIComm.get()
    logger = logging.getLogger('rec_utils')

    logger.info("Rank %d: paint_cat_to_gridk" % comm.rank)

    # check if catalog path exists
    if PaintGrid_config.has_key('DataSource'):
        if PaintGrid_config['DataSource'].has_key('path'):
            tmp_path = PaintGrid_config['DataSource']['path']
            if not os.path.exists(tmp_path):
                raise Exception('File does not exist: %s'%str(tmp_path))

    # nbkit code is in /Users/msl2/anaconda/anaconda/envs/nbodykit-0.3.7-env/lib/python2.7/site-packages/nbodykit/
    if True:
        # check nbkit version is new enough
        import nbodykit
        if comm.rank == 0:
            logger.info("nbkit version: %s" % str(nbodykit.__version__))
        if not nbodykit.__version__.startswith('0.3.'):
            raise Exception("Please use nbodykit 0.3.7 or higher. Maybe have to run with python, not pythonw")

    #  check PaintGrid_config is supported here
    if PaintGrid_config['Painter']['plugin'] != 'DefaultPainter':
        raise Exception("nbkit 0.3 wrapper not implemented for plugin %s" % str(
            PaintGrid_config['plugin']))
    implemented_painter_keys = ['plugin', 'normalize', 'setMean']
    for k in PaintGrid_config['Painter'].keys():
        if k not in implemented_painter_keys:
            raise Exception("config key %s not implemented in wrapper" % str(k))

    ## Do the painting

    if PaintGrid_config['DataSource']['plugin'] == 'Subsample':
        
        ## Read hdf5 catalog file and paint

        # TODO: get 'same value error' when using mass weighted columns b/c nbk03
        # thinks two columns have same name. probably string issue.
        if comm.rank == 0:
            logger.info("Try reading %s" % PaintGrid_config['DataSource']['path'])
        cat_nbk = HDFCatalog(PaintGrid_config['DataSource']['path'],
                             root='Subsample/')
        # Positions in Marcel hdf5 catalog files go from 0 to 1 but nbkit requires 0 to boxsize
        maxpos = get_cstat(cat_nbk['Position'].compute(), 'max')
        if (maxpos < 0.1*np.min(cat_nbk.attrs['BoxSize'][:])) or maxpos < 1.5:
            cat_nbk['Position'] = cat_nbk['Position'] * cat_nbk.attrs['BoxSize']
        if comm.rank == 0:
            logger.info("cat: %s" % str(cat_nbk))
            logger.info("columns: %s" % str(cat_nbk.columns))

        # paint catalog to grid
        if grid_ptcle2grid_deconvolution is None:
            to_mesh_kwargs={
                'window': 'cic', 'compensated': False, 'interlaced': False,
                'BoxSize': np.array([boxsize,boxsize,boxsize]),
                'dtype': 'f8'
            }
        else:
            raise Exception("todo: specify to_mesh_kwargs properly")
        #print(type(cat_nbk))
        
        # init CatalogMesh object (not painted yet)
        #mesh = cat_nbk.to_mesh(Nmesh=Ngrid, weight='log10M', **to_mesh_kwargs)
        mesh = cat_nbk.to_mesh(Nmesh=Ngrid, **to_mesh_kwargs)
        if comm.rank == 0:
            logger.info("mesh type: %s" % str(type(mesh)))
            logger.info("mesh attrs: %s" % str(mesh.attrs))


        # Paint. If normalize=True, outfield = 1+delta; if normalize=False: outfield=rho
        normalize = PaintGrid_config['Painter'].get('normalize',True)
        outfield = mesh.to_real_field(normalize=normalize)


    elif PaintGrid_config['DataSource']['plugin'] == 'BigFileGrid':
        ## Read bigfile grid (mesh) directly, no painting required, e.g. for linear density.        
        if comm.rank == 0:
            logger.info("Try reading %s" % PaintGrid_config['DataSource']['path'])
        mesh = BigFileMesh(PaintGrid_config['DataSource']['path'],
                           dataset=PaintGrid_config['DataSource']['dataset'])

        # Paint. If normalize=True, outfield = 1+delta; if normalize=False: outfield=rho
        outfield = mesh.to_real_field()
        normalize = PaintGrid_config['Painter'].get('normalize',True)
        if normalize:
            cmean = outfield.cmean()
            if np.abs(cmean < 1e-6):
                raise Exception('Found cmean=%g. Are you sure you want to normalize?' % cmean)
            outfield.apply(lambda x,v: v/cmean, kind="relative", out=outfield)


        #raise Exception('todo: normalize manually')


    else:
        raise Exception("Unsupported DataSource plugin %s" % 
                        PaintGrid_config['DataSource']['plugin'])


    
    # print paint info
    if comm.rank == 0:
        if normalize:
            logger.info('painted 1+delta')
        else:
            logger.info('painted rho (normalize=False)')
        if hasattr(outfield, 'attrs'):
            logger.info("outfield.attrs: %s" % str(outfield.attrs))


    # set the mean
    if 'setMean' in PaintGrid_config['Painter']:
        cmean = outfield.cmean()
        if comm.rank == 0:
            logger.info("mean: %g" % cmean)
        outfield.apply(
            lambda x,v: v - cmean + PaintGrid_config['Painter']['setMean'],
            kind='relative', out=outfield)
        new_mean = outfield.cmean()
        if comm.rank == 0:
            logger.info("setting mean=%s" % str(PaintGrid_config['Painter']['setMean']))
            logger.info("new mean: %s" % str(new_mean))

    if comm.rank == 0:
        if hasattr(outfield, 'attrs'):
            logger.info("outfield.attrs: %s" % str(outfield.attrs))

    # should simplify, keep for backwards compatibility
    if hasattr(outfield, 'attrs'):
        field_attrs = convert_np_arrays_to_lists(outfield.attrs)
    else:
        field_attrs = {}
    infodict = {
        'MS_infodict': {'Nbkit_config': PaintGrid_config,
            'cat_attrs': convert_np_arrays_to_lists(mesh.attrs)},
        'Ntot': mesh.attrs.get('Ntot', np.nan),
        'field_attrs': field_attrs}
    column_info = {'Nbkit_infodict': infodict}


    if comm.rank == 0:
        logger.info("Rescale factor: %g" % rescalefac)
    if rescalefac != 1.0:
        outfield.apply(lambda x,v: v*rescalefac, kind="relative", out=outfield)
        column_info['rescalefac'] = rescalefac
    #print("rescaled delta(x) rms min mean max:", np.mean(delta**2)**0.5, np.min(delta),
    #      np.mean(delta), np.max(delta))

    # return gridx and gridk if they were not passed as args
    return_gridx = (gridx is None)
    return_gridk = (gridk is None)

    
    # Store density in RealGrid instance
    if gridx is None:
        gridx = RealGrid(meshsource=outfield, column=column, Ngrid=Ngrid,
                         column_info=column_info, boxsize=boxsize)
    else:
        gridx.append_column(column, outfield, column_info=column_info)
    del outfield

    if False:
        # TEST FUNCTIONS in new Grid class
        gridx.append_column('bla', gridx.G[column], column_info=column_info)
        gridx.save_to_bigfile('test.bigfile', new_dataset_for_each_column=True)
        tmp_gridx = RealGrid(fname='test.bigfile', read_columns=[column])
        tmp_gridx.append_columns_from_bigfile('test.bigfile', ['bla'])
        tmp_gridk = ComplexGrid(meshsource=tmp_gridx.fft_x2k('bla', drop_column=False), column='bla',
            Ngrid=tmp_gridx.Ngrid, boxsize=tmp_gridx.boxsize)
        tmp_gridk.store_smoothed_gridx(
            col='bla', path='./', fname='test_smoothed_gridx.bigfile', R=50., replace_nan=True,
            helper_gridx=tmp_gridx)
        #tmp_gridx.apply_smoothing('bla', 'Gaussian', R=100.)
        tmp_gridx.convert_to_weighted_uniform_catalog(col='bla', uni_cat_Nptcles_per_dim=gridx.Ngrid,
            fill_value_negative_mass=0.)



    if comm.rank == 0:
        logger.info("column_info: %s" % str(gridx.column_infos[column]))
    
    
    if not skip_fft:

        # Compute FFT of density and store in ComplexGrid
        if gridk is None:
            gridk = ComplexGrid(meshsource=gridx.fft_x2k(column, drop_column=True),
                                column=column, Ngrid=gridx.Ngrid, boxsize=gridx.boxsize,
                                column_info=column_info)
        else:
            gridk.append_column(column, gridx.fft_x2k(column, drop_column=drop_gridx_column),
                                column_info=column_info)

        # Deconvolve CIC from grid
        if grid_ptcle2grid_deconvolution is not None:
            raise Exception("not implemented; use compensated Painter")
            gridk.deconvolve_ptcle2grid_from_grid(column,
                grid_ptcle2grid_deconvolution=grid_ptcle2grid_deconvolution)

        # Zero-pad high k
        if kmax is not None:
            gridk.apply_smoothing(column, mode='Gaussian', R=0.0, kmax=kmax)


        # Apply smoothing
        #if smoothing_kwargs is not None:
        #    raise Exception("not implemented yet")

        
    if return_gridx and return_gridk:
        return gridx, gridk
    elif return_gridx:
        return gridx
    elif return_gridk:
        return gridk



