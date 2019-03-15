#!/usr/bin/env python

# Uses nbodykit 0.3.x

from __future__ import print_function,division

import cPickle
import numpy as np
import os
from collections import OrderedDict
from scipy import interpolate as interp
import random
import glob
import sys

# MS packages
from nbodykit.source.catalog import HDFCatalog
from nbodykit.source.mesh.bigfile import BigFileMesh
from Grid import RealGrid, ComplexGrid
from nbkit03_utils import get_cstat
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
    implemented_painter_keys = ['plugin', 'normalize', 'setMean', 'paint_mode', 'velocity_column']
    for k in PaintGrid_config['Painter'].keys():
        if k not in implemented_painter_keys:
            raise Exception("config key %s not implemented in wrapper" % str(k))

    # TODO: implement keys 'weight' (different particles contribute with different weight or mass)



    ## Do the painting

    # Paint number density (e.g. galaxy overdensity) or divergence of momentum density
    paint_mode = PaintGrid_config['Painter'].get('paint_mode', None)
    
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

        # Paint options
        if grid_ptcle2grid_deconvolution is None:
            to_mesh_kwargs={
                'window': 'cic', 'compensated': False, 'interlaced': False,
                'BoxSize': np.array([boxsize,boxsize,boxsize]),
                'dtype': 'f8'
            }
        else:
            raise Exception("todo: specify to_mesh_kwargs properly")
        
        if paint_mode in [None, 'overdensity']:
            # Paint overdensity.
            # Init CatalogMesh object (not painted yet)
            #mesh = cat_nbk.to_mesh(Nmesh=Ngrid, weight='log10M', **to_mesh_kwargs)
            mesh = cat_nbk.to_mesh(Nmesh=Ngrid, **to_mesh_kwargs)
            if comm.rank == 0:
                logger.info("mesh type: %s" % str(type(mesh)))
                logger.info("mesh attrs: %s" % str(mesh.attrs))

            # Paint. If normalize=True, outfield = 1+delta; if normalize=False: outfield=rho
            normalize = PaintGrid_config['Painter'].get('normalize',True)
            outfield = mesh.to_real_field(normalize=normalize)

        elif paint_mode == 'momentum_divergence':
            # Paint momentum divergence div((1+delta)v/(aH)).
            # See https://nbodykit.readthedocs.io/en/latest/cookbook/painting.html#Painting-the-Line-of-sight-Momentum-Field. 
            assert PaintGrid_config['Painter'].has_key('velocity_column')
            theta_k = None
            for idir in [0,1,2]:
                vi_label = 'tmp_V_%d' % idir
                # this is the velocity / (a*H). units are Mpc/h
                cat[vi_label] = cat[PaintGrid_config['Painter']['velocity_column']][:,idir]
                to_mesh_kwargs.update(dict(position='Position', value=vi_label))
                mesh = cat_nbk.to_mesh(Nmesh=Ngrid, **to_mesh_kwargs)
                if comm.rank == 0:
                    logger.info("mesh type: %s" % str(type(mesh)))
                    logger.info("mesh attrs: %s" % str(mesh.attrs))
                # this is (1+delta)v_i/(aH) in k space  (if normalize were False would get rho*v_i/(aH))
                outfield = mesh.to_complex_field(normalize=True)
                # get nabla_i[(1+delta)v_i/(aH)]
                def grad_i_fcn(k3vec, val, myidir=idir):
                    return -1.0j * k3vec[myidir]*val
                outfield.apply(grad_i_fcn, mode='complex', kind='wavenumber', out=outfield)
                # sum up to get theta(k) = sum_i nabla_i[(1+delta)v_i/(aH)]
                if theta_k is None:
                    theta_k = FieldMesh(outfield.compute('complex'))
                else:
                    theta_k = FieldMesh(theta_k.compute(mode='complex') + outfield.compute(mode='complex'))

            # save theta(x) in outfield
            outfield = FieldMesh(theta_k.compute(mode='real'))

        else:
            raise Exception('Invalid paint_mode %s' % paint_mode)




    elif PaintGrid_config['DataSource']['plugin'] == 'BigFileGrid':
        ## Read bigfile grid (mesh) directly, no painting required, e.g. for linear density.
        if paint_mode not in [None, 'overdensity']:
            raise Exception('Can only paint overdensity when reading BigFileGrid')

        if comm.rank == 0:
            logger.info("Try reading %s" % PaintGrid_config['DataSource']['path'])
        mesh = BigFileMesh(PaintGrid_config['DataSource']['path'],
                           dataset=PaintGrid_config['DataSource']['dataset'])

        if PaintGrid_config['Painter'].get('value',None) is not None:
            raise Exception('value kwarg not allowed in Painter when reading BigFileGrid')


        # Paint. 
        # If normalize=True, divide by the mean. In particular:
        # - If paint_mode=None, overdensity: If normalize=True, outfield = 1+delta; if normalize=False, outfield=rho
        # - If paint_mode=momentum_divergence: Reading from BigFileGrid not implemented
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




def weighted_paint_cat_to_delta(
        cat, weight=None,
        weighted_paint_mode=None,
        Nmesh=None,
        to_mesh_kwargs={'window': 'cic', 'compensated': False, 'interlaced': False},
        set_mean = 0.0,
        verbose=True):

    if weight is None:
        raise Exception("Must specify weight")
    if weighted_paint_mode not in ['sum','avg']:
        raise Exception("Invalid weighted_paint_mode %s" % weighted_paint_mode)
    
    # We want to sum up weight. Use value not weight for this b/c each ptlce should contribute
    # equally. Later we divide by number of contributions.
    meshsource = cat.to_mesh(Nmesh=Nmesh, value=weight, **to_mesh_kwargs)
    meshsource.attrs['weighted_paint_mode'] = weighted_paint_mode

    # get outfield = 1+delta
    outfield = meshsource.paint(mode='real')

    if weighted_paint_mode=='avg':
        # count contributions per cell (no value or weight).
        # outfield_count = 1+delta_unweighted = number of contributions per cell
        outfield_count = cat.to_mesh(Nmesh=Nmesh, **to_mesh_kwargs).paint(mode='real')

    if verbose:
        comm = meshsource.comm
        print("%d: outfield_weighted: min, mean, max, rms(x-1):"%comm.rank, 
              np.min(outfield), np.mean(outfield), np.max(outfield), np.mean((outfield-1.)**2)**0.5)
        if weighted_paint_mode=='avg':
            print("%d: outfield_count: min, mean, max, rms(x-1):"%comm.rank, np.min(outfield_count), 
                  np.mean(outfield_count), np.max(outfield_count), np.mean((outfield_count-1.)**2)**0.5)

    # divide weighted 1+delta by number of contributions
    if weighted_paint_mode=='avg':
        outfield /= outfield_count
        del outfield_count
    
    # set the mean
    outfield = outfield - outfield.cmean() + set_mean

    if verbose:
        # print some info:
        print("%d: outfield weighted/count: min, mean, max, rms(x-1):"%comm.rank, 
              np.min(outfield), np.mean(outfield), np.max(outfield), np.mean((outfield-1.)**2)**0.5)

    return outfield, meshsource.attrs




def convert_np_arrays_to_lists(indict):
    outdict = OrderedDict()
    for k in indict.keys():
        if type(indict[k]) == np.ndarray:
            # convert to list b/c json cannot save numpy arrays
            outdict[k] = indict[k].tolist()
        else:
            outdict[k] = indict[k]
    return outdict

