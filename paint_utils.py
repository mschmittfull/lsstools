#!/usr/bin/env python

# Uses nbodykit 0.3.x

from __future__ import print_function,division

import cPickle
import numpy as np
import os
from collections import OrderedDict
from mpi4py import MPI
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
from nbodykit.source.mesh.field import FieldMesh
from pmesh.pm import RealField, ComplexField
from pm_utils import ltoc_index_arr, cgetitem_index_arr

def read_ms_hdf5_catalog(fname, root='Subsample/'):
    from nbodykit import CurrentMPIComm
    comm = CurrentMPIComm.get()
    logger = logging.getLogger('paint_utils')
    
    if comm.rank == 0:
        logger.info("Try reading %s" % fname)

    cat_nbk = HDFCatalog(fname, root=root)
    # Positions in Marcel hdf5 catalog files go from 0 to 1 but nbkit requires 0 to boxsize
    maxpos = get_cstat(cat_nbk['Position'].compute(), 'max')
    if (maxpos < 0.1*np.min(cat_nbk.attrs['BoxSize'][:])) or maxpos < 1.5:
        cat_nbk['Position'] = cat_nbk['Position'] * cat_nbk.attrs['BoxSize']
    if comm.rank == 0:
        logger.info("cat: %s" % str(cat_nbk))
        logger.info("columns: %s" % str(cat_nbk.columns))
    return cat_nbk



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
    logger = logging.getLogger('paint_utils')

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
    implemented_painter_keys = ['plugin', 'normalize', 'setMean', 'paint_mode', 'velocity_column',
        'fill_empty_cells', 'randseed_for_fill_empty_cells']
    for k in PaintGrid_config['Painter'].keys():
        if k not in implemented_painter_keys:
            raise Exception("config key %s not implemented in wrapper" % str(k))

    # TODO: implement keys 'weight' or weight_ptcles_by (different particles contribute with different weight or mass)



    ## Do the painting

    # Paint number density (e.g. galaxy overdensity) or divergence of momentum density
    paint_mode = PaintGrid_config['Painter'].get('paint_mode', None)
    
    if PaintGrid_config['DataSource']['plugin'] == 'Subsample':
        
        ## Read hdf5 catalog file and paint

        # TODO: get 'same value error' when using mass weighted columns b/c nbk03
        # thinks two columns have same name. probably string issue.
        cat_nbk = read_ms_hdf5_catalog(PaintGrid_config['DataSource']['path'])

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
            assert PaintGrid_config['Painter']['velocity_column'] is not None
            theta_k = None
            for idir in [0,1,2]:
                vi_label = 'tmp_V_%d' % idir
                # this is the velocity / (a*H). units are Mpc/h
                cat_nbk[vi_label] = cat_nbk[PaintGrid_config['Painter']['velocity_column']][:,idir]
                to_mesh_kwargs.update(dict(position='Position', value=vi_label))
                mesh = cat_nbk.to_mesh(Nmesh=Ngrid, **to_mesh_kwargs)
                if comm.rank == 0:
                    logger.info("mesh type: %s" % str(type(mesh)))
                    logger.info("mesh attrs: %s" % str(mesh.attrs))
                # this is (1+delta)v_i/(aH) (if normalize were False would get rho*v_i/(aH))
                outfield = FieldMesh(mesh.to_real_field(normalize=True))
                # get nabla_i[(1+delta)v_i/(aH)]
                def grad_i_fcn(k3vec, val, myidir=idir):
                    return -1.0j * k3vec[myidir]*val
                outfield = outfield.apply(grad_i_fcn, mode='complex', kind='wavenumber')
                # sum up to get theta(k) = sum_i nabla_i[(1+delta)v_i/(aH)]
                if theta_k is None:
                    theta_k = FieldMesh(outfield.compute('complex'))
                else:
                    theta_k = FieldMesh(theta_k.compute(mode='complex') + outfield.compute(mode='complex'))

            # save theta(x) in outfield  (check data types again)
            outfield = FieldMesh(theta_k.compute(mode='real')).to_real_field()


        elif paint_mode == 'velocity_divergence':
            # paint div v, and fill empty cells according to some rule
            assert PaintGrid_config['Painter']['velocity_column'] is not None
            assert PaintGrid_config['Painter']['fill_empty_cells'] is not None
            assert PaintGrid_config['Painter']['randseed_for_fill_empty_cells'] is not None

            vi_labels = []
            for idir in [0,1,2]:
                vi_label = 'tmp_V_%d' % idir
                vi_labels.append(vi_label)
                # this is the velocity / (a*H). units are Mpc/h
                cat_nbk[vi_label] = cat_nbk[PaintGrid_config['Painter']['velocity_column']][:,idir]
            
            # call extra function to do the painting, saving result in gridx.G[chi_cols]
            paint_chicat_to_gridx(
                chi_cols=vi_labels, 
                cat=cat_nbk,
                weight_ptcles_by=PaintGrid_config['Painter'].get('weight_ptcles_by', None), 
                fill_empty_chi_cells=PaintGrid_config['Painter']['fill_empty_cells'],
                RandNeighbSeed=PaintGrid_config['Painter']['randseed_for_fill_empty_cells'],
                gridx=gridx, gridk=gridk,
                cache_path=cache_path, do_plot=False, Ngrid=Ngrid, kmax=kmax)

            # delete catalog columns b/c not needed any more
            for vi in vi_labels:
                del cat_nbk[vi]

            # get divergence
            theta_k = None
            for idir, vi in enumerate(vi_labels):
                # this is v_i/(aH)
                outfield = FieldMesh(gridx.G[vi].compute(mode='real'))
                # get nabla_i[(1+delta)v_i/(aH)]
                def grad_i_fcn(k3vec, val, myidir=idir):
                    return -1.0j * k3vec[myidir]*val
                outfield = outfield.apply(grad_i_fcn, mode='complex', kind='wavenumber')
                # sum up to get theta(k) = sum_i nabla_i[(1+delta)v_i/(aH)]
                if theta_k is None:
                    theta_k = FieldMesh(outfield.compute('complex'))
                else:
                    theta_k = FieldMesh(theta_k.compute(mode='complex') + outfield.compute(mode='complex'))

            # save theta(x) in outfield  (check data types again)
            outfield = FieldMesh(theta_k.compute(mode='real')).to_real_field()

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
        if paint_mode in [None,'overdensity']:
            if normalize:
                logger.info('painted 1+delta')
            else:
                logger.info('painted rho (normalize=False)')
        elif paint_mode == 'momentum_divergence':
            logger.info('painted div[(1+delta)v/(aH)]')
        else:
            logger.info('painted with paint_mode %s' % paint_mode)
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

    # Save attrs. Should simplify, keep for backwards compatibility
    if hasattr(outfield, 'attrs'):
        field_attrs = convert_np_arrays_to_lists(outfield.attrs)
    else:
        field_attrs = {}
    Ntot = None
    if 'mesh' in vars():
        Ntot = mesh.attrs.get('Ntot', np.nan)
    elif 'cat_nbk' in vars():
        Ntot = cat_nbk.attrs.get('Ntot', np.nan)
    infodict = {
        'MS_infodict': {'Nbkit_config': PaintGrid_config},
        'field_attrs': field_attrs,
        'Ntot': Ntot}
    if 'mesh' in vars():
        infodict['MS_infodict']['cat_attrs'] = convert_np_arrays_to_lists(mesh.attrs)
    elif 'cat_nbk' in vars():
        infodict['MS_infodict']['cat_attrs'] = convert_np_arrays_to_lists(cat_nbk.attrs)

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




def paint_chicat_to_gridx(chi_cols=None, cat=None, gridx=None, gridk=None,
                          weight_ptcles_by=None, 
                          cache_path=None, do_plot=False,
                          Ngrid=None, fill_empty_chi_cells='RandNeighb',
                          RandNeighbSeed=1234,
                          kmax=None):
    """
    Helper function that reads in displacement field 'chi_col_{0,1,2}' from catalog
    and paints it to a regular grid. 
    The result is stored in gridx.G['chi_col_{0,1,2}'].
    Assume that catalog has 'weighted_chi_col_{0,1,2}' column which is used for painting.

    chi_cols : 3-tuple of strings, (chi_col_0, chi_col_1, chi_col_2)

    fill_empty_chi_cells : string
        - None, 'SetZero': Set empty cells to 0.
        - 'RandNeighbReadout': Fill using value at random neighbor cells (fast, using readout).
        - 'RandNeighb': Fill using value at random neighbor cells (same as RandNeighbReadout
            but slower b/c of manual MPI communication).
        - 'AvgAndRandNeighb': Not implemented in parallel version.
    """
    from nbodykit import CurrentMPIComm
    from nbodykit.mpirng import MPIRandomState

    comm = CurrentMPIComm.get()
    logger = logging.getLogger('paint_utils')

    # Read catalog
    #cat = read_ms_hdf5_catalog(cat_fname)

    ## Get mass density rho so we can normalize chi later. Assume mass=1, or given by
    # weight_ptcles_by.
    # This is to get avg chi if multiple ptcles are in same cell.
    # 1 Sep 2017: Want chi_avg = sum_i m_i chi_i / sum_j m_i where m_i is particle mass,
    # because particle mass says how much the average should be dominated by a single ptcle
    # that can represent many original no-mass particles.
    
    # Compute rho4chi = sum_i m_i
    rho4chi, rho4chi_attrs = weighted_paint_cat_to_delta(
        cat, 
        weight=weight_ptcles_by,
        weighted_paint_mode='sum',
        normalize=False, # want rho not 1+delta
        Nmesh=Ngrid,
        set_mean=None)



    ## Paint chi to grid using nbodykit. Do usual painting of particles to grid, but
    # use chi component as weight. 
    # This should give rho(x)chi(x) = sum_i m_i chi(x_i) where m_i is ptcle mass.
    for chi_col in chi_cols:
        print("Paint chi %s" % chi_col)
        # compute chi weighted by ptcle mass chi(x)m(x)
        weighted_col = 'TMP weighted %s' % chi_col
        if weight_ptcles_by is not None:
            cat[weighted_col] = cat[weight_ptcles_by] * cat[chi_col]
        else:
            # weight 1 for each ptcle
            cat[weighted_col] = cat[chi_col]
        thisChi, thisChi_attrs = weighted_paint_cat_to_delta(
            cat, 
            weight=weighted_col, # chi weighted by ptcle mass
            weighted_paint_mode='sum',
            normalize=False, # want rho not 1+delta (TODO: check)
            Nmesh=Ngrid,
            set_mean=None)


        # Normalize Chi by dividing by rho: So far, our chi will get larger if there are 
        # more particles, because it sums up displacements over all particles. 
        # To normalize, divide by rho (=mass density on grid if all ptcles have mass m=1).
        # (i.e. divide by number of contributions to a cell)
        if fill_empty_chi_cells in [None, 'SetZero']:
            # Set chi=0 if there are not ptcles in grid cell. Used until 7 April 2017.
            # Seems ok for correl coeff and BAO, but gives large-scale bias in transfer
            # function or broad-band power because violates mass conservation.
            thisChi = FieldMesh(np.where(
                rho4chi.compute(mode='real')==0,
                rho4chi.compute(mode='real')*0,
                thisChi.compute(mode='real')/rho4chi.compute(mode='real')))
            #thisChi = np.where(gridx.G['rho4chi']==0, thisChi*0, thisChi/gridx.G['rho4chi'])

        elif fill_empty_chi_cells in ['RandNeighb', 'RandNeighbReadout', 'AvgAndRandNeighb']:
            # Set chi in empty cells equal to a random neighbor cell. Do this until all empty
            # cells are filled.
            # First set all empty cells to nan.
            #thisChi = np.where(gridx.G['rho4chi']==0, thisChi*0+np.nan, thisChi/gridx.G['rho4chi'])
            thisChi = thisChi/rho4chi # get nan when rho4chi=0
            if True:
                # test if nan ok
                ww1 = np.where(rho4chi==0)
                #ww2 = np.where(np.isnan(thisChi.compute(mode='real')))
                ww2 = np.where(np.isnan(thisChi))
                assert np.allclose(ww1, ww2)
                del ww1, ww2

            # Progressively replace nan by random neighbors:
            Ng = Ngrid
            #thisChi = thisChi.reshape((Ng,Ng,Ng))
            logger.info('thisChi.shape: %s' % str(thisChi.shape))
            #assert thisChi.shape == (Ng,Ng,Ng)
            # indices of empty cells on this rank
            ww = np.where(np.isnan(thisChi))
            # number of empty cells across all ranks
            Nfill = comm.allreduce(ww[0].shape[0], op=MPI.SUM)
            have_empty_cells = (Nfill > 0)

            if fill_empty_chi_cells in ['RandNeighb','RandNeighbReadout']:
                i_iter = -1
                while have_empty_cells:
                    i_iter += 1
                    if comm.rank == 0:
                        logger.info("Fill %d empty chi cells (%g percent) using random neighbors" % (
                            Nfill, Nfill/float(Ng)**3*100.))
                    if Nfill/float(Ng)**3 >= 0.999:
                        raise Exception("Stop because too many empty chi cells")
                    # draw -1,0,+1 for each empty cell, in 3 directions
                    # r = np.random.randint(-1,2, size=(ww[0].shape[0],3), dtype='int')
                    rng = MPIRandomState(comm, seed=RandNeighbSeed+i_iter*100, size=ww[0].shape[0], chunksize=100000)
                    r = rng.uniform(low=-2, high=2, dtype='int', itemshape=(3,))
                    assert np.all(r>=-1)
                    assert np.all(r<=1)

                    # Old serial code to replace nan by random neighbors.
                    # thisChi[ww[0],ww[1],ww[2]] = thisChi[(ww[0]+r[:,0])%Ng, (ww[1]+r[:,1])%Ng, (ww[2]+r[:,2])%Ng]
                   
                    if fill_empty_chi_cells == 'RandNeighbReadout':
                        # New parallel code, 1st implementation.
                        # Use readout to get field at positions [(ww+rank_offset+r)%Ng] dx.
                        BoxSize = cat.attrs['BoxSize']
                        dx = BoxSize/(float(Ng))
                        #pos_wanted = ((np.array(ww).transpose() + r) % Ng) * dx   # ranges from 0 to BoxSize
                        # more carefully:
                        pos_wanted = np.zeros((ww[0].shape[0],3))+np.nan
                        for idir in [0,1,2]:
                            pos_wanted[:,idir] = ( (np.array(ww[idir]+thisChi.start[idir]) + r[:,idir]) % Ng ) * dx[idir] # ranges from 0..BoxSize

                        # use readout to get neighbors
                        readout_window = 'nnb'
                        layout = thisChi.pm.decompose(pos_wanted, smoothing=readout_window)
                        # interpolate field to particle positions (use pmesh 'readout' function)
                        thisChi_neighbors = thisChi.readout(pos_wanted, resampler=readout_window, layout=layout)
                        if False:
                            # print dbg info
                            for ii in range(10000,10004):
                                if comm.rank == 1:
                                    logger.info('chi manual neighbor: %g' %  
                                        thisChi[(ww[0][ii]+r[ii,0])%Ng, (ww[1][ii]+r[ii,1])%Ng, (ww[2][ii]+r[ii,2])%Ng])
                                    logger.info('chi readout neighbor: %g' % thisChi_neighbors[ii])
                        thisChi[ww] = thisChi_neighbors

                    elif fill_empty_chi_cells == 'RandNeighb':
                        # New parallel code, 2nd implementation.
                        # Use collective getitem and only work with indices.
                        # http://rainwoodman.github.io/pmesh/pmesh.pm.html#pmesh.pm.Field.cgetitem.
                        
                        # Note ww are indices of local slab, need to convert to global indices.
                        thisChi_neighbors = None
                        my_cindex_wanted = None
                        for root in range(comm.size):
                            # bcast to all ranks b/c must call cgetitem collectively with same args on each rank
                            if comm.rank == root:
                                # convert local index to collective index using ltoc which gives 3 tuple
                                assert len(ww) == 3
                                wwarr = np.array(ww).transpose()
                               
                                #cww = np.array([ 
                                #    ltoc(field=thisChi, index=[ww[0][i],ww[1][i],ww[2][i]]) 
                                #    for i in range(ww[0].shape[0]) ])
                                cww = ltoc_index_arr(field=thisChi, lindex_arr=wwarr)
                                #logger.info('cww: %s' % str(cww))

                                #my_cindex_wanted = [(cww[:,0]+r[:,0])%Ng, (cww[1][:]+r[:,1])%Ng, (cww[2][:]+r[:,2])%Ng]
                                my_cindex_wanted = (cww+r) % Ng
                                #logger.info('my_cindex_wanted: %s' % str(my_cindex_wanted))
                            cindex_wanted = comm.bcast(my_cindex_wanted, root=root)
                            glob_thisChi_neighbors = cgetitem_index_arr(thisChi, cindex_wanted) 

                            # slower version doing the same
                            # glob_thisChi_neighbors = [
                            #     thisChi.cgetitem([cindex_wanted[i,0], cindex_wanted[i,1], cindex_wanted[i,2]]) 
                            #     for i in range(cindex_wanted.shape[0]) ]


                            if comm.rank == root:
                                thisChi_neighbors = np.array(glob_thisChi_neighbors)
                            #thisChi_neighbors = thisChi.cgetitem([40,42,52])
                        
                        #print('thisChi_neighbors:', thisChi_neighbors)

                        if False:
                            # print dbg info (rank 0 ok, rank 1 fails)
                            for ii in range(11000,11004):
                                if comm.rank == 1:
                                    logger.info('ww: %s' % str([ww[0][ii], ww[1][ii], ww[2][ii]]))
                                    logger.info('chi[ww]: %g' % thisChi[ww[0][ii], ww[1][ii], ww[2][ii]])
                                    logger.info('chi manual neighbor: %g' %  
                                        thisChi[(ww[0][ii]+r[ii,0])%Ng, (ww[1][ii]+r[ii,1])%Ng, (ww[2][ii]+r[ii,2])%Ng])
                                    logger.info('chi bcast neighbor: %g' % thisChi_neighbors[ii])
                            raise Exception('just dbg')
                        thisChi[ww] = thisChi_neighbors

                    ww = np.where(np.isnan(thisChi))
                    Nfill = comm.allreduce(ww[0].shape[0], op=MPI.SUM)
                    have_empty_cells = (Nfill > 0)
                    comm.barrier()



            elif fill_empty_chi_cells == 'AvgAndRandNeighb':
                raise Exception('Not implemented any more')
                # while have_empty_cells:
                #     print("Fill %d empty chi cells (%g percent) using avg and random neighbors" % (
                #         ww[0].shape[0],ww[0].shape[0]/float(Ng)**3*100.))
                #     # first take average (only helps empty cells surrounded by filled cells)
                #     thisChi[ww[0],ww[1],ww[2]] = 0.0
                #     for r0 in range(-1,2):
                #         for r1 in range(-1,2):
                #             for r2 in range(-1,2):
                #                 if (r0==0) and (r1==0) and (r2==0):
                #                     # do not include center point in avg b/c this is nan
                #                     continue
                #                 else:
                #                     # average over 27-1 neighbor points
                #                     thisChi[ww[0],ww[1],ww[2]] += thisChi[(ww[0]+r0)%Ng, (ww[1]+r1)%Ng, (ww[2]+r2)%Ng]/26.0
                #     # get indices of cells that are still empty (happens if a neighbor was nan above)
                #     ww = np.where(np.isnan(thisChi))
                #     have_empty_cells = (ww[0].shape[0] > 0)
                #     if have_empty_cells:
                #         # draw -1,0,+1 for each empty cell, in 3 directions
                #         r = np.random.randint(-1,2, size=(ww[0].shape[0],3), dtype='int')
                #         # replace nan by random neighbors
                #         thisChi[ww[0],ww[1],ww[2]] = thisChi[(ww[0]+r[:,0])%Ng, (ww[1]+r[:,1])%Ng, (ww[2]+r[:,2])%Ng]
                #         # recompute indices of nan cells
                #         ww = np.where(np.isnan(thisChi))
                #         have_empty_cells = (ww[0].shape[0] > 0)
                
                
        else:
            raise Exception("Invalid fill_empty_chi_cells option: %s" % str(
                fill_empty_chi_cells))
        # Save as RealGrid entry. 
        gridx.append_column(chi_col, thisChi)

        # release memory
        del thisChi
        
        # If kmax is not None, apply smoothing
        if kmax is not None:
            col = chi_col
            gridk.append_column(col, gridx.fft_x2k(col, drop_column=True))
            gridk.apply_smoothing(col, mode='Gaussian', R=0.0, kmax=kmax)
            gridx.append_column(col, gridk.fft_k2x(col, drop_column=True))


        # plot slice
        if do_plot:
            gridx.plot_slice(chi_col, 'slice4chi_%s.pdf'%chi_col)

    gridx.drop_column('rho4chi')
    # output is stored in gridx.G['chi_col_{0,1,2}'], nothing to return.



def weighted_paint_cat_to_delta(
        cat, weight=None,
        weighted_paint_mode=None,
        normalize=True,
        Nmesh=None,
        to_mesh_kwargs={'window': 'cic', 'compensated': False, 'interlaced': False},
        set_mean = None,
        verbose=True):
    """
    - weighted_paint_mode='sum': In each cell, sum up the weight of all particles in the cell.
        So this gets larger if there are more particles in a cell. 
    - weighted_paint_mode='avg': In each cell, sum up the weight of all particles in the cell
        and divide by the number of contributions. This does not increase if there are more
        particles in a cell with the same weight.
    Note: In nbodykit nomenclature this is called 'value' instead of 'weight', but only implements
        our 'sum' not 'avg' mode (it seems).

    NOTE: Looks like this is actually not used anywhere so far.
    """

    if weighted_paint_mode not in ['sum','avg']:
        raise Exception("Invalid weighted_paint_mode %s" % weighted_paint_mode)

    assert 'value' not in to_mesh_kwargs.keys()
    
    # We want to sum up weight. Use value not weight for this b/c each ptlce should contribute
    # equally. Later we divide by number of contributions.
    if weight is not None:
        meshsource = cat.to_mesh(Nmesh=Nmesh, value=weight, **to_mesh_kwargs)
    else:
        # no weight so assume each ptcle has weight 1
        meshsource = cat.to_mesh(Nmesh=Nmesh, **to_mesh_kwargs)
    meshsource.attrs['weighted_paint_mode'] = weighted_paint_mode

    # get outfield = 1+delta
    #outfield = meshsource.paint(mode='real')
    # Paint. If normalize=True, outfield = 1+delta; if normalize=False: outfield=rho
    outfield = meshsource.to_real_field(normalize=normalize)

    if weighted_paint_mode=='avg':
        # count contributions per cell (no value or weight).
        # outfield_count = 1+delta_unweighted = number of contributions per cell
        # (or rho_unweighted if normalize=False)
        #outfield_count = cat.to_mesh(Nmesh=Nmesh, **to_mesh_kwargs).paint(mode='real')
        outfield_count = cat.to_mesh(Nmesh=Nmesh, **to_mesh_kwargs).to_real_field(normalize=normalize)

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
    if set_mean is not None:
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

