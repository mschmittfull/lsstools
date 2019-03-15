#!/usr/bin/env python
#
# Marcel Schmittfull 2018 (mschmittfull@gmail.com)
#
# Utilities to call nbodykit 0.3
#

from __future__ import print_function,division


from nbodykit.lab import *
import numpy as np
import os

# MS code
from cosmo_model import CosmoModel
from gen_cosmo_fcns import generate_calc_Da
from nbodykit.base.mesh import MeshSource
from pmesh.pm import RealField, ComplexField


def get_cstat(data, statistic, comm=None):
    """
    Compute a collective statistic across all ranks and return as float.
    Must be called by all ranks.
    """
    #if isinstance(data, MeshSource):
    #    data = data.compute().value
    if isinstance(data, RealField) or isinstance(data, ComplexField):
        data = data.value
    else:
        assert type(data) == np.ndarray
    if comm is None:
        from nbodykit import CurrentMPIComm
        comm = CurrentMPIComm.get()

    if statistic == 'min':
        return comm.allreduce(data.min(), op=MPI.MIN)
    elif statistic  == 'max':
        return comm.allreduce(data.max(), op=MPI.MAX)
    elif statistic == 'mean':
        # compute the mean
        csum = comm.allreduce(data.sum())
        csize = comm.allreduce(data.size)
        return csum / csize
    elif statistic == 'rms':
        rsum = comm.allreduce((data**2).sum())
        csize = comm.allreduce(data.size)
        rms = (rsum / csize)**0.5
        return rms
    else:
        raise Exception("Invalid statistic %s" % statistic)

def get_cmean(data, comm=None):
    return get_cstat(data, 'mean', comm=comm)

def get_cmin(data, comm=None):
    return get_cstat(data, 'min', comm=comm)

def get_cmax(data, comm=None):
    return get_cstat(data, 'max', comm=comm)

def get_crms(data, comm=None):
    return get_cstat(data, 'rms', comm=comm)


def get_cstats_string(data, comm=None):
    """
    Get collective statistics (rms, min, mean, max) of data and return as string.
    Must be called by all ranks.
    """
    from collections import OrderedDict
    stat_names = ['rms', 'min', 'mean', 'max']
    cstats = OrderedDict()
    iscomplex = False
    for s in stat_names:
        cstats[s] = get_cstat(data, s)
        if np.iscomplex(cstats[s]):
            iscomplex = True

    if iscomplex:
        return 'rms, min, mean, max: %s %s %s %s' % (
            str(cstats['rms']), str(cstats['min']), str(cstats['mean']), str(cstats['max']))
    else:
        return 'rms, min, mean, max: %g %g %g %g' % (
            cstats['rms'], cstats['min'], cstats['mean'], cstats['max'])


def print_cstats(data, prefix="", logger=None, comm=None):
    """
    Must be called by all ranks.
    """
    if comm is None:
        from nbodykit import CurrentMPIComm
        comm = CurrentMPIComm.get()
    if logger is None:
        from nbodykit import logging
        logger = logging.getLogger("nbkit03_utils")
    cstats = get_cstats_string(data, comm)
    if comm.rank == 0:
        logger.info('%s%s' % (prefix,cstats))


def interpolate_pm_rfield_to_catalog(rfield, catalog, catalog_column_to_save_to,
                                     window='linear', verbose=True):
    """
    Given a pmesh RealField rfield, interpolate to positions of particles
    in a catalog, and save as a column in the catalog.

    The interpolation from regular grid to particle positions is done using
    pmesh RealField.readout, see
    http://rainwoodman.github.io/pmesh/pmesh.pm.html?highlight=readout#pmesh.pm.RealField.readout

    Parameters
    ----------
    rfield : pmesh.RealField

    catalog : nbodykit CatalogSource

    catalog_column_to_save_to : string
    
    window : string
        Can be 'nearest', 'linear', 'cic'.
        For more see http://rainwoodman.github.io/pmesh/_modules/pmesh/window.html#ResampleWindow.

    Returns
    -------
    Nothing; save result in catalog[catalog_column_to_save_to].


    TODO: maybe think about units of Position?
    """

    # get a layout (need window to determine buffer region)
    layout = rfield.pm.decompose(catalog['Position'], smoothing=window)

    # interpolate field to particle positions (use pmesh 'readout' function)
    samples = rfield.readout(catalog['Position'], resampler=window, layout=layout)

    # save into catalog column
    catalog[catalog_column_to_save_to] = samples

    if verbose:
        # print info
        from nbodykit import CurrentMPIComm
        comm = CurrentMPIComm.get()
        #print("%d: rfield read out at catalog:" % comm.rank, type(samples), samples.shape)
        print("%d: interpolated field to catalog and saved to column '%s'" % (comm.rank,catalog_column_to_save_to))




def get_rfield_from_bigfilemesh_file(in_fname, dataset_name='Field', file_scale_factor=None,
                                     desired_scale_factor=None, cosmo_params=None,
                                     normalize=True, set_mean=0.0):
    """
    Read mesh from bigfile. Paint it to get pmesh RealField object. Optionally, linearly rescale
    to desired redshift.

    normalize : boolean
        Normalize the field to set mean == 1. Applied before anything else.

    set_mean : float
        Set the mean. Applied after normalize.
    """
    if not os.path.exists(in_fname):
        raise Exception("could not find %s" % in_fname)
    
    
    bfmesh = BigFileMesh(in_fname, dataset_name)
    if bfmesh.comm.rank==0:
        print("Successfully read %s" % in_fname)
    rfield = bfmesh.paint(mode='real')

    rfield_print_info(rfield, bfmesh.comm, 'File: ')
    
    if normalize:
        cmean = rfield.cmean()
        if cmean != 1.0:
            rfield /= cmean
    if set_mean is not None:
        cmean = rfield.cmean()
        rfield = rfield - cmean + set_mean
        
    rfield_print_info(rfield, bfmesh.comm, 'After normalize and set_mean: ')

    # rescale to desired redshift
    rescalefac = linear_rescale_fac(
        file_scale_factor, desired_scale_factor, cosmo_params=cosmo_params)
    if rescalefac != 1.0:
        if set_mean == 0.0:
            rfield *= rescalefac
        else:
            raise Exception("Must use set_mean=0 to be able to rescale redshift")

    if bfmesh.comm.rank == 0:
        print("%d: Linearly rescale field from a=%g to a=%g, rescalefac=%g" % (
            bfmesh.comm.rank,file_scale_factor, desired_scale_factor, rescalefac))
    
    rfield_print_info(rfield, bfmesh.comm, 
                      'After scaling to redshift z=%g: '%(1./desired_scale_factor-1.))
    
    return rfield


def smoothen_cfield(in_pm_cfield, mode='Gaussian', R=0.0, kmax=None):

    pm_cfield = in_pm_cfield.copy()
    
    # zero pad all k>=kmax
    if kmax is not None:
        def kmax_fcn(k,v,kmax=kmax):
            k2 = sum(ki**2 for ki in k)
            return np.where(k2<kmax**2, v, 0.0*v)
        pm_cfield = pm_cfield.apply(kmax_fcn, out=Ellipsis)
            
    # apply smoothing
    if mode == 'Gaussian':
        if R != 0.0:
            def smoothing_fcn(k,v,R=R):
                k2 = sum(ki**2 for ki in k)
                W = np.exp(-0.5*k2*R**2)
                #print("smoothing: k:", k)
                #print("smoothing: :", W)
                return v*W
            pm_cfield = pm_cfield.apply(smoothing_fcn, out=Ellipsis)
    elif mode == '1-Gaussian':
        if R == 0.0:
            # W=1 so 1-W=0 and (1-W)delta=0
            pm_cfield = 0*pm_cfield
        else:
            def OneMinusW_smoothing_fcn(k,v,R=R):
                k2 = sum(ki**2 for ki in k)
                W = np.exp(-0.5*k2*R**2)
                #print("smoothing: k:", k)
                #print("smoothing: :", W)
                return v*(1.0-W)
            pm_cfield = pm_cfield.apply(OneMinusW_smoothing_fcn, out=Ellipsis)
        
    else:
        raise Exception("Invalid smoothing mode %s" % str(mode))

    return pm_cfield


def calc_quadratic_field(base_rfield=None, base_cfield=None,
                         quadfield=None,
                         smoothing_of_base_field=None,
                         return_in_k_space=False, verbose=False):

    if base_rfield is None and base_cfield is None:
        raise Exception("Only one of base_rfield and base_cfield can be not None")

    if base_rfield is not None:
        # go to k space
        tmp_field = base_rfield.copy()
        base_cfield = tmp_field.r2c()
        del tmp_field

    if verbose:
        from nbodykit import CurrentMPIComm
        comm = CurrentMPIComm.get()

    # apply smoothing
    base_cfield = smoothen_cfield(base_cfield, **smoothing_of_base_field)
    
    # compute quadratic field
    if quadfield == 'growth':
        out_rfield = base_cfield.c2r()
        out_rfield = out_rfield**2

    elif quadfield == 'growth-mean':
        out_rfield = base_cfield.c2r()
        out_rfield = out_rfield**2
        mymean = out_rfield.cmean()
        out_rfield -= mymean

    elif quadfield == 'cube-mean':
        out_rfield = base_cfield.c2r()
        out_rfield = out_rfield**3
        mymean = out_rfield.cmean()
        out_rfield -= mymean

    elif quadfield == 'tidal_G2':
        # Get G2[delta] = d_ijd_ij - delta^2

        # Compute -delta^2(\vx)
        out_rfield = -(base_cfield.copy().c2r())**2

        # Compute d_ij(x). It's symmetric in i<->j so only compute j>=i.
        # d_ij = k_ik_j/k^2*basefield(\vk).
        for idir in range(3):
            for jdir in range(idir,3):
                def my_transfer_function(k, v, idir=idir, jdir=jdir):
                    k2 = sum(kk**2 for kk in k)
                    return np.where(k2 == 0.0, 0*v, k[idir]*k[jdir]*v / (k2))
                dij_k = base_cfield.copy().apply(my_transfer_function)
                del my_transfer_function
                dij_x = dij_k.c2r()
                if verbose:
                    rfield_print_info(dij_x, comm, 'd_%d%d: '%(idir,jdir))

                # Add \sum_{i,j=0..2} d_ij(\vx)d_ij(\vx) 
                #   = [d_00^2+d_11^2+d_22^2 + 2*(d_01^2+d_02^2+d_12^2)]
                if jdir == idir:
                    fac = 1.0
                else:
                    fac = 2.0
                out_rfield += fac * dij_x**2
                del dij_x, dij_k               

    else:
        raise Exception("quadfield %s not implemented" % str(quadfield))
    
    if return_in_k_space:
        return out_rfield.r2c()
    else:
        return out_rfield
    

def get_displacement_from_density_rfield(in_density_rfield, component=None, Psi_type=None,
                                         smoothing=None):
    """
    Given density delta(x) in real space, compute Zeldovich displacemnt Psi_component(x)
    given by Psi_component(\vk) = k_component / k^2 * W(k) * delta(\vk),
    where W(k) is smoothing window.

    Follow http://rainwoodman.github.io/pmesh/intro.html.
    """
    assert (component in [0,1,2])
    assert Psi_type in ['Zeldovich','2LPT','-2LPT']

    # copy so we don't do any in-place changes by accident
    density_rfield = in_density_rfield.copy()

    
    if Psi_type in ['Zeldovich','2LPT','-2LPT']:

        # get zeldovich displacement in direction given by component

        def potential_transfer_function(k, v):
            k2 = sum(ki**2 for ki in k)
            return np.where(k2 == 0.0, 0*v, v / (k2))
            #return v / k2
            #return k[0]

        # get potential pot = delta/k^2
        pot_k = density_rfield.r2c().apply(potential_transfer_function)
        #print("pot_k head:\n", pot_k[:2,:2,:2])

        # apply smoothing
        if smoothing is not None:
            pot_k = smoothen_cfield(pot_k, **smoothing)

            #print("pot_k head2:\n", pot_k[:2,:2,:2])

        # get zeldovich displacement
        def force_transfer_function(k, v, d=component):
            # MS: not sure if we want a factor of -1 here
            return k[d] * 1j * v
        Psi_component_rfield = pot_k.apply(force_transfer_function).c2r()

        
        if Psi_type in ['2LPT','-2LPT']:

            # add 2nd order Psi on top of Zeldovich

            # compute G2
            G2_cfield = calc_quadratic_field(
                base_rfield=in_density_rfield, quadfield='tidal_G2',
                smoothing_of_base_field=smoothing, return_in_k_space=True)

            # compute Psi_2ndorder = -3/14 ik/k^2 G2(k). checked sign: improves rcc with deltaNL
            # if we use -3/14, but get worse rcc when using +3/14.
            Psi_2ndorder_rfield = -3./14. * (
                G2_cfield.apply(potential_transfer_function).apply(force_transfer_function).c2r())

            if Psi_type == '-2LPT':
                # this is just to test sign
                Psi_2ndorder_rfield *= -1.0

            # add 2nd order to Zeldoivhc displacement
            Psi_component_rfield += Psi_2ndorder_rfield
            
        
    return Psi_component_rfield

    

def shift_catalog_by_psi_grid(
        cat=None, pos_column='Position',
        in_displacement_rfields=None,
        pos_units='Mpc/h', displacement_units='Mpc/h', boxsize=None, verbose=False):

    assert type(in_displacement_rfields) in [tuple,list]
    assert len(in_displacement_rfields)==3

    # copy so we don't do any in-place changes by accident
    displacement_rfields = [in_displacement_rfields[0].copy(),
                            in_displacement_rfields[1].copy(),
                            in_displacement_rfields[2].copy()]
    
    for direction in range(3):
        if cat.comm.rank==0:
            print("Get Psi in direction:", direction)
        displacement_rfield = displacement_rfields[direction]

        # convert units so that displacement and pos have consistent units
        if displacement_units == 'Mpc/h':
            if pos_units=='Mpc/h':
                pass
            elif pos_units=='0to1':
                displacement_rfield /= boxsize
            else:
                raise Exception("Invalid units: %s" % str(pos_units))
        else:
            raise Exception("Invalid units: %s" % str(displacement_units))

        if verbose:
            print("%d: displacement_rfield_%d: min, mean, max, rms:"%(cat.comm.rank,direction),
                  np.min(displacement_rfield), np.mean(displacement_rfield), np.max(displacement_rfield),
                  np.mean(displacement_rfield**2)**0.5)

        # interpolate Psi_i(x) to catalog ptcle positions (this shouldn't use weights)
        interpolate_pm_rfield_to_catalog(
            displacement_rfield, cat, catalog_column_to_save_to='TMP_Psi_%d'%direction)

        # find max displacement and print
        maxpos = np.max(np.abs(cat[pos_column]))
        if verbose:      
            print("%d: Catalog displacement_%d min, mean, max, rms:" % (cat.comm.rank, direction),
                np.min(np.array(cat['TMP_Psi_%d'%direction])),
                np.mean(np.array(cat['TMP_Psi_%d'%direction])),
                np.max(np.array(cat['TMP_Psi_%d'%direction])),
                np.mean(np.array(cat['TMP_Psi_%d'%direction])**2)**0.5)
    
    # shift positions
    if cat.comm.rank==0:
        print("add psi")
    cat[pos_column] = transform.StackColumns(
        cat[pos_column][:,0] + cat['TMP_Psi_0'],
        cat[pos_column][:,1] + cat['TMP_Psi_1'],
        cat[pos_column][:,2] + cat['TMP_Psi_2'])
    print("%d: done adding psi" % cat.comm.rank)
    
    # save memory
    #for direction in range(3):
    #    cat['TMP_Psi_%d'%direction] = 0.0


    # box wrap shifted positions to [mincoord,L[
    if pos_units=='0to1':
        cat[pos_column] = cat[pos_column] % 1.0
        assert np.all(cat[pos_column]<=1.0)
        assert np.all(cat[pos_column]>=0.0)
    elif pos_units=='Mpc/h':
        cat[pos_column] = cat[pos_column] % boxsize
        assert np.all(cat[pos_column]<=boxsize)
        assert np.all(cat[pos_column]>=0.0)
    else:
        raise Exception("invalid units %s" % str(pos_units))

    print("%d: done shifting catalog" % cat.comm.rank)


def rfield_cmin(rfield, comm):
    """
    collective min
    """
    return comm.allreduce(rfield.value.min(), op=MPI.MIN)

def rfield_cmax(rfield, comm):
    """
    collective max
    """
    return comm.allreduce(rfield.value.max(), op=MPI.MAX)


def rfield_print_info(rfield, comm, text=''):
    cmean = rfield.cmean()
    #cmin = rfield_cmin(rfield, comm) # crashes in big jobs
    #cmax = rfield_cmax(rfield, comm)
    if comm.rank==0:
        if len(np.array(rfield))>0:
            local_min = np.min(np.array(rfield))
            local_max = np.max(np.array(rfield))
        else:
            local_min = None
            local_max = None
        print("%d: %s: local_min=%s, cmean=%g, local_max=%s" % (
            comm.rank, text, str(local_min),cmean,str(local_max)))



def linear_rescale_fac(current_scale_factor, desired_scale_factor,
                   cosmo_params=None):
    if desired_scale_factor is None or current_scale_factor is None:
        raise Exception("scale factors must be not None")
    if desired_scale_factor > 1.0 or current_scale_factor > 1.0:
        raise Exception("scale factors must be <=1")

    if desired_scale_factor == current_scale_factor:
        rescalefac = 1.0
    else:
        # Factor to linearly rescale delta to desired redshift                           
        assert (cosmo_params is not None)
        cosmo = CosmoModel(**cosmo_params)
        calc_Da = generate_calc_Da(cosmo=cosmo, verbose=False)
        rescalefac = calc_Da(desired_scale_factor) / calc_Da(current_scale_factor)
        #del cosmo
    return rescalefac



