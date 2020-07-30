from __future__ import print_function, division

from nbodykit.lab import *
import numpy as np
import os
from copy import copy

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
    elif statistic == 'max':
        return comm.allreduce(data.max(), op=MPI.MAX)
    elif statistic == 'mean':
        # compute the mean
        csum = comm.allreduce(data.sum(), op=MPI.SUM)
        csize = comm.allreduce(data.size, op=MPI.SUM)
        return csum / float(csize)
    elif statistic == 'rms':
        mean = get_cmean(data, comm=comm)
        rsum = comm.allreduce(((data-mean)**2).sum())
        csize = comm.allreduce(data.size)
        rms = (rsum / float(csize))**0.5
        return rms
    elif statistic == 'sum':
        csum = comm.allreduce(data.sum(), op=MPI.SUM)
        return csum
    elif statistic == 'sqsum':
        csqsum = comm.allreduce((data**2).sum(), op=MPI.SUM)
        return csqsum
    else:
        raise Exception("Invalid statistic %s" % statistic)


def get_cmean(data, comm=None):
    return get_cstat(data, 'mean', comm=comm)

def get_csum(data, comm=None):
    return get_cstat(data, 'sum', comm=comm)

def get_csqsum(data, comm=None):
    return get_cstat(data, 'sqsum', comm=comm)

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
        return 'rms, min, mean, max: %s %s %s %s' % (str(
            cstats['rms']), str(cstats['min']), str(
                cstats['mean']), str(cstats['max']))
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
        logger.info('%s%s' % (prefix, cstats))
        
        print('%s%s' % (prefix, cstats))



def interpolate_pm_rfield_to_catalog(rfield,
                                     catalog,
                                     catalog_column_to_save_to,
                                     window='linear',
                                     verbose=True):
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
    samples = rfield.readout(catalog['Position'],
                             resampler=window,
                             layout=layout)

    # save into catalog column
    catalog[catalog_column_to_save_to] = samples

    if verbose:
        # print info
        from nbodykit import CurrentMPIComm
        comm = CurrentMPIComm.get()
        #print("%d: rfield read out at catalog:" % comm.rank, type(samples), samples.shape)
        print("%d: interpolated field to catalog and saved to column '%s'" %
              (comm.rank, catalog_column_to_save_to))


def get_rfield_from_bigfilemesh_file(in_fname,
                                     dataset_name='Field',
                                     file_scale_factor=None,
                                     desired_scale_factor=None,
                                     cosmo_params=None,
                                     normalize=True,
                                     set_mean=0.0):
    """
    Read mesh from bigfile. Paint it to get pmesh RealField object. Optionally, 
    linearly rescale to desired redshift.

    normalize : boolean
        Normalize the field to set mean == 1. Applied before anything else.

    set_mean : float
        Set the mean. Applied after normalize.
    """
    if not os.path.exists(in_fname):
        raise Exception("could not find %s" % in_fname)

    bfmesh = BigFileMesh(in_fname, dataset_name)
    if bfmesh.comm.rank == 0:
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
    rescalefac = linear_rescale_fac(file_scale_factor,
                                    desired_scale_factor,
                                    cosmo_params=cosmo_params)
    if rescalefac != 1.0:
        if set_mean == 0.0:
            rfield *= rescalefac
        else:
            raise Exception(
                "Must use set_mean=0 to be able to rescale redshift")

    if bfmesh.comm.rank == 0:
        print("%d: Linearly rescale field from a=%g to a=%g, rescalefac=%g" %
              (bfmesh.comm.rank, file_scale_factor, desired_scale_factor,
               rescalefac))

    rfield_print_info(
        rfield, bfmesh.comm,
        'After scaling to redshift z=%g: ' % (1. / desired_scale_factor - 1.))

    return rfield


def smoothen_cfield(in_pm_cfield, mode='Gaussian', R=0.0, kmax=None):

    pm_cfield = in_pm_cfield.copy()

    # zero pad all k>=kmax
    if kmax is not None:

        def kmax_fcn(k, v, kmax=kmax):
            k2 = sum(ki**2 for ki in k)
            return np.where(k2 < kmax**2, v, 0.0 * v)

        pm_cfield = pm_cfield.apply(kmax_fcn, out=Ellipsis)

    # apply smoothing
    if mode == 'Gaussian':
        if R != 0.0:

            def smoothing_fcn(k, v, R=R):
                k2 = sum(ki**2 for ki in k)
                W = np.exp(-0.5 * k2 * R**2)
                #print("smoothing: k:", k)
                #print("smoothing: :", W)
                return v * W

            pm_cfield = pm_cfield.apply(smoothing_fcn, out=Ellipsis)
    elif mode == '1-Gaussian':
        if R == 0.0:
            # W=1 so 1-W=0 and (1-W)delta=0
            pm_cfield = 0 * pm_cfield
        else:

            def OneMinusW_smoothing_fcn(k, v, R=R):
                k2 = sum(ki**2 for ki in k)
                W = np.exp(-0.5 * k2 * R**2)
                #print("smoothing: k:", k)
                #print("smoothing: :", W)
                return v * (1.0 - W)

            pm_cfield = pm_cfield.apply(OneMinusW_smoothing_fcn, out=Ellipsis)

    else:
        raise Exception("Invalid smoothing mode %s" % str(mode))

    return pm_cfield


def apply_smoothing(mesh_source=None,
                    mode='Gaussian',
                    R=0.0,
                    kmax=None,
                    additional_props=None):
    """
    Apply smoothing to a mesh_source field.

    Parameters
    ----------
    mode: string
        'Gaussian' or 'SharpK' or 'kstep' or 'InverseGaussian'.

    kmax : float
        Additionally to smoothing, set field to 0 if k>kmax.

    Returns
    -------
    New mesh_source object that contains the smoothed field.
    """
    out = copy(mesh_source)

    if kmax is not None:
        # zero pad all k>kmax
        def kmax_cutter(k3vec, val):
            # k3vec = [k_x, k_y, k_z]
            absk = np.sqrt(sum(ki**2 for ki in k3vec))  # absk on the mesh
            #absk = (sum(ki ** 2 for ki in k3vec))**0.5 # absk on the mesh
            return np.where(absk <= kmax,
                            val,
                            np.zeros(val.shape, dtype=val.dtype))

        # append column
        # self.append_column(column,
        #     self.G[column].apply(kmax_cutter, mode='complex', kind='wavenumber'),
        #     column_info=column_info)
        # directly modify column (add action)
        out = out.apply(kmax_cutter, mode='complex', kind='wavenumber')

    if mode == 'Gaussian':
        if R != 0.0:

            def smoothing_fcn(k3vec, val):
                absk = np.sqrt(sum(ki**2 for ki in k3vec))  # absk on the mesh
                return np.exp(-(R * absk)**2 / 2.0) * val

            #self.G[column] = self.G[column].apply(smoothing_fcn, kind='wavenumber', mode='complex')
            out = out.apply(smoothing_fcn,
                                    kind='wavenumber',
                                    mode='complex')
            #print_cstats(out.compute(mode='complex'), prefix='gridk after smoothing ', logger=self.logger)

    elif mode == 'InverseGaussian':
        # divide by Gaussian smoothing kernel; set to 0 at high k
        if R != 0.0:

            def smoothing_fcn(k3vec, val):
                absk = np.sqrt(sum(ki**2 for ki in k3vec))  # absk on the mesh
                return np.where(R * absk <= 5.0,
                                np.exp(+(R * absk)**2 / 2.0) * val, 0.0 * val)

            out = out.apply(smoothing_fcn,
                                    kind='wavenumber',
                                    mode='complex')

    elif mode == 'kstep':
        assert type(additional_props) == dict
        assert additional_props.has_key('step_kmin')
        assert additional_props.has_key('step_kmax')
        step_kmin = additional_props['step_kmin']
        step_kmax = additional_props['step_kmax']

        #self.compute_helper_grid('ABSK')

        def kstep_cutter(k3vec, val):
            # k3vec = [k_x, k_y, k_z]
            absk = np.sqrt(sum(ki**2 for ki in k3vec))  # absk on the mesh
            return np.where((absk >= step_kmin) & (absk < step_kmax), val,
                            np.zeros(val.shape, dtype=val.dtype))

        out = out.apply(kstep_cutter, mode='complex', kind='wavenumber')

    else:
        raise Exception('Invalid smoothing mode %s' % mode)


    return out


def calc_quadratic_field(
        base_field_mesh=None,
        second_base_field_mesh=None,
        quadfield=None,
        smoothing_of_base_field=None,
        #return_in_k_space=False,
        verbose=False):
    """
    Calculate quadratic field, essentially by squaring base_field_mesh
    with filters applied before squaring. 

    Parameters
    ----------
    base_field_mesh : MeshSource object, typically a FieldMesh object
        Input field that will be squared.

    second_base_field_mesh : MeshSource object, typically a FieldMesh object
        Use this to multiply two fields, e.g. delta1*delta2 or G2[delta1,delta2].
        Only implemented for tidal_G2 at the moment.

    quadfield : string
        Represents quadratic field to be calculated. Can be
        - 'tidal_s2': Get s^2 = 3/2*s_ij*s_ij = 3/2*[d_ij d_ij - 1/3 delta^2] 
                      = 3/2*d_ijd_ij - delta^2/2,
                      where s_ij = (k_ik_j/k^2-delta_ij^K/3)basefield(\vk) and
                      d_ij = k_ik_j/k^2*basefield(\vk).
        - 'tidal_G2': Get G2[delta] = d_ij d_ij - delta^2. This is orthogonal to
                      delta^2 at low k which can be useful; also see Assassi et al (2014).
        - 'shift': Get shift=\vPsi\cdot\vnabla\basefield(\vx), where vPsi=-ik/k^2*basefield(k).
        - 'PsiNablaDelta': Same as 'shift'
        - 'growth': Get delta^2(\vx)
        - 'F2': Get F2[delta] = 17/21 delta^2 + shift + 4/21 tidal_s2
                              = delta^2 + shift + 2/7 tidal_G2

    Returns
    -------
    Return the calculated (Ngrid,Ngrid,Ngrid) field as a FieldMesh object.

    """
    comm = CurrentMPIComm.get()

    if second_base_field_mesh is not None:
        if not quadfield.endswith('two_meshs'):
            raise Exception(
                'second_base_field_mesh not implemented for quadfield %s' 
                % quadfield)

    # apply smoothing
    if smoothing_of_base_field is not None:
        base_field_mesh = apply_smoothing(mesh_source=base_field_mesh,
                                          **smoothing_of_base_field)
        if second_base_field_mesh is not None:
            second_base_field_mesh = apply_smoothing(mesh_source=second_base_field_mesh,
                                          **smoothing_of_base_field)


    # compute quadratic (or cubic) field
    if quadfield == 'growth':
        out_rfield = base_field_mesh.compute(mode='real')**2

    elif quadfield == 'growth_two_meshs':
        out_rfield = (
            base_field_mesh.compute(mode='real')
            *second_base_field_mesh.compute(mode='real'))

    elif quadfield == 'growth-mean':
        out_rfield = base_field_mesh.compute(mode='real')**2
        mymean = out_rfield.cmean()
        out_rfield -= mymean

    elif quadfield == 'cube-mean':
        out_rfield = base_field_mesh.compute(mode='real')**3
        mymean = out_rfield.cmean()
        out_rfield -= mymean

    elif quadfield == 'tidal_G2':
        # Get G2[delta] = d_ijd_ij - delta^2

        if second_base_field_mesh is None:

            # Compute -delta^2(\vx)
            out_rfield = -base_field_mesh.compute(mode='real')**2

            # Compute d_ij(x). It's symmetric in i<->j so only compute j>=i.
            # d_ij = k_ik_j/k^2*basefield(\vk).
            for idir in range(3):
                for jdir in range(idir, 3):

                    def my_transfer_function(k3vec, val, idir=idir, jdir=jdir):
                        kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                        kk[kk == 0] = 1
                        return k3vec[idir] * k3vec[jdir] / kk * val

                    dij_k = base_field_mesh.apply(my_transfer_function,
                                                  mode='complex',
                                                  kind='wavenumber')
                    del my_transfer_function
                    # do fft and convert field_mesh to RealField object
                    dij_x = dij_k.compute(mode='real')
                    if verbose:
                        rfield_print_info(dij_x, comm, 'd_%d%d: ' % (idir, jdir))

                    # Add \sum_{i,j=0..2} d_ij(\vx)d_ij(\vx)
                    #   = [d_00^2+d_11^2+d_22^2 + 2*(d_01^2+d_02^2+d_12^2)]
                    if jdir == idir:
                        fac = 1.0
                    else:
                        fac = 2.0
                    out_rfield += fac * dij_x**2
                    del dij_x, dij_k

        else:
            raise Exception('use tidal_G2_two_meshs')

    elif quadfield == 'tidal_G2_two_meshs':
        # use second_base_field_mesh
        # Compute -delta1*delta2(\vx)
        out_rfield = -(
            base_field_mesh.compute(mode='real')
            * second_base_field_mesh.compute(mode='real') )

        # Compute d_ij(x). It's not symmetric in i<->j so compute all i,j
        # d_ij = k_ik_j/k^2*basefield(\vk).
        for idir in range(3):
            for jdir in range(3):

                def my_transfer_function(k3vec, val, idir=idir, jdir=jdir):
                    kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                    kk[kk == 0] = 1
                    return k3vec[idir] * k3vec[jdir] / kk * val

                dij_k = base_field_mesh.apply(
                    my_transfer_function,
                    mode='complex',
                    kind='wavenumber')
                second_dij_k = second_base_field_mesh.apply(
                    my_transfer_function,
                    mode='complex',
                    kind='wavenumber')
                del my_transfer_function
                
                # do fft and convert field_mesh to RealField object
                dij_x = dij_k.compute(mode='real')
                second_dij_x = second_dij_k.compute(mode='real')
                if verbose:
                    rfield_print_info(
                        dij_x, comm, 'd_%d%d: ' % (idir, jdir))
                    rfield_print_info(
                        second_dij_x, comm, 'd_%d%d: ' % (idir, jdir))

                # Add \sum_{i,j=0..2} d_ij(\vx)second_d_ij(\vx)
                # 29 Jul 2020: Had bug before, assumign symmetry.
                out_rfield += dij_x * second_dij_x

                del dij_x, dij_k, second_dij_x, second_dij_k


    elif quadfield == 'tidal_s2':
        # Get s^2 = 3/2*d_ijd_ij - delta^2/2
        # Compute -delta^2(\vx)/2
        out_rfield = -base_field_mesh.compute(mode='real')**2 / 2.0

        # Compute d_ij(x). It's symmetric in i<->j so only compute j>=i.
        # d_ij = k_ik_j/k^2*basefield(\vk).
        for idir in range(3):
            for jdir in range(idir, 3):

                def my_transfer_function(k3vec, val, idir=idir, jdir=jdir):
                    kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                    kk[kk == 0] = 1
                    return k3vec[idir] * k3vec[jdir] * val / kk

                dij_k = base_field_mesh.apply(my_transfer_function,
                                              mode='complex',
                                              kind='wavenumber')
                del my_transfer_function
                dij_x = dij_k.compute(mode='real')
                if verbose:
                    rfield_print_info(dij_x, comm, 'd_%d%d: ' % (idir, jdir))

                # Add \sum_{i,j=0..2} d_ij(\vx)d_ij(\vx)
                #   = [d_00^2+d_11^2+d_22^2 + 2*(d_01^2+d_02^2+d_12^2)]
                if jdir == idir:
                    fac = 1.0
                else:
                    fac = 2.0
                out_rfield += fac * 1.5 * dij_x**2
                del dij_x, dij_k

    elif quadfield in ['shift', 'PsiNablaDelta']:
        # Get shift = \vPsi\cdot\nabla\delta
        for idir in range(3):
            # compute Psi_i
            def Psi_i_fcn(k3vec, val, idir=idir):
                kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                kk[kk == 0] = 1
                return -1.0j * k3vec[idir] * val / kk

            Psi_i_x = base_field_mesh.apply(
                Psi_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')

            # compute nabla_i delta
            def grad_i_fcn(k3vec, val, idir=idir):
                return -1.0j * k3vec[idir] * val

            nabla_i_delta_x = base_field_mesh.apply(
                grad_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')

            # multiply and add up in x space
            if idir == 0:
                out_rfield = Psi_i_x * nabla_i_delta_x
            else:
                out_rfield += Psi_i_x * nabla_i_delta_x

    elif quadfield in ['shift_two_meshs', 'PsiNablaDelta_two_meshs']:
        # Get shift = 0.5(\vPsi_1\cdot\nabla\delta_2 + \vPsi_2\cdot\nabla\delta_1)
        out_rfield = None
        for idir in range(3):
            
            def Psi_i_fcn(k3vec, val, idir=idir):
                kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                kk[kk == 0] = 1
                return -1.0j * k3vec[idir] * val / kk
            
            def grad_i_fcn(k3vec, val, idir=idir):
                return -1.0j * k3vec[idir] * val

            # compute Psi_i of mesh1
            Psi_i_x = base_field_mesh.apply(
                Psi_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')

            # compute nabla_i delta of mesh2
            nabla_i_delta_x = second_base_field_mesh.apply(
                grad_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')

            # multiply and add up in x space
            if out_rfield is None:
                out_rfield = 0.5 * Psi_i_x * nabla_i_delta_x
            else:
                out_rfield += 0.5 * Psi_i_x * nabla_i_delta_x
            del Psi_i_x, nabla_i_delta_x

            # also swap mesh1 and mesh2
            # compute Psi_i of mesh2
            Psi_i_x_prime = second_base_field_mesh.apply(
                Psi_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')

            # compute nabla_i delta of mesh1
            nabla_i_delta_x_prime = base_field_mesh.apply(
                grad_i_fcn, mode='complex',
                kind='wavenumber').compute(mode='real')
            
            out_rfield += 0.5 * Psi_i_x_prime * nabla_i_delta_x_prime


    # elif quadfield in ['tidal_G2_par_LOS100', 'tidal_G2_par_LOS010', 'tidal_G2_par_LOS001']:
    #     if quadfield.endswith('100'):
    #         LOS_dir = 0
    #     elif quadfield.endswith('010'):
    #         LOS_dir = 1
    #     elif quadfield.endswith('001'):
    #         LOS_dir = 2
    #     else:
    #         raise Exception('invalid LOS')

    #     # compute nabla_parallel^2/nabla^2 G2
    #     G2_mesh = calc_quadratic_field(
    #         quadfield='tidal_G2',
    #         base_field_mesh=base_field_mesh,
    #         smoothing_of_base_field=smoothing_of_base_field,
    #         verbose=verbose)

    #     def my_parallel_transfer_function(k3vec, val, idir=LOS_dir, jdir=LOS_dir):
    #         kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
    #         kk[kk == 0] = 1
    #         # MS 25 Jul 2020: Include minus sign because (ik_2)^2=-k_2^2.
    #         return -k3vec[idir] * k3vec[jdir] * val / kk

    #     G2_parallel = G2_mesh.apply(my_parallel_transfer_function,
    #                                   mode='complex',
    #                                   kind='wavenumber')
    #     del my_parallel_transfer_function

    #     out_rfield = G2_parallel.compute(mode='real')
    #     del G2_parallel, G2_mesh



    elif quadfield == 'F2':
        # F2 = delta^2...
        out_rfield = calc_quadratic_field(
            quadfield='growth',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... - shift
        out_rfield -= calc_quadratic_field(
            quadfield='shift',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... + 2/7 tidal_G2
        out_rfield += 2. / 7. * calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')

    elif quadfield == 'F2_two_meshs':
        # F2 = delta^2...
        out_rfield = calc_quadratic_field(
            quadfield='growth_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... - shift
        out_rfield -= calc_quadratic_field(
            quadfield='shift_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... + 2/7 tidal_G2
        out_rfield += 2. / 7. * calc_quadratic_field(
            quadfield='tidal_G2_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')

    elif quadfield == 'velocity_G2':
        # G2 = delta^2...
        out_rfield = calc_quadratic_field(
            quadfield='growth',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... - shift
        out_rfield -= calc_quadratic_field(
            quadfield='shift',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... + 4/7 tidal_G2
        out_rfield += 4. / 7. * calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')

    elif quadfield == 'velocity_G2_two_meshs':
        # G2 = delta^2...
        out_rfield = calc_quadratic_field(
            quadfield='growth_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... - shift
        out_rfield -= calc_quadratic_field(
            quadfield='shift_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')
        # ... + 4/7 tidal_G2
        out_rfield += 4. / 7. * calc_quadratic_field(
            quadfield='tidal_G2_two_meshs',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')

    elif quadfield in ['delta_G2', 'G2_delta']:
        # Get G2[delta] * delta
        out_rfield = (
            base_field_mesh.compute(mode='real')
            * calc_quadratic_field(
                quadfield='tidal_G2',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose).compute(mode='real'))

        # take out the mean (already close to 0 but still subtract)
        mymean = out_rfield.cmean()
        if comm.rank == 0:
            print('Subtract mean of G2*delta: %g' % mymean)
        out_rfield -= mymean 


    elif quadfield in ['delta_G2_par_LOS100', 'delta_G2_par_LOS010', 'delta_G2_par_LOS001']:
        # Get delta * G2_parallel[delta]
        out_rfield = (
            base_field_mesh.compute(mode='real')
            * calc_quadratic_field(
                quadfield='tidal_G2_par_LOS%s' % (quadfield[-3:]),
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose).compute(mode='real'))

        # take out the mean (already close to 0 but still subtract)
        mymean = out_rfield.cmean()
        if comm.rank == 0:
            print('Subtract mean of G2par*delta: %g' % mymean)
        out_rfield -= mymean 


    elif quadfield in ['tidal_G3', 'tidal_G3_flipped_dij_sign']:
        # Get G3[delta]

        # Have 3/2 G2 delta = 3/2 (p1.p2)^2/(p1^2 p2^2) - 3/2
        # so
        # G3 = 3/2 G2 delta + delta^3 - (p1.p2)(p2.p3)(p1.p3)/(p1^2 p2^2 p3^2)

        # Compute 1 * delta^3(\vx)
        out_rfield = base_field_mesh.compute(mode='real')**3

        # Add 3/2 delta * G2[delta]
        out_rfield += (3./2. 
            * base_field_mesh.compute(mode='real')
            * calc_quadratic_field(
                quadfield='tidal_G2',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose).compute(mode='real'))

        # Compute ppp = (p1.p2)(p2.p3)(p1.p3)/(p1^2 p2^2 p3^2)
        # = k.q k.p q.p / (k^2 q^2 p^2)
        # = k_i q_i k_j p_j q_l p_l / (k^2 q^2 p^2)
        # = sum_ijl d_ij(k) d_il(q) d_jl(p)
        # where we denoted p1=k, p2=q, p3=p.

        # Compute d_ij(x). It's symmetric in i<->j so only compute j>=i.
        # d_ij = k_ik_j/k^2*basefield(\vk).
        dij_x_dict = {}
        for idir in range(3):
            for jdir in range(idir, 3):

                def my_transfer_function(k3vec, val, idir=idir, jdir=jdir):
                    kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                    kk[kk == 0] = 1
                    return k3vec[idir] * k3vec[jdir] * val / kk

                dij_k = base_field_mesh.apply(my_transfer_function,
                                              mode='complex',
                                              kind='wavenumber')
                del my_transfer_function
                # do fft and convert field_mesh to RealField object
                dij_x = dij_k.compute(mode='real')
                del dij_k

                if verbose:
                    rfield_print_info(dij_x, comm, 'd_%d%d: ' % (idir, jdir))

                if quadfield == 'tidal_G3_flipped_dij_sign':
                    # flip sign of dij
                    dij_x *= -1.0

                dij_x_dict[(idir,jdir)] = dij_x
                del dij_x

        # get j<i by symmetry
        def get_dij_x(idir, jdir):
            if jdir>=idir:
                return dij_x_dict[(idir,jdir)]
            else:
                return dij_x_dict[(jdir,idir)]

        # Compute - sum_ijl d_ij(k) d_il(q) d_jl(p)
        for idir in range(3):
            for jdir in range(3):
                for ldir in range(3):
                    out_rfield -= (
                          get_dij_x(idir,jdir)
                        * get_dij_x(idir,ldir)
                        * get_dij_x(jdir,ldir) )

        # take out the mean (already close to 0 but still subtract)
        mymean = out_rfield.cmean()
        if comm.rank == 0:
            print('Subtract mean of G3: %g' % mymean)
        out_rfield -= mymean 

        if verbose:
            rfield_print_info(out_rfield, comm, 'G3: ')


    elif quadfield == 'Gamma3':
        # Get Gamma3[delta] = -4/7 * G2[G2[delta,delta],delta]

        tmp_G2 = calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        out_rfield = -4./7. * calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=tmp_G2,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose).compute(mode='real')


    elif quadfield in [
        'delta1_par_LOS100', 'delta1_par_LOS010', 'delta1_par_LOS001', 
        'tidal_G2_par_LOS100', 'tidal_G2_par_LOS010', 'tidal_G2_par_LOS001', 
        'Gamma3_par_LOS100', 'Gamma3_par_LOS010', 'Gamma3_par_LOS001',
        'tidal_G3_par_LOS100', 'tidal_G3_par_LOS010', 'tidal_G3_par_LOS001', 
        'tidal_G3_flipped_dij_sign_par_LOS100', 'tidal_G3_flipped_dij_sign_par_LOS010', 'tidal_G3_flipped_dij_sign_par_LOS001'
    ]:
        if quadfield.endswith('LOS100'):
            LOS_dir = 0
        elif quadfield.endswith('LOS010'):
            LOS_dir = 1
        elif quadfield.endswith('LOS001'):
            LOS_dir = 2
        else:
            raise Exception('invalid LOS string in %s' % str(quadfield))


        if quadfield.startswith('delta1_par'):
            tmp_base_quadfield_str = 'delta1'
        elif quadfield.startswith('tidal_G2_par'):
            tmp_base_quadfield_str = 'tidal_G2'
        elif quadfield.startswith('Gamma3_par'):
            tmp_base_quadfield_str = 'Gamma3'
        elif quadfield.startswith('tidal_G3_par'):
            tmp_base_quadfield_str = 'tidal_G3'
        elif quadfield.startswith('tidal_G3_flipped_dij_sign_par'):
            tmp_base_quadfield_str = 'tidal_G3_flipped_dij_sign'
        else:
            raise Exception('Invalid quadfield %s' % str(quadfield))

        # compute nabla_parallel^2/nabla^2 tmp_base_quadfield
        if tmp_base_quadfield_str == 'delta1':
            tmp_base_quadfield_mesh = FieldMesh(base_field_mesh.compute(
                mode='real'))
        else:
            tmp_base_quadfield_mesh = calc_quadratic_field(
                quadfield=tmp_base_quadfield_str,
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose)

        def my_parallel_transfer_function(k3vec, val, idir=LOS_dir, jdir=LOS_dir):
            kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
            kk[kk == 0] = 1
            # MS 25 Jul 2020: Include minus sign because (ik_2)^2=-k_2^2.
            return -k3vec[idir] * k3vec[jdir] * val / kk

        tmp_base_quadfield_parallel = tmp_base_quadfield_mesh.apply(
            my_parallel_transfer_function,
            mode='complex',
            kind='wavenumber')
        del my_parallel_transfer_function

        out_rfield = tmp_base_quadfield_parallel.compute(mode='real')
        del tmp_base_quadfield_parallel, tmp_base_quadfield_mesh


    elif quadfield in ['velocity_G2_par_LOS001_two_meshs']:
        LOS_dir = 2
        tmp_base_quadfield_str = 'velocity_G2_two_meshs'

        # compute nabla_parallel^2/nabla^2 tmp_base_quadfield
        tmp_base_quadfield_mesh = calc_quadratic_field(
            quadfield=tmp_base_quadfield_str,
            base_field_mesh=base_field_mesh,
            second_base_field_mesh=second_base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        def my_parallel_transfer_function(k3vec, val, idir=LOS_dir, jdir=LOS_dir):
            kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
            kk[kk == 0] = 1
            # MS 29 Jul 2020: No minus sign because -(ik_2)^2/k^2=k_2^2/k^2.
            return k3vec[idir] * k3vec[jdir] * val / kk

        tmp_base_quadfield_parallel = tmp_base_quadfield_mesh.apply(
            my_parallel_transfer_function,
            mode='complex',
            kind='wavenumber')
        del my_parallel_transfer_function

        out_rfield = tmp_base_quadfield_parallel.compute(mode='real')
        del tmp_base_quadfield_parallel, tmp_base_quadfield_mesh

    elif quadfield == 'delta1_par_G2_par_LOS001':
        out_rfield = (
            calc_quadratic_field(
                quadfield='delta1_par_LOS001',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose).compute(mode='real')
            * calc_quadratic_field(
                quadfield='tidal_G2_par_LOS001',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose).compute(mode='real') )


    elif quadfield in ['RSDLPT_Q2_LOS001']:
        # Q2 = sum_i d_i2 d_i2
        # d_ij = -k_ik_j/k^2*basefield(\vk).

        out_rfield = None
        jdir = 2
        for idir in range(3):

            def my_transfer_function_dij(k3vec, val, idir=idir, jdir=jdir):
                kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                kk[kk == 0] = 1
                return -k3vec[idir] * k3vec[jdir] * val / kk

            dij_k = base_field_mesh.apply(my_transfer_function_dij,
                                          mode='complex',
                                          kind='wavenumber')
            del my_transfer_function_dij
            # do fft and convert field_mesh to RealField object
            dij_x = dij_k.compute(mode='real')
            del dij_k

            if verbose:
                rfield_print_info(dij_x, comm, 'd_%d%d: ' % (idir, jdir))

            if out_rfield is None:
                out_rfield = dij_x**2
            else:
                out_rfield += dij_x**2


    elif quadfield in ['RSDLPT_K3_LOS001']:
        # K3 = sum_i d_i2 dG2_i2
        # where d_ij = -k_ik_j/k^2*basefield(\vk)
        # and dG2_ij = -k_ik_j/k^2*G2(\vk).
        tmp_G2_mesh = calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        out_rfield = None
        jdir = 2
        for idir in range(3):

            def my_transfer_function_dij(k3vec, val, idir=idir, jdir=jdir):
                kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                kk[kk == 0] = 1
                return -k3vec[idir] * k3vec[jdir] * val / kk

            dij_x = base_field_mesh.apply(my_transfer_function_dij,
                                          mode='complex',
                                          kind='wavenumber').compute(mode='real')

            dijG2_x = tmp_G2_mesh.apply(my_transfer_function_dij,
                                          mode='complex',
                                          kind='wavenumber').compute(mode='real')

            del my_transfer_function_dij

            if out_rfield is None:
                out_rfield = dij_x * dijG2_x
            else:
                out_rfield += dij_x * dijG2_x


    elif quadfield in ['RSDLPT_S3Ia','RSDLPT_S3IIa_LOS001']:
        # S3Ia = sum_i d_i where d_i = - (k_i/k^2 G2) (k_i delta_1)
        # (or delta_1 -> delta_1 parallel for S3IIa)
        tmp_G2_mesh = calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        if quadfield == 'RSDLPT_S3Ia':
            tmp_delta1_mesh = FieldMesh(base_field_mesh.compute(mode='real'))
        elif quadfield == 'RSDLPT_S3IIa_LOS001':
            tmp_delta1_mesh =  calc_quadratic_field(
                quadfield='delta1_par_LOS001',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose)

        out_rfield = None
        for idir in range(3):

            def my_trf_for_G2(k3vec, val, idir=idir):
                kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                kk[kk == 0] = 1
                return k3vec[idir] * val / kk

            def my_trf_for_delta1(k3vec, val, idir=idir):
                return k3vec[idir] * val

            G2_filtered = tmp_G2_mesh.apply(my_trf_for_G2,
                                          mode='complex',
                                          kind='wavenumber').compute(mode='real')

            delta1_filtered = tmp_delta1_mesh.apply(my_trf_for_delta1,
                                          mode='complex',
                                          kind='wavenumber').compute(mode='real')

            del my_trf_for_G2, my_trf_for_delta1

            if out_rfield is None:
                out_rfield = -G2_filtered * delta1_filtered
            else:
                out_rfield -= G2_filtered * delta1_filtered


    elif quadfield in ['RSDLPT_S3Ib_LOS001','RSDLPT_S3IIb_LOS001']:
        # S3Ia = - (k_2/k^2 G2) (k_2 delta_1)

        tmp_G2_mesh = calc_quadratic_field(
            quadfield='tidal_G2',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        if quadfield == 'RSDLPT_S3Ib_LOS001':
            tmp_delta1_mesh = FieldMesh(base_field_mesh.compute(mode='real'))
        elif quadfield == 'RSDLPT_S3IIb_LOS001':
            tmp_delta1_mesh =  calc_quadratic_field(
                quadfield='delta1_par_LOS001',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose)

        idir = 2

        def my_trf_for_G2(k3vec, val, idir=idir):
            kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
            kk[kk == 0] = 1
            return k3vec[idir] * val / kk

        def my_trf_for_delta1(k3vec, val, idir=idir):
            return k3vec[idir] * val

        G2_filtered = tmp_G2_mesh.apply(my_trf_for_G2,
                                      mode='complex',
                                      kind='wavenumber').compute(mode='real')

        delta1_filtered = tmp_delta1_mesh.apply(my_trf_for_delta1,
                                      mode='complex',
                                      kind='wavenumber').compute(mode='real')

        del my_trf_for_G2, my_trf_for_delta1

        out_rfield = -G2_filtered * delta1_filtered


    elif quadfield in ['RSDLPT_calQ2_LOS001']:
        # calQ2 = Q2 - delta1*delta1_parallel
        out_rfield = calc_quadratic_field(
            quadfield='RSDLPT_Q2_LOS001',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        out_rfield -= (
            base_field_mesh.compute(mode='real')
            * calc_quadratic_field(
            quadfield='delta1_par_LOS001',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose))

    elif quadfield in ['RSDLPT_calQ3_LOS001']:
        # calQ3 = Q3 - delta1 Q2 - 1/2 delta1_par G2
        out_rfield = calc_quadratic_field(
            quadfield='RSDLPT_Q3_LOS001',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose)

        out_rfield -= (
            base_field_mesh.compute(mode='real')
            * calc_quadratic_field(
            quadfield='RSDLPT_Q2_LOS001',
            base_field_mesh=base_field_mesh,
            smoothing_of_base_field=smoothing_of_base_field,
            verbose=verbose))

        out_rfield -= 0.5 * (
            calc_quadratic_field(
                quadfield='delta1_par_LOS001',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose)
            * calc_quadratic_field(
                quadfield='tidal_G2',
                base_field_mesh=base_field_mesh,
                smoothing_of_base_field=smoothing_of_base_field,
                verbose=verbose)
            )


    elif quadfield in ['RSDLPT_Q3_LOS001']:
        # Q3 = sum_m,n d_m2 d_mn d_n2
        # d_ij = -k_ik_j/k^2*basefield(\vk).
        # Compute d_ij(x). It's symmetric in i<->j so only compute j>=i.
        dij_x_dict = {}
        for idir in range(3):
            for jdir in range(idir, 3):

                def my_dij_transfer_function(k3vec, val, idir=idir, jdir=jdir):
                    kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                    kk[kk == 0] = 1
                    return -k3vec[idir] * k3vec[jdir] * val / kk

                dij_k = base_field_mesh.apply(my_dij_transfer_function,
                                              mode='complex',
                                              kind='wavenumber')
                del my_dij_transfer_function
                # do fft and convert field_mesh to RealField object
                dij_x = dij_k.compute(mode='real')
                del dij_k

                if verbose:
                    rfield_print_info(dij_x, comm, 'd_%d%d: ' % (idir, jdir))

                dij_x_dict[(idir,jdir)] = dij_x
                del dij_x

        # get j<i by symmetry
        def get_dij_x(idir, jdir):
            if jdir>=idir:
                return dij_x_dict[(idir,jdir)]
            else:
                return dij_x_dict[(jdir,idir)]

        # Compute sum_nm d_2m(k) d_mn(q) d_n2(p), assume LOS=z direction.
        out_rfield = 0.0 * base_field_mesh.compute(mode='real')
        for mdir in range(3):
            for ndir in range(3):
                out_rfield += (
                      get_dij_x(2,mdir)
                    * get_dij_x(mdir,ndir)
                    * get_dij_x(ndir,2) )

        # take out the mean (already close to 0 but still subtract)
        mymean = out_rfield.cmean()
        if comm.rank == 0:
            print('Subtract mean of %s: %g' % (quadfield, mymean))
        out_rfield -= mymean




    elif quadfield.startswith('PsiDot1_'):
        # get PsiDot \equiv \sum_n n*Psi^{(n)} up to 1st order
        assert quadfield in ['PsiDot1_0', 'PsiDot1_1', 'PsiDot1_2']
        component = int(quadfield[-1])
        out_rfield = get_displacement_from_density_rfield(
            base_field_mesh.compute(mode='real'),
            component=component,
            Psi_type='Zeldovich',
            smoothing=smoothing_of_base_field,
            RSD=False
            )

    elif quadfield.startswith('PsiDot2_'):
        # get PsiDot \equiv \sum_n n*Psi^{(n)} up to 2nd order
        assert quadfield in ['PsiDot2_0', 'PsiDot2_1', 'PsiDot2_2']
        component = int(quadfield[-1])

        # PsiDot \equiv \sum_n n*Psi^{(n)}
        out_rfield = get_displacement_from_density_rfield(
            base_field_mesh.compute(mode='real'),
            component=component,
            Psi_type='2LPT',
            prefac_Psi_1storder=1.0,
            prefac_Psi_2ndorder=2.0,  # include n=2 factor
            smoothing=smoothing_of_base_field,
            RSD=False
            )

    else:
        raise Exception("quadfield %s not implemented" % str(quadfield))

    return FieldMesh(out_rfield)


def calc_divergence_of_3_meshs(meshsource_tuple, prefactor=1.0):
    """
    Compute divergence of 3 MeshSource objects.

    Parameters
    ----------
    meshsource_tuple : 3-tuple of MeshSource objects
    """
    out_field = None
    for direction in [0,1,2]:
        # copy so we don't modify the input
        cfield = meshsource_tuple[direction].compute(mode='complex').copy()

        def derivative_function(k, v, d=direction):
            return prefactor * k[d] * 1j * v

        # i k_d field_d
        if out_field is None:
            out_field = cfield.apply(derivative_function)
        else:
            out_field += cfield.apply(derivative_function)

        del cfield
        
    return FieldMesh(out_field.c2r())


def get_displacement_from_density_rfield(in_density_rfield,
                                         in_density_cfield=None,
                                         component=None,
                                         Psi_type=None,
                                         smoothing=None,
                                         smoothing_Psi3LPT=None,
                                         prefac_Psi_1storder=1.0,
                                         prefac_Psi_2ndorder=1.0,
                                         prefac_Psi_3rdorder=1.0,
                                         RSD=False,
                                         RSD_line_of_sight=None,
                                         RSD_f_log_growth=None):
    """
    Given density delta(x) in real space, compute Zeldovich displacemnt Psi_component(x)
    given by Psi_component(\vk) = k_component / k^2 * W(k) * delta(\vk),
    where W(k) is smoothing window.

    For Psi_type='Zeldovich' compute 1st order displacement.
    For Psi_type='2LPT' compute 1st plus 2nd order displacement.
    etc

    Supply either in_density_rfield or in_density_cfield.

    Multiply 1st order displacement by prefac_Psi_1storder, 2nd order by 
    prefac_Psi_2ndorder, etc. Use this for getting time derivative of Psi.


    Follow http://rainwoodman.github.io/pmesh/intro.html.

    Parameters
    ----------
    RSD : boolean
        If True, include RSD by displacing by \vecPsi(q)+f (\e_LOS.\vecPsi(q)) \e_LOS, 
        where \ve_LOS is unit vector in line of sight direction.

    RSD_line_of_sight : array_like, (3,)
        Line of sight direction, e.g. [0,0,1] for z axis.
    """
    assert (component in [0, 1, 2])
    assert Psi_type in ['Zeldovich', '2LPT', '-2LPT', '3LPT', '-3LPT', '3LPT_v2',
        '3LPT_v3']

    from nbodykit import CurrentMPIComm
    comm = CurrentMPIComm.get()

    if in_density_cfield is None:
        # copy so we don't do any in-place changes by accident
        density_rfield = in_density_rfield.copy()
        density_cfield = density_rfield.r2c()
    else:
        assert in_density_rfield is None
        density_cfield = in_density_cfield.copy()


    if Psi_type in ['Zeldovich', '2LPT', '-2LPT', '3LPT', '-3LPT', '3LPT_v2',
        '3LPT_v3']:

        # get zeldovich displacement in direction given by component

        def potential_transfer_function(k, v):
            k2 = sum(ki**2 for ki in k)
            return np.where(k2 == 0.0, 0 * v, v / (k2))

        # get potential pot = delta/k^2
        #pot_k = density_rfield.r2c().apply(potential_transfer_function)
        pot_k = density_cfield.apply(potential_transfer_function)
        #print("pot_k head:\n", pot_k[:2,:2,:2])

        # apply smoothing
        if smoothing is not None:
            pot_k = smoothen_cfield(pot_k, **smoothing)

            #print("pot_k head2:\n", pot_k[:2,:2,:2])

        # get zeldovich displacement
        def force_transfer_function(k, v, d=component):
            # MS: not sure if we want a factor of -1 here.
            return k[d] * 1j * v

        Psi_component_rfield = pot_k.apply(force_transfer_function).c2r()

        if RSD:
            # Add linear RSD displacement f (\e_LOS.\vecPsi^(1)(q)) \e_LOS.
            assert RSD_f_log_growth is not None
            if RSD_line_of_sight in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]:
                # If [0,0,1] simply shift by Psi_z along z axis. Similarly in the other cases.
                if RSD_line_of_sight[component] == 0:
                    # nothing to do in this direction
                    pass
                elif RSD_line_of_sight[component] == 1:
                    # add f Psi_component(q)
                    Psi_component_rfield += RSD_f_log_growth * Psi_component_rfield
                    if comm.rank == 0:
                        print('%d: Added RSD in direction %d' %
                              (comm.rank, component))
            else:
                # Need to compute (\e_LOS.\vecPsi(q)) which requires all Psi components.
                raise Exception('RSD_line_of_sight %s not implemented' %
                                str(RSD_line_of_sight))

        Psi_component_rfield *= prefac_Psi_1storder


        # if comm.rank == 0:
        #     print('mean, rms, max Psi^{1}_%d: %g, %g, %g' % (
        #         component, np.mean(Psi_component_rfield), 
        #         np.mean(Psi_component_rfield**2)**0.5,
        #         np.max(Psi_component_rfield)))



        if Psi_type in ['2LPT', '-2LPT', '3LPT', '-3LPT', '3LPT_v2', '3LPT_v3']:

            # add 2nd order Psi on top of Zeldovich

            if in_density_rfield is not None:
                in_density_fieldmesh = FieldMesh(in_density_rfield)
            else:
                in_density_fieldmesh = FieldMesh(in_density_cfield)

            # compute G2
            G2_cfield = calc_quadratic_field(
                base_field_mesh=in_density_fieldmesh,
                quadfield='tidal_G2',
                smoothing_of_base_field=smoothing).compute(mode='complex')

            # compute Psi_2ndorder = -3/14 ik/k^2 G2(k). checked sign: improves rcc with deltaNL
            # if we use -3/14, but get worse rcc when using +3/14.
            Psi_2ndorder_rfield = -3. / 14. * (
                G2_cfield.apply(potential_transfer_function).apply(
                    force_transfer_function).c2r())
            del G2_cfield


            if Psi_type == '-2LPT':
                # this is just to test sign
                Psi_2ndorder_rfield *= -1.0


            if RSD:
                # Add 2nd order RSD displacement 2*f*(\e_LOS.\vecPsi^(2)(q)) \e_LOS.
                # Notice factor of 2 b/c \dot\psi enters for RSD.
                if RSD_line_of_sight in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]:
                    # If [0,0,1] simply shift by Psi_z along z axis. Similarly in the other cases.
                    if RSD_line_of_sight[component] == 0:
                        # nothing to do in this direction
                        pass
                    elif RSD_line_of_sight[component] == 1:
                        # add 2 f Psi^{(2)}_component(q)
                        Psi_2ndorder_rfield += (
                            2.0 * RSD_f_log_growth * Psi_2ndorder_rfield)
                        if comm.rank == 0:
                            print('%d: Added 2nd order RSD in direction %d' %
                                  (comm.rank, component))
                else:
                    # Need to compute (\e_LOS.\vecPsi(q)) which requires all Psi components.
                    raise Exception('RSD_line_of_sight %s not implemented' %
                                    str(RSD_line_of_sight))


            # if comm.rank == 0:
            #     print('mean, rms, max Psi^{2}_%d: %g, %g, %g' % (
            #         component, np.mean(Psi_2ndorder_rfield), 
            #         np.mean(Psi_2ndorder_rfield**2)**0.5,
            #         np.max(Psi_2ndorder_rfield)))

            Psi_2ndorder_rfield *= prefac_Psi_2ndorder

            # add 2nd order to Zeldoivhc displacement
            Psi_component_rfield += Psi_2ndorder_rfield
            del Psi_2ndorder_rfield


        if Psi_type in ['3LPT', '-3LPT', '3LPT_v2', '3LPT_v3']:

            # add 3nd order Psi on top of Zeldovich

            # compute G3
            if Psi_type == '3LPT_v3':
                # flip sign of dij
                G3_cfield = calc_quadratic_field(
                        base_field_mesh=FieldMesh(in_density_rfield),
                        quadfield='tidal_G3_flipped_dij_sign',
                        smoothing_of_base_field=smoothing_Psi3LPT).compute(mode='complex')

            else:
                G3_cfield = calc_quadratic_field(
                        base_field_mesh=FieldMesh(in_density_rfield),
                        quadfield='tidal_G3',
                        smoothing_of_base_field=smoothing_Psi3LPT).compute(mode='complex')

                if Psi_type == '3LPT_v2':
                    # switch sign of G3
                    G3_cfield *= -1.0

            Psi_3rdorder_rfield = 1./9. * (
                G3_cfield.apply(potential_transfer_function).apply(
                    force_transfer_function).c2r())
            del G3_cfield

            # add Gamma3
            Gamma3_cfield = calc_quadratic_field(
                base_field_mesh=FieldMesh(in_density_rfield),
                quadfield='Gamma3',
                smoothing_of_base_field=smoothing_Psi3LPT).compute(mode='complex')

            Psi_3rdorder_rfield -= -5./24. * (
                Gamma3_cfield.apply(potential_transfer_function).apply(
                    force_transfer_function).c2r())
            del Gamma3_cfield

            if Psi_type == '-3LPT':
                # this is just to test sign
                Psi_3rdorder_rfield *= -1.0

            if RSD:
                # Add 3rd order RSD displacement 3*f*(\e_LOS.\vecPsi^(3)(q)) \e_LOS.
                # Notice factor of 3 b/c \dot\psi enters for RSD.
                if RSD_line_of_sight in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]:
                    # If [0,0,1] simply shift by Psi_z along z axis. Similarly in the other cases.
                    if RSD_line_of_sight[component] == 0:
                        # nothing to do in this direction
                        pass
                    elif RSD_line_of_sight[component] == 1:
                        # add 3 f Psi^{(3)}_component(q)
                        Psi_3rdorder_rfield += (
                            3.0 * RSD_f_log_growth * Psi_3rdorder_rfield)
                        if comm.rank == 0:
                            print('%d: Added 3rd order RSD in direction %d' %
                                  (comm.rank, component))
                else:
                    # Need to compute (\e_LOS.\vecPsi(q)) which requires all Psi components.
                    raise Exception('RSD_line_of_sight %s not implemented' %
                                    str(RSD_line_of_sight))

            # if comm.rank == 0:
            #     print('mean, rms, max Psi^{3}_%d: %g, %g, %g' % (
            #         component, np.mean(Psi_3rdorder_rfield), 
            #         np.mean(Psi_3rdorder_rfield**2)**0.5,
            #         np.max(Psi_3rdorder_rfield)))

            Psi_3rdorder_rfield *= prefac_Psi_3rdorder

            # add 3rd order to displacement
            Psi_component_rfield += Psi_3rdorder_rfield
            del Psi_3rdorder_rfield


    return Psi_component_rfield


def shift_catalog_by_psi_grid(cat=None,
                              pos_column='Position',
                              in_displacement_rfields=None,
                              pos_units='Mpc/h',
                              displacement_units='Mpc/h',
                              boxsize=None,
                              verbose=False):
    """
    Changes catalog in-place.

    If in_displacement_rfields is None, do not shift.
    If an entry in in_displacement_rfields is None, do not shift in that
    direction.
    """

    if in_displacement_rfields is None:
        # Keep catalog unchanged
        return

    assert type(in_displacement_rfields) in [tuple, list]
    assert len(in_displacement_rfields) == 3

    # copy so we don't do any in-place changes by accident
    displacement_rfields = []
    for i in range(3):
        if in_displacement_rfields[i] is None:
            displacement_rfields.append(None)
        else:
            displacement_rfields.append(in_displacement_rfields[i].copy())

    for direction in range(3):

        displacement_rfield = displacement_rfields[direction]
        if displacement_rfield is None:
            # Move by 0 if in_field is None
            cat['TMP_Psi_%d' % direction] = 0.0
            continue

        if cat.comm.rank == 0:
            print("Get Psi in direction:", direction)

        # convert units so that displacement and pos have consistent units
        if displacement_units == 'Mpc/h':
            if pos_units == 'Mpc/h':
                pass
            elif pos_units == '0to1':
                displacement_rfield /= boxsize
            else:
                raise Exception("Invalid units: %s" % str(pos_units))
        else:
            raise Exception("Invalid units: %s" % str(displacement_units))

        if verbose:
            print("%d: displacement_rfield_%d: min, mean, max, rms:" %
                  (cat.comm.rank, direction), np.min(displacement_rfield),
                  np.mean(displacement_rfield), np.max(displacement_rfield),
                  np.mean(displacement_rfield**2)**0.5)

        # interpolate Psi_i(x) to catalog ptcle positions (this shouldn't use weights)
        interpolate_pm_rfield_to_catalog(
            displacement_rfield,
            cat,
            catalog_column_to_save_to='TMP_Psi_%d' % direction)

        # find max displacement and print
        maxpos = np.max(np.abs(cat[pos_column]))
        if verbose:
            print("%d: Catalog displacement_%d min, mean, max, rms:" %
                  (cat.comm.rank, direction),
                  np.min(np.array(cat['TMP_Psi_%d' % direction])),
                  np.mean(np.array(cat['TMP_Psi_%d' % direction])),
                  np.max(np.array(cat['TMP_Psi_%d' % direction])),
                  np.mean(np.array(cat['TMP_Psi_%d' % direction])**2)**0.5)

    # shift positions
    if cat.comm.rank == 0:
        print("add psi")

    cat[pos_column] = transform.StackColumns(
        cat[pos_column][:, 0] + cat['TMP_Psi_0'],
        cat[pos_column][:, 1] + cat['TMP_Psi_1'],
        cat[pos_column][:, 2] + cat['TMP_Psi_2'])
    print("%d: done adding psi" % cat.comm.rank)

    # save memory
    #for direction in range(3):
    #    cat['TMP_Psi_%d'%direction] = 0.0

    # box wrap shifted positions to [mincoord,L[
    if pos_units == '0to1':
        cat[pos_column] = cat[pos_column] % 1.0
        assert np.all(cat[pos_column] <= 1.0)
        assert np.all(cat[pos_column] >= 0.0)
    elif pos_units == 'Mpc/h':
        cat[pos_column] = cat[pos_column] % boxsize
        assert np.all(cat[pos_column] <= boxsize)
        assert np.all(cat[pos_column] >= 0.0)
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
    if comm.rank == 0:
        if len(np.array(rfield)) > 0:
            local_min = np.min(np.array(rfield))
            local_max = np.max(np.array(rfield))
        else:
            local_min = None
            local_max = None
        print("%d: %s: local_min=%s, cmean=%g, local_max=%s" %
              (comm.rank, text, str(local_min), cmean, str(local_max)))


def linear_rescale_fac(current_scale_factor,
                       desired_scale_factor,
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
        rescalefac = calc_Da(desired_scale_factor) / calc_Da(
            current_scale_factor)
        #del cosmo
    return rescalefac

def catalog_persist(cat, columns=None):
    """
    Return a CatalogSource, where the selected columns are
    computed and persist in memory.
    """

    import dask.array as da
    if columns is None:
        columns = cat.columns

    r = {}
    for key in columns:
        if key in cat.columns:
            r[key] = cat[key]

    r = da.compute(r)[0] # particularity of dask

    from nbodykit.source.catalog.array import ArrayCatalog
    c = ArrayCatalog(r, comm=cat.comm)
    c.attrs.update(cat.attrs)

    return c


def convert_nbk_cat_to_np_array(cat, ID_as_int=False, skip_columns=None,
    position_as_xyz=False):
    """
    Convert nbodykit catalog to structured numpy array.
    """
    if skip_columns is None:
        skip_columns = []

    struc_dtype = []
    for col in cat.columns:
        if col in skip_columns:
            continue
        shape = cat[col].shape
        dtype = cat[col].dtype
        if ID_as_int:
            if 'ID' in col:
                dtype = np.longlong
        if len(shape) == 1:
            struc_dtype.append( (col, dtype) )
        elif len(shape) == 2:
            struc_dtype.append( (col, dtype, shape[1]) )
        else:
            raise Exception('Invalid column shape %s' % str(shape))

    if position_as_xyz:
        # also store position as xyz
        struc_dtype.append( ('xPos', 'f8') )
        struc_dtype.append( ('yPos', 'f8') )
        struc_dtype.append( ('zPos', 'f8') )
            
    print(struc_dtype)
    
    # list of columns
    out = np.empty(cat.csize, dtype=struc_dtype)
    for col in cat.columns:
        if col not in skip_columns:
            out[col] = cat[col]

    if position_as_xyz:
        out['xPos'] = out['Position'][:,0]
        out['yPos'] = out['Position'][:,1]
        out['zPos'] = out['Position'][:,2]

    return out


