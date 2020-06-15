# Marcel Schmittfull 2020 (mschmittfull@gmail.com)
from __future__ import print_function,division

from argparse import ArgumentParser
from collections import OrderedDict
import h5py
import json
import numpy as np
import os

from nbodykit.lab import BigFileCatalog
from nbodykit.lab import FOF
from nbodykit.lab import HaloCatalog
from nbodykit.cosmology import Planck15
from nbodykit import setup_logging
from nbodykit.hod import Zheng07Model


def main():
    """ 
    Script to compute HOD galaxies from FOF halo catalog with mvir.

    For batch runs, use e.g.

    for SEED in {0..4}; do python main_run_hod.py --fof_halos_mvir "/data/mschmittfull/lss/ms_gadget/run4/0000040${SEED}-01536-500.0-wig/nbkit_fof_0.6250/ll_0.200_nmin25_mvir/" --RSD 0; done
    """
    setup_logging()

    ap = ArgumentParser()
    ap.add_argument(
        '--fof_halos_mvir', 
        help=('Directory of halo catalog with mvir Mass, e.g.'
            '/data/mschmittfull/lss/ms_gadget/run4/00000400-01536-500.0-wig/nbkit_fof_0.6250/ll_0.200_nmin25_mvir/'),
        #default='/data/mschmittfull/lss/ms_gadget/run4/00000400-01536-500.0-wig/nbkit_fof_0.6250/ll_0.200_nmin25_mvir/'
        default='/Users/mschmittfull/scratch_data/lss/ms_gadget/run4/00000400-01536-500.0-wig/nbkit_fof_0.6250/ll_0.200_nmin25_mvir'
        )
    ap.add_argument(
        '--RSD', help='Add RSD to positions if not 0',
        type=int,
        default=0)
    ap.add_argument(
        '--HOD_model_name', help='Name of HOD model',
        default='Zheng07_HandSeljak17_v2')

    args = ap.parse_args()
    RSD_LOS = np.array([0,0,1])

    # load input halo catalog
    print('Read halos from %s' % args.fof_halos_mvir) 
    cat = BigFileCatalog(args.fof_halos_mvir)

    # run hod to get galaxy catalog
    galcat = run_hod(
        cat, 
        add_RSD=args.RSD, 
        RSD_LOS=RSD_LOS,
        HOD_model_name=args.HOD_model_name)

    if True:
        # save to hdf5
        out_fname = os.path.join(args.fof_halos_mvir, 
            '/HOD_%s' % args.HOD_model_name)
        if args.RSD:
            assert np.all(RSD_LOS == np.array([0,0,1]))
            out_fname += '_RSD001'
        out_fname += '.hdf5'
        save_galcat_to_hdf5(galcat, out_fname=out_fname)
        print('Wrote %s' % out_fname)

    # save to bigfile
    out_fname = '%s_HOD_%s' % (args.fof_halos_mvir, args.HOD_model_name)
    if args.RSD:
        assert np.all(RSD_LOS == np.array([0,0,1]))
        out_fname += '_RSD001'
    out_fname += '.bigfile'
    galcat.save(out_fname, columns=galcat.columns)
    print('Wrote %s' % out_fname)




def run_hod(cat, HOD_model_name=None, hod_seed=42,
    add_RSD=False, RSD_LOS=None):
    """
    Run HOD to get galaxy catalog from input halo catalog.

    Parameters
    ----------
    cat : nbodykit Catalog object
        Input halo catalog, should use virial mass as 'Mass' column.
    """
    if cat.comm.rank == 0:
        print('HOD model: %s' % HOD_model_name)
    cat.attrs['BoxSize']  = np.ones(3) * cat.attrs['BoxSize'][0]
    cat.attrs['Nmesh']  = np.ones(3) * 512.0    # in TreePM catalog, there is no 'NC' attribute
    
    cosmo = Planck15.match(Omega0_m=cat.attrs['Omega0'])
    # In TreePM, we need to use 'Omega0' instead of 'OmegaM' in FastPM.
    # csize is the total number of particles
    M0 = (cat.attrs['Omega0'][0] * 27.75 * 1e10 * cat.attrs['BoxSize'].prod() 
            / cat.csize)
    redshift = 1.0/cat.attrs['Time'][0]-1.0

    # convert to HaloCatalog
    halos = HaloCatalog(cat, cosmo, redshift)

    if cat.comm.rank == 0:
        print('BoxSize', halos.attrs['BoxSize'])
        print('attrs', halos.attrs.keys())
        print('RSDFactor', halos.attrs['RSDFactor'])
        print('Columns', halos.columns)

    # Define HOD
    if HOD_model_name in [
        'Zheng07_HandSeljak17_v2',
        'Zheng07_HandSeljak17_centrals_v2',
        'Zheng07_HandSeljak17_sats_v2',
        'Zheng07_HandSeljak17_parent_halos_v2']:

        # (1) Hand & Seljak 1706.02362:  
        # Uses {log10 Mmin, sigma log10 M, log10 M1, alpha, log10 Mcut} = {12.99, 0.308, 14.08, 0.824, 13.20}.
        # See Reid et al https://arxiv.org/pdf/1404.3742.pdf eq 17-19

        # (2) halotools docs on zheng07 model:
        #  See https://halotools.readthedocs.io/en/stable/quickstart_and_tutorials/tutorials/model_building/preloaded_models/zheng07_composite_model.html#zheng07-parameters):
        # logMmin - Minimum mass required for a halo to host a central galaxy.
        # sigma_logM - Rate of transition from <Ncen>=0 -> <Ncen=1>.
        # alpha - Power law slope of the relation between halo mass and <Nsat>.
        # logM0 - Low-mass cutoff in <Nsat>.
        # logM1 - Characteristic halo mass where <Nsat> begins to assume a power law form.

        # 11 June 2020: Zheng07_HandSeljak17_v2 uses fixed RSDFactor, which was wrong by factor of 1/a before.

        hodmodel = Zheng07Model.to_halotools(cosmo=cosmo, redshift=redshift, mdef='vir')

        # HOD parameters from Hand & Seljak 1706.02362
        hodmodel.param_dict['logMmin'] = 12.99
        hodmodel.param_dict['sigma_logM'] = 0.308
        hodmodel.param_dict['logM1'] = 14.08
        hodmodel.param_dict['alpha'] = 1.06
        hodmodel.param_dict['logM0'] = 13.20 # this is called Mcut in Hand et al and Reid et al.

        if cat.comm.rank == 0:
            print('Use zheng07model with:', hodmodel.param_dict)

        # Run HOD
        galcat = halos.populate(hodmodel, seed=hod_seed)

        # select which galaxies to keep
        if HOD_model_name == 'Zheng07_HandSeljak17_v2':
            # keep all
            pass

        elif HOD_model_name == 'Zheng07_HandSeljak17_centrals_v2':
            # select only centrals
            ww = galcat['gal_type'] == 0  # 0: central, 1: satellite
            galcat = galcat[ww]

        elif HOD_model_name == 'Zheng07_HandSeljak17_sats_v2':
            # select only satellites
            ww = galcat['gal_type'] == 1  # 0: central, 1: satellite
            galcat = galcat[ww]

        elif HOD_model_name == 'Zheng07_HandSeljak17_parent_halos_v2':
            # select centrals
            ww = galcat['gal_type'] == 0  # 0: central, 1: satellite
            galcat = galcat[ww]

            # set position to that of parent halo (in Mpc/h)
            halo_pos = galcat['Position'].compute() + np.nan
            halo_pos[:,0] = galcat['halo_x'].compute()
            halo_pos[:,1] = galcat['halo_y'].compute()
            halo_pos[:,2] = galcat['halo_z'].compute()
            galcat['Position'] = halo_pos
            del halo_pos

            # set velocity to that of parent halo (in km/s)
            halo_vel = galcat['Velocity'].compute() + np.nan
            halo_vel[:,0] = galcat['halo_vx'].compute()
            halo_vel[:,1] = galcat['halo_vy'].compute()
            halo_vel[:,2] = galcat['halo_vz'].compute()
            galcat['Velocity'] = halo_vel
            del halo_vel

            # Get RSD displacement = v_z/(aH(a)), where v_z is halo velocity.
            # Compute rsd_factor = 1/(aH(a)) = (1+z)/H(z)
            # see https://nbodykit.readthedocs.io/en/latest/catalogs/common-operations.html#Adding-Redshift-space-Distortions
            rsd_factor = (1.+redshift) / (100. * cosmo.efunc(redshift))
            raise Exception('this is not correct for ms_gadget which has a^2 dx/dt for velocity.')
            galcat['VelocityOffset'] = rsd_factor * galcat['Velocity']

            # columns: ['Position', 'Selection', 'Value', 'Velocity', 'VelocityOffset', 'Weight', 'conc_NFWmodel', 'gal_type', 'halo_hostid', 'halo_id', 'halo_mvir', 'halo_num_centrals', 'halo_num_satellites', 'halo_rvir', 'halo_upid', 'halo_vx', 'halo_vy', 'halo_vz', 'halo_x', 'halo_y', 'halo_z', 'host_centric_distance', 'vx', 'vy', 'vz', 'x', 'y', 'z']

    else:
        raise Exception('Unknown hod_model %s' % HOD_model_name)


    if add_RSD:
        assert type(RSD_LOS)==np.ndarray
        assert RSD_LOS.shape==(3,)
        print('cat attrs:', galcat.attrs)


        # It seems like halos.populate gives satellite velocity in km/s by drawing from NFW profile, and sets central velocity equal to halo velocity.
        # But not sure what units are assumed for halo velocity. Note we have different velocity a prefactor in ms_gadget and new MP-Gadget format.
        # Also, should probably use peak velocity instead of bulk velocity of halos for the centrals velocity.
        # So HOD just seems screwed up.
        raise Exception('todo: use RSDFactor of the catalog! VelocityOffset can be wrong by factor of a if catalog has a^2 dx/dt (ms_gadget) instead of a dx/dt.')


        galcat['Position'] = (
            galcat['Position'] + galcat['VelocityOffset'] * RSD_LOS)

    if cat.comm.rank == 0:
        print('galcat', galcat)
        print('attrs', galcat.attrs)
        print('columns', galcat.columns)
        print('fsat', galcat.attrs['fsat'])

    return galcat
   


def save_galcat_to_hdf5(galcat, out_fname):
    """
    Save to hdf5 format that's used by other parts of the code.
    Should get rid of this at some point and work directly with catalogs
    or bigfiles.
    """
    Ntot = galcat.csize

    print('Ntot', Ntot)

    # init out_array
    out_dtype = np.dtype([('Position', ('f4', 3)), ('Velocity', ('f4', 3)),
                          ('halo_mvir', 'f4')])
    out_array = np.empty((Ntot,), dtype=out_dtype)
    print("out_array dtype: ", out_array.dtype)
    print("out_array shape: ", out_array.shape)

    # fill out_array
    assert np.all(galcat.attrs['BoxSize'] == galcat.attrs['BoxSize'][0])
    out_array['Position'] = galcat['Position'] / galcat.attrs['BoxSize'][
        0]  # out_pos ranges from 0 to 1
    out_array['Velocity'] = galcat['Velocity'] # not sure what units, maybe Mpc/h
    out_array['halo_mvir'] = galcat['halo_mvir']

    # box wrap to [mincoord,1[
    print("min, max pos: %.15g, %.15g" %
          (np.min(out_array['Position']), np.max(out_array['Position'])))
    mincoord = 1e-11
    if (not np.all(out_array['Position'] >= mincoord)) or (
            not np.all(out_array['Position'] < 1.0)):
        # box wrap
        print("box wrap")
        out_array['Position'][np.where(out_array['Position'] >= 1.0)] -= 1.0
        out_array['Position'][np.where(out_array['Position'] < 0.0)] += 1.0
        assert np.all(out_array['Position'] >= 0.)
        out_array['Position'][np.where(
            out_array['Position'] < mincoord)] = mincoord
        print("min, max pos after box wrap: %.15g, %.15g" %
              (np.min(out_array['Position']), np.max(out_array['Position'])))
        assert np.all(out_array['Position'] >= mincoord)
        #assert np.all(out_array['Position']<1.0)
        assert np.all(out_array['Position'] <= 1.0)

    # hdf5 attrs
    bigfile_attrs_dict = OrderedDict()
    for k in galcat.attrs.keys():
        if type(galcat.attrs[k]) == np.ndarray:
            # convert to list b/c json cannot save numpy arrays
            bigfile_attrs_dict[k] = galcat.attrs[k].tolist()
        else:
            bigfile_attrs_dict[k] = galcat.attrs[k]
    attrs_dict = {
        'BoxSize': galcat.attrs['BoxSize'],
        'Ntot': Ntot,
        'bigfile_attrs': '' #json.dumps(bigfile_attrs_dict)
    }

    #out_dataset_name = 'FOFGroups'
    # Marcel's hdf5 files always assume data is stored in 'Subsample' dataset.
    out_dataset_name = 'Subsample'
    #print("Output file:", ofile)

    if os.path.exists(out_fname):
        os.remove(out_fname)
    f_out = h5py.File(out_fname, 'w')
    dataset = f_out.create_dataset(name=out_dataset_name,
                                   dtype=out_array.dtype,
                                   shape=(Ntot,))
    # should probably use attrs.create() for this
    for key, v in attrs_dict.items():
        print("type", attrs_dict[key])
        dataset.attrs[key] = attrs_dict[key]
    dataset[:] = out_array[:]

    f_out.close()
    print("Wrote %s" % out_fname)


if __name__ == '__main__':
    main()

