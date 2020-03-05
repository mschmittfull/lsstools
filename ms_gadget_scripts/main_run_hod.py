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
    """
    setup_logging()

    ap = ArgumentParser()
    ap.add_argument(
        '--fof_halos_mvir', 
        help=('Directory of halo catalog with mvir Mass, e.g.'
            '/data/mschmittfull/lss/ms_gadget/run4/00000400-01536-500.0-wig/nbkit_fof_0.6250/ll_0.200_nmin25_mvir/'),
        #default='/data/mschmittfull/lss/ms_gadget/run4/00000400-01536-500.0-wig/nbkit_fof_0.6250/ll_0.200_nmin25_mvir/'
        default='/Users/mschmittfull/scratch_data/lss/ms_gadget/run4/00000400-01536-500.0-wig/nbkit_fof_0.6250/ll_0.200_nmin25_mvir/'
        )
    ap.add_argument(
        '--RSD', help='Add RSD to positions if not 0',
        type=int,
        default=0)
    ap.add_argument(
        '--HOD_model_name', help='Name of HOD model',
        default='Zheng07_HandSeljak17')

    args = ap.parse_args()
    RSD_LOS = np.array([1,0,0])

    # load input halo catalog
    print('Read halos from %s' % args.fof_halos_mvir) 
    cat = BigFileCatalog(args.fof_halos_mvir)

    # run hod to get galaxy catalog
    galcat = run_hod(
        cat, 
        add_RSD=args.RSD, 
        RSD_LOS=RSD_LOS,
        HOD_model_name=args.HOD_model_name)

    # save to hdf5
    out_fname = os.path.join(args.fof_halos_mvir, 
        'HOD_%s' % args.HOD_model_name)
    if args.RSD:
        assert np.all(RSD_LOS == np.array([1,0,0]))
        out_fname += '_RSD100'
    out_fname += '.hdf5'

    save_galcat_to_hdf5(galcat, out_fname=out_fname)




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

    # Define HOD
    if HOD_model_name == 'Zheng07_HandSeljak17':

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

        hodmodel = Zheng07Model.to_halotools(cosmo=cosmo, redshift=redshift, mdef='vir')

        # HOD parameters from Hand & Seljak 1706.02362
        hodmodel.param_dict['logMmin'] = 12.99
        hodmodel.param_dict['sigma_logM'] = 0.308
        hodmodel.param_dict['logM1'] = 14.08
        hodmodel.param_dict['alpha'] = 1.06
        hodmodel.param_dict['logM0'] = 13.20 # this is called Mcut in Hand et al and Reid et al.

        if cat.comm.rank == 0:
            print('Use zheng07model with:', hodmodel.param_dict)

    else:
        raise Exception('Unknown hod_model %s' % HOD_model_name)

    # Run HOD
    galcat = halos.populate(hodmodel, seed=hod_seed)

    if add_RSD:
        assert type(RSD_LOS)==np.ndarray
        assert RSD_LOS.shape==(3,)
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

