# Marcel Schmittfull 2020 (mschmittfull@gmail.com)
from __future__ import print_function,division

import os
import numpy as np
from nbodykit.lab import BigFileCatalog
from nbodykit.lab import FOF
from nbodykit.lab import HaloCatalog
from nbodykit.lab import KDDensity
from argparse import ArgumentParser
from nbodykit.cosmology import Planck15
from nbodykit import setup_logging


def main():
    """ 
    Script to compute FOF halos from treepm DM catalog. Compute virial mass
    which is needed for HOD models. 

    Note: Before March 2020, used mass given by number of particles in halo,
    see psiRec/psirec/main_ms_gadget_fof_halofinder_nbkit0.3.py.
    """

    ap = ArgumentParser()
    ap.add_argument('treepm', 
        help='Directory of TreePM matter field, e.g. /scratch/treepm_0.1000/')
    ap.add_argument('ll', type=float, 
        help='Linking length of finding halos, e.g. 0.2 or 0.168', 
        default=0.2)
    ap.add_argument('fof', 
        help=('Output directory of halo catalogs, e.g. '
              '/scratch/treepm_0.1000/fof . Will write to {fof}/{ll_nmin_mvir}'))
    ap.add_argument('--nmin', type=int, default=20, 
        help='min number of particles to be in the catalogue')

    ns = ap.parse_args()

    cat = BigFileCatalog(ns.treepm, header='Header', dataset='1/')

    cat.attrs['BoxSize']  = np.ones(3) * cat.attrs['BoxSize'][0]
    cat.attrs['Nmesh'] = 512

    cosmo = Planck15.match(Omega0_m=cat.attrs['Omega0'])
    # In TreePM, we need to use 'Omega0' instead of 'OmegaM' in FastPM.
    # csize is the total number of particles
    M0 = (cat.attrs['Omega0'][0] * 27.75 * 1e10 * cat.attrs['BoxSize'].prod() 
            / cat.csize)

    if cat.comm.rank == 0:
        print('BoxSize', cat.attrs['BoxSize'])
        print('Mass of a particle', M0)
        print('OmegaM', cosmo.Om0)
        print('Redshift', cat.attrs['Redshift'])

    # Halos which have more than nmin particles are selected.
    fof = FOF(cat, linking_length=ns.ll, nmin=ns.nmin)  

    # Compute halo catalog. Mass column contains virial mass, which is needed
    # to get concentration needed for hod.
    halos = fof.to_halos(
        cosmo=cosmo, 
        redshift=cat.attrs['Redshift'],
        particle_mass=M0,
        mdef='vir')

    halos['log10M'] = np.log10(halos['Mass'])

    # print info
    if fof.comm.rank == 0:
        print('Total number of halos found', halos.csize)
        print('Saving columns', halos.columns)
        if not os.path.exists(ns.fof):
            os.makedirs(ns.fof)

    # Save the halo catalog to disk so can easily load it later to populate
    # galaxies with hod.
    out_fname = ns.fof + '/ll_{0:.3f}_nmin{1}_mvir'.format(ns.ll, ns.nmin+1)
    halos.save(out_fname)

    if cat.comm.rank == 0:
        print('Saved HaloCatalog to %s' % out_fname)

   
if __name__ == '__main__':
    main()

