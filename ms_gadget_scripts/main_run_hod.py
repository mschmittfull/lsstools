# Marcel Schmittfull 2020 (mschmittfull@gmail.com)
from __future__ import print_function,division

import os
import numpy as np
from nbodykit.lab import BigFileCatalog
from nbodykit.lab import FOF
from nbodykit.lab import HaloCatalog
#from nbodykit.lab import HODCatalog


from nbodykit.lab import KDDensity
from argparse import ArgumentParser
from nbodykit.cosmology import Planck15
from nbodykit import setup_logging


def main():
    """ 
    Script to compute HOD galaxies from FOF halo catalog with mvir.
    """
    setup_logging()

    ap = ArgumentParser()
    ap.add_argument('--fof_halos_mvir', 
        help=('Directory of halo catalog with mvir Mass, e.g.'
            '/data/mschmittfull/lss/ms_gadget/run4/00000400-01536-500.0-wig/nbkit_fof_0.6250/ll_0.200_nmin25_mvir/'),
        default='/data/mschmittfull/lss/ms_gadget/run4/00000400-01536-500.0-wig/nbkit_fof_0.6250/ll_0.200_nmin25_mvir/')

    ns = ap.parse_args()

    cat = BigFileCatalog(ns.fof_halos_mvir)


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

    # run hod
    #halotools_halos = halos.to_halotools()
    from nbodykit.hod import Zheng07Model
    hodmodel = Zheng07Model.to_halotools(cosmo=cosmo, redshift=redshift, mdef='vir')
    if cat.comm.rank == 0:
        print('zheng07model default:', hodmodel.param_dict)

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

    # HOD parameters from Hand & Seljak 1706.02362
    hodmodel.param_dict['logMmin'] = 12.99
    hodmodel.param_dict['sigma_logM'] = 0.308
    hodmodel.param_dict['logM1'] = 14.08
    hodmodel.param_dict['alpha'] = 1.06
    hodmodel.param_dict['logM0'] = 13.20 # this is called Mcut in Hand et al and Reid et al.

    if cat.comm.rank == 0:
        print('Use zheng07model with:', hodmodel.param_dict)

    galcat = halos.populate(hodmodel, seed=41)
    #hod = HODCatalog(halotools_halos)

    #galcat.repopulate(alpha=0.9, logMmin=13.5, seed=42)


    if cat.comm.rank == 0:
        print('galcat', galcat)

   
if __name__ == '__main__':
    main()

