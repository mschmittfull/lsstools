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

    if cat.comm.rank == 0:
        print('BoxSize', halos.attrs['BoxSize'])
        print('attrs', halos.attrs.keys())

    # run hod
    halotools_halos = halos.to_halotools()
    hod = HODCatalog(halotools_halos)

    if cat.comm.rank == 0:
        print('hod', hod)

   
if __name__ == '__main__':
    main()

