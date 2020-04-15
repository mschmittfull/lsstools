# Marcel Schmittfull 2020 (mschmittfull@gmail.com)
from __future__ import print_function,division

from argparse import ArgumentParser
from collections import OrderedDict
import json
import numpy as np
import os

from lsstools.nbkit03_utils import catalog_persist
from nbodykit.source.catalog import ArrayCatalog



def main():
    """ 
    Script to convert Rockstar halo catalog to bigfile catalog.

    For batch runs, use e.g.

    for SEED in {0..4}; do python main_rockstar_catalog_to_bigfile.py --rockstar_halos "/data/mschmittfull/lss/ms_gadget/run4/0000040${SEED}-01536-500.0-wig/snap_0.6250.gadget3/rockstar_out_0.list" --RSD 0; done
    """
    setup_logging()

    ap = ArgumentParser()
    ap.add_argument(
        '--rockstar_halos', 
        help=('File name of Rockstar halo catalog, e.g.'
            '/data/mschmittfull/lss/ms_gadget/run4/00000400-01536-500.0-wig/snap_0.6250.gadget3/rockstar_out_0.list'),
        default='/data/mschmittfull/lss/ms_gadget/run4/00000404-01536-500.0-wig/snap_0.6250.gadget3/rockstar_out_0.list'
        )
    ap.add_argument(
        '--RSD', help='Add RSD to positions if not 0',
        type=int,
        default=0)
    
    args = ap.parse_args()
    RSD_LOS = np.array([0,0,1])

    # load input halo catalog
    print('Read halos from %s' % args.rockstar_halos)

    # read header
    with open(args.rockstar_halos) as myfile:
        header = [next(myfile) for x in xrange(16)]
    header = ''.join(header)
    print('Header:')
    print(header)

    # read data
    print('Reading data')
    np_cat = np.genfromtxt(args.rockstar_halos, names=True, max_rows=10)
    print('Done reading data')
    print(np_cat[:10])

    # convert to arraycatalog
    cat = ArrayCatalog(np_cat)

    # fill position and velocity
    pos = np.empty(cat.csize, dtype=[('Position', ('f8', 3))])
    pos['Position'][:,0] = cat['X']
    pos['Position'][:,1] = cat['Y']
    pos['Position'][:,2] = cat['Z']
    cat['Position'] = pos['Position']
    del pos

    vel = np.empty(cat.csize, dtype=[('Velocity', ('f8', 3))])
    vel['Velocity'][:,0] = cat['VX']
    vel['Velocity'][:,1] = cat['VY']
    vel['Velocity'][:,2] = cat['VZ']
    # todo: what units?
    cat['Velocity'] = vel['Velocity']
    del vel

    # Keep only some columns
    keep_columns = ['Position', 'Velocity', 'Mvir']
    cat = catalog_persist(cat, keep_columns)
    cat.attrs['rockstar_header'] = header

    if args.RSD:
        raise Exception('RSD not implemented')



    # save to bigfile
    out_fname = args.rockstar_halos + '.bigfile'
    if cat.comm.rank == 0:
        print('Writing to %s' % out_fname)
    cat.save(out_fname)
    if cat.comm.rank == 0:
        print('Wrote %s' % out_fname)


if __name__ == '__main__':
    main()

