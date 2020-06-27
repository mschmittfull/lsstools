# Marcel Schmittfull 2020 (mschmittfull@gmail.com)
from __future__ import print_function,division

from argparse import ArgumentParser
from collections import OrderedDict
import json
import numpy as np
import os
from shutil import rmtree

from lsstools.nbkit03_utils import catalog_persist
from nbodykit.source.catalog import ArrayCatalog
from nbodykit import setup_logging



def main():
    """ 
    Script to convert Rockstar halo catalog to bigfile catalog.
    Login to a single node on helios and run there on command line.

    For batch runs, use e.g.

    for SEED in {0..1}; do python main_rockstar_catalog_to_bigfile.py --rockstar_halos "/scratch/mschmittfull/lss/ms_gadget/run4/0000040${SEED}-01536-1500.0-wig/snap_0.6250.gadget3/rockstar_out_0.list" --max_rows 5 --include_parent_ID; done
    """
    setup_logging()

    ap = ArgumentParser()
    ap.add_argument(
        '--rockstar_halos', 
        help=('File name of Rockstar halo catalog, e.g.'
            '/data/mschmittfull/lss/ms_gadget/run4/00000400-01536-500.0-wig/snap_0.6250.gadget3/rockstar_out_0.list'),
        default='/scratch/mschmittfull/lss/ms_gadget/run4/00000400-01536-500.0-wig/snap_0.6250.gadget3/rockstar_out_0.list.parents'
        )

    ap.add_argument('--add_RSD', dest='RSD', action='store_true', help='Add RSD to position')

    ap.add_argument('--include_parent_ID', dest='include_parent_ID', 
        action='store_true', help='Include ID and parent ID in bigfile.')

    # ap.add_argument(
    #     '--RSD', help='Add RSD to positions if not 0',
    #     type=int,
    #     default=0)

    ap.add_argument(
        '--max_rows', help='Max number of rows to read. Read all if 0.',
        type=int,
        default=0)

    ap.set_defaults(RSD=False, include_parent_ID=False)

    
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

    # get names of columns
    np_cat1 = np.genfromtxt(args.rockstar_halos, names=True, max_rows=1)
    names = np_cat1.dtype.names
    # keep only a subset
    usecol_names = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'Mvir']
    if args.include_parent_ID:
        usecol_names += ['ID', 'PID']

    usecols = []
    for column_number, name in enumerate(names):
        if name in usecol_names:
            usecols.append(column_number)

    print('usecols:', usecols)
    print([names[usecol] for usecol in usecols])



    # read data. 
    print('Reading data')
    if args.max_rows == 0:
        max_rows = None
    else:
        max_rows = args.max_rows

    # TODO: np.loadtxt should be faster, but now take 5 minutes so probably ok.
    np_cat = np.genfromtxt(
        args.rockstar_halos, names=True, max_rows=max_rows, usecols=usecols)

    print('Read data:')
    print(np_cat[:5])

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

    cat['log10Mvir'] = np.log10(cat['Mvir'])


    # Keep only some columns
    keep_columns = ['Position', 'Velocity', 'log10Mvir']
    if args.include_parent_ID:
        # also keep halo ID and parent ID
        keep_columns += ['ID', 'PID']

    cat = catalog_persist(cat, keep_columns)
    cat.attrs['rockstar_header'] = header

    if args.RSD:
        raise Exception('RSD not implemented')


    print('Will write data:')
    for c in keep_columns:
        print('%s:' % c, cat[c])

    # save to bigfile
    if max_rows is None:
        out_fname = '%s.bigfile' % args.rockstar_halos
    else:
        out_fname = '%s_max_rows%d.bigfile' % (args.rockstar_halos, max_rows)
    
    if os.path.exists(out_fname):
        rmtree(out_fname)

    if cat.comm.rank == 0:
        print('Writing to %s' % out_fname)
    cat.save(out_fname, columns=keep_columns)
    if cat.comm.rank == 0:
        print('Wrote %s' % out_fname)


if __name__ == '__main__':
    main()

