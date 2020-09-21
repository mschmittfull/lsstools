from __future__ import print_function,division

from nbodykit.lab import *
import numpy as np
import os
from shutil import rmtree
from nbodykit.source.catalog.uniform import MPIRandomState
from argparse import ArgumentParser
from nbodykit.utils import GatherArray



def main():

    """
    Read a bigfile catalog, take a random subsample, and save it to hdf5.
    """

    # #####################
    # COMMAND LINE OPTIONS
    # #####################

    ap = ArgumentParser()
    ap.add_argument('sim_seed', type=int, help='Simulation seed')
    ap.add_argument('--save_bigfile', type=int, default=1, help='Save subsample to bigfile (should run with many cores)')
    ap.add_argument('--save_hdf5', type=int, default=0, help='Read subsample from bigfile and save as hdf5 (must run with 1 core)')
    
    cmd_args = ap.parse_args()
    sim_seed = cmd_args.sim_seed


    # #####################
    # OPTIONS
    # #####################
    boxsize = 500.0

    if True:
        # big sim
        Nptcles_per_dim = 1536
        in_file = 'snap_0.6250'
    else:
        # small sim
        Nptcles_per_dim = 32
        in_file = 'snap_000'

    in_dir = '/scratch/mschmittfull/lss/ms_gadget/run4/00000%d-%05d-%.1f-wig/' % (
        sim_seed,Nptcles_per_dim,boxsize)
    

    out_dir = in_dir

    ## subsample options
    sub_ssseed = 40000+sim_seed
    # subsample ratio. 0.025 corresponds to 90e6 ptcles which is similar to 
    # 1% subsample of 2048**3.
    # Sep 2020: Use 0.0015 for laptop and 0.04 for cluster.
    subsample_ratio = 0.00025

    # write subsample to bigfile (should run with many cores)
    save_bigfile = bool(cmd_args.save_bigfile)
    # convert bigfile subsample to hdf5 file (must run with 1 core)
    save_hdf5 = bool(cmd_args.save_hdf5)
    columns_to_write_to_hdf5 = ['Position','Velocity','ID']

    if (not save_bigfile) and (not save_hdf5):
        print("Nothing to do, you should set save_bigfile or save_hdf5 to 1")


    # #####################
    # START PROGRAM
    # #####################
    in_fname = os.path.join(in_dir,in_file)

    fname_subsample_bigfile = '%s_sub_sr%g_ssseed%d.bigfile' % (in_fname,
        subsample_ratio, sub_ssseed)

    # read catalog, subsample, and save to bigfile (can do in parallel)
    if save_bigfile:
        # read catalog
        cat = BigFileCatalog(in_fname, dataset='1', header='Header')
        rank = cat.comm.rank
        if rank>0 and save_hdf5:
            raise Exception("must run with a single core to be able to save hdf5")
        if rank==0:
            print("Done reading BigFileCatalog from %s" % in_fname)
            print("Cat:", cat)

        # create random subsample
        rng = MPIRandomState(cat.comm, seed=sub_ssseed, size=cat.size)
        rr = rng.uniform(0.0, 1.0, itemshape=(1,))
        # r is (cat.size,1) array.
        mysubsample = cat[rr[:,0] < subsample_ratio]
        if rank==0:
            print("mysubsample:", mysubsample)

        # write to bigfile
        if rank==0:
            print("Writing to %s" % fname_subsample_bigfile)
        mysubsample.save(fname_subsample_bigfile,mysubsample.columns)
        if rank==0:
            print("Wrote %s" % fname_subsample_bigfile)


    # read bigfile and write to hdf5 (must run on single core)
    if save_hdf5:

        # read subsample from subsample bigfile
        if not os.path.exists(fname_subsample_bigfile):
            raise Exception("First run with --save_bigfile; could not find subsample bigfile %s"
                            % fname_subsample_bigfile)
        if 'mysubsample' in vars():
            del mysubsample
        mysubsample_serial = BigFileCatalog(fname_subsample_bigfile, dataset='./', header='Header')
        rank = mysubsample_serial.comm.rank
        if rank==0:
            print("\nHave read %d particles from %s\n" % (len(mysubsample_serial), fname_subsample_bigfile))
        #print("mysubsample.attrs:\n", mysubsample.attrs)


        # use subsample created above (NOT WORKING b/c can only gather numpy arrays)
        # gather subsample to one rank b/c want to write to hdf5 serially
        #mysubsample_serial = GatherArray(mysubsample, mysubsample.comm, root=0)
        #print("Rank %d: mysubsample_serial=%s" % (rank,str(mysubsample_serial)))

        if rank>0:
            raise Exception("must run with a single core to be able to save hdf5")
        
        if rank==0:
            # create marcel Catalog obect from subsample
            import Catalog_nbk01 as Catalog

            subCat = Catalog.Catalog()
            subCat.sim_Ngrid = Nptcles_per_dim
            subCat.sim_boxsize = boxsize
            subCat.dataset_attrs = mysubsample_serial.attrs
            subCat.N_objects = len(mysubsample_serial)

            # convert unicode attrs b/c cannot be easily saved to hdf5 
            for key in subCat.dataset_attrs.keys():
                myattr = subCat.dataset_attrs[key]
                if type(myattr) in [list,np.ndarray] and (myattr.dtype=='<U1'):
                    subCat.dataset_attrs[key] = [a.encode('utf8') for a in myattr]
                    print("Converted attribute %s to utf8" % key)
            print("subCat.dataset_attrs:\n", subCat.dataset_attrs)

            # create structured numpy array P
            print("Columns in bigfile:\n", mysubsample_serial.columns)
            print("Columns to write to hdf5:\n", columns_to_write_to_hdf5)

            dtype = Catalog.Catalog.get_dtype_of_columns(columns_to_write_to_hdf5)
            subCat.P = np.empty(len(mysubsample_serial), dtype=dtype)
            for col in columns_to_write_to_hdf5:
                print("Filling column %s" % col)
                print("type:", type(mysubsample_serial[col]))
                subCat.P[col] = mysubsample_serial[col]
            # convert position to range from 0 to 1
            subCat.P['Position'] *= (1.0/boxsize)
            print("subCat.P['Position'] head:\n", subCat.P['Position'][:10])

            # check position makes sense
            minpos = np.min(subCat.P['Position'])
            maxpos = np.max(subCat.P['Position'])
            print('Pos min, max:', minpos, maxpos)
            if minpos < -0.1:
                print("Warning: Catalog has min(pos)=%s but should be >=-0.1" % str(minpos))
            if maxpos > 1.5:
                print("WARNING: Catalog has max(pos)=%s but should be <=1.5" % str(maxpos))
            if maxpos < 0.5:
                print("WARNING: Catalog has max(pos)=%s but should be >=0.5" % str(maxpos))

            subCat.update_N_objects()

            # box_wrap
            print("box wrap")
            subCat.box_wrap(mincoord=1e-11, maxcoord=1.0, period=1.0)

            # save to hdf5 
            out_fname2 = '%s_sub_bw_sr%g_ssseed%d.hdf5' % (in_fname, subsample_ratio, sub_ssseed)
            subCat.save_to_hdf5(out_fname2)



    
if __name__ == '__main__':
    main()

