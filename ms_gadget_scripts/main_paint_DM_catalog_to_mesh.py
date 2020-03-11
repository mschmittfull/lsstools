#!/home/mschmittfull/anaconda2/envs/nbodykit-0.3-env/bin/python

# NOTE: We load nbodykit-0.3 environment above, so can call with ./main_test_nbkit0.3py.
# Better: Use run.sh script.

# Should make sure that PYTHONPATH="".

# Run with
#   ./main_test_nbkit0.3.py
# or
#   ./run.sh main_test_nbkit0.3py
# but NOT with
#   python main_test_nbkit0.3.py


# Marcel Schmittfull 2018 (mschmittfull@gmail.com)


from __future__ import print_function,division

from nbodykit.lab import *
import numpy as np
import os
from shutil import rmtree
from argparse import ArgumentParser


def main():

    """
    Read positions of particles and paint to grid.
    This includes window effects, (small) shot noise, and gridding effects.

    Follow http://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.base.catalogmesh.html#nbodykit.base.catalogmesh.CatalogMesh
    """
    ap = ArgumentParser()
    ap.add_argument('sim_seed', type=int, default=0, help='Simulation seed')
    ap.add_argument('Nptcles_per_dim', type=int, default=0,
        help='Number of particles per dimension in input sim')
    ap.add_argument('Ngrid_out', type=int, default=0, help='Ngrid of file to save')
    ap.add_argument('in_file', default='', help='Name of input file')
    ap.add_argument('--RSD', default=0, type=int, help='Include RSD if not zero.')
    
    cmd_args = ap.parse_args()
    sim_seed = cmd_args.sim_seed
    Nptcles_per_dim = cmd_args.Nptcles_per_dim
    Ngrid_out = cmd_args.Ngrid_out
    in_file = cmd_args.in_file
    add_RSD = cmd_args.RSD
    
    ### OPTIONS
    boxsize = 500.0
    RSD_LOS = np.array([0,0,1])

    # if True:
    #     # big sim
    #     sim_seed = 400
    #     Nptcles_per_dim = 1536
    #     Ngrid_out = 64
    # else:
    #     # small sim
    #     sim_seed = 400
    #     Nptcles_per_dim = 32
    #     Ngrid_out = 16

    in_dir = '/scratch/mschmittfull/lssbisp2013/ms_gadget/run4/00000%d-%05d-%.1f-wig/' % (
        sim_seed,Nptcles_per_dim,boxsize)
    
    ## Which file to read: IC, ZA, or nonlinear DM snapshot
    #in_file = 'IC'
    #in_file = 'ZA_0.6250'
    #in_file = 'snap_0.6250'
    #in_file = 'snap_000'

    
    out_file = '%s_PtcleDensity_Ng%d' % (in_file,Ngrid_out)
    out_dir = in_dir

    
    
    ### START PROGRAM
    in_fname = os.path.join(in_dir,in_file)

    out_fname = os.path.join(out_dir,out_file)

    # load only dataset '1' which contains DM
    cat = BigFileCatalog(in_fname, dataset='1', header='Header')
    print("cat vars:", dir(cat))
    rank = cat.comm.rank

    if os.path.exists(out_fname):
        if rank==0:
            rmtree(out_fname)
    
    if rank==0:
        print("in_fname:", in_fname)
        #print("cat:", cat)
        #print("cat vars:\n)", dir(cat))
        print("cat attrs:\n", cat.attrs)
        print("cat columns:\n", cat.columns)

    # add RSD
    if add_RSD:
        out_fname += '_RSD%d%d%d' % (RSD_LOS[0], RSD_LOS[1], RSD_LOS[2])
        

    # paint to mesh
    meshsource = cat.to_mesh(Nmesh=Ngrid_out, window='cic', compensated=False, interlaced=False,
                             position='Position')
    outmesh = FieldMesh(meshsource.paint(mode='real'))

    # copy MeshSource attrs
    for k,v in meshsource.attrs.items():
        outmesh.attrs['MeshSource_%s'%k] = v
    if rank==0:
        print("outmesh.attrs:\n", outmesh.attrs)
    
    
    # save to bigfile
    if rank==0:
        print("Writing to %s" % out_fname)
    outmesh.save(out_fname, mode='real')
    if rank==0:
        print("Wrote %s" % out_fname)


    
if __name__ == '__main__':
    main()

