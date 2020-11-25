# Marcel Schmittfull 2018 (mschmittfull@gmail.com)
from __future__ import print_function,division

from nbodykit.lab import *
import numpy as np
import os
from shutil import rmtree
from scipy.interpolate import interp1d
from argparse import ArgumentParser


def main():

    """
    Read a linear power spectrum file and create a linear Gaussian field on a mesh
    using that power spectrum. Also specify seed.

    This has no CIC window effects or shot noise because we never use particles anywhere.

    See http://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.source.mesh.linear.html?highlight=linearmesh#nbodykit.source.mesh.linear.LinearMesh
    for documenation.

    To run on helios:
    salloc --exclusive --nodes=1 --time=01:00:00
    ssh nodeXY.hpc.sns.ias.edu
    cd ~/CODE/lsstools/ms_gadget_scripts
    srun -n 14 python main_create_LinearMesh.py 400 1536
    """

    # read command line arguments
    ap = ArgumentParser()
    ap.add_argument('sim_seed', type=int, help='Simulation seed')
    ap.add_argument('Ngrid_out', type=int, help='Ngrid_out')

    ns = ap.parse_args()
    sim_seed = ns.sim_seed
    Ngrid_out = ns.Ngrid_out
    
    
    ### OPTIONS
    wiggle = False
    boxsize = 1500.0
    Nptcles_per_dim_for_out_dir = 1536  # only used to get name of output folder
    if wiggle:
        Plin_fname = '/home/mschmittfull/CODE/MP-Gadget_msrunscripts/ms_gadget/run4/planck_camb_56106182_matterpower_z0.dat'
    else:
        Plin_fname = '/home/mschmittfull/CODE/MP-Gadget_msrunscripts/ms_gadget/run4/planck_camb_56106182_matterpower_smooth_z0.dat'
    

    # Linear ICs
    if wiggle:
        out_dir = '/scratch/mschmittfull/lss/ms_gadget/run4/00000%d-%05d-%.1f-wig/' % (
            sim_seed,Nptcles_per_dim_for_out_dir,boxsize)
    else:
        out_dir = '/scratch/mschmittfull/lss/ms_gadget/run4/00000%d-%05d-%.1f-now/' % (
            sim_seed,Nptcles_per_dim_for_out_dir,boxsize)
    out_file = 'IC_LinearMesh_z0_Ng%d' % Ngrid_out
    out_fname = os.path.join(out_dir,out_file)
    
    ### START PROGRAM


    # interpolate input power spectrum
    # read Plin
    kPk = np.genfromtxt(Plin_fname)
    myk = kPk[:,0]  # in h/Mpc
    myP = kPk[:,1]  # in (Mpc/h)^3

    # interpolate (in log-log space)
    interp_lnP_at_lnk = interp1d(
        np.log(myk), np.log(myP),
        kind='linear', bounds_error=True, fill_value=-1.0e100)
    def interp_Plin(k):
        if type(k)==np.ndarray:
            Pout = 0.0*k+np.nan
            ww = np.where(k==0.0)[0]
            Pout[ww] = 0.0
            ww = np.where(k>0.0)[0]
            Pout[ww] = np.exp(interp_lnP_at_lnk(np.log(k[ww])))
            return Pout
        else:
            if k==0.0:
                return 0.0
            else:
                return np.exp(interp_lnP_at_lnk(np.log(k)))

    # create linear mesh 
    meshsource = LinearMesh(interp_Plin, np.ones(3)*boxsize, Ngrid_out, seed=sim_seed)
    rank = meshsource.comm.rank

    # remove out file if it exists (actually don't b/c causes issues when running in parallel)
    # if os.path.exists(out_fname):
    #     if rank==0:
    #         rmtree(out_fname)

    # paint to mesh and save on disk
    if rank==0:
        print("Writing to %s" % out_fname)
    meshsource.save(out_fname)
    if rank==0:
        print("Wrote %s" % out_fname)

    
if __name__ == '__main__':
    main()

