from __future__ import print_function,division

from nbodykit.lab import *
import numpy as np
import os
import h5py
from collections import OrderedDict
import json
from argparse import ArgumentParser

from lsstools.nbkit03_utils import get_cstats_string


def main():
    """
    - Read bigfile halo catalog. 
    - Apply mass cut
    - Add RSD
    - Save as hdf5.
    """

    opts = OrderedDict()


    # command line args
    ap = ArgumentParser()

    # optional arguments
    ap.add_argument('--MinMass', type=float, default='12.8',
        help='Minimum halo mass (given as log10M). Set to 0 use no lower bound.')
    ap.add_argument('--MaxMass', type=float, default='13.8',
        help='Maximum halo mass (given as log10M). Set >= 20 to use no upper bound.')
    ap.add_argument('--SimSeed', type=int, default=400, help='Simulation seed to load.')
    ap.add_argument('--RSD', type=int, default=0, help="Include RSD if non-zero.")

    
    # copy args
    args = ap.parse_args()
    opts['sim_seed'] = args.SimSeed
    if args.MinMass == 0. and args.MaxMass >= 20.:
        opts['halo_mass_string'] = ''
    else:
        opts['halo_mass_string'] = '%.1f_%.1f' % (args.MinMass, args.MaxMass)
    opts['log10Mmin'] = args.MinMass
    opts['log10Mmax'] = args.MaxMass
    opts['RSD'] = bool(args.RSD)
    opts['RSD_line_of_sight'] = [0,0,1]


    
    # ################################################################################
    # OPTIONS (not from command line)
    # ################################################################################
    
    # ms_gadget L=500 sim
    opts['sim_name'] = 'ms_gadget'
    opts['boxsize'] = 500.0
    opts['sim_scale_factor'] = 0.6250


    opts['basepath'] = os.path.expandvars('$SCRATCH/lss/ms_gadget/run4/00000%d-01536-%.1f-wig/' % (
        opts['sim_seed'], opts['boxsize']))


    idir_fof = os.path.join(opts['basepath'], 'nbkit_fof_%.4f/ll_0.200_nmin25' % opts['sim_scale_factor'])

    print("Input dir:", idir_fof)

    if not os.path.exists(os.path.join(idir_fof,'Header')):
        raise Exception('%s does not look like a bigfile (missing header)' % idir_fof)

    cat_fof = BigFileCatalog(idir_fof, dataset='./', header='Header')
    print("Found columns:\n", cat_fof.columns)
    print("Read attrs:\n", cat_fof.attrs)
    center_pos = cat_fof['CMPosition'].compute()
    # MS: RSDFactor comes from MP-Gadget. It is probably 1/(a*aH) because UsePeculiarVelocity=0
    # so need to divide by a to go from internal to RSD velocity. The conversion is presumably done to 
    # get v/(aH), which is velocity in Mpc/h.
    # Confirmed that this is correct:
    # Halo catalog has v/(aH), which is velocity in Mpc/h.
    # Expect \theta = div v = -f a H \delta on large scales (in absence of velocity bias).
    # Then, \theta/(aH) = div center_velocity = -f\delta.
    # So expect P_theta/Plin = f^2. 
    # In ms_gadget, measured P_theta/Plin=0.6 and have f^2=0.61826 so looks ok.
    center_vel = cat_fof['CMVelocity'].compute() * cat_fof.attrs['RSDFactor'][0]
    length = cat_fof['Length']
    log10M = cat_fof['log10M']  ## TMP: SHOULD USE cat_fof['log10M']

    #print("first few lengths:", length[:5])
    # In TreePM, we need to use 'Omega0' instead of 'OmegaM' in FastPM.
  
          

    # jerry old
    # data_m = np.hstack((center_pos, center_vel))
    # data_m = np.vstack((data_m.T, length)).T


    if opts['halo_mass_string'] == '':
        # keep all halos with length>0
        ww = np.where(length>0)[0]
    else:
        # apply mass cut
        ww = np.where( (log10M >= opts['log10Mmin']) & (log10M <= opts['log10Mmax']) )[0]

    if opts['RSD']:
        # add RSD
        # Halo catalog saved by nbodykit fof has v/(aH), which is velocity in Mpc/h.
        # So we can just add this to the position array which is also in Mpc/h.
        print('Print adding RSD along line of sight ', opts['RSD_line_of_sight'])
        print('RSD displacement in Mpc/h: ', get_cstats_string(center_vel * opts['RSD_line_of_sight']))
        center_pos += center_vel * opts['RSD_line_of_sight']

    Ntot = ww.shape[0]

    # init out_array
    out_dtype = np.dtype( 
        [ ('Position', ('f4', 3)), ('Velocity', ('f4', 3)), ('log10M', 'f4') ] )
    out_array = np.empty( (Ntot,), dtype=out_dtype)
    print("out_array dtype: ", out_array.dtype)
    print("out_array shape: ", out_array.shape)

    # fill out_array
    assert np.all(cat_fof.attrs['BoxSize'] == cat_fof.attrs['BoxSize'][0])
    out_array['Position'] = center_pos[ww] / cat_fof.attrs['BoxSize'][0]  # out_pos ranges from 0 to 1
    out_array['Velocity'] = center_vel[ww]  # not sure what units, probably Mpc/h
    out_array['log10M'] = log10M[ww]

    # box wrap to [mincoord,1[
    print("min, max pos: %.15g, %.15g" %(np.min(out_array['Position']), np.max(out_array['Position'])))
    mincoord = 1e-11
    if (not np.all(out_array['Position']>=mincoord)) or (not np.all(out_array['Position']<1.0)):
        # box wrap
        print("box wrap")
        out_array['Position'][np.where(out_array['Position']>=1.0)] -= 1.0
        out_array['Position'][np.where(out_array['Position']<0.0)] += 1.0
        assert np.all(out_array['Position']>=0.)
        out_array['Position'][np.where(out_array['Position']<mincoord)] = mincoord
        print("min, max pos after box wrap: %.15g, %.15g" % (np.min(out_array['Position']), np.max(out_array['Position'])))
        assert np.all(out_array['Position']>=mincoord)
        #assert np.all(out_array['Position']<1.0)
        assert np.all(out_array['Position']<=1.0)


    # hdf5 attrs
    bigfile_attrs_dict = OrderedDict()
    for k in cat_fof.attrs.keys():
        if type(cat_fof.attrs[k]) == np.ndarray:
            # convert to list b/c json cannot save numpy arrays
            bigfile_attrs_dict[k] = cat_fof.attrs[k].tolist()
        else:
            bigfile_attrs_dict[k] = cat_fof.attrs[k]
    attrs_dict = {'BoxSize': cat_fof.attrs['BoxSize'], 'Ntot': Ntot,
                  'bigfile_attrs': json.dumps(bigfile_attrs_dict)}
    

    ## write to hdf5
    odir = idir_fof
    out_hdf5_fname = os.path.join(odir, 'fof_nbkfmt.hdf5')
    if opts['halo_mass_string'] != '':
        out_hdf5_fname += '_BOUNDS_log10M_%s.hdf5' % opts['halo_mass_string']
    if opts['RSD']:
        if opts['RSD_line_of_sight'] in [[0,0,1],[0,1,0],[1,0,0]]:
            out_hdf5_fname += '_RSD%d%d%d.hdf5' % (
                opts['RSD_line_of_sight'][0], opts['RSD_line_of_sight'][1],
                opts['RSD_line_of_sight'][2])
        else:
            out_hdf5_fname += '_RSD_%.2f_%.2f_%.2f.hdf5' % (
                opts['RSD_line_of_sight'][0], opts['RSD_line_of_sight'][1],
                opts['RSD_line_of_sight'][2])


    #out_dataset_name = 'FOFGroups'
    # Marcel's hdf5 files always assume data is stored in 'Subsample' dataset.
    out_dataset_name = 'Subsample'
    #print("Output file:", ofile)

    if os.path.exists(out_hdf5_fname):
        os.remove(out_hdf5_fname)
    f_out = h5py.File(out_hdf5_fname, 'w')
    dataset = f_out.create_dataset(
        name=out_dataset_name, dtype=out_array.dtype, shape=(Ntot,))
    # should probably use attrs.create() for this
    for key, v in attrs_dict.items():
        print("type", attrs_dict[key])
        dataset.attrs[key] = attrs_dict[key]
    dataset[:] = out_array[:]

    f_out.close()
    print("Wrote %s" % out_hdf5_fname)

    
    # data_m = np.zeros((ww.shape[0],7))
    # data_m[ww,0:3] = center_pos[ww]
    # data_m[ww,3:6] = center_vel[ww]
    # data_m[ww,6] = length[ww] #log10M[ww] 

    # remove halo with 0 mass
    #if data_m[0,6] == 0:
    #    data_m = data_m[1:,:]



    # # Write to ascii. Combine data of position, velocity and number of particles and output them.
    # ofile = odir + 'fof_PVL10M.dat'
    # print("Write ascii to %s" % ofile)
    # if os.path.exists(ofile):
    #     os.remove(ofile)
    # data_m = np.zeros((Ntot,7))
    # data_m[:,0:3] = out_array['Position']
    # data_m[:,3:6] = out_array['Velocity']
    # data_m[:,6] = out_array['log10M']
    # np.savetxt(ofile, data_m[:], fmt='%.8e %.8e %.8e %.8e %.8e %.8e %8f', newline='\n')
    # print("Wrote %s" % ofile)



    
if __name__ == '__main__':
    main()

