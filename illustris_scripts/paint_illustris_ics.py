from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import os
import h5py

from nbodykit.lab import *
from nbodykit.source.catalog import HDFCatalog
from nbodykit import CurrentMPIComm

from lsstools.nbkit03_utils import get_cmean, get_cstats_string



def main():
    """
    Read illustris ICs
    """

    ####
    # OPTIONS
    ####
    Nmesh_lst = [2500]
   
    if False:
        # Illustris-3
        # https://www.illustris-project.org/data/downloads/Illustris-3/
        fname = '/data/mschmittfull/lss/IllustrisTNG/Illustris-3/output/snap_ics.hdf5'
        
    else:
        # TNG300-1 / L205n2500TNG
        fname = '/data/mschmittfull/lss/IllustrisTNG/L205n2500TNG/output/snap_ics.hdf5'

    ####
    # Run code
    ####

    comm = CurrentMPIComm.get()
    print('Greetings from rank %d' % comm.rank)

    for Nmesh in Nmesh_lst:

        if comm.rank==0:
            print('Reading header from %s' % fname)

        # Read header
        f = h5py.File(fname, 'r')
        if comm.rank == 0:
            print('header: ', f['Header'].attrs.keys())

        attrs = dict(
            Redshift = f['Header'].attrs[u'Redshift'],
            Time = f['Header'].attrs[u'Time'],
            BoxSize  = f['Header'].attrs[u'BoxSize']/1e3 * np.ones(3),  #convert to Mpc/h
            NumFilesPerSnapshot = f['Header'].attrs[u'NumFilesPerSnapshot'],
            NumPart_ThisFile = f['Header'].attrs[u'NumPart_ThisFile'],
            Omega0  = f['Header'].attrs[u'Omega0'],
            OmegaLambda  = f['Header'].attrs[u'OmegaLambda'],
            HubbleParam        = f['Header'].attrs[u'HubbleParam'],
            MassTable   = f['Header'].attrs[u'MassTable'],
            TotNumPart     = f['Header'].attrs[u'NumPart_Total']
        )

        if comm.rank==0:
            print('attrs:')
            print(attrs)

        f.close()

        # read particles positions
        if comm.rank==0:
            print('Reading particle data from %s' % fname)

        cat = HDFCatalog(fname, exclude=['PartType1/ParticleIDs', 'PartType1/Velocities'],
                       attrs=attrs)
        if comm.rank==0:
            print('columns: ', cat.columns)

        # convert kpc/h to Mpc/h
        cat['Position'] = cat['PartType1/Coordinates'] / 1e3

        if comm.rank==0:
            print('paint...')
        catmesh = cat.to_mesh(Nmesh=Nmesh,
                              window='cic', compensated=False, interlaced=False
                              )

        if False:
            # crashes when using too many cores (empty arrays)
            rfield = catmesh.compute() 
            stats = get_cstats_string(rfield)
            if comm.rank==0:
                print('density stats: %s' % stats)

        # save to bigfile
        out_fname = '%s_PtcleDensity_z%d_Ng%d' % (fname, int(catmesh.attrs['Redshift']), Nmesh)
        if comm.rank==0:
            print('Writing to %s' % out_fname)
        catmesh.save(out_fname)
        if comm.rank==0:
            print('Wrote %s' % out_fname)

if __name__ == '__main__':
    main()


