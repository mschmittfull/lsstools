from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import os
import pandas as pd
import shutil


from nbodykit.lab import *
from nbodykit.source.catalog import HDFCatalog
from nbodykit import CurrentMPIComm

import illustris_python as il

from lsstools.nbkit03_utils import get_cmean, get_cstats_string



def main():
    """
    Read Illustris group catalog.
    """
    basePath = '/data/mschmittfull/lss/IllustrisTNG/L205n2500TNG/output/'
    #basePath = '/data/mschmittfull/lss/IllustrisTNG/Illustris-3/output/'
    #basePath = '/Users/mschmittfull/scratch_data/lss/IllustrisTNG/Illustris-3/output'
    fields = ['SubhaloMass',
              'SubhaloMassType',
              'SubhaloSFR',
              'SubhaloFlag',
              'SubhaloPos',  # pos of most bound particle
              'SubhaloCM', # pos of center of mass
              'SubhaloVel', 
              'SubhaloVelDisp',
              'SubhaloStellarPhotometrics']

    #snapNum = 1
    snapNum = 67

    # read group catalog
    print('Reading snap %d group catalog from %s' % (snapNum, basePath))

    # load header
    header = il.groupcat.loadHeader(basePath, snapNum)
    print('header: ', header)


    attrs = dict(
        Redshift = header[u'Redshift'],
        Time = header[u'Time'],
        BoxSize  = header[u'BoxSize']/1e3 * np.ones(3),  #convert to Mpc/h
        Ngroups_Total = header[u'Ngroups_Total'],
        Nsubgroups_Total = header[u'Nsubgroups_Total'],
        Omega0  = header[u'Omega0'],
        OmegaLambda  = header[u'OmegaLambda'],
        HubbleParam        = header[u'HubbleParam'],
    )
    print('attrs: ', attrs)


    # load subfind subhalos
    subhalos = il.groupcat.loadSubhalos(basePath, snapNum, fields=fields)
    del subhalos['count']

    # convert to nbkit catalog and save to disk
    # see https://nbodykit.readthedocs.io/en/latest/catalogs/reading.html#array-data
    cat = ArrayCatalog(subhalos, **attrs)

    print(cat)

    # save to disk
    outfile = os.path.join(basePath, 'groups_%03d.bigfile' % snapNum)
    print('Writing to %s' % outfile)
    if os.path.exists(outfile):
        shutil.rmtree(outfile)
    cat.save(outfile, columns=cat.columns)
    print('Wrote %s' % outfile)
    


if __name__ == '__main__':
    main()
