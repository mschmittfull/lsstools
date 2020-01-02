from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import os
import pandas as pd
from astropy.table import Table

from nbodykit.lab import *
from nbodykit.source.catalog import HDFCatalog
from nbodykit import CurrentMPIComm

import illustris_python as il

from lsstools.nbkit03_utils import get_cmean, get_cstats_string



def main():
    """
    Read Illustris subhalos from bigfile and select
    """
    basePath = '/data/mschmittfull/lss/IllustrisTNG/L205n2500TNG/output/'
    #basePath = '/data/mschmittfull/lss/IllustrisTNG/Illustris-3/output/'
    #basePath = '/Users/mschmittfull/scratch_data/lss/IllustrisTNG/Illustris-3/output'
    fields = ['SubhaloMass','SubhaloMassType','SubhaloSFRinRad','SubhaloFlag']

    snapNum = 1
    #snapNum = 67


    todooo... read bigfile

    # only keep cosmological subhalos (no disk fragments found by subfind)
    ww = np.where( shtable['SubhaloFlag']==1 )[0]
    shtable = shtable[ww]


    # PartType0 - GAS
    # PartType1 - DM
    # PartType2 - (unused)
    # PartType3 - TRACERS
    # PartType4 - STARS & WIND PARTICLES
    # PartType5 - BLACK HOLES

    # select stellar mass of each object, in 1e10 Msun/h
    smass = shtable['SubhaloMassType'][:,4]
    print(smass.shape)

    # only keep objects where stars have formed
    smass = smass[smass > 0.0]


    #TODO: cut on subhaloflag!
    #https://www.tng-project.org/data/docs/background/#sec1

    print('stellar mass stats: min, mean, max:', np.min(smass),
          np.mean(smass), np.max(smass))

    sm_df = pd.DataFrame(smass)
    print(sm_df.describe())

    # want to resemble boss
    # e.g. maraston+ https://arxiv.org/pdf/1207.6114.pdf 
    print(len(np.where(smass > (10.0**10.0)/1e10)[0]))


    ## Load FoF halos=halos=groups
    # Let us get a list of primary subhalo IDs by loading the GroupFirstSub field from the FoF groups.
    GroupFirstSub = il.groupcat.loadHalos(basePath, snapNum, fields=['GroupFirstSub'])
    GroupFirstSub.dtype
    print(GroupFirstSub.shape)
    
    


if __name__ == '__main__':
    main()
