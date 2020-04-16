from __future__ import print_function, division

from copy import copy
import json
import numpy as np
from scipy.special import erf

from lsstools.nbkit03_utils import catalog_persist, get_cstats_string
from nbodykit.mpirng import MPIRandomState
from nbodykit import CurrentMPIComm


class SimGalaxyCatalogCreator(object):
    """
    Class for generating a simulated galaxy catalog based on an input source
    catalog from simulations, e.g. DM-only halo catalog or full hydro galaxy
    catalog.
    """

    def __init__(self):
        raise NotImplementedError

    def get_galaxy_catalog_from_source_catalog(self, source_cat):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    def __str__(self):
        return json.dumps(self.to_dict())

    def __repr__(self):
        return self.__str__()


class PTChallengeGalaxiesFromRockstarHalos(SimGalaxyCatalogCreator):
    """
    Use prescription from PT challenge paper to populate rockstar halos with
    galaxies, based on virial mass of rockstar halos, using core position and 
    core velocity of rockstar halos.

    mass_column should be virial mass from rockstar. Position and Velocity
    should be core position and velocity from rockstar.
    """
    def __init__(
        self,
        name=None,
        RSD=False,
        RSD_los=None,
        log10M_column=None,
        log10Mmin=None,
        sigma_log10M=None
        ):

        self.name = name
        self.RSD = RSD
        self.RSD_los = RSD_los
        self.log10M_column = log10M_column
        self.log10Mmin = log10Mmin
        self.sigma_log10M = sigma_log10M

    def get_galaxy_catalog_from_source_catalog(
        self,
        source_cat,
        rand_seed_for_galaxy_sampling=123):
        
        assert self.log10M_column in source_cat.columns
        #cat = deepcopy(source_cat)
        cat = copy(source_cat)
        comm = CurrentMPIComm.get()

        # For each halo draw a random number RAND b/w 0 and 1.
        # For each halo, compute prob to be a galaxy.
        # Keep only halos where RAND<=prob_gal, remove rest from catalog.
        # This is our galaxy catalog.

        # Draw random number b/w 0 and 1
        rng = MPIRandomState(comm,
                             seed=rand_seed_for_galaxy_sampling,
                             size=cat.size,
                             chunksize=100000)

        cat['RAND'] = rng.uniform(low=0.0, high=1.0, dtype='f8')
        #print(cat[self.log10M_column])
        #cat['PROB_GAL'] = 0.0 #cat[self.log10M_column]
        cat['PROB_GAL'] = 0.5 * (
            1.0 + erf( (cat[self.log10M_column].compute()-self.log10Mmin)
                /self.sigma_log10M ) )
        print('Nhalos:', cat.csize)

        cat = cat[ cat['RAND']<=cat['PROB_GAL'] ]

        print('Ngalaxies:', cat.csize)
        print('Galaxy mass: ', get_cstats_string(cat[self.log10M_column].compute()))

        return cat


    def to_dict(self):
        return {
            'name': self.name,
            'RSD': self.RSD,
            'RSD_los': self.RSD_los,
            'log10M_column': self.log10M_column,
            'log10Mmin': self.log10Mmin,
            'sigma_log10M': self.sigma_log10M
        }



    
