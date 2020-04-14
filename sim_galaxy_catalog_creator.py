from __future__ import print_function, division

from collections import Counter, OrderedDict
import json
import numpy as np

from lsstools.nbkit03_utils import catalog_persist, get_cstats_string
from nbodykit.lab import *


class SimGalaxyCatalogCreator(object):
    """
    Class for generating a simulated galaxy catalog based on an input source
    catalog from simulations, e.g. DM-only halo catalog or full hydro galaxy
    catalog.
    """

    def __init__(
        self,
        name=None,
        RSD=None,
        RSD_los=None
        ):
        self.name = name
        self.RSD = RSD
        self.RSD_los = RSD_los

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
    """
    def __init__(
        self,
        name=None,
        RSD=None,
        RSD_los=None,
        mass_column=None,
        log10Mmin=None,
        sigma_log10M=None
        ):

        super(SimGalaxyCatalogCreator, self).__init__(
            name=name, RSD=RSD, RSD_los=RSD_los)
        self.mass_column = mass_column
        self.log10Mmin = log10Mmin
        self.sigma_log10M = sigma_log10M

    def get_galaxy_catalog_from_source_catalog(self, source_cat):
        assert self.mass_column in source_cat.columns



    def to_dict(self):
        return {
            'name': self.name,
            'RSD': self.RSD,
            'RSD_los': self.RSD_los,
            'mass_column': self.mass_column,
            'log10Mmin': self.log10Mmin,
            'sigma_log10M': self.sigma_log10M
        }



    