from __future__ import print_function, division
from collections import namedtuple, OrderedDict

from parameters import SimOpts


class RunPBSimOpts(SimOpts):
    def __init__(self, boxsize, sim_scale_factor, cosmo_params, **kwargs):
        """
        Simulations options for Martin White's RunPB sims.
        """
        super(MSGadgetSimOpts, self).__init__(
            boxsize,
            sim_scale_factor,
            cosmo_params,
            **kwargs
        )

    @staticmethod
    def load_default_opts(**kwargs):
        """See parent class.
        """
        default = {}
        default['sim_name'] = 'RunPB'

        # RunPB by Martin White; read cosmology from Martin email
        #omega_bh2=0.022
        #omega_m=0.292
        #h=0.69
        # Martin rescaled sigma8 from camb from 0.84 to 0.82 to set up ICs.
        # But no need to include here b/c we work with that rescaled 
        # deltalin directly (checked that linear power agrees with nonlinear
        # one if rescaled deltalin rescaled by D(z).
        default['cosmo_params'] = dict(Om_m=0.292,
                                       Om_L=1.0 - 0.292,
                                       Om_K=0.0,
                                       Om_r=0.0,
                                       h0=0.69)

        raise Exception('TODO: include more default params for RunPB')
