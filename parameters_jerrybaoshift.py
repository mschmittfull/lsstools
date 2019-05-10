from __future__ import print_function, division
from collections import namedtuple, OrderedDict

from parameters import SimOpts


class JerryBAOShiftSimOpts(SimOpts):
    def __init__(self, boxsize, sim_scale_factor, cosmo_params, **kwargs):
        """
        Simulations options for Jerry's BAO shift sims.
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
        default['sim_name'] = 'jerryou_baoshift'
        
        # cosmology
        # omega_m = 0.307494
        # omega_bh2 = 0.022300
        # omega_ch2 = 0.118800
        # h = math.sqrt((omega_bh2 + omega_ch2) / omega_m) = 0.6774
        default['cosmo_params'] = dict(Om_m=0.307494,
                                       Om_L=1.0 - 0.307494,
                                       Om_K=0.0,
                                       Om_r=0.0,
                                       h0=0.6774)

        raise Exception('TODO: include more default params')
