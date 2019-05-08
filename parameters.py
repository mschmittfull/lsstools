from __future__ import print_function, division
from collections import namedtuple


# Bunch parameters together to simplify arguments of functions.

GridOpts = namedtuple('GridOpts', ['Ngrid', 'kmax',
    'grid_ptcle2grid_deconvolution'])

TrfFcnOpts = namedtuple('TrfFncOpts', ['Rsmooth_for_quadratic_sources',
    'Rsmooth_for_quadratic_sources2', 'N_ortho_iter', 'orth_method',
    'interp_kind'])


class PowerOpts(object):
    def __init__(self, 
                 k_bin_width=1.0,
                 Pk_1d_2d_mode='1d', 
                 RSD_poles=None,
                 RSD_Nmu=None,
                 RSD_los=None,
                 Pk_ptcle2grid_deconvolution=None):
        """Object to bundle options for power spectrum measurements.

        Attributes
        ----------
        k_bin_width : float
            Width of each k bin, in units of k_fundamental=2pi/L. Must be >=1.

        Pk_1d_2d_mode : string
            '1d' or '2d' (use for anisotropic power e.g. due to RSD)

        RSD_poles : list
            Multipoles to measure if mode='2d'. E.g. [0,2,4].

        RSD_Nmu : int
            If not None, measure also P(k,mu), in Nmu mu bins.

        RSD_los : list
            Direction of line of sight if `Pk_1d_2d_mode`=='2d', e.g. [0,0,1]
            for z direction.
        """
        self.k_bin_width = k_bin_width
        self.Pk_1d_2d_mode = Pk_1d_2d_mode
        self.RSD_poles = RSD_poles
        self.RSD_Nmu = RSD_Nmu
        self.RSD_los = RSD_los
        self.Pk_ptcle2grid_deconvolution = Pk_ptcle2grid_deconvolution


class SimOpts(object):
    def __init__(self, 
                 boxsize,
                 sim_scale_factor,
                 cosmo_params,
                 **kwargs):
        """
        Simulation options.

        Parameters
        ----------
        boxsize : float
            Boxsize in Mpc/h.
        sim_scale_factor : float
            Simulation scale factor a=1/(1+z). Used to generate linear density
            at the same redshift.
        cosmo_params : dict
            Cosmological parameters used to run the simulation. Used to scale linear
            density to the simulation redshift.
        kwargs : arbitrary
            Additional, optional options to be stored as attributes.
        """
        self.boxsize = boxsize
        self.sim_scale_factor = sim_scale_factor
        self.cosmo_params = cosmo_params
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def load_default_opts(sim_name, **kwargs):
        """Load default options of a simulation. They can be overwritten using
        kwargs.

        Parameters
        ----------
        sim_name : string

        Returns
        -------
        sim_opts : SimOpts object
        """
        if sim_name == 'ms_gadget_test_data':
            # L=500 ms_gadget sims produced with MP-Gadget, 1536^3 particles,
            # 64^3 test data.
            default = {}
            default['sim_name'] = sim_name
            default['boxsize'] = 500.0
            default['sim_scale_factor'] = 0.6250
            default['sim_irun'] = 4
            default['sim_seed'] = 403
            # seed used to draw subsample
            default['ssseed'] = 40000 + default['sim_seed']
            # Nbody, so used thousands of time steps
            default['sim_Ntimesteps'] = None  
            default['sim_Nptcles'] = 1536
            default['sim_wig_now_string'] = 'wig'
            # halo mass
            default['halo_mass_string'] = '13.8_15.1'

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

        else:
            raise Exception('Invalid sim_name %s' % sim_name)

        # update keys given in kwargs
        opts_dict = default.copy()
        opts_dict.update(kwargs)

        sim_opts = SimOpts(**opts_dict)
        return sim_opts
