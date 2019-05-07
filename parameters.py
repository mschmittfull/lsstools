from __future__ import print_function, division
from collections import namedtuple


# Bunch parameters together to simplify arguments of functions.

GridOpts = namedtuple('GridOpts', ['Ngrid', 'kmax',
    'grid_ptcle2grid_deconvolution'])

SimOpts = namedtuple('SimOpts', ['boxsize', 'f_log_growth',
    'sim_scale_factor', 'cosmo_params', 'ssseed'])

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
