from __future__ import print_function, division
from collections import namedtuple, OrderedDict

# Bunch parameters together to simplify arguments of functions.

GridOpts = namedtuple('GridOpts',
                      ['Ngrid', 'kmax', 'grid_ptcle2grid_deconvolution'])


class TrfFcnOpts(object):
    def __init__(self,
                 Rsmooth_for_quadratic_sources=0.1,
                 Rsmooth_for_quadratic_sources2=0.1,
                 orth_method='CholeskyDecomp',
                 N_ortho_iter=1,
                 interp_kind='manual_Pk_k_bins'):
        """
        Object bundling transfer function options.

        Parameters
        ----------
        Rsmooth_for_quadratic_sources : float
            Smoothing sclae for quadratic fields, in Mpc/h.

        Rsmooth_for_quadratic_sources2 : float
            Alternative smoothing scale that can be used optionally by quadr.
            fields. In Mpc/h.

        orth_method : string
            Orthogonalization method used for fields when computing trf fcns.
            Only used if `N_ortho_iter`>=1. This affects trf fcns but not final
            noise curves.
            - 'EigenDecomp': Use eigenvalue decomposition of S matrix; each 
                orthogonal field is a combination of all original fields, so not
                great.
            - 'CholeskyDecomp': Use Cholesky decomposition of S matrix so that 
                span of first k orthogonal fields is the same as span of first k
                original fields. This is the default.

        N_ortho_iter : int
            Number of times to run orthogonalization algorithm. 1 is usually ok.

        interp_kind : string
            Interpolation used for transfer functions. 'nearest', 'linear', or
            'manual_Pk_k_bins' (=most accurate).
        """
        self.Rsmooth_for_quadratic_sources = Rsmooth_for_quadratic_sources
        self.Rsmooth_for_quadratic_sources2 = Rsmooth_for_quadratic_sources2
        self.orth_method = orth_method
        self.N_ortho_iter = N_ortho_iter
        self.interp_kind = interp_kind


class PowerOpts(object):
    def __init__(self,
                 k_bin_width=1.0,
                 Pk_1d_2d_mode='1d',
                 RSD_poles=None,
                 RSD_Nmu=None,
                 RSD_los=None,
                 Pk_ptcle2grid_deconvolution=None):
        """Object bundling options for power spectrum measurements.

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
                 boxsize=None,
                 sim_scale_factor=None,
                 cosmo_params=None,
                 **kwargs):
        """
        Simulation options. For each simulation, write a subclass.

        Parameters
        ----------
        boxsize : float
            Boxsize in Mpc/h.
        sim_scale_factor : float
            Simulation scale factor a=1/(1+z). Used to generate linear density
            at the same redshift.
        cosmo_params : dict
            Cosmological parameters used to run the simulation. Used to scale
            linear density to the simulation redshift.
        kwargs : arbitrary
            Additional, optional options to be stored as attributes.
        """
        self.boxsize = boxsize
        self.sim_scale_factor = sim_scale_factor
        self.cosmo_params = cosmo_params
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def load_default_opts(**kwargs):
        """Load default options of a simulation. They can be overwritten using
        kwargs. Should be implemented by child classes.

        Parameters
        ----------
        sim_name : string

        Returns
        -------
        sim_opts : SimOpts object
        """
        raise NotImplementedError("To be implemented by child classes.")

    def get_default_ext_grids_to_load(self,
                                      Ngrid,
                                      include_shifted_fields=True,
                                      shifted_fields_RPsi=0.23,
                                      shifted_fields_Np=1536,
                                      shifted_fields_Nmesh=1536):
        """
        Specify default list of external grids to load with the simulation.

        Parameters
        ----------
        Ngrid : int
            Number of grid points per dimension of the grids to load.
        shifted_fields_RPsi : float
            Psi smoothing used in shifting code. In Mpc/h.
        shifted_fields_Np : int
            Nptcles_per_dim used in shifting code.
        shifted_fields_Nmesh : int
            Internal Nmesh used in shifting code

        Returns
        -------
        ext_grids : dict
            Dictionary with keys labeling the external grids and values
            specifying how to load the external grids.
        """
        raise NotImplementedError("To be implemented by child classes.")

    def get_default_catalogs(self):
        """Default catalogs to load for given sims.
        """
        raise NotImplementedError("To be implemented by child classes.")

