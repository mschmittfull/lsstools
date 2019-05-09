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


class MSGadgetSimOpts(SimOpts):
    def __init__(self, boxsize, sim_scale_factor, cosmo_params, **kwargs):
        """
        Simulations options for ms_gadget sims.
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
        # L=500 ms_gadget sims produced with MP-Gadget, 1536^3 particles,
        # 64^3 test data.
        default = {}
        default['sim_name'] = 'ms_gadget'
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
        default['f_log_growth'] = None
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

        # update with kwargs and return
        default.update(kwargs)
        return MSGadgetSimOpts(**default)

    def get_default_ext_grids_to_load(self,
                                      Ngrid,
                                      include_shifted_fields=True,
                                      shifted_fields_RPsi=0.23,
                                      shifted_fields_Np=1536,
                                      shifted_fields_Nmesh=1536):
        """See parent class.
        """
        ext_grids = OrderedDict()

        if True:
            # linear density (ICs of the sims)
            # deltalin from mesh (created on mesh, no particles involved)
            ext_grids['deltalin'] = {
                'dir': 'IC_LinearMesh_z0_Ng%d' % Ngrid,
                'file_format': 'nbkit_BigFileGrid',
                'dataset_name': 'Field',
                'scale_factor': 1.0,
                'nbkit_normalize': True,
                'nbkit_setMean': 0.0
            }

        if False:
            # deltalin from ptcles (created from particle snapshot so
            # includes CIC artifacts)
            # on 64^3, noise curves looked the same as with linearMesh
            ext_grids['deltalin_PtcleDens'] = {
                'dir': 'IC_PtcleDensity_Ng%d' % Ngrid,
                'file_format': 'nbkit_BigFileGrid',
                'dataset_name': 'Field',
                'scale_factor': 1.0 / (1.0 + 99.0),  # ICs are at z=99
                'nbkit_normalize': True,
                'nbkit_setMean': 0.0
            }

        if False:
            # delta_ZA, created by moving 1536^3 ptcles with NGenic
            # (includes CIC artifacts, small shot noise)
            ext_grids['deltaZA'] = {
                'dir':
                'ZA_%.4f_PtcleDensity_Ng%d' %
                (self.sim_scale_factor, Ngrid),
                'file_format':
                'nbkit_BigFileGrid',
                'dataset_name':
                'Field',
                'scale_factor':
                self.sim_scale_factor,
                'nbkit_normalize':
                True,
                'nbkit_setMean':
                0.0
            }

        if True:
            # deltanonl painted from all 1536^3 DM particles (includes CIC
            # artifacts, small shot noise)
            ext_grids['delta_m'] = {
                'dir':
                'snap_%.4f_PtcleDensity_Ng%d' %
                (self.sim_scale_factor, Ngrid),
                'file_format':
                'nbkit_BigFileGrid',
                'dataset_name':
                'Field',
                'scale_factor':
                self.sim_scale_factor,
                'nbkit_normalize':
                True,
                'nbkit_setMean':
                0.0
            }

        if include_shifted_fields:

            ## Shifted fields
            #for psi_type_str in ['','Psi2LPT_']:
            psi_type_str = ''

            # 1 shifted by deltalin_Zeldovich displacement (using nbkit0.3;
            # same as delta_ZA)
            ext_grids['1_SHIFTEDBY_%sdeltalin' % psi_type_str] = {
                'dir':
                '1_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum'
                % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                   shifted_fields_Np, shifted_fields_Nmesh, Ngrid),
                'file_format':
                'nbkit_BigFileGrid',
                'dataset_name':
                'Field',
                'scale_factor':
                self.sim_scale_factor,
                'nbkit_normalize':
                True,
                'nbkit_setMean':
                0.0
            }

            # deltalin shifted by deltalin_Zeldovich displacement (using 
            # nbkit0.3)
            ext_grids['deltalin_SHIFTEDBY_%sdeltalin' % psi_type_str] = {
                'dir':
                'IC_LinearMesh_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum'
                % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                   shifted_fields_Np, shifted_fields_Nmesh, Ngrid),
                'file_format':
                'nbkit_BigFileGrid',
                'dataset_name':
                'Field',
                'scale_factor':
                self.sim_scale_factor,
                'nbkit_normalize':
                True,
                'nbkit_setMean':
                0.0
            }

            # deltalin^2 shifted by deltalin_Zeldovich displacement (using 
            # nbkit0.3)
            ext_grids[
                'deltalin_growth-mean_SHIFTEDBY_%sdeltalin' % psi_type_str] = {
                    'dir':
                    'IC_LinearMesh_growth-mean_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum'
                    % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                       shifted_fields_Np, shifted_fields_Nmesh, Ngrid),
                    'file_format':
                    'nbkit_BigFileGrid',
                    'dataset_name':
                    'Field',
                    'scale_factor':
                    self.sim_scale_factor,
                    'nbkit_normalize':
                    True,
                    'nbkit_setMean':
                    0.0
                }

            # G2[deltalin] shifted by deltalin_Zeldovich displacement (using 
            # nbkit0.3)
            ext_grids['deltalin_G2_SHIFTEDBY_%sdeltalin' % psi_type_str] = {
                'dir':
                'IC_LinearMesh_tidal_G2_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum'
                % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                   shifted_fields_Np, shifted_fields_Nmesh, Ngrid),
                'file_format':
                'nbkit_BigFileGrid',
                'dataset_name':
                'Field',
                'scale_factor':
                self.sim_scale_factor,
                'nbkit_normalize':
                True,
                'nbkit_setMean':
                0.0
            }

            # deltalin^3 shifted by deltalin_Zeldovich displacement (using 
            # nbkit0.3)
            ext_grids['deltalin_cube-mean_SHIFTEDBY_%sdeltalin' % psi_type_str] = {
                'dir':
                'IC_LinearMesh_cube-mean_intR0.00_0.50_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum'
                % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                   shifted_fields_Np, shifted_fields_Nmesh, Ngrid),
                'file_format':
                'nbkit_BigFileGrid',
                'dataset_name':
                'Field',
                'scale_factor':
                self.sim_scale_factor,
                'nbkit_normalize':
                True,
                'nbkit_setMean':
                0.0
            }

        return ext_grids

    def get_default_catalogs(self):
        """Default catalogs to load for ms_gadget sims.
        """
        cats = OrderedDict()

        tmp_halo_dir = 'nbkit_fof_%.4f/ll_0.200_nmin25' % (
            self.sim_scale_factor)

        ## nonuniform catalogs without ptcle masses
        if True:
            # halos without mass weight, narrow mass cuts: 10.8..11.8..12.8
            # ..13.8..15.1
            cats['delta_h'] = {
                'in_fname':
                "%s/fof_nbkfmt.hdf5_BOUNDS_log10M_%s.hdf5" %
                (tmp_halo_dir, self.halo_mass_string),
                'weight_ptcles_by':
                None
            }

        if False:
            # halos not weighted by mass but including mass info in file,
            # broad mass cut
            # TODO: looks like nbodykit 0.3 does not read this properly b/c
            # of hdf5. Should switch all files to bigfile at some point.
            cats['delta_h'] = {
                'in_fname':
                "%s/fof_nbkfmt.hdf5_WithMassCols.hdf5_BOUNDS_log10M_%s.hdf5" %
                (tmp_halo_dir, self.halo_mass_string),
                'weight_ptcles_by':
                None
            }

        if False:
            # halos in narrow mass bins, no mass weights
            cats['delta_h_M10.8-11.8'] = {
                'in_fname':
                "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_10.8_11.8.hdf5" %
                tmp_halo_dir,
                'weight_ptcles_by':
                None
            }
            cats['delta_h_M11.8-12.8'] = {
                'in_fname':
                "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_11.8_12.8.hdf5" %
                tmp_halo_dir,
                'weight_ptcles_by':
                None
            }
            cats['delta_h_M12.8-13.8'] = {
                'in_fname':
                "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_12.8_13.8.hdf5" %
                tmp_halo_dir,
                'weight_ptcles_by':
                None
            }
            cats['delta_h_M13.8-15.1'] = {
                'in_fname':
                "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_13.8_15.1.hdf5" %
                tmp_halo_dir,
                'weight_ptcles_by':
                None
            }

        return cats


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


