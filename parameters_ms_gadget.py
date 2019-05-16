from __future__ import print_function, division
from collections import namedtuple, OrderedDict

from parameters import SimOpts


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
        #default['ssseed'] = 40000 + default['sim_seed']
        default['ssseed'] = None
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
                                      shifted_fields_Nmesh=1536,
                                      RSDstrings=None):
        """See parent class.
        """
        ext_grids = OrderedDict()
        if RSDstrings is None:
            RSDstrings = ['']

        if True:
            # Linear density (ICs of the sims).
            # This is deltalin from mesh (created on mesh, without particles).
            # This has no RSD.
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
            ext_grids['deltaZA_NORSD'] = {
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
            # artifacts, small shot noise).
            # This has no RSD.
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

            for RSDstring in RSDstrings:

                # 1 shifted by deltalin_Zeldovich displacement (using nbkit0.3;
                # same as delta_ZA)
                ext_grids['1_SHIFTEDBY_%sdeltalin%s' % (
                    psi_type_str, RSDstring
                )] = {
                    'dir':
                    '1_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                    % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                       shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                       RSDstring),
                    'file_format': 'nbkit_BigFileGrid',
                    'dataset_name': 'Field',
                    'scale_factor': self.sim_scale_factor,
                    'nbkit_normalize': True,
                    'nbkit_setMean': 0.0
                }

                # deltalin shifted by deltalin_Zeldovich displacement (using 
                # nbkit0.3)
                ext_grids['deltalin_SHIFTEDBY_%sdeltalin%s' % (
                    psi_type_str, RSDstring
                )] = {
                    'dir':
                    'IC_LinearMesh_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                    % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                       shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                       RSDstring),
                    'file_format': 'nbkit_BigFileGrid',
                    'dataset_name': 'Field',
                    'scale_factor': self.sim_scale_factor,
                    'nbkit_normalize': True,
                    'nbkit_setMean': 0.0
                }

                # deltalin^2 shifted by deltalin_Zeldovich displacement (using 
                # nbkit0.3)
                ext_grids[
                    'deltalin_growth-mean_SHIFTEDBY_%sdeltalin%s' % (
                        psi_type_str, RSDstring
                    )] = {
                        'dir':
                        'IC_LinearMesh_growth-mean_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                        % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                           shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                           RSDstring),
                        'file_format': 'nbkit_BigFileGrid',
                        'dataset_name': 'Field',
                        'scale_factor': self.sim_scale_factor,
                        'nbkit_normalize': True,
                        'nbkit_setMean': 0.0
                    }

                # G2[deltalin] shifted by deltalin_Zeldovich displacement (using 
                # nbkit0.3)
                ext_grids['deltalin_G2_SHIFTEDBY_%sdeltalin%s' % (
                    psi_type_str, RSDstring)] = {
                    'dir':
                    'IC_LinearMesh_tidal_G2_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                    % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                       shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                       RSDstring),
                    'file_format': 'nbkit_BigFileGrid',
                    'dataset_name': 'Field',
                    'scale_factor': self.sim_scale_factor,
                    'nbkit_normalize': True,
                    'nbkit_setMean': 0.0
                }

                # deltalin^3 shifted by deltalin_Zeldovich displacement (using 
                # nbkit0.3)
                ext_grids['deltalin_cube-mean_SHIFTEDBY_%sdeltalin%s' % (
                    psi_type_str, RSDstring)] = {
                    'dir':
                    'IC_LinearMesh_cube-mean_intR0.00_0.50_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                    % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                       shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                       RSDstring),
                    'file_format': 'nbkit_BigFileGrid',
                    'dataset_name': 'Field',
                    'scale_factor': self.sim_scale_factor,
                    'nbkit_normalize': True,
                    'nbkit_setMean': 0.0
                }

        return ext_grids

    def get_default_catalogs(self, RSDstrings=None):
        """Default catalogs to load for ms_gadget sims.
        """
        cats = OrderedDict()
        if RSDstrings is None:
            RSDstrings = ['']

        halo_dir = 'nbkit_fof_%.4f/ll_0.200_nmin25' % (
            self.sim_scale_factor)

        for RSDstring in RSDstrings:
            if RSDstring == '':
                RSDfilestr = ''
            else:
                RSDfilestr = '%s.hdf5' % RSDstring

            ## nonuniform catalogs without ptcle masses
            if True:
                # halos without mass weight, narrow mass cuts: 10.8..11.8..12.8
                # ..13.8..15.1
                cats['delta_h%s' % RSDstring] = {
                    'in_fname':
                    "%s/fof_nbkfmt.hdf5_BOUNDS_log10M_%s.hdf5%s" %
                    (halo_dir, self.halo_mass_string, RSDfilestr),
                    'weight_ptcles_by':
                    None
                }

            if False:
                # halos not weighted by mass but including mass info in file,
                # broad mass cut
                # TODO: looks like nbodykit 0.3 does not read this properly b/c
                # of hdf5. Should switch all files to bigfile at some point.
                cats['delta_h%s' % RSDstring] = {
                    'in_fname':
                    "%s/fof_nbkfmt.hdf5_WithMassCols.hdf5_BOUNDS_log10M_%s.hdf5%s" %
                    (halo_dir, self.halo_mass_string, RSDfilestr),
                    'weight_ptcles_by':
                    None
                }

            if False:
                # halos in narrow mass bins, no mass weights
                cats['delta_h_M10.8-11.8%s' % RSDstring] = {
                    'in_fname':
                    "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_10.8_11.8.hdf5%s" %
                    (halo_dir, RSDfilestr),
                    'weight_ptcles_by':
                    None
                }
                cats['delta_h_M11.8-12.8%s' % RSDstring] = {
                    'in_fname':
                    "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_11.8_12.8.hdf5%s" %
                    (halo_dir, RSDfilestr),
                    'weight_ptcles_by':
                    None
                }
                cats['delta_h_M12.8-13.8%s' % RSDstring] = {
                    'in_fname':
                    "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_12.8_13.8.hdf5%s" %
                    (halo_dir, RSDfilestr),
                    'weight_ptcles_by':
                    None
                }
                cats['delta_h_M13.8-15.1%s' % RSDstring] = {
                    'in_fname':
                    "%s/fof_nbkfmt.hdf5_WithMassCols_BOUNDS_log10M_13.8_15.1.hdf5%s" %
                    (halo_dir, RSDfilestr),
                    'weight_ptcles_by':
                    None
                }

        return cats
