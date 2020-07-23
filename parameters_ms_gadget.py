from __future__ import print_function, division
from collections import namedtuple, OrderedDict
import numpy as np

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
        # L=500,1500 ms_gadget sims produced with MP-Gadget, 1536^3 particles,
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
        # halo mass
        default['halo_mass_string'] = '13.8_15.1'
        default['hod_model_name'] = 'Zheng07_HandSeljak17'

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

        # calculate D and f
        cosmo = CosmoModel(**default['cosmo_params'])
        calc_Da = generate_calc_Da(cosmo=cosmo)
        f_log_growth = calc_f_log_growth_rate(
            a=sim_opts.sim_scale_factor,
            calc_Da=calc_Da,
            cosmo=cosmo,
            do_test=True
            )
        # save in opts so we can easily access it throughout code (although strictly
        # speaking it is not a free option but derived from cosmo_params)
        default['f_log_growth'] = f_log_growth

        # update with kwargs and return
        default.update(kwargs)
        return MSGadgetSimOpts(**default)

    def get_default_ext_grids_to_load(self,
                                      Ngrid,
                                      include_shifted_fields=True,
                                      shifted_fields_RPsi=0.23,
                                      shifted_fields_Np=1536,
                                      shifted_fields_Nmesh=1536,
                                      include_2LPT_shifted_fields=False,
                                      include_3LPT_shifted_fields=False,
                                      include_minus_3LPT_shifted_fields=False,
                                      include_div_shifted_PsiDot1=False,
                                      include_div_shifted_PsiDot2=False,
                                      include_shifted_PsiDot1=False,
                                      RSDstrings=None):
        """See parent class.
        """
        ext_grids = OrderedDict()
        if RSDstrings is None:
            RSDstrings = [('','')]

        if False:
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

        if False:
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

        if (include_shifted_fields 
            or include_2LPT_shifted_fields 
            or include_3LPT_shifted_fields):

            ## Shifted fields
            psi_type_strings = ['']
            if include_2LPT_shifted_fields:
                psi_type_strings.append('Psi2LPT_')
            if include_3LPT_shifted_fields:
                psi_type_strings.append('Psi3LPT_')

            for psi_type_str in psi_type_strings:
        

                for target_RSDstring, model_RSDstring in RSDstrings:

                    # 1 shifted by deltalin_Zeldovich displacement (using nbkit0.3;
                    # same as delta_ZA)
                    ext_grids['1_SHIFTEDBY_%sdeltalin%s' % (
                        psi_type_str, model_RSDstring
                    )] = {
                        'dir':
                        '1_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                        % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                           shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                           model_RSDstring),
                        'file_format': 'nbkit_BigFileGrid',
                        'dataset_name': 'Field',
                        'scale_factor': self.sim_scale_factor,
                        'nbkit_normalize': True,
                        'nbkit_setMean': 0.0
                    }

                    # deltalin shifted by deltalin_Zeldovich displacement (using 
                    # nbkit0.3)
                    ext_grids['deltalin_SHIFTEDBY_%sdeltalin%s' % (
                        psi_type_str, model_RSDstring
                    )] = {
                        'dir':
                        'IC_LinearMesh_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                        % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                           shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                           model_RSDstring),
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
                            psi_type_str, model_RSDstring
                        )] = {
                            'dir':
                            'IC_LinearMesh_growth-mean_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                            % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                               shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                               model_RSDstring),
                            'file_format': 'nbkit_BigFileGrid',
                            'dataset_name': 'Field',
                            'scale_factor': self.sim_scale_factor,
                            'nbkit_normalize': True,
                            'nbkit_setMean': 0.0
                        }

                    # G2[deltalin] shifted by deltalin_Zeldovich displacement (using 
                    # nbkit0.3)
                    ext_grids['deltalin_G2_SHIFTEDBY_%sdeltalin%s' % (
                        psi_type_str, model_RSDstring)] = {
                        'dir':
                        'IC_LinearMesh_tidal_G2_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                        % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                           shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                           model_RSDstring),
                        'file_format': 'nbkit_BigFileGrid',
                        'dataset_name': 'Field',
                        'scale_factor': self.sim_scale_factor,
                        'nbkit_normalize': True,
                        'nbkit_setMean': 0.0
                    }

                    # deltalin^3 shifted by deltalin_Zeldovich displacement (using 
                    # nbkit0.3)
                    ext_grids['deltalin_cube-mean_SHIFTEDBY_%sdeltalin%s' % (
                        psi_type_str, model_RSDstring)] = {
                        'dir':
                        'IC_LinearMesh_cube-mean_intR0.00_0.50_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                        % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                           shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                           model_RSDstring),
                        'file_format': 'nbkit_BigFileGrid',
                        'dataset_name': 'Field',
                        'scale_factor': self.sim_scale_factor,
                        'nbkit_normalize': True,
                        'nbkit_setMean': 0.0
                    }

                    if False:
                        # additional cubic fields

                        # G2*deltalin shifted by deltalin_Zeldovich displacement (using 
                        # nbkit0.3)
                        ext_grids['deltalin_G2_delta_SHIFTEDBY_%sdeltalin%s' % (
                            psi_type_str, model_RSDstring)] = {
                            'dir':
                            'IC_LinearMesh_G2_delta_intR0.00_0.50_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                            % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                               shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                               model_RSDstring),
                            'file_format': 'nbkit_BigFileGrid',
                            'dataset_name': 'Field',
                            'scale_factor': self.sim_scale_factor,
                            'nbkit_normalize': True,
                            'nbkit_setMean': 0.0
                        }

                        # G3 shifted by deltalin_Zeldovich displacement (using 
                        # nbkit0.3)
                        ext_grids['deltalin_G3_SHIFTEDBY_%sdeltalin%s' % (
                            psi_type_str, model_RSDstring)] = {
                            'dir':
                            'IC_LinearMesh_tidal_G3_intR0.00_0.50_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                            % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                               shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                               model_RSDstring),
                            'file_format': 'nbkit_BigFileGrid',
                            'dataset_name': 'Field',
                            'scale_factor': self.sim_scale_factor,
                            'nbkit_normalize': True,
                            'nbkit_setMean': 0.0
                        }

                        # Gamma3 shifted by deltalin_Zeldovich displacement (using 
                        # nbkit0.3)
                        ext_grids['deltalin_Gamma3_SHIFTEDBY_%sdeltalin%s' % (
                            psi_type_str, model_RSDstring)] = {
                            'dir':
                            'IC_LinearMesh_Gamma3_intR0.00_0.50_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                            % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                               shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                               model_RSDstring),
                            'file_format': 'nbkit_BigFileGrid',
                            'dataset_name': 'Field',
                            'scale_factor': self.sim_scale_factor,
                            'nbkit_normalize': True,
                            'nbkit_setMean': 0.0
                        }


        if include_div_shifted_PsiDot1:

            psi_type_str = ''
            for target_RSDstring, model_RSDstring in RSDstrings:

                # div of PsiDot1 shifted by deltalin_Zeldovich displacement
                ext_grids['div_PsiDot1_SHIFTEDBY_%sdeltalin%s' % (
                    psi_type_str, model_RSDstring
                )] = {
                    'dir':
                    'div_IC_LinearMesh_PsiDot1_0_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                    % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                       shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                       model_RSDstring),
                    'file_format': 'nbkit_BigFileGrid',
                    'dataset_name': 'Field',
                    'scale_factor': self.sim_scale_factor,
                    'nbkit_normalize': False, # not sure
                    'nbkit_setMean': 0.0
                }

        if include_div_shifted_PsiDot2:

            psi_type_str = ''
            for target_RSDstring, model_RSDstring in RSDstrings:

                # div of PsiDot2 shifted by deltalin_Zeldovich displacement
                ext_grids['div_PsiDot2_SHIFTEDBY_%sdeltalin%s' % (
                    psi_type_str, model_RSDstring
                )] = {
                    'dir':
                    'div_IC_LinearMesh_PsiDot2_0_intR0.00_extR0.00_SHIFTEDBY_%sIC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum%s'
                    % (psi_type_str, shifted_fields_RPsi, self.sim_scale_factor,
                       shifted_fields_Np, shifted_fields_Nmesh, Ngrid,
                       model_RSDstring),
                    'file_format': 'nbkit_BigFileGrid',
                    'dataset_name': 'Field',
                    'scale_factor': self.sim_scale_factor,
                    'nbkit_normalize': False, # not sure
                    'nbkit_setMean': 0.0
                }


        return ext_grids


    def get_default_catalogs(self, RSDstrings=None):
        """Default catalogs to load for ms_gadget sims.
        """
        cats = OrderedDict()
        if RSDstrings is None:
            RSDstrings = [('', '')]

        target_RSDstrings = [ tup[0] for tup in RSDstrings ]

        # FOF halos, mass given by number of particles in halo
        halo_dir = 'nbkit_fof_%.4f/ll_0.200_nmin25' % (
            self.sim_scale_factor)

        for RSDstring in target_RSDstrings:
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
                # Halo momentum divergence theta_pi = div[(1+delta)v]/(faH).
                # narrow mass cuts: 10.8..11.8..12.8..13.8..15.1
                # Halo catalog has v/(aH), which is velocity in Mpc/h. We divide by f.
                # Expect \theta_\pi = div pi/(faH) ~ div v/(faH) = -\delta on large scales (in absence of velocity bias).
                # Then, \theta_pi = div center_velocity/f = -\delta.
                # So expect P_{theta_\pi}/Plin = 1. Looks ok in ms_gadget.
                cats['thetapi_h%s' % RSDstring] = {
                    'in_fname':
                    "%s/fof_nbkfmt.hdf5_BOUNDS_log10M_%s.hdf5%s" %
                    (halo_dir, self.halo_mass_string, RSDfilestr),
                    'weight_ptcles_by':
                    None,
                    'paint_mode':
                    'momentum_divergence',
                    'velocity_column':
                    'Velocity'
                }

            if True:
                # Halo velocity divergence theta_v = div v/(faH).
                # narrow mass cuts: 10.8..11.8..12.8..13.8..15.1
                # Halo catalog has v/(aH), which is velocity in Mpc/h. We divide by f in paint_utils.py.
                # Expect \theta_v = div v/(faH) = -\delta on large scales (in absence of velocity bias).
                # Then, \theta_v = div center_velocity/f = -\delta.
                # So expect P_theta/Plin = 1. Looks ok in ms_gadget sims.
                # TODO: should just specify Painter dict here and don't copy in combine_fields...
                cats['thetav_h%s' % RSDstring] = {
                    'in_fname':
                    "%s/fof_nbkfmt.hdf5_BOUNDS_log10M_%s.hdf5%s" %
                    (halo_dir, self.halo_mass_string, RSDfilestr),
                    'weight_ptcles_by':
                    None,
                    'paint_mode':
                    'velocity_divergence',
                    'velocity_column':
                    'Velocity',
                    'fill_empty_cells':
                    'RandNeighbReadout',  # RandNeighb or RandNeighbReadout
                    'randseed_for_fill_empty_cells':
                    1000 + self.sim_seed,
                    'raise_exception_if_too_many_empty_cells':
                    False,
                    'save_to_disk': True,
                    'out_fname':
                    "%s/fof_nbkfmt.hdf5_BOUNDS_log10M_%s.hdf5%s_thetav%s" %
                    (halo_dir, self.halo_mass_string, RSDfilestr, RSDstring),
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


        # FOF halos, virial mass (used for HOD)
        hod_dir = 'nbkit_fof_%.4f/ll_0.200_nmin25_mvir' % (
            self.sim_scale_factor)

        for RSDstring in target_RSDstrings:
            
            ## nonuniform catalogs without ptcle masses
            if True:
                # HOD galaxies
                hod_model_name = self.hod_model_name
                cats['delta_g%s' % RSDstring] = {
                    'in_fname':
                    "%s/HOD_%s%s.hdf5" %
                    (hod_dir, hod_model_name, RSDstring),
                    'weight_ptcles_by':
                    None
                }

                # HOD central galaxies
                hod_model_name = self.hod_model_name + '_centrals'
                cats['delta_gc%s' % RSDstring] = {
                    'in_fname':
                    "%s/HOD_%s%s.hdf5" %
                    (hod_dir, hod_model_name, RSDstring),
                    'weight_ptcles_by':
                    None
                }

                # HOD satellite galaxies
                hod_model_name = self.hod_model_name + '_sats'
                cats['delta_gs%s' % RSDstring] = {
                    'in_fname':
                    "%s/HOD_%s%s.hdf5" %
                    (hod_dir, hod_model_name, RSDstring),
                    'weight_ptcles_by':
                    None
                }

                # HOD parent halos of centrals
                hod_model_name = self.hod_model_name + '_parent_halos'
                cats['delta_gp%s' % RSDstring] = {
                    'in_fname':
                    "%s/HOD_%s%s.hdf5" %
                    (hod_dir, hod_model_name, RSDstring),
                    'weight_ptcles_by':
                    None
                }


        for RSDstring in target_RSDstrings:

            if True:                  
                # PT Challenge galaxies from rockstar halos. 

                # Rockstar gives core positions and velocities.
                # Units: 1/(aH) = 1./(a * H0*np.sqrt(Om_m/a**3+Om_L)) * (H0/100.) in Mpc/h / (km/s)
                # = 1/(100*a*sqrt(Om_m/a**3+Om_L)).
                # For ms_gadget, get 1/(aH) = 0.01145196 Mpc/h/(km/s) = 0.0183231*0.6250 Mpc/h/(km/s).
                # Note that our MP-Gadget files have RSDFactor=1/(a^2H)=0.0183231 for a=0.6250 b/c they use a^2\dot x for Velocity.
                # Rockstar 'Velocity' column is v=a\dot x in km/s ("Velocities in 
                # km / s (physical, peculiar)")
                #
                # Minimum mass in L=1500, Np=1536 sims is Mmin_1500 = 15.9e10 = 10**11.2

                from perr_private.model_target_pair import Target
                from sim_galaxy_catalog_creator import PTChallengeGalaxiesFromRockstarHalos

                a = self.sim_scale_factor
                RSDFactor_rockstar = 1./(100.*a * np.sqrt(
                        self.cosmo_params['Om_m']/a**3 
                        + self.cosmo_params['Om_K']/a**2
                        + self.cosmo_params['Om_L']))

                # for a=0.625 get RSDFactor_rockstar=0.01145196
                assert np.isclose(RSDFactor_rockstar, 0.01145196)

                if RSDstring == '':
                    apply_RSD_to_position = False
                    RSD_los = None
                elif RSDstring == '_RSD001':
                    apply_RSD_to_position = True
                    RSD_los = [0,0,1]

                cats['delta_gPTC%s' % RSDstring] = Target(
                    name='delta_gPTC%s' % RSDstring,
                    in_fname='snap_%.4f.gadget3/rockstar_out_0.list.bigfile' % (
                        self.sim_scale_factor),
                    position_column='Position',
                    velocity_column='Velocity', 
                    apply_RSD_to_position=apply_RSD_to_position,
                    RSD_los=RSD_los,
                    RSDFactor=RSDFactor_rockstar, # to convert velocity to RSD displacement in Mpc/h
                    cuts=[PTChallengeGalaxiesFromRockstarHalos(
                            log10M_column='log10Mvir', log10Mmin=12.97, sigma_log10M=0.35, RSD=False)
                         ]
                    )

                cats['delta_gPTC_11.5%s' % RSDstring] = Target(
                    name='delta_gPTC%s' % RSDstring,
                    in_fname='snap_%.4f.gadget3/rockstar_out_0.list.bigfile' % (
                        self.sim_scale_factor),
                    position_column='Position',
                    velocity_column='Velocity', 
                    apply_RSD_to_position=apply_RSD_to_position,
                    RSD_los=RSD_los,
                    RSDFactor=RSDFactor_rockstar, # to convert velocity to RSD displacement in Mpc/h
                    cuts=[PTChallengeGalaxiesFromRockstarHalos(
                            log10M_column='log10Mvir', log10Mmin=11.5, sigma_log10M=0.35, RSD=False)
                         ]
                    )

                if False:
                    # god sample
                    assert RSD_los in [None, [0,0,1]]
                    cats['delta_gPTC_GodPsiDot1%s' % RSDstring] = Target(
                        name='delta_gPTC_GodPsiDot1%s' % RSDstring,
                        in_fname='snap_%.4f.gadget3/rockstar_out_0.list.bigfile_RESID_PsiDot1_D2.bf' % (
                            self.sim_scale_factor),
                        position_column='Position',
                        velocity_column='Velocity', 
                        apply_RSD_to_position=apply_RSD_to_position,
                        RSD_los=RSD_los,
                        RSDFactor=RSDFactor_rockstar, # to convert velocity to RSD displacement in Mpc/h
                        cuts=[('residual_D2', 'min', -2.0),  # script already applied galaxy selection, only need to cut on residual
                              ('residual_D2', 'max', 2.0)
                             ]
                        )

        return cats
