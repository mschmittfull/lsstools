#!/usr/bin/env python
# coding: utf-8

# # Get PT challenge centrals catalog and save to disk#
# Use different defnitions of centrals:
  
# - (1) God sample: Keep all galaxies whose residual deisplacement w.r.t. to zeldovich displacement is > 2Mpc/h
# - (2) Neighbor sample: Keep all galaxies that don't have a more massive galaxy within their virial radius (PT challenge paper recipe)

# (2) is not implemented yet


from __future__ import print_function, division


#from nbodykit import style

#import matplotlib.pyplot as plt
import numpy as np
import os

from lsstools.cosmo_model import CosmoModel
from lsstools.gen_cosmo_fcns import calc_f_log_growth_rate, generate_calc_Da
from lsstools.model_spec import get_trf_spec_from_list
from lsstools.paint_utils import mass_weighted_paint_cat_to_delta
from lsstools.results_db import retrieve_pickles
from lsstools.results_db.io import Pickler
from perr.path_utils import get_in_path
from lsstools.nbkit03_utils import get_csum, get_csqsum, apply_smoothing, catalog_persist, get_cstats_string, linear_rescale_fac, get_crms,convert_nbk_cat_to_np_array
from perr_private.model_target_pair import ModelTargetPair, Model, Target
from lsstools.sim_galaxy_catalog_creator import SimGalaxyCatalogCreator, PTChallengeGalaxiesFromRockstarHalos
from perr_private.read_utils import readout_mesh_at_cat_pos

#from nbodykit.lab import ArrayCatalog, BigFileMesh

#plt.style.use(style.notebook)
#colors = [d['color'] for d in style.notebook['axes.prop_cycle']]


# ## Global params ##

# In[2]:


# path
sim_seed = 400
basedir = '/Users/mschmittfull/scratch_data/lss/ms_gadget/run4/00000%d-01536-500.0-wig/' % sim_seed
sim_scale_factor = 0.625

# cosmology of ms_gadget sims (to compute D_lin(z))
# omega_m = 0.307494
# omega_bh2 = 0.022300
# omega_ch2 = 0.118800
# h = math.sqrt((omega_bh2 + omega_ch2) / omega_m) = 0.6774
cosmo_params = dict(Om_m=0.307494,
                   Om_L=1.0 - 0.307494,
                   Om_K=0.0,
                   Om_r=0.0,
                   h0=0.6774)

f_log_growth = np.sqrt(0.61826)

# Smoothings in lagrangian space, in Mpc/h
Rsmooth_density_to_shift = 0.0   # 0.0 before 30/3/2020
Rsmooth_displacement_source = 0.23   # 0.23 before 30/3/2020

Ngrid = 512

# Options for shifted field
ShiftedFieldsNp = 1536
ShiftedFieldsNmesh = 1536


# avg or sum. Should use avg to get correct velocity model.
PsiDot_weighted_CIC_mode = 'avg'


# Below, 'D' stands for RSD displacement in Mpc/h: D=v/(aH)=f*PsiDot.
tex_names = {}

## Targets
# DM subsample
DM_D0 = Target(
    name='DM_D0',
    in_fname=os.path.join(basedir, 'snap_%.4f_sub_sr0.0015_ssseed40400.bigfile' % sim_scale_factor),
    position_column='Position',
    val_column='Velocity',
    val_component=0,
    rescale_factor='RSDFactor'
)




# PT Challenge galaxies from rockstar halos. Rockstar gives core positions and velocities.
# Units: 1/(aH) = 1./(a * H0*np.sqrt(Om_m/a**3+Om_L)) * (H0/100.) in Mpc/h / (km/s).
# For ms_gadget, get 1/(aH) = 0.01145196 Mpc/h/(km/s) = 0.0183231*0.6250 Mpc/h/(km/s).
# Note that MP-Gadget files have RSDFactor=1/(a^2H)=0.0183231 for a=0.6250 b/c they use a^2\dot x for Velocity.
assert sim_scale_factor == 0.625
gal_ptchall_D0 = Target(
    name='gal_ptchall_D0',
    in_fname=os.path.join(basedir, 'snap_%.4f.gadget3/rockstar_out_0.list.parents.bigfile' % sim_scale_factor),
    position_column='Position',
    val_column='Velocity', # This is rockstar velocity, which is v=a\dot x in km/s ("Velocities in km / s (physical, peculiar)")
    val_component=0,
    rescale_factor=0.01145196, # RSD displacement in Mpc/h is D=v/(aH)=0.01145196*v. 
    RSDFactor=0.01145196, # used internally to get redshift space positions if needed (e.g. for cylinder cuts)
    cuts=[PTChallengeGalaxiesFromRockstarHalos(
            log10M_column='log10Mvir', log10Mmin=12.97, sigma_log10M=0.35, RSD=False)
         ]
    )
gal_ptchall_D2 = Target(
    name='gal_ptchall_D2',
    in_fname=os.path.join(basedir, 'snap_%.4f.gadget3/rockstar_out_0.list.parents.bigfile' % sim_scale_factor),
    position_column='Position',
    val_column='Velocity', # This is rockstar velocity, which is v=a\dot x in km/s ("Velocities in km / s (physical, peculiar)")
    val_component=2,
    rescale_factor=0.01145196, # RSD displacement in Mpc/h is D=v/(aH)=0.01145196*v. 
    RSDFactor=0.01145196, # used internally to get redshift space positions if needed (e.g. for cylinder cuts)
    cuts=[PTChallengeGalaxiesFromRockstarHalos(
            log10M_column='log10Mvir', log10Mmin=12.97, sigma_log10M=0.35, RSD=False)
         ]
    )

# subbox of PT challenge galaxies, x component
assert sim_scale_factor == 0.625
gal_ptchall_subbox_D0 = Target(
    name='gal_ptchall_D0',
    in_fname=os.path.join(basedir, 'snap_%.4f.gadget3/rockstar_out_0.list.parents.bigfile' % sim_scale_factor),
    position_column='Position',
    val_column='Velocity', # This is rockstar velocity, which is v=a\dot x in km/s ("Velocities in km / s (physical, peculiar)")
    val_component=0,
    rescale_factor=0.01145196, # RSD displacement in Mpc/h is D=v/(aH)=0.01145196*v. 
    RSDFactor=0.01145196, # used internally to get redshift space positions if needed
    cuts=[PTChallengeGalaxiesFromRockstarHalos(
            log10M_column='log10Mvir', log10Mmin=12.97, sigma_log10M=0.35, RSD=False),
          ('Position', 'max', [100.,100.,100.])
         ]
    )

# subbox of PT challenge galaxies, x component, apply RSD to position (TEST)
assert sim_scale_factor == 0.625
gal_ptchall_subbox_D0_RSDtest = Target(
    name='gal_ptchall_D0_RSDtest',
    in_fname=os.path.join(basedir, 'snap_%.4f.gadget3/rockstar_out_0.list.parents.bigfile' % sim_scale_factor),
    position_column='Position',
    velocity_column='Velocity', 
    apply_RSD_to_position=True,
    RSD_los=[1,0,0],
    RSDFactor=0.01145196,
    val_column='Velocity', # This is rockstar velocity, which is v=a\dot x in km/s ("Velocities in km / s (physical, peculiar)")
    val_component=0,
    rescale_factor=0.01145196, # RSD displacement in Mpc/h is D=v/(aH)=0.01145196*v. 
    cuts=[PTChallengeGalaxiesFromRockstarHalos(
            log10M_column='log10Mvir', log10Mmin=12.97, sigma_log10M=0.35, RSD=False),
          ('Position', 'max', [100.,100.,20.])
         ]
    )

# subbox of PT challenge galaxies, y component
assert sim_scale_factor == 0.625
gal_ptchall_subbox_D1 = Target(
    name='gal_ptchall_D1',
    in_fname=os.path.join(basedir, 'snap_%.4f.gadget3/rockstar_out_0.list.parents.bigfile' % sim_scale_factor),
    position_column='Position',
    val_column='Velocity', # This is rockstar velocity, which is v=a\dot x in km/s ("Velocities in km / s (physical, peculiar)")
    val_component=1,
    rescale_factor=0.01145196, # RSD displacement in Mpc/h is D=v/(aH)=0.01145196*v. 
    cuts=[PTChallengeGalaxiesFromRockstarHalos(
            log10M_column='log10Mvir', log10Mmin=12.97, sigma_log10M=0.35, RSD=False),
          ('Position', 'max', [500.,500.,20.])
         ]
    )


# In[3]:


### Models

# 1st order PsiDot shifted by linear Psi
s = ('IC_LinearMesh_PsiDot1_0_intR0.00_extR%.2f_SHIFTEDBY_'
     'IC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICavg' % (
         Rsmooth_density_to_shift,
         Rsmooth_displacement_source,
         sim_scale_factor,
         ShiftedFieldsNp, 
         ShiftedFieldsNmesh,
         Ngrid))
PsiDot1_D0 = Model(
    name='PsiDot1_D0',
    in_fname=os.path.join(basedir, s),
    rescale_factor=f_log_growth,
    read_mode='velocity',
    filters=None,
    readout_window='cic')

# y direction
s = ('IC_LinearMesh_PsiDot1_1_intR0.00_extR%.2f_SHIFTEDBY_'
     'IC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICavg' % (
         Rsmooth_density_to_shift,
         Rsmooth_displacement_source,
         sim_scale_factor,
         ShiftedFieldsNp, 
         ShiftedFieldsNmesh,
         Ngrid))
PsiDot1_D1 = Model(
    name='PsiDot1_D1',
    in_fname=os.path.join(basedir, s),
    rescale_factor=f_log_growth,
    read_mode='velocity',
    filters=None,
    readout_window='cic')

# z direction
s = ('IC_LinearMesh_PsiDot1_2_intR0.00_extR%.2f_SHIFTEDBY_'
     'IC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICavg' % (
         Rsmooth_density_to_shift,
         Rsmooth_displacement_source,
         sim_scale_factor,
         ShiftedFieldsNp, 
         ShiftedFieldsNmesh,
         Ngrid))
PsiDot1_D2 = Model(
    name='PsiDot1_D2',
    in_fname=os.path.join(basedir, s),
    rescale_factor=f_log_growth,
    read_mode='velocity',
    filters=None,
    readout_window='cic')


# 2nd order PsiDot shifted by 2nd order Psi
s = ('IC_LinearMesh_PsiDot2_0_intR0.00_extR%.2f_SHIFTEDBY_'
     'Psi2LPT_IC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICavg' % (
         Rsmooth_density_to_shift,
         Rsmooth_displacement_source,
         sim_scale_factor,
         ShiftedFieldsNp, 
         ShiftedFieldsNmesh,
         Ngrid))
PsiDot2_D0 = Model(
    name='PsiDot2_D0',
    in_fname=os.path.join(basedir, s),
    rescale_factor=f_log_growth,
    read_mode='velocity',
    filters=None,
    readout_window='cic')

# y direction
s = ('IC_LinearMesh_PsiDot2_1_intR0.00_extR%.2f_SHIFTEDBY_'
     'Psi2LPT_IC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICavg' % (
         Rsmooth_density_to_shift,
         Rsmooth_displacement_source,
         sim_scale_factor,
         ShiftedFieldsNp, 
         ShiftedFieldsNmesh,
         Ngrid))
PsiDot2_D1 = Model(
    name='PsiDot2_D1',
    in_fname=os.path.join(basedir, s),
    rescale_factor=f_log_growth,
    read_mode='velocity',
    filters=None,
    readout_window='cic')


# 2nd order PsiDot shifted by 1st order Psi
s = ('IC_LinearMesh_PsiDot2_0_intR0.00_extR%.2f_SHIFTEDBY_'
     'IC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICavg' % (
         Rsmooth_density_to_shift,
         Rsmooth_displacement_source,
         sim_scale_factor,
         ShiftedFieldsNp, 
         ShiftedFieldsNmesh,
         Ngrid))
PsiDot2_shiftedbyPsi1_D0 = Model(
    name='PsiDot2_shiftedbyPsi1_D0',
    in_fname=os.path.join(basedir, s),
    rescale_factor=f_log_growth,
    read_mode='velocity',
    filters=None,
    readout_window='cic')

# y direction
s = ('IC_LinearMesh_PsiDot2_1_intR0.00_extR%.2f_SHIFTEDBY_'
     'IC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICavg' % (
         Rsmooth_density_to_shift,
         Rsmooth_displacement_source,
         sim_scale_factor,
         ShiftedFieldsNp, 
         ShiftedFieldsNmesh,
         Ngrid))
PsiDot2_shiftedbyPsi1_D1 = Model(
    name='PsiDot2_shiftedbyPsi1_D1',
    in_fname=os.path.join(basedir, s),
    rescale_factor=f_log_growth,
    read_mode='velocity',
    filters=None,
    readout_window='cic')


# k/k^2 deltaZ, x direction
def k0ovksq_filter_fcn(k, v, d=0):
    k2 = sum(ki**2 for ki in k)
    return np.where(k2 == 0.0, 0*v, 1j*k[d] * v / (k2))

s = '1_intR0.00_extR%.2f_SHIFTEDBY_IC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum' % (
        Rsmooth_density_to_shift, Rsmooth_displacement_source, sim_scale_factor,
        ShiftedFieldsNp, ShiftedFieldsNmesh, Ngrid)
deltaZ_D0 = Model(
    name='deltaZ_D0',
    in_fname=os.path.join(basedir, s),
    rescale_factor=f_log_growth,
    #read_mode='density',
    read_mode='delta from 1+delta',
    filters=[k0ovksq_filter_fcn],
    readout_window='cic')

# k/k^2 deltaZ, y direction
def k1ovksq_filter_fcn(k, v, d=1):
    k2 = sum(ki**2 for ki in k)
    return np.where(k2 == 0.0, 0*v, 1j*k[d] * v / (k2))

s = '1_intR0.00_extR%.2f_SHIFTEDBY_IC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum' % (
        Rsmooth_density_to_shift, Rsmooth_displacement_source, sim_scale_factor,
        ShiftedFieldsNp, ShiftedFieldsNmesh, Ngrid)
deltaZ_D1 = Model(
    name='deltaZ_D1',
    in_fname=os.path.join(basedir, s),
    rescale_factor=f_log_growth,
    read_mode='delta from 1+delta',
    filters=[k1ovksq_filter_fcn],
    readout_window='cic')


# k/k^2 delta2LPT, x direction
def k0ovksq_filter_fcn(k, v, d=0):
    k2 = sum(ki**2 for ki in k)
    return np.where(k2 == 0.0, 0*v, 1j*k[d] * v / (k2))

s = '1_intR0.00_extR%.2f_SHIFTEDBY_Psi2LPT_IC_LinearMeshR%.2f_a%.4f_Np%d_Nm%d_Ng%d_CICsum' % (
        Rsmooth_density_to_shift, Rsmooth_displacement_source, sim_scale_factor,
        ShiftedFieldsNp, ShiftedFieldsNmesh, Ngrid)
delta2LPT_D0 = Model(
    name='delta2LPT_D0',
    in_fname=os.path.join(basedir, s),
    rescale_factor=f_log_growth,
    #read_mode='density',
    read_mode='delta from 2+delta',
    filters=[k0ovksq_filter_fcn],
    readout_window='cic')


# k/k^2 delta_lin
z_rescalefac = linear_rescale_fac(current_scale_factor=1.0,
                                  desired_scale_factor=sim_scale_factor,
                                  cosmo_params=cosmo_params)
deltalin_D0 = Model(
    name='deltalin_D0',
    in_fname=os.path.join(basedir, 'IC_LinearMesh_z0_Ng%d' % Ngrid),
    rescale_factor=f_log_growth*z_rescalefac,
    read_mode='density',
    filters=[k0ovksq_filter_fcn],
    readout_window='cic')
deltalin_D1 = Model(
    name='deltalin_D1',
    in_fname=os.path.join(basedir, 'IC_LinearMesh_z0_Ng%d' % Ngrid),
    rescale_factor=f_log_growth*z_rescalefac,
    read_mode='density',
    filters=[k1ovksq_filter_fcn],
    readout_window='cic')

deltalin = Model(
    name='deltalin',
    in_fname=os.path.join(basedir, 'IC_LinearMesh_z0_Ng%d' % Ngrid),
    rescale_factor=z_rescalefac,
    read_mode='delta from 1+delta',
    filters=None,
    readout_window='cic')


# ## Compute model displacements at target positions ##

# In[4]:


# Use LOS=[0,0,1] along z direction
LOS = np.array([0,0,1])

# get the catalog and model
mtp = ModelTargetPair(model=PsiDot1_D2, target=gal_ptchall_D2)
#mtp = ModelTargetPair(model=delta2LPT_D0, target=gal_ptchall_D0)
#mtp = ModelTargetPair(model=delta2LPT_D0, target=gal_ptchall_subbox_D0) # careful with subbox: this makes box nonperiodic!

BoxSize = np.array([500.,500.,500.])
#BoxSize = np.array([100.,100.,100.])

# residual displacement at target position
Dresidual = mtp.get_target_val_at_target_pos().compute() - mtp.readout_model_at_target_pos()
print('TARGET-MODEL: ', get_cstats_string(Dresidual))

# catalog in real space
cat = mtp.target.get_catalog(keep_all_columns=True)
print(cat.attrs)
cat['residual_D2'] = Dresidual

# add redshift space positions, assuming LOS is in z direction
cat['RSDPosition'] = cat['Position'] + cat['Velocity']*mtp.target.RSDFactor * LOS

print(cat.columns)  
cat = catalog_persist(cat, columns=['ID','PID','Position','RSDPosition','Velocity',
                                    'log10Mvir','residual_D2'])
cat.attrs['RSDFactor'] = mtp.target.RSDFactor
print(cat.columns)


# In[5]:


cat.attrs


# In[ ]:





# ## store as bigfile to calculate Perror externally ##

# In[10]:


mtp.target.in_fname


# In[11]:


cat.columns


# In[12]:


out_fname = '%s_RESID_%s.bf' % (mtp.target.in_fname, mtp.model.name)
cat.save(out_fname, cat.columns)
print('Wrote %s' % out_fname)


if False:
    # ## store as ascii for ML applications ##

    # In[15]:


    cat.columns


    # In[16]:


    cat['RSDPosition'].compute()


    # In[17]:


    cat['Position'].compute()


    # In[19]:


    out_dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('residual_D2', 'f8'), ('bad', int)]
    out_arr = np.empty((cat.csize,), dtype=out_dtype)
    out_arr['x'] = cat['RSDPosition'][:,0]
    out_arr['y'] = cat['RSDPosition'][:,1]
    out_arr['z'] = cat['RSDPosition'][:,2]
    out_arr['residual_D2'] = cat['residual_D2'][:]
    out_arr['bad'] = np.where(np.abs(cat['residual_D2'][:])>2, np.ones(cat.csize, dtype=int), np.zeros(cat.csize, dtype=int))
    out_fname = 'data/galaxies_mtp_%s_%s_LOS%g%g%g.txt' % (mtp.model.name, mtp.target.name, LOS[0], LOS[1], LOS[2])
    header = ','.join(out_arr.dtype.names)
    np.savetxt(out_fname, out_arr, fmt=('%.8g','%.8g','%.8g','%.6g','%d'), header=header)
    print('Wrote %s' % out_fname)


    # In[20]:


    len(np.where(out_arr['bad']==0)[0])


    # In[21]:


    out_arr.dtype.names






