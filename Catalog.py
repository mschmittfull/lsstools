from __future__ import print_function,division

import cPickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from collections import OrderedDict
from matplotlib import rcParams
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
#from scipy import interpolate
#from scipy.interpolate import RectBivariateSpline
#import time
#import re
import h5py


# MS packages
#import constants
#import shift_catalog


class Catalog:
    """
    A class for storing particle catalogs.
    """
    def __init__(self):
        # Number of grid points used for sim, e.g. 2048. Must be integer.
        # Actually this is number of particles used in the original simulation!
        self.sim_Ngrid = None
        # simulation box size in Mpc/h
        self.sim_boxsize = None
        # Nbodykit CatalogSource object with particle positions, IDs, velocities, etc
        self.P = None
        # dataset attributes read from file or to write to file
        self.dataset_attrs = None
        # number of objects in catalog
        self.N_objects = None
        # mass of a DM particle in 1e10 Msun/h
        self.DMpmass_in_1e10Msunh = None
        

    def read_from_hdf5(self, fname, dataset_name='Subsample',
                       columns=None, verbose=False, sim_Nptcles_for_PID=None):
        """
        Read columns from a DM or halo catalog stored in hdf5 file.

        Parameters
        ----------
        fname : string
            Filename of hdf5 catalog
        dataset_name : string
            Dataset to read from hdf5 file, e.g. 'Subsample' or 'FOFGroups'
        columns : list of strings
            Columns to read from dataset
        """

        raise Exception("TODOOO: upgrade to nbk03 below. Use CatalogSource class.")
        #https://nbodykit.readthedocs.io/en/latest/catalogs/overview.html?highlight=catalogsource
        
        # read from hdf5 file
        print("Read", fname)
        if not os.path.exists(fname):
            raise Exception("File not found: %s" % str(fname))
        f = h5py.File(fname, "r")
        Pin = f[dataset_name]
        self.dataset_attrs = dict(Pin.attrs)
        file_columns = Pin.dtype.names

        print("File columns:", file_columns)
        print("Attrs:", self.dataset_attrs)

        if columns is None:
            # read all columns
            columns=file_columns

        # check Nmesh and save
        if 'Nmesh' in Pin.attrs:
            if not type(Pin.attrs['Nmesh']) in [type(int), int, np.int32, np.int64]:
                print('Type of Nmesh:', type(Pin.attrs['Nmesh']))
                raise Exception("Nmesh in hdf5 file must be integer")    
            self.sim_Ngrid = Pin.attrs['Nmesh']

        # check boxsize and save
        self.sim_boxsize = Pin.attrs['BoxSize'][0]
        if ((Pin.attrs['BoxSize'][1]!=self.sim_boxsize) or 
            (Pin.attrs['BoxSize'][2]!=self.sim_boxsize)):
            raise Exception("BoxSize in hdf5 file is not cubic")
            
        # print some info
        if verbose:
            print('sim_Ngrid=', self.sim_Ngrid)
            print('sim_boxsize=', self.sim_boxsize)
            print('Pos shape:', Pin['Position'][:].shape)
            print('Pos examples:\n', Pin['Position'][10000:10006])

        # Position should range from 0..1 if units are correct (though some overshoot
        # is ok b/c we do not impose box wrapping here)
        # Print warnings if units seem fishy.
        minpos = np.min(Pin['Position'])
        maxpos = np.max(Pin['Position'])
        print('Pos min, max:', minpos, maxpos)
        if minpos < -0.1:
            print("Warning: Catalog has min(pos)=%s but should be >=-0.1" % str(minpos))
        if maxpos > 1.5:
            print("WARNING: Catalog has max(pos)=%s but should be <=1.5" % str(maxpos))
        if maxpos < 0.5:
            print("WARNING: Catalog has max(pos)=%s but should be >=0.5" % str(maxpos))
       
        # Get dtype of structured numpy array
        dtype = Catalog.get_dtype_of_columns(columns)
        self.P = np.zeros(Pin['Position'].shape[0], dtype=dtype)
            
        # Store particle data in self.P
        for col in columns:
            if col in file_columns:
                self.P[col] = Pin[col].copy()
            elif col == 'InitialPosition':
                # try to get initial positions later from ID
                pass
            else:
                raise Exception("Invalid column %s" % str(col))

        # release some memory
        del Pin
        f.close()

        # If necessary, get initial positions from ptcle IDs and save in self.P['InitialPosition']
        if ('InitialPosition' in columns) and ('InitialPosition' not in file_columns):
            self.get_InitialPosition_from_ID(verbose=verbose, sim_Nptcles_for_PID=sim_Nptcles_for_PID)

        self.update_N_objects()
        

    def fill_with_random_positions(self, Nptcles=None, mincoord=0.0, maxcoord=1.0,
                                   boxsize=None, seed=2):
        # fill some attributes
        self.boxsize = boxsize
        self.dataset_attrs = {'boxsize': float(boxsize),
                              'BoxSize': np.array([float(boxsize),float(boxsize),float(boxsize)])}

        # fill self.P
        assert self.P is None
        self.P = np.empty(Nptcles, dtype=Catalog.get_dtype_of_columns(['Position']))

        # draw random pos between mincoord and maxcoord
        print("Generate random catalog with %d ptcles" % Nptcles)
        rr = np.random.RandomState(seed)
        for idir in range(3):
            self.P['Position'][:,idir] = rr.uniform(size=(Nptcles,), low=mincoord, high=maxcoord) 
        print("Done")
        print("Example pos:\n", self.P['Position'][:5])

       
        self.update_N_objects()
        
        
    def box_wrap(self, mincoord=None, maxcoord=None, period=None,
                 check_before_wrap=True):
        # 23 jan 2018: box wrap to [mincoord,maxcoord[
        if check_before_wrap:
            assert maxcoord >= 0.95 * np.max(self.P['Position'])
            assert maxcoord <= 1.05 * np.max(self.P['Position'])
            assert np.all(self.P['Position'] >= -0.05*maxcoord)
            assert np.all(self.P['Position'] <= 1.05*maxcoord)
            assert period >= 0.95 * np.max(self.P['Position'])
            assert period <= 1.05 * np.max(self.P['Position'])

        print("min, max pos: %.15g, %.15g" %(np.min(self.P['Position']), np.max(self.P['Position'])))
        if (not np.all(self.P['Position']>=mincoord)) or (not np.all(self.P['Position']<maxcoord)):
            # box wrap
            print("box wrap")
            self.P['Position'][np.where(self.P['Position']<0.0)] += period
            self.P['Position'][np.where(self.P['Position']>=maxcoord)] -= period
            assert np.all(self.P['Position']>=0.)
            self.P['Position'][np.where(self.P['Position']<mincoord)] = mincoord
            print("min, max pos after box wrap: %.15g, %.15g" % (np.min(self.P['Position']), np.max(self.P['Position'])))
            assert np.all(self.P['Position']>=mincoord)
            assert np.all(self.P['Position']<maxcoord)

                          
        
    def update_N_objects(self):
        self.N_objects = self.P['Position'].shape[0]        


    def calc_DMpmass(self, Om_m=None, sim_boxsize=None, sim_Ngrid=None):
        """
        Calculate DM particle mass.
        sim_boxsize : float, in Mpc/h, e.g. 1380.
        sim_Ngrid : int or float, e.g. 2048
        """
        if Om_m is None:
            raise Exception("Need Om_m to compute particle mass")
        if sim_boxsize is None:
            sim_boxsize = self.sim_boxsize
        if sim_Ngrid is None:
            sim_Ngrid = self.sim_Ngrid
        # DM particle mass in 1e10 Msun/h, see code in
        # https://github.com/rainwoodman/fastpm/blob/87a5ffeccb1e923d665501b6820446edd74bebc7/src/fastpm.c
        # ("mass of a particle is...")
        # expect 2.61022 1e10 Msun/h for Omega_m=0.307494, L=1380, Ngrid=2048
        rho_crit = 27.7455
        DMpmass_in_1e10Msunh = Om_m*rho_crit*(float(sim_boxsize)/float(sim_Ngrid))**3
        self.DMpmass_in_1e10Msunh = DMpmass_in_1e10Msunh
        # print info
        print("DM particle has mass %g 1e10 Msun/h (log10M=%g)" % (
            self.DMpmass_in_1e10Msunh, 10.+np.log10(self.DMpmass_in_1e10Msunh)))
        if self.has_column('Length'):
            print("Length min, max, rms:", np.min(self.P['Length']), np.max(self.P['Length']),
                  float(np.mean(self.P['Length']**2))**0.5)
            print("min Length > 0:", np.min(self.P['Length'][np.where(self.P['Length']>0)]))
            # print("log10M mass of lightest and heaviest halo:",
            #      10.+np.log10(self.DMpmass_in_1e10Msunh*np.min(self.P['Length'])),
            #      10.+np.log10(self.DMpmass_in_1e10Msunh*np.max(self.P['Length'])))
        return DMpmass_in_1e10Msunh


    def convert_halo_lengths_to_log10M(self):
        """
        FastPM saves number of DM particles in each halo in Length column.
        Convert this to halo mass, log10(M[Msun/h]), here.
        """
        print("Convert halo lengths to log10M masses...")
        if self.DMpmass_in_1e10Msunh is None:
            raise Exception("Must call calc_DMpmass first")
        log10M = 10.0 + np.log10(self.P['Length'] * self.DMpmass_in_1e10Msunh)
        self.drop_column('Length')
        self.append_column('log10M', log10M)
        print("Converted halo lengths to log10M masses")

        
    def apply_selection(self, selector):
        """
        Apply selection to the catalog, keeping only certain objects.
        For example only keep halos above certain mass.

        Parameters
        ----------
        selector : HaloSelector instance
        """
        if selector is None:
            return
        else:
            self.P = selector.apply_bounds_to_named_array(self.P)
            self.update_N_objects()
   
            
    def save_to_hdf5(self, fname, dataset_name='Subsample',
                     columns=None, verbose=False):
        """
        Save to hdf5 file similarly to nbodykit.core.algorithms.Subsample.py.
        Allow additional columns to store for example psi_x, psi_y, psi_z.

        First remove file if it exists.
        """
        if columns is None:
            # wite all columns
            columns = self.P.dtype.names
        dtype = Catalog.get_dtype_of_columns(columns)

        # print info
        print("Try writing columns:", columns)
        print("dtype:", dtype)
        size = self.P['Position'].shape[0]
                    
        # the data to write
        if columns == self.P.dtype.names:
            data = self.P
        else:
            # copy only columns we want
            data = np.empty(size, dtype=dtype)
            for col in columns:
                data[col][:] = self.P[col][:]

        # remove file if it exists
        if os.path.exists(fname):
            os.remove(fname)
            
        # actually write attributes and data to hdf5 file
        print("Writing to %s" % fname)
        ff = h5py.File(fname, 'w')
        # create dataset
        dataset = ff.create_dataset(name=dataset_name,
                                    dtype=dtype, shape=(size,))
        # attributes
        for k in self.dataset_attrs.keys():
            if k == 'Ntot':
                dataset.attrs['Ntot'] = size
            else:
                print("attr %s: "%k, self.dataset_attrs[k])
                dataset.attrs[k] = self.dataset_attrs[k]
        print("dataset.attrs:", str({k: dataset.attrs[k] for k in dataset.attrs.keys()}))
        if False:
            # dataset.attrs['Ratio'] = self.dataset_attrs.Ratio
            # dataset.attrs['CommSize'] = self.dataset_attrs.CommSize
            # dataset.attrs['Seed'] = self.dataset_attrs.Seed
            dataset.attrs['Smoothing'] = self.dataset_attrs['Smoothing']
            dataset.attrs['Nmesh'] = self.dataset_attrs['Nmesh']
            # dataset.attrs['Original'] = self.dataset_attrs.Original
            dataset.attrs['BoxSize'] = self.dataset_attrs['BoxSize']
            dataset.attrs['PsiSource'] = 'Position_minus_InitialPosition'

        #with h5py.File(fname, 'r+') as ff:
        #dataset = ff['Subsample']
        #print("data:", data[:10])
        dataset[:] = data[:]
        #print("dataset:", dataset[:10])

        ff.close()
            
        print("Wrote %d particles to %s" % (size,fname))
        #raise Exception("dbg save_to_hdf5")

            

    @staticmethod
    def get_dtype_of_columns(columns):
        dtype_list = []
        for col in columns:
            #if col in Pin.dtype.names:
            #    dtype_list.append( (col, Pin.dtype.fields[col][0]) )
            if col in ['Position', 'InitialPosition', 'Velocity']:
                dtype_list.append( (col, ('f4', 3)) )
            elif col in ['Density', 'Psi_0', 'Psi_1', 'Psi_2', 'Mass', 
                         'chi_0', 'chi_1', 'chi_2',
                         'chi_x_0','chi_x_1','chi_x_2', 
                         'chi_y_0','chi_y_1','chi_y_2',
                         'weighted_chi_x_0','weighted_chi_x_1','weighted_chi_x_2', 
                         'weighted_chi_y_0','weighted_chi_y_1','weighted_chi_y_2',
                         'SimPsiLagr_0','SimPsiLagr_1','SimPsiLagr_2',
                         'PsiResEul_0','PsiResEul_1','PsiResEul_2',
                         'log10M',
                         'Potential', 'Value', 'Weight']:                         
                dtype_list.append( (col, 'f4') )
            elif col in ['Mass[1e10Msun/h]','Mass[1e10Msun/h]^2','Mass[1e10Msun/h]^3']:
                dtype_list.append( (col, 'f8') )
            elif col.startswith('MassWith'):
                dtype_list.append( (col, 'f8') )
            elif col in ['ID','Length','GroupID']:
                dtype_list.append( (col, 'u8') )
            else:
                raise Exception("Unsupported column: %s" % str(col))
        dtype = np.dtype(dtype_list)
        print("Created dtype:", dtype)
        return dtype

    def append_column(self, column_name, column_data):
        """
        column_name : string
            String corresponding to the name
            of the new field.
        data : array
            Array storing the field to add to the base.
        """
        #from numpy.lib import recfunctions
        dtype = Catalog.get_dtype_of_columns( (column_name,) )[0]
        #dtype = Catalog.get_dtype_of_columns( (column_name,) )
        if (self.P is None) or (len(self.P.dtype.names) == 0):
            self.P = np.empty(column_data.shape, dtype=dtype)
            self.P[column_name][...] = column_data[...]
        else:
            if self.has_column(column_name):
                # overwrite
                self.P[column_name][...] = column_data[...]
            else:
                # Append column manually. Note that numpy.lib.recfunctions.append_fields 
                # makes trouble when the new column is itself a structured array.
                # For example add_dtype = [('ShiftedPosition', 'f4', 3)]
                # new_dtype = np.dtype( cat.P.dtype.descr + [('ShiftedPosition', 'f4', 3)] )
                add_dtype = Catalog.get_dtype_of_columns( (column_name,) )
                new_dtype = np.dtype( self.P.dtype.descr + add_dtype.descr )
                Pnew = np.empty(self.P.shape, dtype=new_dtype)
                # copy old columns (can probably speed this up)
                for col in self.P.dtype.names:
                    Pnew[col] = self.P[col]
                # add new column
                Pnew[column_name][...] = column_data[...]
                self.P = Pnew
                
                # print("attempt to append dtypes:", dtype)
                # print("columN_data:", column_data.shape, column_data.dtype)
                # tmp = np.empty((column_data.shape[0],), dtype=dtype)
                # print("tmp shape:", tmp.shape, tmp.dtype)
                # #tmp[column_name][...] = column_data[...]
                # tmp[...] = column_data[...]
                # #self.P = recfunctions.append_fields(self.P, column_name, tmp, dtypes=dtype)
                # #self.P = recfunctions.append_fields(self.P, (column_name,), (column_data,), dtypes=np.dtype(dtype))
                # #self.P = recfunctions.append_fields(self.P, column_name, tmp[column_name][...], dtypes=dtype)
                # self.P = recfunctions.append_fields(self.P, column_name, tmp, dtypes=dtype)
                # #
                # tmp = np.zeros(cat.P.shape, np.dtype( [('ShiftedPosition', 'f4', 3)] ))
                
        print("Appended column %s to catalog" % column_name)
        print("min, max, rms:", np.min(column_data), np.max(column_data), 
              np.sqrt(np.mean(column_data**2)))
        print("New dtype:", self.P.dtype)


    def drop_column(self, column_name):
        """
        column_name : string
            String corresponding to the name
            of the field to be dropped.
        """
        from numpy.lib import recfunctions
        self.P = recfunctions.drop_fields(self.P, (column_name,))
        print("Dropped column %s from catalog" % column_name)       
        
    def has_column(self, column_name, throw_exception_if_False=False):
        return (column_name in self.P.dtype.names)

    def check_column_exists(self, column_name):
        if not self.has_column(column_name):
            raise Exception("Could not find column %s. Allowed columns: %s" % (
                column_name, str(self.P.dtype.names)))

        
    def shift_column_by_psi_grid(self, column, shifted_column=None, psigrids_3tuple=(None,None,None),
                                 Ngrid=None, boxsize=None,
                                 max_fractional_displacement_in_box=0.4,
                                 Psi_interp_method='nearest', use_mpi=False):
        """
        Shift a column containing particle positions by a displacement field defined
        on a regular grid. The result is saved in shifted_column. If shifted_colum
        is None, overwrite column.

        Parameters
        ----------
        column : string
            Column containing particle positions to be shifted. Units must be
            such that coordinates range from 0 to 1.
        psigrids_3tuple : tuple
            Components of displacement field. Must be 3-tuple (Psi_x, Psi_y, Psi_z),
            where each Psi_i is None or np.ndarray with shape (Ngrid,Ngrid,Ngrid).
            Units of Psi_i must be Mpc/h.
        shifted_colum : None or string
            Column to save shifted particle positions. If None, overwrite input 
            column.
        Psi_interp_method : string
            'nearest' (fast) or 'linear' (better but slower)
        """
        if shifted_column is None:
            # overwrite input column
            shifted_column = column

        if 'Position' not in shifted_column:
            raise Exception("shifted_column must contain 'Position' in column name.")
            
        # check psigrids_3tuple makes sense
        assert type(psigrids_3tuple) == tuple
        assert len(psigrids_3tuple) == 3
        for psiDir in range(3):
            print("type of psigrids_3tuple[psiDir]:", type(psigrids_3tuple[psiDir]))
            #assert type(psigrids_3tuple[psiDir]) == np.ndarray
            assert psigrids_3tuple[psiDir].shape == (Ngrid, Ngrid, Ngrid)

        # check units of input column
        assert np.max(self.P[column]) <= 1.1
        assert np.min(self.P[column]) >= -0.1

        # check shape of input column
        nptcles = self.P.shape[0]
        assert self.P[column].shape == (nptcles,3)

        self.append_column(shifted_column, 
                           1.0/boxsize * shift_catalog.get_shifted_pos_from_orig_pos(
            positions=self.P[column][...]*boxsize, 
            positions_units='Mpc/h',
            Psigrids=psigrids_3tuple, 
            Psigrids_units='Mpc/h',
            L=boxsize, ng=Ngrid,
            max_fractional_displacement_in_box=max_fractional_displacement_in_box,
            Psi_interp_method=Psi_interp_method, use_mpi=use_mpi))
        
            


    def take_subsample(self, sample_method='random', 
                       fsamp=0.01, seed=1,
                       modulo_divisor=101, modulo_remainder=0,
                       overflow=1.25):
        """
        Take a subsample of the catalog and throw away the rest.
        Subsample particles will be saved to self.P.

        sample_method: 

            'random': use fsamp and seed to draw uniform random subsample

            'IDmodulo': select particles with ID%modulo_divisor==modulo_remainder
        """
        if sample_method == 'random':
            print("Get random subsample with fsamp=%g and seed=%d" % (fsamp,seed))
            rr = np.random.RandomState(seed)
            npart = self.P['Position'].shape[0]
            # indices of ptcles to keep
            keep = np.nonzero(rr.uniform(size=npart) < fsamp)[0]
        elif sample_method == 'IDmodulo':
            print("Get IDmodulo subsample with modulo_divisor=%d and modulo_remainder=%d" % (
                modulo_divisor, modulo_remainder))
            keep = np.where(self.P['ID']%modulo_divisor == modulo_remainder)[0]
        else:
            raise Exception("Invalid sample_method %s" % str(sample_method))
        print("Keeping %d=%g**3 particles" % (len(keep), float(len(keep))**(1./3.)))
        newP = np.zeros(len(keep), dtype=self.P.dtype)
        newP[:] = self.P[keep]
        self.P = newP
        
            
    def get_InitialPosition_from_ID(self, verbose=False, sim_Nptcles_for_PID=2048):
        """
        Get initial particle positions from ids (Yu Feng mail from 3 Jan 2017).
        Store in self.P['InitialPosition'].
        """
        Np = sim_Nptcles_for_PID
        print("Get initial particle positions from their IDs...")
        print("WARNING: On 16 June 2017 changed to use sim_Nptcles_for_PID=2048; was read from file before")
        if 'ID' not in self.P.dtype.names:
            raise Exception("Need to have particle IDs to infer initial positions.")
        if 'InitialPosition' not in self.P.dtype.names:
            raise Exception("Must init InitialPosition column before filling it.")
        self.P['InitialPosition'] = np.zeros((len(self.P['ID']), 3), 'f4') + np.nan
        if not type(Np) in [type(int), int, np.int32, np.int64]:
            raise Exception("sim_Nptcles_for_PID must be integer")
        print("sim_Nptcles_for_PID:", Np)
        nc = Np
        id = self.P['ID'].copy()
        for d in [2, 1, 0]:
            self.P['InitialPosition'][:, d] = id % nc
            id[:] //= nc
        self.P['InitialPosition'][:] += 0.5
        # convert so it ranges from 0..1
        self.P['InitialPosition'][:] /= float(Np)
        if verbose:
            i1 = 1
            i2 = 7
            print("Some IDs:\n", self.P['ID'][i1:i2])
            print("InitialPosition:\n", self.P['InitialPosition'][i1:i2,:])
            print("InitialPosition Mpc/h:\n", self.P['InitialPosition'][i1:i2,:]*self.sim_boxsize)
            print("InitialPosition int:\n", self.P['InitialPosition'][i1:i2,:]*Np-0.5)
            print("Position in Mpc/h:\n", self.P['Position'][i1:i2,:]*self.sim_boxsize)
            
            print("Shape:", self.P['InitialPosition'].shape)
            print("Min, max init pos:", np.min(self.P['InitialPosition']), 
                  np.max(self.P['InitialPosition']))

        print("DONE: Get initial particle positions from their IDs")

                
    
