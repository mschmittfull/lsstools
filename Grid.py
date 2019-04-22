from __future__ import print_function,division

import cPickle
import numpy as np
import os
from collections import OrderedDict, namedtuple
import h5py
import json
from copy import copy


# MS packages
from nbodykit.base.mesh import MeshSource
from nbodykit.source.mesh.field import FieldMesh
from pmesh.pm import RealField, ComplexField
from nbodykit import logging
from nbkit03_utils import get_cstat, get_cstats_string, print_cstats
import nbkit03_utils
from MeasuredPower import MeasuredPower1D, MeasuredPower2D

"""
Store a collection of nbdodykit MeshSource objects, e.g. RealField objects.

The MeshSource objects are saved as columns of Grid, with names given by strings,
so we can easily apply actions on many MeshSource objects.

Note: We store the same instances, so modifying meshsource elsewhere will change
the data stored here. We recommend deleting meshsource always after storing it here.
"""



class Grid(object):
    """
    A class for storing and manipulating collections of 3D grids (instances of MeshSource).
    """
    def __init__(self, 
                 fname=None, read_columns=None,
                 meshsource=None, column=None, column_info=None, Ngrid=None, boxsize=None):
        """
        Initialize an instance of the class. Have several options:
        1) Specify fname to read data from file. Can use read_columns to select columns.
        2) Specify meshsource, column, Ngrid and boxsize to fill data.

        In earlier version, used numpy structured array. Now just use dict.

        Parameters
        ----------
        fname : None or string
            File name of hdf5 file that contains a grid to be read in.
        read_columns : None or sequence of strings
            Columns to read from file. If None, read all columns.
            Example: read_columns=('delta_x',).
        meshsource : None or instance of nbodykit MeshSource or pmesh.pm.RealField
                     or pmesh.pm.ComplexField
            Initial MeshSource shape (Ngrid,Ngrid,Ngrid). Will be stored
            in self.G[column].
        column : None or string
            Column where to store grid_array.
        column_info : None or dict
            Optional dict to save info about the stored field.
        Ngrid : None or integer
            Number of grid points per dimension, e.g. 128.
        boxsize : None or float
            Box size in Mpc/h.
        """
        from nbodykit import CurrentMPIComm
        self.comm = CurrentMPIComm.get()
        self.logger = logging.getLogger(str(self.__class__.__name__))

        # check args a bit (not extensively so be careful)
        if not ( (read_columns is None) or (type(read_columns) == tuple)
                 or type(read_columns) == list):
            print("read_columns:", read_columns)
            raise Exception("read_columns must be None or sequence")
       
        # initialize
        if fname is not None:
            # read grid and store it in self.G
            self.init_grid_from_bigfile(fname, columns=read_columns, mode='real')
        else:
            # init from grid_array supplied as argument, which should be MeshSource object
            print("type of meshsource:", type(meshsource))
            if isinstance(meshsource, RealField) or isinstance(meshsource, ComplexField):
                # convert to nbkit FieldMesh (which is subclass of MeshSource)
                meshsource = FieldMesh(meshsource)
            assert isinstance(meshsource, MeshSource)
            assert np.all(meshsource.attrs['Nmesh']==np.array([Ngrid,Ngrid,Ngrid]))
            assert np.all(meshsource.attrs['BoxSize']==np.array([boxsize,boxsize,boxsize]))
            self._Ngrid = Ngrid
            self._boxsize = boxsize
            self._base_dtype = meshsource.dtype          
            #dtype = np.dtype( [ (column, self.base_dtype) ] )
            #self.G = np.empty( (Ngrid**3,), dtype=dtype )
            # store as dict instead of structured array
            self.G = OrderedDict()
            self.G[column] = copy(meshsource)
            self.column_infos = {}
            self.column_infos[column] = column_info
    
            if isinstance(self, RealGrid):
                self._mode = 'real'
            elif isinstance(self, ComplexGrid):
                self._mode = 'complex'
            else:
                raise Exception("Must be RealGrid or ComplexGrid")
            print_cstats(self.G[column].compute(mode=self.mode), prefix="Init with column %s. " % column,
                    logger=self.logger)
    


    # Some read-only properties that shall not be changed once instance is initialized.
    @property
    def Ngrid(self):
        return self._Ngrid
    @property
    def boxsize(self):
        return self._boxsize
    @property
    def mode(self):
        return self._mode
    # @property
    # def base_dtype(self):
    #     return self._base_dtype
    # @property
    # def shape(self):
    #     return self._shape


    # @staticmethod
    def get_dtype_of_columns(self, columns):
        pass
        # dtype_list = []
        # for col in columns:
        #     dtype_list.append( (col, self.base_dtype) )
        # dtype = np.dtype(dtype_list)
        # print("Created dtype:", dtype)
        # return dtype

    def save_to_bigfile(self, fname, columns=None, new_dataset_for_each_column=False,
                     overwrite_file_if_exists=True, print_column_info=False):
        """
        Save grid columns to bigfile file. If columns is None, save all columns.
        If file exists, overwrite.

        Columns here mean entries of grid.G. They are not bigfile columns. They are 
        saved as different blocks in bigfile file.
        """
        # see https://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.base.mesh.html#nbodykit.base.mesh.MeshSource.save
        import bigfile
        import warnings
        import json
        from nbodykit.utils import JSONEncoder
        from pmesh.pm import BaseComplexField

        if self.comm.rank == 0:
            self.logger.info("Try writing to %s" % fname)

        if isinstance(self, RealGrid):
            mode = 'real'
        elif isinstance(self, ComplexGrid):
            mode = 'complex'
        else:
            raise Exception("can only save RealGrid or ComplexGrid")

        # fname = os.path.expandvars(fname)
        # print("Writing %s" % fname)
        # if overwrite_file_if_exists and os.path.exists(fname):
        #     os.remove(fname)
        # if os.path.dirname(fname) != '':
        #     if not os.path.exists(os.path.dirname(fname)):
        #         os.makedirs(os.path.dirname(fname))

        if columns is None:
            columns = self.G.keys()
        if self.comm.rank == 0:
            self.logger.info("Writing GridColumns %s" % str(columns))

        if overwrite_file_if_exists or (not os.path.exists(fname)):
            # write to new file
            create = True
        else:
            create = False

        if not new_dataset_for_each_column:
            raise Exception("only implemented new_dataset_for_each_column=True")

        # figure out bigfile issue
        # https://github.com/rainwoodman/bigfile/blob/b135ea5ef7d5bbdcab0a0086058bb99ebc95684f/bigfile/__init__.py#L179
        assert self.comm == self.G[columns[0]].comm
        tmp_cmean = self.G[columns[0]].to_field().cmean()
        self.logger.info('cmean0: %s' % tmp_cmean)
        self.logger.info('fname: %s' % fname)
        myset = set(self.comm.allgather(fname))
        self.logger.info('myset: %s' % myset)
        

        # save to file
        with bigfile.FileMPI(self.comm, fname, create=create) as ff:

            for grid_col in columns:
                dataset = grid_col
                # convert MeshSource to field
                field = self.G[grid_col].compute(mode=mode)
                data = np.empty(shape=field.size, dtype=field.dtype)
                # Ravel the field to 'C'-order, partitioned by MPI ranks. Save the result to flatiter.
                field.ravel(out=data)
                with ff.create_from_array(dataset, data) as bb:
                    if isinstance(field, RealField):
                        bb.attrs['ndarray.shape'] = field.pm.Nmesh
                        bb.attrs['BoxSize'] = field.pm.BoxSize
                        bb.attrs['Nmesh'] = field.pm.Nmesh
                    elif isinstance(field, BaseComplexField):
                        bb.attrs['ndarray.shape'] = field.Nmesh, field.Nmesh, field.Nmesh // 2 + 1
                        bb.attrs['BoxSize'] = field.pm.BoxSize
                        bb.attrs['Nmesh'] = field.pm.Nmesh

                    for key in field.attrs:
                        # do not override the above values -- they are vectors (from pm)
                        if key in bb.attrs: continue
                        value = field.attrs[key]
                        try:
                            bb.attrs[key] = value
                        except ValueError:
                            try:
                                json_str = 'json://'+json.dumps(value, cls=JSONEncoder)
                                bb.attrs[key] = json_str
                            except:
                                warnings.warn("attribute %s of type %s is unsupported and lost while saving MeshSource" % (key, type(value)))
        
                    bb.attrs['Ngrid'] = self.Ngrid
                    bb.attrs['boxsize'] = self.boxsize
                    if self.column_infos not in [None,{}]:
                        if print_column_info:
                            self.logger.info("column_info: %s" % str(self.column_infos[grid_col]))
                        bb.attrs['grid_column_info'] = json.dumps(self.column_infos[grid_col])

        if self.comm.rank == 0:
            self.logger.info("Wrote GridColumns %s" % str(columns))
            self.logger.info("to bigfile %s" % fname)



    def init_grid_from_bigfile(self, fname, columns=None, mode='real',
        print_column_info=True):
        """
        Read GridColumns columns from bigfile blocks and initialize grid instance.
        If columns is None, read all columns.
        """
        from nbodykit.source.mesh.bigfile import BigFileMesh

        self._mode = mode

        if self.comm.rank == 0:
            self.logger.info("Try to init grid by reading %s" % fname)

        if columns is None:
            # read all columns
            subfolders = [di for di in os.listdir(fname) if os.path.isdir(os.path.join(fname,di))]
            columns = subfolders

        if self.comm.rank == 0:
            self.logger.info("Columns to read: %s" % str(columns))

        counter = -1
        for column in columns:
            counter += 1
            bfmesh = BigFileMesh(path=fname, dataset=column)
            # scalar attrs are saved and read as 0-dimensional arrays. convert them back to scalars
            for k,v in bfmesh.attrs.items():
                if type(v)==np.ndarray and v.shape==():
                    bfmesh.attrs[k] = v[()]
            #if self.comm.rank == 0:
            #    self.logger.info("Attrs of %s:\n%s" % (column,str(bfmesh.attrs)))
            if counter == 0:
                self._Ngrid = bfmesh.attrs['Ngrid']
                self._boxsize = bfmesh.attrs['boxsize']
                self.G = OrderedDict()
                self.column_infos = OrderedDict()
            else:
                assert self._Ngrid == bfmesh.attrs['Ngrid']
                assert self._boxsize == bfmesh.attrs['boxsize']

            colinfo = {}
            if 'grid_column_info' in bfmesh.attrs:
                colinfo = json.loads(bfmesh.attrs['grid_column_info'])
            # read the data, convert to RealField or ComplexField, then to FieldMesh object, and store in self.G
            self.append_column(column, FieldMesh(bfmesh.compute(mode=mode)), column_info=colinfo,
                print_column_info=print_column_info)

            assert isinstance(self.G[column], MeshSource)
            #print("shape:", self.G[column].to_field().shape)
            assert np.all(self.G[column].attrs['Nmesh']==np.array([self.Ngrid,self.Ngrid,self.Ngrid]))
            assert np.all(self.G[column].attrs['BoxSize']==np.array([self.boxsize,self.boxsize,self.boxsize]))
            
        if self.comm.rank == 0:
            self.logger.info("Read Ngrid=%d, boxsize=%g Mpc/h" % (self.Ngrid, self.boxsize))
            self.logger.info("Read columns: %s" % str(self.G.keys()))
            self.logger.info("Done reading %s" % fname)


    def append_column(self, column_name, column_data, column_info=None,
        print_column_info=True):
        """
        column_name : string
            String corresponding to the name of the new column.
        column_data : MeshSource object, or RealField or ComplexField
            Data to store.
        """
        if isinstance(column_data, RealField) or isinstance(column_data, ComplexField):
            print('%d ya' % self.comm.rank)
            self.G[column_name] = FieldMesh(copy(column_data))
        elif isinstance(column_data, MeshSource):
            self.G[column_name] = copy(column_data)
        else:
            raise Exception("Invalid data type for column_data. Name: %s, type: %s" % (
                column_name, type(column_data)))
        #self.G[column_name] = copy(column_data)
        self.column_infos[column_name] = column_info

        print_cstats(self.G[column_name].compute(mode=self.mode), 
            prefix='Append %s. '%column_name, logger=self.logger)
        if print_column_info:
            if self.comm.rank == 0:
                self.logger.info("attrs: %s" % str(self.G[column_name].attrs))
                self.logger.info("column_info: %s" % str(self.column_infos[column_name]))

        # np.sqrt(np.mean(column_data**2)), np.min(column_data),
        # np.mean(column_data), np.max(column_data))

    def print_summary_stats(self, column):
        print_cstats(self.G[column].compute(), prefix="Stats of %s: " % column, logger=self.logger)
        
        # column_data = self.G[column]
        # print(str(self.__class__.__name__)+": Summary stats of "+column+": rms, min, mean, max:",
        #       np.sqrt(np.mean(column_data**2)), np.min(column_data),
        #       np.mean(column_data), np.max(column_data))
        
        
    def drop_column(self, column_name):
        """
        column_name : string
            String corresponding to the name of the column to be dropped.
        """
        if self.has_column(column_name):
            del self.G[column_name]
            del self.column_infos[column_name]
            if self.comm.rank == 0:
                self.logger.info("Dropped column %s" % column_name)

    def drop_all_except(self, columns_to_keep):
        """
        columns_to_keep : list
            List of columns to keep. All other columns will be dropped.
        """
        colnames = self.G.keys()
        for col in colnames:
            if col not in columns_to_keep:
                self.drop_column(col)
        
    def has_column(self, column_name, throw_exception_if_False=False):
        if self.G is None:
            return False
        else:
            return (column_name in self.G.keys())

    def rename_column(self, orig_name, new_name):
        self.append_column(new_name, self.G[orig_name])
        self.drop_column(orig_name)
        
    def check_column_exists(self, column_name):
        if not self.has_column(column_name):
            raise Exception("Could not find column %s. Allowed columns: %s" % (
                column_name, str(self.G.keys())))
        
    def G3d(self, column):
        """
        Return self.G[column], reshaped to 3d array (Ngrid,Ngrid,Ngrid).
        """
        raise Exception("G3d(column) is not implemented any more b/c we store FieldMesh object now")
        #return self.G[column].reshape((self.Ngrid,self.Ngrid,self.Ngrid))


        
    def append_columns_from_bigfile(self, fname, columns=None, replace_existing_col=True,
                                 raise_exception_if_column_does_not_exist=True):
        """
        Read bigfile grid and append to existing grid instance.
        """
        if self.comm.rank == 0:
            self.logger.info("Append columns from %s" % fname)

        if isinstance(self, RealGrid):
            mode = 'real'
        elif isinstance(self, ComplexGrid):
            mode = 'complex'
        else:
            raise Exception('Must be instance of RealGrid or ComplexGrid')

        subfolders = [di for di in os.listdir(fname) if os.path.isdir(os.path.join(fname,di))]

        if columns is None:
            # read all columns
            columns = subfolders

        if self.comm.rank == 0:
            self.logger.info("Reading columns %s" % str(columns))

        for col in columns:
            if (not replace_existing_col) and (self.G is not None) and (col in self.G.keys()):               
                print("Column %s already on grid, not reading again" % col)
            else:
                if self.has_column(col):
                    self.drop_column(col)
                if col not in subfolders: 
                    raise Exception("Column %s is not in file" % str(col))
                # read the columns into new grid instance
                if mode == "real":
                    tmp_grid = RealGrid(fname=fname, read_columns=columns)
                elif mode == "complex":
                    tmp_grid = ComplexGrid(fname=fname, read_columns=columns)
                else:
                    raise Exception('Mode must be real or complex')
                assert self.Ngrid == tmp_grid.Ngrid
                assert self.boxsize == tmp_grid.boxsize
                assert type(self) == type(tmp_grid)
                self.append_column(
                    col, tmp_grid.G[col], column_info=tmp_grid.column_infos.get(col,None))
                del tmp_grid
        if self.comm.rank == 0:
            self.logger.info("Done reading columns %s" % str(columns))
            #self.logger.info("Done reading %s" % fname)


        
      

        
class RealGrid(Grid):
    """
    A class for storing and manipulating 3D grids in configuration space.
    """

    def __init__(self, 
                 fname=None, read_columns=None,
                 meshsource=None, column=None, column_info=None,
                 Ngrid=None, boxsize=None):
        Grid.__init__(self, 
                      fname=fname, read_columns=read_columns, 
                      meshsource=meshsource, column=column, column_info=column_info,
                      Ngrid=Ngrid, boxsize=boxsize)

    def print_info(self):
        pass
        # print("")
        # if self.G is None:
        #     print("RealGrid data: None")
        # else:
        #     print("RealGrid data:", self.G.dtype)
        #     print("RealGrid column_infos:", self.column_infos)
        # print("")

        
    def fft_x2k(self, column, drop_column=False):
        """
        Compute FFT of a column and return it as FieldMesh object (storing ComplexField).
        """
        if self.comm.rank == 0:
            self.logger.info('Do FFT x2k')
        self.check_column_exists(column)
        # Convert meshsource to RealField. What normalization is used here?
        #rfield = self.G[column].to_real_field()
        #cfield = rfield.r2c()
        cfield = self.G[column].compute(mode='complex')
        if self.comm.rank == 0:
            self.logger.info('Done FFT x2k')

        if drop_column:
            self.drop_column(column)
        print_cstats(cfield, prefix="%s after fft " % column, logger=self.logger)
        return FieldMesh(cfield)


    def apply_smoothing(self, column, mode='Gaussian', R=0.0, helper_gridk=None, additional_props=None):
        if self.comm.rank == 0:
            self.logger.info("Apply smoothing to %s. Mode=%s, R=%s" % (column, mode, R))
        column_info = self.column_infos[column]
        # field_k is a FieldMesh object
        field_k = self.fft_x2k(column)
        if helper_gridk is None:
            helper_gridk = ComplexGrid(meshsource=field_k, column=column,
                                       Ngrid=self.Ngrid, boxsize=self.boxsize)
        else:
            helper_gridk.append_column(column, field_k)
        del field_k
        helper_gridk.apply_smoothing(column, mode=mode, R=R, additional_props=additional_props)
        print_cstats(helper_gridk.G[column].compute(mode="complex"), prefix="gridk after fft result ")
        # field_x is a FieldMesh object
        field_x = helper_gridk.fft_k2x(column, drop_column=True)
        self.append_column(column, field_x, column_info=column_info)

    def apply_fx(self, fx):
        # todo: implement using nbkit apply function
        raise Exception("Not implemented")

    def plot_slice(self, column, fname, vmin=None, vmax=None, mode=0,
                   title='AUTO', title_fs=12):
        """
        Mode: 0: Use nbodykit mesh preview.
        """
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,1)

        if mode == 0:
            plt.imshow(self.G[column].preview(axes=[0,1]))
            # old Marcel plot code
            # x_of_slice = (0,self.boxsize)
            # y_of_slice = (0,self.boxsize)
            # if vmin is None:
            #     vmin = np.min(image)
            # if vmax is None:
            #     vmax = np.max(image)
            # # good: RdBu_r, coolwarm
            # myplot = ax.imshow(image, vmin=vmin, vmax=vmax,
            #                    interpolation='gaussian',
            #                    cmap='RdBu_r',
            #                    extent=(x_of_slice[0], x_of_slice[1], y_of_slice[0], y_of_slice[1]))
            
        # colorbar
        myfontsize=14
        cbar = plt.colorbar(myplot, ax=ax, shrink=0.8, orientation='vertical')
        #pad=0.175, fraction=0.1, aspect=10)
        #cbar.set_ticks(np.linspace(vmin, vmax, 3))
               
        # cosmetics
        if title == 'AUTO':
            ax.set_title(column, y=1.01, fontsize=title_fs)
        elif title is not None:
            ax.set_title(title, y=1.01, fontsize=title_fs)
        #ax.set_xlabel('$x\,[\\mathrm{Mpc}/h]$')
        #ax.set_ylabel('$y\,[\\mathrm{Mpc}/h]$')
        plt.tight_layout()
        plt.savefig(fname)
        print("Made %s" % fname)


    def store_smoothed_gridx(self, col, path, fname,
                             helper_gridk=None, smoothing_mode='Gaussian', R=None,
                             plot=False, replace_nan=False):
        """
        Copy column col, FFT to k space, apply smoothing, FFT to x space,
        save to disk.
        """
        assert helper_gridk is not None
        tmpcol = 'tmp4storage'
        if fname is None:
            fname = col           
        if R is not None:
            if replace_nan is False:
                helper_gridk.append_column(tmpcol, self.fft_x2k(col, drop_column=False))
            else:
                # replace nan by value of replace_nan
                def replace_fcn(x,v):
                    return np.where(
                        np.isnan(v), 
                        np.zeros(v.shape, dtype=v.dtype)+replace_nan,
                        v)
                outfield = self.G[col].to_real_field()
                outfield.apply(replace_fcn, kind='relative', out=outfield)
                self.append_column(tmpcol, FieldMesh(outfield))
                del outfield
                helper_gridk.append_column(tmpcol, self.fft_x2k(tmpcol, drop_column=True))
            helper_gridk.apply_smoothing(tmpcol, mode=smoothing_mode, R=R)
            self.append_column(tmpcol, helper_gridk.fft_k2x(tmpcol, drop_column=True))
        else:
            if replace_nan is False:
                self.append_column(tmpcol, self.G[col])
            else:
                # replace nan by value of replace_nan
                def replace_fcn(x,v):
                    return np.where(
                        np.isnan(v), 
                        np.zeros(v.shape, dtype=v.dtype)+replace_nan,
                        v)
                outfield = self.G[col].to_real_field()
                outfield.apply(replace_fcn, kind='relative', out=outfield)
                self.append_column(tmpcol, FieldMesh(outfield))
                del outfield

        out_fname = os.path.join(path, '%s_R%s.bigfile' % (fname, str(R)))
        self.save_to_bigfile(out_fname, columns=[tmpcol], new_dataset_for_each_column=True)
        if plot:
            self.plot_slice(
                tmpcol, 'liveslice_%s_R%s.png'%(col,str(R)))
        self.drop_column(tmpcol)



    def convert_to_weighted_uniform_catalog(self, col=None, uni_cat_Nptcles_per_dim=None,
                                            fill_value_negative_mass=None, add_const_to_mass=0.0):
        """
        Convert 3D overdensity in gridx.G[col] to a uniform (=regular) catalog of ptcles with mass
        gridx.G[col]. Each particle sits at a node of the regular grid.

        Note: Our uniform catalog means sth different than nbodykit's UniformCatalog which uses
        random positions.
        
        Similar to nbkit03_utils.interpolate_pm_rfield_to_catalog.

        fill_value_negative_mass : if not None, set mass ofparticles with negative to mass to this value.
        """
        raise Exception('TODO: Own Catalog class not needed here, should implement without it.')

        import Catalog
        uniform_cat = Catalog.Catalog()
        uniform_cat.sim_Ngrid = None
        uniform_cat.sim_boxsize = self.boxsize
        grid_infodict = self.column_infos.get(col, {})
        uniform_cat.dataset_attrs = {'Smoothing': np.nan, 
                            'Nmesh': uni_cat_Nptcles_per_dim, 
                             'BoxSize': [self.boxsize, self.boxsize, self.boxsize],
                             'grid_infodict': json.dumps(grid_infodict)}

        # generate uniform catalog with particles on a regular grid
        # see https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/mockmaker.html#poisson_sample_to_points
        # and http://rainwoodman.github.io/pmesh/_modules/pmesh/pm.html#ParticleMesh.generate_uniform_particle_grid

        rfield = self.G[col].to_real_field()
        comm = rfield.pm.comm
        # ranges from 0 to 1?
        pos = rfield.pm.generate_uniform_particle_grid(shift=0.0, dtype='f4', return_id=False)

        if comm.rank == 0:
            self.logger.info("uniform regular catalog produced.")

        if True:
            # Sort, similar to https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/mockmaker.html#poisson_sample_to_points)
            # Not sure if needed.

            # generate linear ordering of the positions.
            # this should have been a method in pmesh, e.g. argument
            # to genereate_uniform_particle_grid(return_id=True);

            # grid separation H=Delta x
            H = rfield.BoxSize / rfield.Nmesh

            # FIXME: after pmesh update, remove this
            orderby = np.int64(pos[:, 0] / H[0] + 0.5)
            for i in range(1, rfield.ndim):
                orderby[...] *= rfield.Nmesh[i]
                orderby[...] += np.int64(pos[:, i] / H[i] + 0.5)

            # sort by ID to maintain MPI invariance.
            import mpsort
            pos = mpsort.sort(copy(pos), orderby=orderby, comm=comm)

            if comm.rank == 0:
                self.logger.info("sorting done")

            
        # normalize so position goes from 0 to 1
        pos = pos/rfield.BoxSize


        #dtype = np.dtype( [ ('Position', ('f4', 3)),
        #                    ('Mass', 'f4'),
        #                     ] )
        #uniform_cat.P = np.empty( (uni_cat_Nptcles_per_dim**3,), dtype=dtype )
        from nbodykit.source.catalog.array import ArrayCatalog
        # uniform_cat.P is a CatalogSource object
        uniform_cat.P = ArrayCatalog({'Position': pos}, comm=comm,
            **uniform_cat.dataset_attrs # store info in uniform_cat.P.attrs
            )


        if comm.rank == 0:
            self.logger.info("position min, max (expect b/w 0 and 1) = %s, %s" % (
                uniform_cat.P.compute(uniform_cat.P['Position'].min()),
                uniform_cat.P.compute(uniform_cat.P['Position'].max())))
                

        # x components in units such that box ranges from 0 to 1. Note dx=1/Np.
        #x_components_1d = np.linspace(0.0, (Ng-1)*(L/float(Ng)), num=Ng, endpoint=True)/L
        #x_components_1d = np.linspace(0.0, (Np-1)/float(Np), num=Np, endpoint=True)
        #ones_1d = np.ones(x_components_1d.shape)


        # Put particles on the regular grid
        #uniform_cat.P['Position'][:,0] = np.einsum('a,b,c->abc', x_components_1d, ones_1d, ones_1d).reshape((Np**3,))
        #uniform_cat.P['Position'][:,1] = np.einsum('a,b,c->abc', ones_1d, x_components_1d, ones_1d).reshape((Np**3,))
        #uniform_cat.P['Position'][:,2] = np.einsum('a,b,c->abc', ones_1d, ones_1d, x_components_1d).reshape((Np**3,))

        # assign mass = delta = gridx.G[col]
        if uni_cat_Nptcles_per_dim != self.Ngrid:
            raise Exception("Only implemented uni_cat_Nptcles_per_dim=Ngrid for now; otherwise need to use nbkit0.3 readout.")

        #uniform_cat.P['Mass'][:] = add_const_to_mass + self.G[col].reshape((self.Ngrid**3,))

        # get a layout (need window to determine buffer region)
        window = 'cic'
        layout = rfield.pm.decompose(uniform_cat.P['Position']*rfield.BoxSize, smoothing=window)

        # interpolate field to particle positions (use pmesh 'readout' function)
        # code:https://github.com/rainwoodman/pmesh/blob/54447c8b47085b82c72130074f6ce33cea9e4b21/pmesh/_window.pyx#L156
        # https://github.com/rainwoodman/pmesh/blob/54447c8b47085b82c72130074f6ce33cea9e4b21/pmesh/_window_imp.c
        # https://github.com/rainwoodman/pmesh/blob/master/pmesh/cic.py#L83
        samples_of_delta = rfield.readout(uniform_cat.P['Position']*rfield.BoxSize, resampler=window, layout=layout)

        # save into catalog column
        if add_const_to_mass != 0.0:
            if self.comm.rank == 0:
                self.logger.info('add_const_to_mass: %s' % str(add_const_to_mass))
        uniform_cat.P['Mass'] = add_const_to_mass + samples_of_delta

        print_cstats(uniform_cat.P['Mass'].compute(), 'Mass of ptcles ', self.logger)

        # check mass >= 0
        minmass = get_cstat(uniform_cat.P['Mass'].compute(), 'min')

        if minmass < 0.0:
            # check for negative mass particles
            mass_rfield = uniform_cat.P['Mass'].compute()
            #print('mass rfield:', mass_rfield[:])
            ww = np.where(mass_rfield < 0.0)[0]
            #print("ww ", ww)
            Nneg = self.comm.allreduce(np.array([ww.shape[0]]))
            Ntot = self.comm.allreduce(uniform_cat.P['Mass'].compute().size)
            if self.comm.rank == 0:
                self.logger.info("WARNING: %d cells have negative mass (%.3g percent of all cells)" % (
                    Nneg, 100.*float(Nneg)/float(Ntot)))
            if fill_value_negative_mass is not None:
                # set negative mass to fill_value_negative_mass (is there something better we can do?)
                if self.comm.rank == 0:
                    self.logger.info("Replace negative mass by %g" % fill_value_negative_mass)
                uniform_cat.P['Mass'] = np.where(mass_rfield < 0.0,
                    np.zeros(mass_rfield.shape, dtype=mass_rfield.dtype) + fill_value_negative_mass,
                    mass_rfield)
            else:
                print("Not doing anything about negative ptcle masses (ok when using sum-CIC painting?)")

        return uniform_cat
        




class ComplexGrid(Grid):
    """
    A class for storing and manipulating complex 3D grids in Fourier (k-)space.
    """
    def __init__(self, 
                 fname=None, read_columns=None,
                 meshsource=None, column=None, column_info=None,
                 Ngrid=None, boxsize=None):
        Grid.__init__(self, 
                      fname=fname, read_columns=read_columns, 
                      meshsource=meshsource, column=column, column_info=column_info,
                      Ngrid=Ngrid, boxsize=boxsize)

        # construct k component in each direction
        #self.k_component_1d = fft_ms_v2.get_1d_k_cpts(Ngrid=self.Ngrid, boxsize=self.boxsize)
        #assert self.k_component_1d.shape == (self.Ngrid,)
        #self._helper_grid_names = ['INV_ABSK', 'ABSK']

    def print_info(self):
        pass
        # print("")
        # if self.G is None:
        #     print("ComplexGrid data: None")
        # else:
        #     print("ComplexGrid data:", self.G.dtype)
        # print("")


    def compute_helper_grid(self, column_name=None, DCmode=-1.0e-50):
        pass
        #     """
        #     Return array of shape (Ngrid**3,) containing 1/k at each entry.
        #     """
        #     if not column_name in self._helper_grid_names:
        #         raise Exception("Invalid helper grid column_name %s" % str(column_name))
        #     if self.has_column(column_name):
        #         return
        #     if column_name == 'ABSK':
        #         # compute k = sqrt(kx**2+ky**2+kz**2)
        #         Ngrid = self.Ngrid
        #         ones = np.ones( (Ngrid**3,), dtype='c16')
        #         ones = ones.reshape( (Ngrid,Ngrid,Ngrid) )
        #         absk = np.zeros( (Ngrid**3,), dtype='c16')
        #         absk += (np.einsum('a,a,abc->abc',
        #                            self.k_component_1d, self.k_component_1d, ones)).reshape((Ngrid**3,))
        #         absk += (np.einsum('b,b,abc->abc',
        #                            self.k_component_1d, self.k_component_1d, ones)).reshape((Ngrid**3,))
        #         absk += (np.einsum('c,c,abc->abc',
        #                            self.k_component_1d, self.k_component_1d, ones)).reshape((Ngrid**3,))
        #         #absk = np.sqrt(absk)
        #         absk = absk**0.5
        #         absk[0] = 0.0
        #         #absk2 = absk.reshape( (Ngrid**3,) ).copy()     
        #         #absk = np.ma.filled(absk, fill_value=np.nan)
        #         #absk2 = np.ma.getdata(absk2)
        #         #absk2 = np.ascontiguousarray(absk2)
        #         self.append_column('ABSK', absk)
        #     elif column_name == 'INV_ABSK':
        #         if not self.has_column('ABSK'):
        #             self.compute_helper_grid('ABSK')
        #         absk = self.G['ABSK']
        #         self.append_column('INV_ABSK', np.where(absk==0, DCmode*np.ones(absk.shape), 1.0/absk))



    def drop_helper_grids(self):
        """
        Empty cached helper arrays, like inv_absk.
        """
        pass
        # for col in self._helper_grid_names:
        #     if self.has_column(col):
        #         self.drop_column(col)
            

    def fft_k2x(self, column, drop_column=False):
        """
        Compute FFT of a column and return it as FieldMesh object (storing RealField).
        """
        self.check_column_exists(column)
        if self.comm.rank == 0:
            self.logger.info('Do FFT k2x')

        rfield = self.G[column].compute(mode='real')

        if self.comm.rank == 0:
            self.logger.info('Done FFT k2x')

        # # todo: check if we got a real field with no imag part
        # max_imag = np.max(np.abs(np.imag(field_x)))
        # avg_real = np.mean(np.abs(np.real(field_x)))
        # print("max_imag, avg_real:", max_imag, avg_real)
        # if max_imag > 1e-3 * avg_real:
        #     raise Exception("Real field has too large imaginary part. Consider smoothing with R>=%g or %g Mpc/h." % (
        #         1.5*self.boxsize/float(self.Ngrid), 2.*self.boxsize/float(self.Ngrid)))
        if drop_column:
            self.drop_column(column)
        return FieldMesh(rfield)

        

    def apply_smoothing(self, column, mode='Gaussian', R=0.0, kmax=None,
                        additional_props=None, just_return_window_fcn=False):
        """
        mode: 'Gaussian' or 'SharpK' or 'kstep'.

        Result overwrites data in grid.G[column].
        """
        if just_return_window_fcn:
            # just return a fcn computing the smoothing kernel WR
            def calc_WR(kvec):                
                if mode=='Gaussian':
                    if R!=0.0:
                        WR = np.exp(-(R*kvec)**2/2.0)
                    else:
                        WR = np.ones(kvec.shape)
                elif mode=='InverseGaussian':
                    raise Exception("todo")
                elif mode=='kstep':
                    step_kmin = additional_props['step_kmin']
                    step_kmax = additional_props['step_kmax']
                    WR = np.where( (kvec>=step_kmin) & (kvec<step_kmax), 
                                   np.ones(kvec.shape), np.zeros(kvec.shape) )
                else:
                    raise Exception("just_return_window_fcn not implemented for %s" % mode)
            
                if kmax is not None:
                    WR = np.where(kvec<=kmax, WR, np.zeros(WR.shape))
                return WR
            return calc_WR

        # actually apply smoothing to the field
        if self.comm.rank == 0:
            self.logger.info("Apply smoothing to %s with mode=%s, R=%g Mpc/h, kmax=%s h/Mpc" % (
                column, mode, R, str(kmax)))

        column_info = self.column_infos[column]
        
        self.append_column(
            column,
            nbkit03_utils.apply_smoothing(self.G[column], mode=mode, R=R, kmax=kmax,
                additional_props=additional_props),
            column_info=column_info)
      



    def calc_all_power_spectra(self, columns=None, Pk_ptcle2grid_deconvolution=None,
                               k_bin_width=1.0, Pkmeas=None, verbose=False,
                               mode='1d', poles=None, Nmu=5, line_of_sight=None):
        """
        Calculate power spectra between columns. If columns=None, compute power spectra
        of all columns of the Grid object.

        k_bin_width : float
            Width of each k bin, in units of k_fundamental=2pi/L. Must be >=1.

        mode : string
            '1d' or '2d' (use for anisotropic power e.g. due to RSD)

        poles : list
            Multipoles to measure if mode='2d'. E.g. [0,2,4].

        Nmu : int
            If not None, measure also P(k,mu), in Nmu mu bins.

        line_of_sight : list
            Direction of line of sight if mode='2d', e.g. [0,0,1] for z direction.
        """
        from nbodykit.algorithms.fftpower import FFTPower
        if columns is None and (self.G is not None):
            columns = self.G.keys()
        if Pkmeas is None:
            Pkmeas = OrderedDict()
        if Pk_ptcle2grid_deconvolution is not None:
            raise Exception("Not implemented")
        for id1 in columns:
            for id2 in columns:
                if (not self.has_column(id1)) or (not self.has_column(id2)):
                    print("WARNING: Could not measure power of [%s] X [%s] b/c not on grid." % (
                        id1,id2))
                    print("Allowed grid entries:", self.G.keys())
                    continue
                if (id1,id2) in Pkmeas:
                    pass
                elif (id2,id1) in Pkmeas:
                    Pkmeas[(id1,id2)] = Pkmeas[(id2,id1)]
                else:
                    # Estimate power.
                    # Marcel code until Jan 2019 used dk=2pi/L and kmin=dk/2.
                    # Checked that measured power spectrum agrees with old code.
                    Pk_dk = 2.0*np.pi/self.boxsize * k_bin_width
                    Pk_kmin = 2.0*np.pi/self.boxsize / 2.0
                    if mode == '1d':
                        if id1==id2:
                            Pkresult = FFTPower(first=self.G[id1], 
                                mode=mode, dk=Pk_dk, kmin=Pk_kmin)
                        else:
                            Pkresult = FFTPower(first=self.G[id1], second=self.G[id2],
                                mode=mode, dk=Pk_dk, kmin=Pk_kmin)
                    else:
                        if id1==id2:
                            Pkresult = FFTPower(first=self.G[id1], 
                                mode=mode, dk=Pk_dk, kmin=Pk_kmin, 
                                poles=poles, Nmu=Nmu, los=line_of_sight)
                        else:
                            Pkresult = FFTPower(first=self.G[id1], second=self.G[id2],
                                mode=mode, dk=Pk_dk, kmin=Pk_kmin,
                                poles=poles, Nmu=Nmu, los=line_of_sight)


                    # print info
                    if verbose:
                        if self.comm.rank == 0:
                            self.logger.info("Pkresult: %s" % str(Pkresult))
                            self.logger.info("Pkresult k: %s" % str(Pkresult.power['k']))
                            self.logger.info("Pkresult Nmodes: %s" % str(Pkresult.power['modes'].real))
                            self.logger.info("Pkresult P: %s" % str(Pkresult.power['power'].real))
                            self.logger.info("Pkresult attrs: %s" % str(Pkresult.attrs))

                    # save info about Pk and fields for convenience
                    info = {'Ngrid': self.Ngrid, 'boxsize': self.boxsize,
                            'Pk_ptcle2grid_deconvolution': Pk_ptcle2grid_deconvolution,
                            'k_bin_width': k_bin_width, 'Pk_attrs': Pkresult.attrs}

                    # save result
                    # kest, Pest, num_summands, info, info_id1, info_id2
                    #Pkmeas[(id1,id2)] = MeasuredPower(
                    #    Pkresult.power['k'], Pkresult.power['power'].real, Pkresult.power['modes'].real,
                    #   info, self.column_infos[id1], self.column_infos[id2])
                    if mode == '1d':
                        Pkmeas[(id1,id2)] = MeasuredPower1D(nbk_binned_stat=Pkresult,
                            info=info, info_id1=self.column_infos[id1], info_id2=self.column_infos[id2])
                    elif mode == '2d':
                        Pkmeas[(id1,id2)] = MeasuredPower2D(nbk_binned_stat=Pkresult,
                            info=info, info_id1=self.column_infos[id1], info_id2=self.column_infos[id2])
                    else:
                        raise Exception('Invalid mode %s' % mode)


                    
        return Pkmeas

        

    def apply_fk(self, fk):
        raise Exception("Not implemented")


    def calc_kappa2(self, field_column, gridx=None):
        """
        Calculate kappa2[f] = kappa_2(q,k-q) f(q) f(k-q),
        where kappa_2(\vp_1,\vp_2) = 1 - (\vp_1/p_1 \cdot \vp_2/p_2)^2
        and the base field is f=self.G[field_column].

        Have kappa2(\vx) = deltalin^2(\vx) - \sum_{i,j=0..2} d_ij(\vx)d_ij(\vx),
        where d_ij(\vk) = k_ik_j/k^2*f(\vk)

        TODO: Could do this much more memory efficient.

        Parameters
        ----------
        field_column : string
            Column of base field.
        gridx : instance of RealGrid
            Used for FFTs to real space. Must have same Ngrid etc as
            this instance.

        Returns
        -------
        Return the calculated Ngrid**3 array containing kappa2[f](\vk).
        """
        # Compute kappa2(\vx) = delta^2(\vx) - \sum_{i,j=0..2} d_ij(\vx)d_ij(\vx).
        # Store this in gridx.G['tmp_kappa2'].
        assert (not self.has_column('tmp_kappa2'))
        gridx.append_column(field_column, self.fft_k2x(field_column, drop_column=False))
        # start with field**2(x)
        gridx.append_column('tmp_kappa2', gridx.G[field_column].apply(
            lambda x3vec, val: val**2, mode='real', kind='relative'))
        gridx.drop_column(field_column)

        # Compute d_ij(x). It's symmetric in i<->j so only compute j>=i.
        #self.compute_helper_grid('ABSK')
        for idir in range(3):
            for jdir in range(idir,3):
                dijcol = 'tmp_d_%d%d'%(idir,jdir)
                assert not self.has_column(dijcol)
                assert not gridx.has_column(dijcol)

                # compute dij
                def dij_fcn(k3vec, val, idir=idir, jdir=jdir):
                    kk = sum(ki ** 2 for ki in k3vec) # k^2 on the mesh
                    kk[kk == 0] = 1
                    return k3vec[idir]*k3vec[jdir]*val/kk

                self.append_column(dijcol, 
                    self.G[field_column].apply(dij_fcn, mode='complex', kind='wavenumber'))
                gridx.append_column(dijcol, self.fft_k2x(dijcol, drop_column=True))

                # i_index = 'abc'[idir]
                # j_index = 'abc'[jdir]
                # tmp_dij = np.einsum('%s,%s,abc->abc' % (i_index,j_index),
                #               self.k_component_1d, 
                #               self.k_component_1d,
                #               self.G3d(field_column)/(self.G3d('ABSK'))**2
                #               )
                # tmp_dij[0,0,0] = 0.0

                # Subtract \sum_{i,j=0..2} d_ij(\vx)d_ij(\vx) 
                #   = d_00^2+d_11^2+d_22^2 + 2*(d_01^2+d_02^2+d_12^2)
                if True:
                    if jdir == idir:
                        fac = 1.0
                    else:
                        fac = 2.0
                    def square_me(x3vec, val):
                        return val**2
                    dij_x_square = gridx.G[dijcol].apply(lambda x3vec, val: val**2, mode='real', kind='relative')
                    gridx.G['tmp_kappa2'] = FieldMesh(
                        gridx.G['tmp_kappa2'].compute(mode='real') - fac * dij_x_square.compute(mode='real'))
                    # free memory
                    gridx.drop_column(dijcol)

        # fft to k space
        return gridx.fft_x2k('tmp_kappa2', drop_column=True)


    def calc_quadratic_field(self, basefield=None, quadfield=None,
        smoothing_of_base_field=None):
        """
        Calculate quadratic field. This is a wrapper of nbkit03_utils.calc_quadratic_field.
        Returns a FieldMesh object.
        """

        if smoothing_of_base_field is not None:
            # First apply smoothing, then square
            assert not self.has_column('TMP_FIELD')
            self.append_column('TMP_FIELD', self.G[basefield])
            self.apply_smoothing('TMP_FIELD', **smoothing_of_base_field)
            return nbkit03_utils.calc_quadratic_field(
                base_field_mesh=self.G['TMP_FIELD'],
                quadfield=quadfield)

        else:
            # no smoothing needed
            return nbkit03_utils.calc_quadratic_field(
                base_field_mesh=self.G[basefield],
                quadfield=quadfield)

  
    def store_smoothed_gridx(self, col, path, fname,
                             helper_gridx=None, R=None,
                             plot=False, replace_nan=False):
        """
        Copy column col, apply smoothing, FFT to x space, save to disk.
        """
        if fname is None:
            fname = col
        tmpcol = 'tmp4storage'
        if replace_nan is False:
            self.append_column(tmpcol, FieldMesh(self.G[col].compute(mode='complex')))
        else:
            def replacer(k3vec, val):
                return np.where(
                    np.isnan(val),
                    np.zeros(val.shape, dtype=val.dtype) + replace_nan,
                    val)

            self.append_column(
                tmpcol,
                self.G[col].apply(replacer, mode='complex', kind='wavenumber'))
        self.apply_smoothing(tmpcol, mode='Gaussian', R=R)
        helper_gridx.append_column(
            tmpcol,
            self.fft_k2x(tmpcol, drop_column=True))
        out_fname = os.path.join(path, '%s_R%s.hdf5' % (fname, str(R)))
        
        if ((replace_nan is not False) 
            and np.isnan(get_cstat(helper_gridx.G[tmpcol].compute(mode='real'), 'mean'))):
            raise Exception("Error when exporting file: found nan")
        helper_gridx.save_to_bigfile(out_fname, columns=[tmpcol], new_dataset_for_each_column=True)
        if plot:
            helper_gridx.plot_slice(
                tmpcol, 'liveslice_%s_R%s.png'%(col,str(R)))
        helper_gridx.drop_column(tmpcol)




    def compute_orthogonalized_fields(self, 
        N_ortho_iter=None, orth_method=None,
        all_in_fields=None, 
        orth_prefix='ORTH s', non_orth_prefix='NON_ORTH s',
        Pkmeas=None,
        Pk_ptcle2grid_deconvolution=None,
        k_bin_width=1.0,        
        Pk_1d_2d_mode='1d', RSD_poles=None, RSD_Nmu=None,
        RSD_los=None,
        interp_kind=None,
        delete_original_fields=False):
        """
        Given all_fields, compute orthogonalized fields using orthogonalization method
        orth_method and N_ortho_iter orthogonalization iterations. Save them on self,
        and return relevant power spectra and info.

        Parameters
        ----------
        orth_method : string
            'CholeskyDecomp' or 'EigenDecomp'

        all_in_fields : list of strings
            Column names of fields in self to be orthogonalized. All of these column names must
            start with orth_prefix (included in orthogonalization) or non_orth_prefix (these will
            not be included in orthogonalization). 
            
        Returns
        -------
        all_fields : list of strings
            Column names of the orthogonal fields we computed.
        """

        from scipy.linalg import cholesky
        from scipy import interpolate as interp
        import interpolation_utils
        
        # make a copy so we don't change input list
        all_fields = all_in_fields[:]

        # number of fields we work with
        Nfields = len(all_fields)

        # initialize rotation matrix of transfer fcns (needed when orthogonalizing)
        ortho_rot_matrix = None


        # rotate fields to orthogonalize them
        for iortho in range(1,N_ortho_iter+1):
            print("\n\nOrthogonalize fields for trf fcns, iteration %d of %d" % (iortho,N_ortho_iter))
            print("Orth_method:", orth_method)

            # get S_ij^{(n)} matrix
            Pkmeas = self.calc_all_power_spectra(
                columns=all_fields,
                Pk_ptcle2grid_deconvolution=Pk_ptcle2grid_deconvolution,
                k_bin_width=k_bin_width,
                mode=Pk_1d_2d_mode, poles=RSD_poles, Nmu=RSD_Nmu,
                line_of_sight=RSD_los,
                Pkmeas=Pkmeas)
            Nfields = len(all_fields)
            kvec = Pkmeas[Pkmeas.keys()[0]].k
            Nk = kvec.shape[0]
            print("Nfields: %d, Nk: %d" % (Nfields,Nk))
            Smat = np.zeros( (Nfields,Nfields,Nk) ) + np.nan
            for ifield, field in enumerate(all_fields):
                for ifield2, field2 in enumerate(all_fields):
                    Smat[ifield,ifield2,:] = Pkmeas[field,field2].P
            # enforce exact symmetry
            for ifield in range(Nfields):
                for ifield2 in range(ifield+1,Nfields):
                    Smat[ifield,ifield2,:] = Smat[ifield2,ifield,:]


            if orth_method == 'CholeskyDecomp':
                # Use Cholesky decomposition of S matrix so that span of first
                # k orthogonal fields is the same as span of first k original fields.

                if iortho > 1:
                    raise Exception("Cholesky only implemented for 1 orth step")

                # get lower-diagonal matrix L that orthogonalizes fields
                inv_Lmat = np.zeros( (Nfields,Nfields,Nk) ) + np.nan
                Cmat = np.zeros( (Nfields,Nfields,Nk) ) + np.nan  #correlation
                inv_sqrt_Sii_vec = np.zeros( (Nfields,Nk) ) + np.nan # normalization D_ij=delta_ij S_ii^{-1/2}
                for ik in range(Nk):
                    if False:
                        # cholesky of S matrix without normalizing before cholesky
                        L = cholesky(Smat[:,:,ik], lower=True)
                        inv_Lmat[:,:,ik] = np.linalg.inv(L)
                        del L
                    else:
                        # cholesky of normalized correl matrix C=S_ij/sqrt(S_ii*S_jj)
                        Sii = np.diag(Smat[:,:,ik])
                        inv_sqrt_Sii_vec[:,ik] = np.where(Sii>0., 1./np.sqrt(Sii), 0*Sii)
                        if not np.all(inv_sqrt_Sii_vec[:,ik]>0.):
                            # one of the vectors has 0 length, so do not rotate
                            print("Warning: not rotating k, Sii, inv_sqrt_Sii_vec:", 
                                kvec[ik], Sii, inv_sqrt_Sii_vec[:,ik])
                            inv_Lmat[:,:,ik] = np.diag(np.ones(Nfields))
                        else:
                            # all vectors have nonzero length so can rotate
                            # get correlmat C
                            Cmat[:,:,ik] = np.dot(np.diag(inv_sqrt_Sii_vec[:,ik]), 
                                                  np.dot(Smat[:,:,ik], np.diag(inv_sqrt_Sii_vec[:,ik])))
                            # do cholesky
                            print("Do cholesky of:\n", Cmat[:,:,ik])
                            L = cholesky(Cmat[:,:,ik], lower=True)
                            inv_Lmat[:,:,ik] = np.linalg.inv(L)
                    if ik<5:
                        print("k, Smat, Lmat^{-1}:")
                        print(kvec[ik])
                        print(Smat[:,:,ik])
                        print(inv_Lmat[:,:,ik])

                # 15 feb 2018: normalize orthogonal vectors by rotating fields using
                # M_ij = (L^{-1})_ij (S_jj)^{-1/2} / [ (L^{-1})_ii (S_ii)^{-1/2}
                Mrotmat = np.zeros( (Nfields,Nfields,Nk) ) + np.nan
                for ifield in range(Nfields):
                    for jfield in range(Nfields):
                        Mrotmat[ifield,jfield,:] = (
                            inv_Lmat[ifield,jfield,:] * inv_sqrt_Sii_vec[jfield,:]
                            / ( inv_Lmat[ifield,ifield,:] * inv_sqrt_Sii_vec[ifield,:] ) )

                # set entries for non_orth_linear_fields by hand
                for ifield,field1 in enumerate(all_fields):
                    for jfield,field2 in enumerate(all_fields):
                        if field1.startswith(non_orth_prefix) or field2.startswith(non_orth_prefix):
                            if jfield == ifield:
                                Mrotmat[ifield,jfield,:] = 0*Mrotmat[ifield,jfield,:] + 1.0
                            else:
                                Mrotmat[ifield,jfield,:] = 0*Mrotmat[ifield,jfield,:]

                # print first entries
                for ik in range(5):
                    print("k, Mrotmat:")
                    print(kvec[ik])
                    print(Mrotmat[:,:,ik])

                # interpolate Mrotmat so we can use it to rotate fields
                # kind='nearest' gives better orthogonalization than 'linear', but
                # expect even better when we use exactly same k binning as the one used to measure P(k).
                interp_Mrotmat = (np.zeros( (Nfields,Nfields) )).tolist()
                for ifield in range(Nfields):
                    for jfield in range(Nfields):
                        if False:
                            # use nearest interp; used until 6 June 2018
                            raise Exception("please use manual_Pk_k_bins interpolation")
                            interp_Mrotmat[ifield][jfield] = interp.interp1d(
                                kvec, Mrotmat[ifield,jfield,:],
                                kind='nearest', # 'nearest' gives better orthogonalization than 'linear'
                                fill_value=(Mrotmat[ifield,jfield,0], Mrotmat[ifield,jfield,-1]),
                                bounds_error=False)
                        else:
                            # use manual k binning interp: gives much better orthogonalization (10^-5 or better)
                            interp_Mrotmat[ifield][jfield] = interpolation_utils.interp1d_manual_k_binning(
                                kvec, Mrotmat[ifield,jfield,:],
                                #kind='manual_Pk_k_bins',
                                kind=interp_kind,
                                fill_value=(Mrotmat[ifield,jfield,0], Mrotmat[ifield,jfield,-1]),
                                bounds_error=False, 
                                Ngrid=self.Ngrid, L=self.boxsize, k_bin_width=k_bin_width,
                                Pkref=Pkmeas[Pkmeas.keys()[0]]
                                )


                # Get new orthogonalized fields, q_i = sum_{j<=i} M_ij s_j.
                # Do not rotate non_orth_linear_fields though.
                old_all_fields = all_fields[:]
                all_fields = []
                for ifield in range(Nfields):
                    #if sources[ifield] not in non_orth_linear_sources:
                    if old_all_fields[ifield].startswith(orth_prefix):
                        # rotate field away from all previous fields
                        ofield = '%s^%d_%d' % (orth_prefix, iortho,ifield)
                        all_fields.append(ofield)
                        self.append_column(ofield,
                            FieldMesh(0.0 * self.G['%s^%d_%d' % (orth_prefix,iortho-1,ifield)].compute(mode='complex')))
                        # 9 May 2018: Only need to rotate away from previous fields, so j<=i.
                        for jfield in range(0,ifield+1):
                            # add s_j
                            # self.G[ofield] += (
                            #     interp_Mrotmat[ifield][jfield](self.G['ABSK'].data)
                            #     * self.G['%s^%d_%d' % (orth_prefix,iortho-1,jfield)] )
                            if interp_kind == 'manual_Pk_k_bins':
                                def to_add_filter(k3vec, val):
                                    absk = np.sqrt(sum(ki ** 2 for ki in k3vec)) # absk on the mesh
                                    return interp_Mrotmat[ifield][jfield](absk) * val
                            elif interp_kind == 'manual_Pk_k_mu_bins':
                                def to_add_filter(k3vec, val):
                                    absk = np.sqrt(sum(ki ** 2 for ki in k3vec)) # absk on the mesh
                                    absk[absk==0] = 1
                                    #with np.errstate(invalid='ignore', divide='ignore'):
                                    mu = sum(k3vec[i]*RSD_los[i] for i in range(3)) / absk
                                    return interp_Mrotmat[ifield][jfield](absk,mu) * val


                            to_add = self.G['%s^%d_%d' % (orth_prefix,iortho-1,jfield)].apply(
                                to_add_filter, mode='complex', kind='wavenumber')
                            self.G[ofield] = FieldMesh(
                                self.G[ofield].compute(mode='complex') + to_add.compute(mode='complex'))

                        # properly add column (print rms etc)
                        self.append_column(ofield,
                                            self.G['%s^%d_%d' % (orth_prefix,iortho,ifield)])

                    elif old_all_fields[ifield].startswith(non_orth_prefix):
                        # do not rotate non_orth fields
                        ofield = '%s^%d_%d' % (non_orth_prefix,iortho,ifield)
                        all_fields.append(ofield)
                        self.append_column(ofield,
                                            self.G['%s^%d_%d' % (non_orth_prefix, iortho-1,ifield)])
                    else:
                        raise Exception("Do not know how to handle field %s" % 
                                        old_all_fields[ifield])



            elif orth_method == 'EigenDecomp':

                # Use eigenvalue decomposition. Each new field will be combi of all
                # original fields.
                raise Exception("Not implemented any more in parallel version of the code.")

            else:
                raise Exception("Invalid orth_method %s" % orth_method)

            # delete old fields
            if delete_original_fields:
                for ifield in range(Nfields):
                    if old_all_fields[ifield].startswith(orth_prefix):
                        self.drop_column('%s^%d_%d' % (orth_prefix,iortho-1,ifield))
                    elif old_all_fields[ifield].startswith(non_orth_prefix):
                        self.drop_column('%s^%d_%d' % (non_orth_prefix,iortho-1,ifield))

        orthogonalization_internals = {}
        if orth_method == 'CholeskyDecomp':
            if N_ortho_iter == 1:
                orthogonalization_internals = {
                    'Smat': Smat, 'inv_Lmat': inv_Lmat, 'Cmat': Cmat,
                    'Mrotmat': Mrotmat,
                    'inv_sqrt_Sii_vec': inv_sqrt_Sii_vec}

        return all_fields, Pkmeas, ortho_rot_matrix, orthogonalization_internals


    def convert_to_weighted_uniform_catalog(self, col=None, uni_cat_Nptcles_per_dim=None,
                                            fill_value_negative_mass=None, helper_gridx=None,
                                            add_const_to_mass=0.0):
        """
        Convert 3D overdensity in gridx.G[col] to a uniform catalog of ptcles with mass
        gridx.G[col].
        Similar to nbkit03_utils.interpolate_pm_rfield_to_catalog.

        fill_value_negative_mass : if not None, set mass ofparticles with negative to mass to this value.
        """
        # convert to x space
        if helper_gridx is None:
            print("nan k:",  np.where(np.isnan(self.G[col].compute(mode='complex'))))
            my_meshsource_x = self.fft_k2x(col, drop_column=False)
            
            helper_gridx = RealGrid(meshsource=my_meshsource_x,
                                    column=col, Ngrid=self.Ngrid,
                                    column_info=self.column_infos.get(col,{}),
                                    boxsize=self.boxsize)
            if np.isnan(get_cstat(helper_gridx.G[col].compute(mode='real'), 'max')):
                raise Exception("err 1: gridx is nan after fft")

        else:
            helper_gridx.append_column(col, self.fft_k2x(col, drop_column=False))

            if np.isnan(get_cstat(helper_gridx.G[col].compute(mode='real'), 'max')):
                raise Exception("err 2: gridx is nan after fft")
            
        # call RealGrid function
        return helper_gridx.convert_to_weighted_uniform_catalog(
            col=col, uni_cat_Nptcles_per_dim=uni_cat_Nptcles_per_dim,
            fill_value_negative_mass=fill_value_negative_mass,
            add_const_to_mass=add_const_to_mass)

        
def main():
    """
    Some examples how to use this class.
    """
    Ngrid = 64
    L = 1380.
    # random array in x space
    gridx_array = np.random.rand(Ngrid**3)
    gridx = RealGrid(grid_array=gridx_array, column='delta_x',
                     Ngrid=Ngrid, boxsize=L)

    ## Check that fft and ifft are inverse of each other
    # fft
    gridk_array = gridx.fft_x2k('delta_x', drop_column=False)
    gridk = ComplexGrid(grid_array=gridk_array,
                        column='delta_k', 
                        Ngrid=gridx.Ngrid, boxsize=gridx.boxsize)
    # inverse fft
    gridx2 = gridk.fft_k2x('delta_k', drop_column=True)
    gridx.append_column('delta_x2', gridx2)
    # print
    print('delta_x:', gridx.G['delta_x'])
    print('delta_x2:', gridx.G['delta_x2'])
    # check input = ifft(fft(input))
    assert np.allclose(gridx.G['delta_x'], gridx.G['delta_x2'])
    gridx.drop_column('delta_x2')

    ## Do the same again with shorter syntax (using initialized grids)
    gridk.append_column('delta_k', gridx.fft_x2k('delta_x'))
    gridx.append_column('delta_x2', gridk.fft_k2x('delta_k'))
    # check input = ifft(fft(input))
    assert np.allclose(gridx.G['delta_x'], gridx.G['delta_x2'])
    gridx.drop_column('delta_x2')

    ## Apply a k filter
    # multiply by k_z
    test_k = np.einsum('c,abc->abc',
                       gridk.k_component_1d,
                       gridk.G['delta_k'].reshape( (Ngrid,Ngrid,Ngrid) ))
    print("test_k:", test_k[:2,:2,:2])
    gridk.compute_helper_grid('INV_ABSK')
    print("inv_absk:", gridk.G['INV_ABSK'][:5])
    print("k_component_1d:", gridk.k_component_1d)
    
    # Apply smoothing
    gridk.apply_smoothing('delta_k', mode='Gaussian', R=40.0)
    
    # Compute Zeldovich displacement
    for direction in [0,1,2]:
        index = 'abc'[direction]
        col = 'PsiZA_%d' % direction
        gridk.append_column(col,
            np.einsum('%s,abc,abc->abc' % index,
                      1j*gridk.k_component_1d,
                      (gridk.G3d('INV_ABSK'))**2,
                      gridk.G3d('delta_k')))
        gridx.append_column(col, gridk.fft_k2x(col, drop_column=True))

    gridk.drop_helper_grids()

    # Compute power spectrum
    # CIC deconvolution at P(k) level: None or 'power_isotropic_and_aliasing'
    kest, Pest, num_summands = gridk.calc_power(
        'delta_k', 'delta_k', Pk_ptcle2grid_deconvolution=None)


    # Playing with I/O: Write to hdf5 and read again
    fname = "tmp_delta_x.hdf5"
    gridx.save_to_hdf5(fname)
    gridx2 = RealGrid(fname=fname, read_columns=('delta_x',))
    assert np.allclose(gridx.G['delta_x'], gridx2.G['delta_x'])



    
if __name__ == '__main__':
    main()


