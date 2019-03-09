#!/usr/bin/env python
#
# Marcel Schmittfull 2017 (mschmittfull@gmail.com)
#
# Python script for comparing displacements for BAO reconstruction.
#



from __future__ import print_function,division

import os
import sys
import cPickle as pickle
import time
import numpy as np



class Pickler(object):
    """
    A class for reading and writing pickle files.
    """
    def __init__(self, path=None, base_fname=None, full_fname=None,
                 file_format='pickle', rand_sleep=True):
        """
        Have 2 options to initialize:
        (a): Specify path and base_fname
        (b): Specify full_fname
        file_format can be 'pickle' or 'h5'.
        """
        self.path = path
        self.base_fname = base_fname
        self.full_fname = full_fname
        self.file_format = file_format

        # check params
        if self.file_format not in ['pickle','h5']:
            raise Exception("Invalid file_format: %s" % str(self.file_format))

        if self.full_fname is None:
            # generate full_fname where to save pickle
            file_exists = True
            if rand_sleep:
                # sleep 0-30 secs to avoid simultaneously getting same file name
                # if multiple jobs run at the same time.
                time.sleep(np.random.randint(30))
                time.sleep(np.random.randint(30))
            while file_exists:
                fname = os.path.join(
                    self.path, '%s_%s_time%s.%s' % (
                        self.base_fname, 
                        time.strftime("%Y_%b_%d_%H:%M:%S", time.gmtime()),
                        str(int(time.time())),
                        self.file_format
                        ))
                file_exists = os.path.isfile(fname)
                if file_exists: 
                    time.sleep(np.random.randint(30))
            # write empty file
            open(fname, "w").close()
            self.full_fname = fname
        print("Pickler initialized: %s" % self.full_fname)

    def write_pickle(self, pickle_dict):
        print("Writing to %s" % self.full_fname)
        if not os.path.exists(os.path.dirname(self.full_fname)):
            #os.system('mkdir -p %s' % os.path.dirname(self.full_fname))
            os.makedirs(os.path.dirname(self.full_fname))
        if os.path.exists(self.full_fname):
            os.remove(self.full_fname)
        #os.system('touch %s' % self.full_fname)
        if self.file_format == 'pickle':
            pickle.dump(pickle_dict, open(self.full_fname, "wb"))
        elif self.file_format == 'h5':
            import hickle
            hickle.dump(pickle_dict, self.full_fname, mode="w")

    def read_pickle(self):
        print("Reading %s" % self.full_fname)
        if self.file_format == 'pickle':
            return pickle.load(open(self.full_fname))
        elif self.file_format == 'h5':
            import hickle
            return hickle.load(self.full_fname)

    def delete_pickle_file(self):
        print("Remove %s" % self.full_fname)
        #os.system('rm %s' % self.full_fname)
        os.remove(self.full_fname)


