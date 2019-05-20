from __future__ import print_function, division

import cPickle as pickle
from collections import OrderedDict
from django.utils.text import get_valid_filename
import numpy as np
import os
import re
from subprocess import call
import sys

try:
    from tinydb import TinyDB, Query
except:
    raise Exception("Could not find tinydb.\n" + "# To install tinydb, do\n" +
                    "pip install tinydb\n" +
                    "# or for a local installation, try\n" +
                    "pip install --root ~/software/python/ tinydb\n" +
                    "# and add ~/software/python/ to your PYTHONPATH.")


class PicklesDB(object):
    """
    A class for databases of pickle files. This is used to monitor
    all pickle files in a folder and do database operations on 
    the comp_key of the pickle files. For example, find all pickles
    that have certain options.
    """

    def __init__(self,
                 path=None,
                 fname_pattern=r'^main_calc_Perr.*.dill$',
                 comp_key='opts',
                 data_keys=['Pkmeas', 'Pkmeas_step'],
                 force_update=False):
        """
        Initialize an instance of PicklesDB class.

        Parameters
        ----------
        path : string
            Path to pickle files (this should be a folder).
        fname_pattern : string
            Regular expression for pickle files to be handled by this db.
        comp_key : string
            Key used to compare different pickle files.
        data_keys : sequence of strings
        force_update : boolean
            Force update of database by reading files from disk.
        """
        self.path = path
        self.fname_pattern = fname_pattern
        self.comp_key = comp_key
        if type(data_keys) == str:
            raise Exception(
                'data_keys must be a sequence of strings, not a string')
        self.data_keys = data_keys
        self.force_update = force_update

        # open the database
        self.db_fname = os.path.join(
            self.path, "%s.db" % get_valid_filename(self.fname_pattern))

        print("PicklesDB: Open %s" % self.db_fname)
        self.db = TinyDB(self.db_fname)

        # check if up to date, and update if necessary
        self.update()

        # Create a query
        self.query = Query()

        # test query
        #test_res = [
        #    entry['pickle_fname']
        #    for entry in self.db.search(self.query['Rsmooth'] == 10.)
        #]

    def update(self):

        if self.force_update or (not self.is_up_to_date()):

            print("PicklesDB: Delete empty pickle files...")
            call('find %s -name "*.pickle" -type f -empty -delete' % self.path,
                 shell=True)
            call('find %s -name "*.dill" -type f -empty -delete' % self.path,
                 shell=True)

            print("PicklesDB: Read all pickles to update db...")
            # loop over all files mathcing the pattern in the folder
            files_in_folder = []
            for fname in os.listdir(self.path):
                if re.search(self.fname_pattern, fname):
                    files_in_folder.append(fname)
            files_in_folder = sorted(files_in_folder)

            # insert all new files
            db_fnames = sorted(
                [entry['pickle_fname'] for entry in self.db.all()])
            for fname in files_in_folder:
                if fname not in db_fnames:
                    # load the pickle
                    full_fname = os.path.join(self.path, fname)
                    print("Try to load %s" % full_fname)
                    if fname.endswith('.pickle'):
                        # pickles before spring 2019 need old modules TrfSpec and Pktuple
                        print('WARNING: should use dill instead of pickle')
                        sys.path.append('/Users/mschmittfull/CODE/lsstools/')
                        from collections import namedtuple
                        Pktuple = namedtuple('Pktuple', [
                            'k', 'P', 'num_summands', 'info', 'info_id1',
                            'info_id2'
                        ])
                        p = pickle.load(open(full_fname))
                    elif fname.endswith('.dill'):
                        import dill
                        p = dill.load(open(full_fname))
                    else:
                        raise Exception('Invalid file ending: %s' % full_fname)
                    #print("keys:", p[(0.1,)].keys())
                    # get dict for comparisons
                    print("comp_dict keys:", p.keys())
                    comp_dict = p[self.comp_key]
                    if 'pickle_fname' not in comp_dict:
                        comp_dict['pickle_fname'] = fname
                    comp_dict = dict(comp_dict)
                    import ujson
                    print('type:', type(comp_dict))
                    self.db.insert(dict(comp_dict))

            # close and open to make sure it is written to disk
            self.db.close()
            self.db = TinyDB(self.db_fname)

        else:
            print("PicklesDB: db is up to date")

        print("PicklesDB: Have %d files" % len(self.db))

    def is_up_to_date(self):
        # all fnames in db
        db_fnames = sorted([entry['pickle_fname'] for entry in self.db.all()])

        # loop over all files mathcing the pattern in the folder
        files_in_folder = []
        for fname in os.listdir(self.path):
            if re.search(self.fname_pattern, fname):
                files_in_folder.append(fname)
        files_in_folder = sorted(files_in_folder)

        return db_fnames == files_in_folder

    def match_ref_dict(self, reference_dict, ignore_keys=[]):
        """
        Return a list of dictionaries representing all db entries that match a 
        reference dict.

        Example
        -------
        # Can run
        entries = pdb.match_ref_dict({'Rsmooth': 10., 'Niter': 4}, ignore_keys=['pickle_fname'])
        fnames = [e['pickle_fname'] for e in entries]
        # to get
        fnames: ['main_do_rec_2017_Jan_26_15:46:23_time1485445583.pickle']
        """
        # find a key that's not ignored
        keys = reference_dict.keys()
        key_counter = 0
        init_key = keys[key_counter]
        while init_key in ignore_keys:
            key_counter += 1
            init_key = keys[key_counter]
        if init_key in ignore_keys:
            raise Exception(
                "Could run match_ref_dict b/c all keys are ignored.")

        # construct the query
        # 1st entry
        myquery = (self.query[init_key] == reference_dict[init_key])
        for key in reference_dict.keys():
            if key in ignore_keys:
                continue
            if key == init_key:
                continue
            # combine query using "and"
            myquery = myquery & (self.query[key] == reference_dict[key])

        # run the query
        return self.db.search(myquery)

    def get_latest_pickle_fname_matching(self,
                                         reference_dict,
                                         ignore_keys=[],
                                         verbose=False):
        """
        Return latest 'pickle_fname' matching a reference_dict.
        """
        entries = self.match_ref_dict(reference_dict, ignore_keys=ignore_keys)
        fnames = [e['pickle_fname'] for e in entries]
        if len(fnames) == 0:
            print("DB ignores keys:", ignore_keys)
            raise Exception(
                "DB could not find pickle for reference_dict:\n%s\n" %
                str(reference_dict))
        if len(fnames) > 1:
            # sort by time and return latest
            fnames = PicklesDB.sort_pickle_fnames_by_time(fnames,
                                                          latest_last=True)
            print("Found multiple matching pickle files. Take latest from",
                  fnames)
        latest_fname = fnames[-1]
        if verbose:
            print("Latest pickle matching reference dict:\n%s\n%s" %
                  (latest_fname, str(reference_dict)))
        return latest_fname

    @staticmethod
    def sort_pickle_fnames_by_time(fnames, latest_last=True):
        # helper class to store time and name
        class TimeAndName(object):

            def __init__(self, time, name):
                self.time = time
                self.name = name

        # get time for each fname by parsing fname (must end with time....pickle)
        tnlist = []
        for f in fnames:
            match = re.search(r'time(\d+)\.dill$', f)
            if match:
                mytime = int(match.group(1))
                tnlist.append(TimeAndName(mytime, f))
            else:
                raise Exception(
                    "Pickle filenames must end with time\d+\.dill. Got %s" %
                    f)
        # sort the tnlist by time
        from operator import attrgetter
        tnlist = sorted(tnlist,
                        key=attrgetter('time'),
                        reverse=(not latest_last))
        sorted_fnames = [tn.name for tn in tnlist]
        return sorted_fnames

    def save_to_disk_and_close(self):
        self.db.close()


def main():
    # run some tests
    path = os.path.expandvars('$SCRATCH/lssbisp2013/psiRec/pickle/')

    pdb = PicklesDB(path=path,
                    fname_pattern=r'^main_do_rec.*.pickle$',
                    comp_key='opts',
                    data_keys=['Pkmeas', 'Pkmeas_step'],
                    force_update=False)

    # Test: find all fnames with some smoothing scale
    fnames = [
        entry['pickle_fname']
        for entry in pdb.db.search((pdb.query['Rsmooth'] == 10.) &
                                   (pdb.query['Riter_fac'] == 0.5))
    ]
    print("Test: fnames with R=10 and Riter_fac=0.5:", len(fnames))

    # Test: find all entries that match a reference dict
    # get the reference dict
    reference_dict = pdb.db.all()[0]
    # do the query
    entries = pdb.match_ref_dict(reference_dict, ignore_keys=['pickle_fname'])
    fnames = sorted([e['pickle_fname'] for e in entries])
    # check if result ok
    if reference_dict['pickle_fname'] in fnames:
        print("Test match_ref_dict: OK")
    else:
        print("Test match_ref_dict: FAILED")


if __name__ == '__main__':
    main()
