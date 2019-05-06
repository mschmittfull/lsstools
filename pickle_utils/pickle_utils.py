from __future__ import print_function,division

import os
import sys
import cPickle as pickle
import time
import copy
from collections import OrderedDict

# MS packages
from lsstools import dict_utils
from lsstools import Pickler
import PicklesDB

def get_stacked_pickles(pickle_path, base_fname, 
                        outer_key_name='sim_wig_now_string',
                        outer_key_values=['wig','now'],
                        comp_key='opts',
                        vary_key='sim_seed',
                        vary_values=None,
                        sim_seeds=[],
                        ignore_pickle_keys=['pickle_fname'],
                        fname_pattern=r'^main_do_rec.*.pickle$',
                        skip_keys_when_stacking=['opts'],
                        return_base_pickle=False,
                        return_base_pickle_opts=False,
                        call_wig_now_example=True,
                        verbose=True, dbg=True, get_in_path=None):
    """
    Get pickles stacked over realizations represented by sim_seeds.
    Or more generally, change the value of vary_key to vary_values entries.

    If vary_key=='sim_seed' and vary_values==None, use vary_values=sim_seeds.
    """
    # check args
    if vary_values is not None:
        assert sim_seeds == []

    if (vary_key == 'sim_seed') and (vary_values is None):
        vary_values = sim_seeds

    if get_in_path is None:
        # had this in same folder previously
        from perr.path_utils import get_in_path
    
    # PicklesDB instance to load options of all pickles needed for our fit
    pdb = PicklesDB.PicklesDB(
        path=pickle_path, fname_pattern=fname_pattern,
        comp_key=comp_key, data_keys=['Pkmeas', 'Pkmeas_step'],
        force_update=False)
    
    # Get the entry of the base_fname
    print("Search for ", base_fname)
    entries = pdb.db.search(pdb.query['pickle_fname'] == base_fname)
    print("entries:", entries)
    assert len(entries) == 1
    base_pickle_opts = entries[0]

    if verbose:
        print("base_pickle_opts:", base_pickle_opts)

    # if outer key is not used, still run over loop once
    if outer_key_name is None:
        outer_key_name = '__NO_OUTER_KEY__'
        outer_key_values = [None]
        
    # Get pickles for all realizations and stack them.
    # For arrays, add new 0st axis labelling realization.
    stacked_pickles = {}
    for outer_key_value in outer_key_values:
        pickles_list = []
        # load all pickles
        for vary_val in vary_values:
            reference_dict = copy.deepcopy(base_pickle_opts)
            if type(vary_key)==str:
                reference_dict[vary_key] = vary_val
            elif type(vary_key)==tuple:
                # nested key
                change_lst = [(vary_key, vary_val)]
                reference_dict = dict_utils.nested_dict_update(reference_dict, change_lst)
            else:
                raise Exception("vary_key must be string or tuple, found %s" % str(vary_key))

            if call_wig_now_example and outer_key_name!='__NO_OUTER_KEY__':
                reference_dict[outer_key_name] = outer_key_value
            if reference_dict.has_key('sim_name'):
                reference_dict['in_path'] = get_in_path(reference_dict)
            try:
                fname = pdb.get_latest_pickle_fname_matching(
                    reference_dict, ignore_keys=ignore_pickle_keys)
            except:
                print("Could not find pickle with %s=%s and %s=%s" % (
                    outer_key_name, str(outer_key_value), 
                    vary_key, str(vary_val)))
                pdb.get_latest_pickle_fname_matching(
                    reference_dict, ignore_keys=ignore_pickle_keys)
            full_fname = os.path.join(pickle_path, fname)
            pickles_list.append( 
                Pickler.Pickler(full_fname=full_fname).read_pickle())
        # stack pickles
        if outer_key_name == '__NO_OUTER_KEY__':
            stacked_pickles = dict_utils.stack_dicts(
                pickles_list, skip_keys = skip_keys_when_stacking)
        else:
            stacked_pickles[outer_key_value] = dict_utils.stack_dicts(
                pickles_list, skip_keys = skip_keys_when_stacking)


    if call_wig_now_example:
        # Print example: Shape is (N_realizations, N_k_bins)
        print("Stacked power shape:", 
              stacked_pickles['wig']['Pkmeas'][(
                  'deltalin_unsmoothed','deltalin_unsmoothed')][1].shape)

        # copy over power spectra from final iteration step
        for outer_key_value in outer_key_values:
            last_step_key = stacked_pickles[outer_key_value]['Pkmeas_step'].keys()[-1]
            for Pk_key in stacked_pickles[outer_key_value]['Pkmeas_step'][last_step_key].keys():
                stacked_pickles[outer_key_value]['Pkmeas'][Pk_key] = (
                    stacked_pickles[outer_key_value]['Pkmeas_step'][last_step_key][Pk_key])

    # base_pickle corresponds to first entry in vary_values or sim_seeds
    base_pickle = pickles_list[0]

    # keep this for backwards compatibility...
    if return_base_pickle and return_base_pickle_opts:
        return stacked_pickles, base_pickle, base_pickle_opts
    elif return_base_pickle and (not return_base_pickle_opts):
        return stacked_pickles, base_pickle
    elif (not return_base_pickle) and return_base_pickle_opts:
        return stacked_pickles, base_pickle_opts
    else:
        return stacked_pickles




def get_stacked_pickles_for_varying_base_opts(
        pickle_path, base_fname, 
        comp_key='opts',
        base_param_names_and_values=None,
        stack_key='sim_seed',
        stack_values=None,
        ignore_pickle_keys=['pickle_fname'],
        fname_pattern=r'^main_do_rec.*.pickle$',
        skip_keys_when_stacking=['opts'],
        return_base_vp_pickles=False,
        return_base_vp_pickle_opts=False,
        return_base_vp_pickle_fnames=False,
        verbose=True, dbg=True, get_in_path=None):
    """
    For each entry base_param_names_values, stack stack_key over stack_values.
    """

    if get_in_path is None:
        # had this in same folder previously
        from path_utils import get_in_path

    # PicklesDB instance to load options of all pickles needed
    pdb = PicklesDB.PicklesDB(
        path=pickle_path, fname_pattern=fname_pattern,
        comp_key=comp_key, data_keys=['Pkmeas', 'Pkmeas_step'],
        force_update=False)
    
    # Get the entry of the base_fname
    print("Search for ", base_fname)
    entries = pdb.db.search(pdb.query['pickle_fname'] == base_fname)
    print("entries:", entries)
    assert len(entries) == 1
    base_fname_pickle_opts = entries[0]

    if verbose:
        print("base_fname_pickle_opts:", base_fname_pickle_opts)

    # Loop over base_param_names_values, and
    # get pickles for all stack_values and stack them.
    # For arrays, add new 0st axis labelling stack index.
    stacked_vp_pickles = OrderedDict()
    base_vp_pickles = OrderedDict()
    base_vp_pickle_opts = OrderedDict()
    base_vp_pickle_fnames = OrderedDict()
    for pnames_and_values in base_param_names_and_values:
        print("pnames_and_values:", pnames_and_values)

        # update opts of base_fname by pnames_and_values
        base_reference_dict = dict_utils.nested_dict_update(base_fname_pickle_opts, pnames_and_values)

        # stack pickles
        pickles_list = []
        # load all pickles
        for istack, stack_val in enumerate(stack_values):
            reference_dict = copy.deepcopy(base_reference_dict)
            if type(stack_key)==str:
                reference_dict[stack_key] = stack_val
            elif type(stack_key)==tuple:
                # nested key
                change_lst = [(stack_key, stack_val)]
                reference_dict = dict_utils.nested_dict_update(reference_dict, change_lst)
            else:
                raise Exception("stack_key must be string or tuple, found %s" % str(vary_key))

            if reference_dict.has_key('sim_name'):
                reference_dict['in_path'] = get_in_path(reference_dict)
            try:
                fname = pdb.get_latest_pickle_fname_matching(
                    reference_dict, ignore_keys=ignore_pickle_keys)
            except:
                print("Could not find pickle with %s, and %s=%s" % (
                    str(pnames_and_values),
                    str(stack_key), str(stack_val)))
                pdb.get_latest_pickle_fname_matching(
                    reference_dict, ignore_keys=ignore_pickle_keys)
            if istack == 0:
                base_vp_pickle_fname = fname
            full_fname = os.path.join(pickle_path, fname)
            pickles_list.append( 
                Pickler.Pickler(full_fname=full_fname).read_pickle())
        # stack pickles
        stacked_vp_pickles[pnames_and_values] = dict_utils.stack_dicts(
            pickles_list, skip_keys = skip_keys_when_stacking)

        # base_pickle corresponds to first entry in vary_values or sim_seeds
        base_vp_pickles[pnames_and_values] = pickles_list[0]

        # this does not have updated pickle_fname...
        base_vp_pickle_opts[pnames_and_values] = copy.deepcopy(base_reference_dict)

        # ...so save it here
        base_vp_pickle_fnames[pnames_and_values] = base_vp_pickle_fname

    # return, keep this for backwards compatibility...
    if return_base_vp_pickle_fnames:
        assert return_base_vp_pickles
        assert return_base_vp_pickle_opts
        return stacked_vp_pickles, base_vp_pickles, base_vp_pickle_opts, base_vp_pickle_fnames
    else:
        # keep this for backwards compatibility...
        if return_base_vp_pickles and return_base_vp_pickle_opts:
            return stacked_vp_pickles, base_vp_pickles, base_vp_pickle_opts
        elif return_base_vp_pickle and (not return_base_vp_pickle_opts):
            return stacked_vp_pickles, base_vp_pickle
        elif (not return_base_vp_pickle) and return_base_vp_pickle_opts:
            return stacked_vp_pickles, base_vp_pickle_opts
        else:
            return stacked_vp_pickles

    
