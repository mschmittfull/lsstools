#!/usr/bin/env python
#
# Marcel Schmittfull 2017 (mschmittfull@gmail.com)
#
# Python script for BAO reconstruction.
#

from __future__ import print_function,division

import numpy as np
from collections import OrderedDict
from copy import deepcopy


def stack_dicts(dicts_list, skip_keys=[], cnvrt_innerlst_to_nparr=False):
    """
    Given a length-n list of dicts with the same keys, create a new dict
    with the same keys and values given by an array, where the 0th axis
    labels the index of the initial dicts_list.

    If the original value of a key is a scalar, then the value of that key
    in the output dict will be a shape (n,) numpy array.

    If the original value of a key is an array of shape (n_1,n_2,...,n_m),
    then the value of that key in the output dict will be a numpy array of
    shape (n, n_1, n_2, ..., n_m). Note that the 0th axis represents the 
    index of the original list.

    We work recursively down the original dicts, so it is fine
    if they are nested.

    Parameters
    ----------
    dicts_list : list of dicts
        Input list of dicts that will be stacked. The dicts must all have
        the same keys.
    skip_keys : list of strings, optional
        Keys to be skipped when forming the stacked dict.
    cnvrt_innerlst_to_nparr : boolean, optional
       If cnvrt_innerlst_to_nparr==True:
           If the original value is a list or a tuple, then it is first converted
           to a numpy array and then stacked on the 0th axis. Note that this
           does not work if the inner list contains dicts because they cannot
           be converted to numpy arrays. Code crashes in that case.
        If cnvrt_innerlst_to_nparr==False:
            If the original value is a list or a tuple, then stack_dicts is
            called on every entry of the list or tuple. This is default behavior.

    Returns
    -------
    stacked_dict : dict
        Dictionary with same values as each dict of the input list
        of dicts and values given by a stack of values of the input
        list of dicts (stacked on 0th axis).    
            
    Example
    -------
    # For example,
    dicts_list = [{'Pk': np.array([0,1,2])}, {'Pk': np.array([3,4,5])}]
    stacked_dict = stack_dicts(dicts_list)
    # gives stacked_dict = {'Pk': np.array([[0,1,2], [3,4,5]])}
    # so that stacked_dict['Pk'][0,:] = np.array([0,1,2]) and
    # stacked_dict['Pk'][1,:] = np.array([3,4,5]).
    # To get the average over the different list entries, do
    np.mean(stacked_dict['Pk'], axis=0)
    # which gives array([ 1.5,  2.5,  3.5]).
    """
    
    assert type(dicts_list) == list
    base_dict = dicts_list[0]

    # check all dicts have the same keys
    for d in dicts_list:
        if sorted(d.keys()) != sorted(base_dict.keys()):
            print(sorted(base_dict.keys()))
            print(sorted(d.keys()))
            raise Exception("All dicts of dicts_list must have the same keys.")
    
    stacked_dict = {}
    for key,value in base_dict.items():
        if key in skip_keys:
            continue
        
        if type(value) in [list, tuple]:
            if cnvrt_innerlst_to_nparr:
                # try to convert list to a numpy array and then stack that.
                try:
                    values_list = [ np.array(d[key]) for d in dicts_list ]
                    # stack the values on the 0th axis
                    stacked_dict[key] = np.stack(values_list, axis=0)
                except:
                    raise Exception("Failed to convert list to np array.")
            else:
                # for every entry, call stack_dicts
                stacked_dict[key] = []
                for counter, entry in enumerate(value):
                    tmp_dicts_list = [ {'tmp': d[key][counter]} for d in dicts_list ]
                    stacked_dict[key].append( stack_dicts(
                        tmp_dicts_list, 
                        cnvrt_innerlst_to_nparr=cnvrt_innerlst_to_nparr)['tmp'] )
        
        elif type(value) == np.ndarray:
            # list of arrays.
            values_list = [ d[key] for d in dicts_list ]
            # stack the arrays on new 0th axis
            stacked_dict[key] = np.stack(values_list, axis=0)
            
        elif np.isscalar(value):
            # create 1d array from scalar
            values_list = [ np.array([d[key],]) for d in dicts_list ]
            # stack the values on the 0th axis
            stacked_dict[key] = np.stack(values_list, axis=0)

        elif type(value) in [dict, OrderedDict]:
            # recursively call stack_dicts on inner list of dicts
            values_list = [ d[key] for d in dicts_list ]
            stacked_dict[key] = stack_dicts(values_list, skip_keys=skip_keys)

        elif value is None:
            # added on 16 jan 2018; not sure if correct
            values_list = [ d[key] for d in dicts_list ]
            stacked_dict[key] = values_list
            
        elif isinstance(value, object):
            # convert to dict and then do the same as for dicts
            try:
                values_list = [ vars(d[key]) for d in dicts_list ]
            except:
                print("value:", value)
                print("type(value):", type(value))
                print("Error when trying to get values list of:")
                print(dicts_list)
                values_list = [ vars(d[key]) for d in dicts_list ]
            stacked_dict[key] = stack_dicts(values_list, skip_keys=skip_keys)
            
        else:
            raise Exception("Value of key %s has invalid type %s" % (key,str(type(value))))

    return stacked_dict


def fill_placeholder_strings(in_dict):
    """
    - Make a copy of in_dict.
    - For each dict value that's a string which contains [[...]], replace
    [[...]] by value of key given in [[...]]. 
    - If value is a dict, go down recursively.
    - Currently only support replacement of a single occurence of [[...]] within
      each string (could improve using re.findall).

    Example:
    If 

      in_dict = {'fname': "my_file_[['ssseed']].txt",
                 'ssseed': 100}

    return
    
      out_dict = {'fname': "my_file_100.txt",
                 'ssseed': 100}
    """

    raise Exception("not fully implemented; use symlinks instead")
    
    from copy import deepcopy
    import re
    out_dict = deepcopy(in_dict)

    for k in out_dict.keys():
        if type(out_dict[k]) == str:
            while ('[[' in out_dict[k]) and (']]' in out_dict[k]):
                # find and replace
                # .*? is non-greedy, any letter
                match = re.search(r'\[\[(.*?)\]\]', out_dict[k])
                lookup_key = match.group(1)
                replace_val = str(out_dict[lookup_key])
                print("Val of key %s before fill: " % k, out_dict[k])
                out_dict[k] = re.sub(r'\[\[(.*?)\]\]', replace_val, out_dict[k])
                print("Val of key %s after fill: " % k, out_dict[k])
        elif type(out_dict[k]) == dict:
            # recursively go in dict
            out_dict[k] = fill_placeholder_strings(out_dict[k])
        elif type(out_dict[k]) == list:
            #TODOOOO : deal with list of dicts
            #ACTUALLY: SHOULD JUST CREATE SUBFOLDER SSSEED... and link subsamples to there
            pass

    print("dict after filling place holders:")
    print(out_dict)

    raise Exception("dbg fill_placeholder_strings")

    return out_dict


def nested_dict_update(in_dict, change_list=None):
    """
    Update dictionary by a list of changes, where each change consists of a 
    key tuple (so can do nested dicts), and a value to change to. Examples:

      change_list = [(('stage0','sim_seed'),400), (('stage0','Rsmooth'),5.0)]
      change_list = [(('noisefac_CMBlensing',),1.0), (('noisefac_CMBlensing',),2.0)]

    Note: We do not change in_dict, but return new dict.
    """

    out_dict = deepcopy(in_dict)
    if change_list not in [[], (), None]:
        print("dict change_list:", change_list)
        for change in change_list:
            assert len(change) == 2
            param_name_tuple, param_val = change[0], change[1]
            if len(param_name_tuple)==1:
                out_dict[param_name_tuple[0]] = param_val
            elif len(param_name_tuple)==2:
                out_dict[param_name_tuple[0]][param_name_tuple[1]] = param_val
            elif len(param_name_tuple)==3:
                out_dict[param_name_tuple[0]][param_name_tuple[1]][param_name_tuple[2]] = param_val
            else:
                raise Exception("Found tuple in change_list key with invalid size: %s" 
                                % str(len(param_name_tuple)))
    return out_dict

def nested_dict_get(in_dict, nested_key, default=None):
    """
    If nested_key is a non-string list or tuple:
        Get dict[nested_key[0]][nested_key[1]]...
    elif nested_key is a string:
        Get dict[nested_key]
    Otherwise get dict[nested_key]
    """
    if (type(nested_key) in [list,tuple]) and (not type(nested_key) == str):
        # nested_key is a non-string list
        if len(nested_key) == 1:
            return in_dict.get(nested_key[0], default)
        elif len(nested_key) == 2:
            if in_dict.has_key(nested_key[0]) and in_dict[nested_key[0]].has_key(nested_key[1]):
                return in_dict[nested_key[0]][nested_key[1]]
            else:
                return default
        elif len(nested_key) == 3:
            if (in_dict.has_key(nested_key[0])
                and in_dict[nested_key[0]].has_key(nested_key[1])
                and in_dict[nested_key[0]][nested_key[1]].has_key(nested_key[2])):
                return in_dict[nested_key[0]][nested_key[1]][nested_key[2]]
            else:
                return default
        else:
            raise Exception("nested_dict_get with nesting depth %d not implemented" % (
                len(nested_key)))
    else:
        return in_dict.get(nested_key, default)
        

def change_list_to_dict(change_list):
    """
    Converts a change_list to a dictionary.
    """
    change_dict = OrderedDict()
    if change_list not in [[], None]:
        for change in change_list:
            assert len(change)==2
            param_name_tuple, param_val = change[0], change[1]
            if len(param_name_tuple)==1:
                change_dict[param_name_tuple[0]] = param_val
            elif len(param_name_tuple)==2:
                if param_name_tuple[0] not in change_dict:
                    change_dict[param_name_tuple[0]] = OrderedDict()
                change_dict[param_name_tuple[0]][param_name_tuple[1]] = param_val
            else:
                raise Exception("Found tuple in change_list key with invalid size: %s" 
                                % str(len(param_name_tuple)))
    return change_dict        


def merge_dicts(dict1, dict2):
    out = dict1.copy()
    out.update(dict2)
    return out


