#!/usr/bin/env python
#
# Marcel Schmittfull 2018 (mschmittfull@gmail.com).
#
from __future__ import print_function,division

from collections import OrderedDict
import json

class TrfSpec:

    def __init__(self, 
                 linear_sources=None, # get trf fcns
                 fixed_linear_sources=[], # no trf fcns
                 non_orth_linear_sources=[], # get trf fcns, but never include in orthogonalization
                 field_to_smoothen_and_square=None,quadratic_sources=None,  # get trf fcns
                 field_to_smoothen_and_square2=None,quadratic_sources2=[],  # get trf fcns
                 sources_for_trf_fcn=None,
                 target_field=None,
                 target_spec=None,
                 export_bestfit_field=False,
                 save_bestfit_field=None):
        self.linear_sources = linear_sources
        self.fixed_linear_sources = fixed_linear_sources
        self.non_orth_linear_sources = non_orth_linear_sources
        self.field_to_smoothen_and_square = field_to_smoothen_and_square
        self.quadratic_sources = quadratic_sources

        # optional 2nd set of quadratic fields
        self.field_to_smoothen_and_square2 = field_to_smoothen_and_square2
        self.quadratic_sources2 = quadratic_sources2

        self.sources_for_trf_fcn = sources_for_trf_fcn
        self.target_field = target_field
        self.target_spec = target_spec
        self.export_bestfit_field = export_bestfit_field
        self.save_bestfit_field = save_bestfit_field

    def to_dict(self):
        mydict = OrderedDict()
        mydict['linear_sources'] = self.linear_sources
        mydict['fixed_linear_sources'] = getattr(self, 'fixed_linear_sources', None)
        if hasattr(self, 'non_orth_linear_sources'):
            mydict['non_orth_linear_sources'] = self.non_orth_linear_sources
        mydict['field_to_smoothen_and_square'] = self.field_to_smoothen_and_square
        if hasattr(self, 'field_to_smoothen_and_square2'):
            mydict['field_to_smoothen_and_square2'] = self.field_to_smoothen_and_square2
        else:
            mydict['field_to_smoothen_and_square2'] = None
        mydict['quadratic_sources'] = self.quadratic_sources
        if hasattr(self, 'quadratic_sources2'):
            mydict['quadratic_sources2'] = self.quadratic_sources2
        else:
            mydict['quadratic_sources2'] = None

        mydict['sources_for_trf_fcn'] = self.sources_for_trf_fcn

        mydict['target_field'] = self.target_field

        if hasattr(self, 'target_spec'):
            mydict['target_spec'] = str(getattr(self,'target_spec'))

        if hasattr(self, 'export_bestfit_field'):
            mydict['export_bestfit_field'] = getattr(self, 'export_bestfit_field')

        mydict['save_bestfit_field'] = self.save_bestfit_field
            
        return mydict

    def __str__(self):
        return json.dumps(self.to_dict())

    def __repr__(self):
        return self.__str__()



class TargetSpec(object):
    """
    Specify target field that's composed of several fields.
    """
    def __init__(self,
                 linear_target_contris=None,     # linear contributions to target field
                 minimization_objective=None,
                 target_norm=None,
                 save_bestfit_target_field=None
                ):
        self.linear_target_contris = linear_target_contris
        self.minimization_objective = minimization_objective
        self.target_norm = target_norm
        self.save_bestfit_target_field = save_bestfit_target_field

    def to_dict(self):
        mydict = OrderedDict()
        for k in ['linear_target_contris','minimization_objective',
                  'target_norm','save_bestfit_target_field']:
            if hasattr(self,k):
                mydict[k] = getattr(self,k)


    def __str__(self):
        return json.dumps(self.to_dict())

    def __repr__(self):
        return self.__str__()
