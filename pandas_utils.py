from __future__ import print_function, division

from astropy.table import Table
from collections import OrderedDict
import numpy as np
import pandas as pd

def convert_table_to_df(table):
    """
    Convert from astropy table to pandas df.
    pandas cannot handle multi-dimensional columns, so remove them.
    """
    keep = []
    for colval, col in zip(table.columns.values(), table.dtype.names):
        if getattr(colval, 'ndim', 1) > 1:
            print('remove column %s: pandas cannot handle multi-dimensional columns' % col)
        else:
            keep.append(col)
    print('Columns kept:')
    print(keep)
    return table[keep].to_pandas()

def convert_catalog_to_df(cat):
    data_dict = OrderedDict()
    for col in cat.columns:
        data_dict[col] = cat[col].compute()
    data_table = Table(data_dict)
    del data_dict
    return convert_table_to_df(data_table)
