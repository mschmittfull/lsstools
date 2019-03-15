#!/usr/bin/env python

from __future__ import print_function,division
from collections import namedtuple

# simple named tuple to store output of power spectrum measurements
Pktuple = namedtuple('Pktuple', ['k', 'P', 'num_summands', 'info', 'info_id1', 'info_id2'])

