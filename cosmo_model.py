from __future__ import print_function, division


class CosmoModel:

    def __init__(self,
                 Om_L=None,
                 Om_m=None,
                 Om_K=None,
                 Om_r=None,
                 h0=None,
                 n_s=None,
                 m_nu=None,
                 fnl=None):
        self.Om_L = Om_L
        self.Om_m = Om_m
        self.Om_K = Om_K
        self.Om_r = Om_r
        self.h0 = h0
        self.n_s = n_s
        self.m_nu = m_nu
        self.fnl = fnl
