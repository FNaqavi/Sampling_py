# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 12:54:54 2021

@author: naqavi
"""


import numpy as np


def RegressionEstimatorIS_scaper(p, q, f, c):  
    """
    Parameters
    ----------
    p : target function. control variate.
    q : sampling function. (p from homezone)
    f : function of interest.
    c : Index of sampled values from q.

    Returns
    -------
    importance sampling estimator of E_p[f].

    """
    # f = f[c]!
    w = p[c]/q[c].flatten()    # sampling weight, i.e., p/q where p is target distribution and q is
                               # distribution from which samples are drawn.              
    mw = np.mean(w)
    # b = (np.sum ((w-mw) * w *f )) / np.sum((w-mw)**2)
    weight = w * (1/len(c) - (((w-mw)*(mw-1)) / np.sum((w-mw)**2)))
    fn = f
    mu = np.matmul(fn, weight)          
    mu = mu 
    return mu 

