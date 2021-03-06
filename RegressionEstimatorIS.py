# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:26:30 2021

@author: naqavi
"""
import numpy as np


def RegressionEstimatorIS(p, q, f, c):  
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
    n = len(w)
    mw = np.mean(w)   
    b = (np.sum ((w-mw) * w *f )) / np.sum((w-mw)**2)
    #mf = np.mean(f)
    #b = np.cov(f,w)/np.var(w)
    #b = np.sum((f-mf)*(w-mw))/np.sum((w-mw)**2)  #this one is my personal effort and Oskar was not involved
    mu_1 =  1/n *  np.sum (w * f)          # what is this?
    mu_2 = 1/n *  np.sum (b * (w - 1))    # what is this?
    mu = mu_1 - mu_2
              
    return mu 