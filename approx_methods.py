# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:21:39 2021

@author: naqavi
"""

import numpy as np
from RejectSampling import RejectSampling
from SelfNormalizedIS import SelfNormalizedIS
from RegressionEstimatorIS import RegressionEstimatorIS
from RegressionEstimatorIS_scaper import RegressionEstimatorIS_scaper


def approx_methods(ph, pr, expt, Nsamp, I, NI, w, c, method, phold, prold, w_all,expt_all):
    """
    
    Parameters
    ----------
    ph : probability from homezone. Used for sampling locations.
    pr : probabilities used for updating corrections .
    expt : np.exp(vr) vr: utilities for updating corrections.
    Nsamp : number of sampled zones.
    I : list of indeces (contains 1240 indeces starting from 0 to 1239).
    NI : number of times each ph is selected (shape (1240,1)).
    w : weights from samplieFromP (ndarray of shape(1240,)).
    c : Index of sampled values.
    method : name of the method 

    Returns
    -------
    approx : weighted exp of (a float number for each method.)

    """
    Ir, NIr, wr = RejectSampling(NI, pr, Nsamp, ph)   # Ir is indeces of non-zero weights (wr)
    w4 = SelfNormalizedIS(w, pr)
    
    if method == "RejectSampling":    
        # approx= np.matmul(wr,expt[Ir.flatten()].T)
        # expt = np.array(expt[~expt.reset_index().duplicated().values])

        expt = np.array(expt.groupby(level=0).first())
        approx= np.matmul(wr,expt.T)

        # approx= np.matmul(wr,expt.T)

    elif method == "SelfNormalized":            
        # approx = np.matmul(w4, expt[Ir.flatten()].T)
        # approx= np.matmul(w4,expt[Ir])
        approx= np.matmul(w4,expt[np.unique(Ir)])

    
    elif method == "RegressionEstimatorIS": 
        prold = np.array(prold).reshape(-1,)     
        approx= RegressionEstimatorIS(prold, phold, expt_all/prold[c], c)
        
    elif method == 'base':
        pholdc = phold[c].reshape(-1,)
        approx = np.mean(expt_all/pholdc)
        
    elif method == "RegressionEstimatorIS_scaper": 
        prold = np.array(prold).reshape(-1,)     
        approx= RegressionEstimatorIS_scaper(prold, phold, expt_all/prold[c], c)
        
        
    return approx 
    