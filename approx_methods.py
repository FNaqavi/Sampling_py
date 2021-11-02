# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:21:39 2021

@author: naqavi
"""

import numpy as np
from SampleFromP import SampleFromP
from RejectSampling import RejectSampling
from SelfNormalizedIS import SelfNormalizedIS
from RegressionEstimatorIS import RegressionEstimatorIS


def approx_methods(ph, pr, expt, Nsamp, I, NI, w, c, method, phold, prold):
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
        approx= np.matmul(wr,expt[Ir.flatten()].T)
        # approx= np.matmul(wr,expt.T)

    elif method == "SelfNormalized":            
        # approx = np.matmul(w4, expt[Ir.flatten()].T)
        approx= np.matmul(w4,expt[Ir])
    
    elif method == "RegressionEstimatorIS":  
        approx= RegressionEstimatorIS(prold, phold, expt/pr, c)
    
    return approx 
    