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


def approx_methods(ph, pr, expt, Nsamp, I, NI, w, c, method):
    
    Ir, NIr, wr = RejectSampling(NI, pr, Nsamp, ph)
    w4 = SelfNormalizedIS(w, pr)
    
    if method == "RejectSampling":    
        approx= np.matmul(wr,expt[Ir.flatten()].T)
        # approx= np.matmul(wr,expt.T)

    elif method == "SelfNormalized":            
        approx = np.matmul(w4, expt[Ir.flatten()].T)
        # approx= np.matmul(wr,expt.T)
    
    elif method == "RegressionEstimatorIS":  
        approx= RegressionEstimatorIS(pr, ph, expt/pr, c)
    
    return approx 
    