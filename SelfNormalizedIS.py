# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 20:37:51 2021

@author: naqavi
"""
import numpy as np

def SelfNormalizedIS(w, pr):

    pr = pr.to_numpy()   
    M = np.sum((w[np.flatnonzero(w)] * pr[np.flatnonzero(w)]))
    w4 = w[np.flatnonzero(w)]/M
    
    return w4