# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 18:59:10 2021

@author: naqavi
"""
import numpy as np


def SampleFromP(p, N):
    """
    Parameters
    ----------
    p : TYPE
        probability
    N : TYPE
        Number of samples

    Returns
    -------
        I  index in p corresponding to NI
       NI number of times alterantive has been sampled
    """    
    edges = np.array(np.cumsum(p))
    edges = np.insert(edges,0,0)
    s = edges[-1]
    eps = 2.2204e-16   # where does this come from?
    if np.abs(s - 1) > eps:
        edges = edges * (1 / s)
        
    # draw bins
    
    rv = np.random.random(N)    
    c = np.histogram(rv, edges)[-2]
    ce = c[-1]
    c[-1] = c[-1] + ce
    NI = c
    I = [x for x in range(0,len(p))]
    w = NI/p / N
    
    xv = np.argwhere(c)
    
    if len(xv) == N:     # each value is sampled at most once
        x = xv
    else:                # some values are sampled more than once    
        xc = []
        for i in range(len(xv)):
            xc.append(c[int(xv[i])])
        d = np.zeros((N,))
        dv = [np.diff(xv, axis = 0)][-1]
#        dv = np.insert(dv, 0, xv[0])
        dp = np.cumsum(xc)-1
        dp = dp[:-1]
        for i in range(len(dv)):          #d(dp) = dv
            idx = dp[i]
            d[idx] = dv[i]
            #print(i, dp, dv, dp.shape)
              
        x = np.cumsum(d)
    
    x = x[np.random.permutation(N)]
    
    return I, NI, w, x


    