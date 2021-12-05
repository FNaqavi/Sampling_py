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
        probability from homezone. Used for sampling locations (ph)
    N : TYPE
        Number of samples 

    Returns
    -------
        I  index in p corresponding to NI
       NI number of times alterantive has been sampled
    """    

    p = p.reshape(-1,)
    edges = np.array(np.cumsum(p))
    edges = np.insert(edges,0,0)  #insert 0 at the 0th position of edges 
    s = edges[-1]
    eps = 2.2204e-16   # system precision?
    if np.abs(s - 1) > eps:
        edges = edges * (1 / s)
        
    # draw bins
    np.random.seed()
    rv = np.random.random(N)    
    c = np.histogram(rv, edges)[-2]   #edges are the bins in histogram
    ce = c[-1]
    if sum(c)<200:
        c[-1] = c[-1] + ce
       
    NI = c
    I = [x for x in range(0,len(p))]
    p = np.reshape(p, (1240,))
    w = NI/p / N
    
    xv = np.argwhere(c)
    if len(xv) == N:     # each value is sampled at most once
        x = xv
    else:                # some values are sampled more than once    
        # x=np.zeros((N,),dtype=int)   
        # count=0
        # for i in range(len(xv)):
        #     for j in range(NI[xv[i][0]]):
        #         x[count]=xv[i,0]
        #         count+=1
        x = np.repeat(xv, NI[xv].flatten())
                
    return I, NI, w, x.astype(int)


