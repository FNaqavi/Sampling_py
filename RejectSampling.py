# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 19:56:44 2021

@author: naqavi
"""
import numpy as np


# def RejectSampling(*args):
def RejectSampling(g, f, N, gaccept):


    I = np.array(np.nonzero(g)).flatten()
    g = g[I]#.flatten()

    
    if len(gaccept) == 1240:
        gaccept = gaccept[I].flatten()
        
    gaccept = np.array(gaccept.groupby(level=0).first())
    f = np.array(f.groupby(level=0).first())
    
    g = g/np.sum(g)
    #nonzero_g_els = np.flatnonzero(g)
    fr = f/np.sum(f)
    # fr = fr.to_numpy()
    NI = np.zeros((len(g)))
    M = np.max(fr/gaccept)
    n = 0
    
    while(n < N):
        samples = np.random.choice(np.arange(len(g)), size=2*N, replace=True, p = g)    # draws the index
        r = M*np.random.random(2*N)              
        samples = samples[r*gaccept[samples] < fr[samples]]
        samples = samples[0:min(N-n, len(samples))]
        NI = NI + np.histogram(samples, len(g))[0]

        n = n + len(samples)

    w = NI/f / N

    return I, NI, w
