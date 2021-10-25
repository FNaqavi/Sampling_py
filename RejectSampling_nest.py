# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:25:28 2021

@author: naqavi
"""
import numpy as np
import pandas as pd

def RejectSampling_nest(g, f, N, gaccept):

    #    g = args[0]
    #    gaccept = args[-1]
    #    f = args[1]
    #    N = args[2]

    I = np.argwhere(g != 0)
    I = I.reshape((len(I),))
    g = g[I]
    gaccept = np.array([gaccept[i] for i in I.flat])
    idx_unique = pd.unique(f.index)
    f = f.loc[idx_unique]
    f = np.array([f[i] for i in I.flat])
    g = g/np.sum(g)
    nonzero_g_els = np.flatnonzero(g)
    fr = f/np.sum(f[nonzero_g_els])
    NI = np.zeros((len(g)))
    M = np.max(fr/gaccept)

    n = 0
    while(n < N):
        samples = np.random.choice(
            np.arange(len(g)), size=2*N, replace=True)    # draws the index
        r = M*np.random.random(2*N)
        samples = samples[r*gaccept[samples] < fr[samples]]
        samples = samples[0:min(N-n, len(samples))]
        NI = NI + np.histogram(samples, len(g))[0]

        n = n + len(samples)

    w = NI/f / N

    return I, NI, w