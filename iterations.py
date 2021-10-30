# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:58:16 2021

@author: naqavi
"""

import numpy as np
import pandas as pd
from Get_a import Get_a
from logsum_methods import logsum_methods
from SampleFromP import SampleFromP

def iterations(times, theta_car, Nalt, hd_lst, rng, home, d):

    v_od = times*theta_car;
    v = v_od[home][0:Nalt] # Utility to homezone. Sampling is done based on this utility
    v_target = v_od[d][0:Nalt] # Utility on which log-sum is supposed to be calculated.
    expv = np.exp(v)
    ph = expv/np.sum(expv) # Probability from homezone. Used for sampling locations (ph)
    ph = ph.values.reshape(-1,1)
    #pt = np.exp(v_target)/(np.sum(np.exp(v_target))) # Probability in zone d
    nit = 100  # number of iterations? 
    Nsamp = 200
    xx = []
    approx = np.zeros((nit,4)) 
    wrs = []
    ws = []
    w4s = []
    w_regests = []
    for i in range(rng):
        print(i)
        a = Get_a(i, rng)   
        vr = v_target*a + v*(1-a)
        pr = np.exp(vr)/(np.sum(np.exp(vr))) # Probabilities used for updating corrections        
        expt = np.exp(v_target)

        real = np.sum(np.exp(v_target))
        
        
        
        for k in range(nit):
            approx, wr, w4, w, w_regest = logsum_methods(ph, pr, expt, Nsamp, approx, k)
            wrs.append(wr)
            w4s.append(w4)
            ws.append(w)
            w_regests.append(w_regest)
            
        
        for j in range (approx.shape[1]):
            val = approx[:,j]
            me = np.mean(val)
            error = real - me
            no = np.sqrt(np.mean((real-val)**2))
            dev = np.sqrt(np.var(val)/(nit))  
            tval = np.abs(error/dev)
            xx.append(np.stack((me, error, no, dev, tval, a)).flatten())

    hd_lst.append(xx)            
    return (hd_lst , approx, wrs, w4s, ws, w_regests)    

