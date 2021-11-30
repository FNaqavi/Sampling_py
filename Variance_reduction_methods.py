# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:55:24 2021

@author: naqavi
"""

import numpy as np
import pandas as pd
from Draw_zones import Draw_zones
from SampleFromP import SampleFromP
from RejectSampling import RejectSampling
from RegressionEstimatorIS import RegressionEstimatorIS
from Get_a import Get_a


def Variance_reduction_methods(times, theta_car, Nalt, hd_lst):
    home, d = Draw_zones(times)
    print(home, d)
    v_od = times*theta_car;
    v = v_od[home][0:Nalt] # Utility to homezone. Sampling is done based on this utility
    v_target = v_od[d][0:Nalt] # Utility on which log-sum is supposed to be calculated.
    expv = np.exp(v)
    p_sample = expv/np.sum(expv) # Probability from homezone. Used for sampling locations (ph)
    pt = np.exp(v_target)/(np.sum(np.exp(v_target))) # Probability in zone d
    
    xx = []
    rng = 11    # how many trial a; see Get_a function
    for i in range(rng):
        print(i)
        a = Get_a(i, rng)   
        vr = v_target*a + v*(1-a)
        pr = np.exp(vr)/(np.sum(np.exp(vr))) # Probabilities used for updating corrections        
        expt = np.exp(v_target)
        Nsamp = 200
        nit=100;
        approx = np.zeros((nit,4))    
        NIrsum=np.zeros((Nalt,))
        real = np.sum(np.exp(v_target))
        
        for k in range(nit):
            
            I, NI, w, c = SampleFromP(p_sample,Nsamp)
            approx[k,0] = np.matmul(w, expt[I].T)
       
            Ir, NIr, wr = RejectSampling(NI,pr,Nsamp,p_sample)
            approx[k,1]= np.matmul(wr,expt[Ir].T)
              
            M = np.sum(pd.Series.multiply(w[np.flatnonzero(w)], pr[np.flatnonzero(w)]))
            w4 = w[np.flatnonzero(w)]/M
            approx[k,2] = np.matmul(w4, expt[Ir].T)
        
            approx[k,3] = RegressionEstimatorIS(pr,p_sample,expt/pr,c)
        
        for j in range (approx.shape[1]):
            val = approx[:,j]
            me = np.mean(val)
            error = real - me
            no = np.sqrt(np.mean((real-val)**2))
            dev = np.sqrt(np.var(val)/(k))       
            tval = np.abs(error/dev)
            xx.append(np.stack((me, error, no, dev, tval, a)).flatten())
    hd_lst.append(xx)            
    return hd_lst     