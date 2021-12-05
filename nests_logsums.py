# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 08:39:28 2021

@author: naqavi
"""


import numpy as np
import pandas as pd
from Draw_zones import Draw_zones
from SampleFromP import SampleFromP
from RejectSampling import RejectSampling
from RegressionEstimatorIS import RegressionEstimatorIS
from Get_a import Get_a

from RejectSampling_nest import RejectSampling_nest



def nests_logsums(times, theta_car, Nalt, hd_lst, rng):

    home, d = Draw_zones(times)
    v_od = times*theta_car;
    v = v_od[home][0:Nalt] # Utility to homezone. Sampling is done based on this utility
    v_target = v_od[d][0:Nalt] # Utility on which log-sum is supposed to be calculated.
    expv = np.exp(v)
    p_sample = expv/np.sum(expv) # Probability from homezone. Used for sampling locations (ph)
    p_sample = p_sample.values.reshape(len(p_sample),1)
    #pt = np.exp(v_target)/(np.sum(np.exp(v_target))) # Probability in zone d
 
    xx = []
    for i in range(rng):
        print(i)
        a = Get_a(i, rng)   
        vr = v_target*a + v*(1-a)
        pr = np.exp(vr)/(np.sum(np.exp(vr))) # Probabilities used for updating corrections        
        expt = np.exp(v_target)
        Nsamp = 200
        nit= 100   # number of iterations
        approx = np.zeros((nit,4))    
        #NIrsum=np.zeros((Nalt,))
        real = np.sum(np.exp(v_target))     
        
        I, NI, w, c = SampleFromP(p_sample, Nsamp)  # in every iteration it returns random draws
        
        u_samp = v_od.iloc[c][c]  # utilities of sampled zones to the sampled zones, size (Nsamp, Nsamp)
        pr1 = np.exp(u_samp) / np.sum(np.exp(u_samp), axis = 0)
        pr_n = pd.DataFrame(0, index = np.arange(1240), columns = np.arange(1240))
        pr1 = pr1.drop_duplicates()   # drop duplicate rows
        pr1 = pr1.loc[:,~pr1.columns.duplicated()]  # drop duplicate columns
        pr_n.loc[pr1.index, pr1.columns] = pr1       
  
        # I, NI, w, c = SampleFromP(pr1, Nsamp)  # in every iteration it returns random draws
        approx[i,0] = np.matmul(w, expt[I].values.reshape(len(expt[I]),1))
        # approx[k,0] = np.matmul(w, expt[I].T)
        
        # M = np.sum(pd.Series.multiply(w[np.flatnonzero(w)], pr1[np.flatnonzero(w)]))
        M = w.multiply(pr1)
        # #w4 = w[np.flatnonzero(w)]/M
        # div_wm = w / M
        # w4 = div_wm.dropna(axis = 1)
        # #approx[k,2] = np.matmul(w4, expt[Ir].values.reshape(len(expt[Ir]),1))
    
        # approx[i,3] = RegressionEstimatorIS(pr,p_sample,expt/pr,c)
 
    
        # Ns2 = Nsamp
        # Ir, NIr, wr = RejectSampling(NI, pr, Ns2, p_sample)
        # approx[i,1]= np.matmul(wr,expt[Ir].T)
        
          

        for j in range (approx.shape[1]):
            val = approx[:,j]
            me = np.mean(val)
            error = real - me
            no = np.sqrt(np.mean((real-val)**2))
            dev = np.sqrt(np.var(val)/(k))       
            tval = np.abs(error/dev)
            xx.append(np.stack((me, error, no, dev, tval, a)).flatten())
    hd_lst.append(xx)            
    return (hd_lst , approx)    