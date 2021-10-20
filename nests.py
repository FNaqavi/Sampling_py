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


def Variance_reduction_methods(times, theta_car, Nalt, hd_lst, rng):
    home, d = Draw_zones(times)
    v_od = times*theta_car;
    v = v_od[home][0:Nalt] # Utility to homezone. Sampling is done based on this utility
    v_target = v_od[d][0:Nalt] # Utility on which log-sum is supposed to be calculated.
    expv = np.exp(v)
    p_sample = expv/np.sum(expv) # Probability from homezone. Used for sampling locations (ph)
    #pt = np.exp(v_target)/(np.sum(np.exp(v_target))) # Probability in zone d
 
    xx = []
        
    for i in range(rng):
        print(i)
        a = Get_a(i, rng)   
        vr = v_target*a + v*(1-a)
        pr = np.exp(vr)/(np.sum(np.exp(vr))) # Probabilities used for updating corrections        
        expt = np.exp(v_target)
        Nsamp = 200
        nit=100   # number of iterations? 
        approx = np.zeros((nit,4))    
        #logsum = np.zeros((nit,4))
        #NIrsum=np.zeros((Nalt,))
        real = np.sum(np.exp(v_target))
        
        for k in range(nit):
            
            I, NI, w, c = SampleFromP(p_sample,Nsamp)
            #logsum[k,0] = np.sum(np.log(expt[I]))
            approx[k,0] = np.matmul(w, expt[I].T)
                
            Ns2=Nsamp
            Ir, NIr, wr = RejectSampling(NI,pr,Ns2,p_sample)
            #logsum[k,1] = np.sum(np.log(expt[Ir]))
            approx[k,1]= np.matmul(wr,expt[Ir].T)
              
            M = np.sum(pd.Series.multiply(w[np.flatnonzero(w)], pr[np.flatnonzero(w)]))
            w4 = w[np.flatnonzero(w)]/M
            #logsum[k,2] = np.sum(np.log(expt[Ir]))
            approx[k,2] = np.matmul(w4, expt[Ir].T)
        
            approx[k,3] = RegressionEstimatorIS(pr,p_sample,expt/pr,c)
            
            ### lower nest
            
            # d1 = Draw_zones(times)[0]
            # v1 = v_od[d][0:Nalt] # Utility to firs selected zone. Sampling is done based on this utility
            # v_target1 = v_od[d1][0:Nalt] # Utility on which log-sum is supposed to be calculated.
            # expv1 = np.exp(v1)
            # p_sample1 = expv1/np.sum(expv1) # Probability from homezone. Used for sampling locations (ph)
            # pt1 = np.exp(v_target1)/(np.sum(np.exp(v_target1))) # Probability in zone d
            
            # rng1 = 1    # how many trial a; see Get_a function
            # for second_a in range(rng1):
            #     print(second_a)
            #     a1 = Get_a(second_a, rng1)   
            #     vr1 = v_target1*a1 + v1*(1-a1)
            #     pr1 = np.exp(vr1)/(np.sum(np.exp(vr1))) # Probabilities used for updating corrections        
            #     expt1 = np.exp(v_target1)
            #     Nsamp1 = 200
            #     nit1=100   # number of iterations?    
            #     logsum1 = np.zeros((nit1,4))
            #     NIrsum1 = np.zeros((Nalt1,))
            #     real1 = np.sum(np.exp(v_target1))
                
            #     for k1 in range(nit):
        
            #     I, NI, w, c = SampleFromP(p_sample1,Nsamp1)
            #     #logsum[k1,0] = np.sum(np.log(expt1[I]))
                   
            #     Ns2_1=Nsamp1
            #     Ir, NIr, wr = RejectSampling(NI,pr,Ns2_1,p_sample1)
            #     #logsum1[k1,1] = np.sum(np.log(expt1[Ir]))
                  
            #     M = np.sum(pd.Series.multiply(w[np.flatnonzero(w)], pr[np.flatnonzero(w)]))
            #     w4 = w[np.flatnonzero(w)]/M
            #     #logsum1[k1,2] = np.sum(np.log(expt1[Ir]))
            
            #     approx[k1,3] = RegressionEstimatorIS(pr1,p_sample1,expt1/pr1,c)
            

        
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