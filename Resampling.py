# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:34:00 2021

@author: naqavi
"""

# %%
import pandas as pd
import numpy as np
from SampleFromP import SampleFromP
from RejectSampling import RejectSampling
from RegressionEstimatorIS import RegressionEstimatorIS

times = pd.read_csv('carTimes_peak.csv',';', index_col=False, header=None)
theta_car = -0.1        # what is this?
Nalt = 1240
v_od = times*theta_car;
home = 3 # Zone used for sampling locations
d=400 # Zone where we want to calculate log-sum

#%%

v = v_od[home][0:Nalt] # Utility to homezone. Sampling is done based on this utility
v_target = v_od[d][0:Nalt] # Utility on which log-sum is supposed to be calculated.
expv = np.exp(v)
p_sample = expv/np.sum(expv) # Probability from homezone. Used for sampling locations (ph)

pt = np.exp(v_target)/(np.sum(np.exp(v_target))) # Probability in zone d

a = 0.5 # If a is 1, we use the probabilites in d. Should give very good approximation of logsums
        # If a is 0, we don't gain any information. Should give no improvement
vr = v_target*a + v*(1-a)
pr = np.exp(vr)/(np.sum(np.exp(vr))) # Probabilities used for updating corrections

expt = np.exp(v_target)

#%%
Nsamp = 200
nit=100;
approx = np.zeros((nit,4))    
NIrsum=np.zeros((Nalt,))
real = np.sum(np.exp(v_target))

for k in range(nit):
    
    I, NI, w, c = SampleFromP(p_sample,Nsamp)
    approx[k,0] = np.matmul(w, expt[I].T)
    
    Ns2=Nsamp
    Ir, NIr, wr = RejectSampling(NI,pr,Ns2,p_sample)
    approx[k,1]= np.matmul(wr,expt[Ir].T)
      
    qi = NI[np.flatnonzero(w)]/Nsamp
    M = np.sum(pd.Series.multiply(w[np.flatnonzero(w)], pr[np.flatnonzero(w)]))
    # w4 is equal to wr as Ns2 goes to infinity when 
    # [Ir,NIr,wr]=RejectSampling(NI,pr,Ns2,ph);
    w4 = w[np.flatnonzero(w)]/M
    approx[k,2] = np.matmul(w4, expt[Ir].T)
    
    #w6 = w * Nsamp / sum(pr[c]/p_sample[c])
    #approx[k,3] = np.matmul(w6, expt.T)
    
    approx[k,3] = RegressionEstimatorIS(pr,p_sample,expt/pr,c)

for i in range (approx.shape[1]):
    val = approx[:,i]
    me = np.mean(val)
    error = real - me
    no = np.sqrt(np.mean((real-val)**2))
    dev = np.sqrt(np.var(val)/k)
    tval = np.abs(error/dev)
    print(i, "   ", Nsamp,"   ", "%.2f" % real,"   ",  "%.2f" % me,"   ", "%.2f" % np.abs(error),"   ", "%.2f" % dev,"   ", "%.2f" % tval, "   ","%.2f" % no,"   ", "%.2f" % np.sqrt(np.var(val)))   
print("i    Nsamp    real       mean  abs(error)   dev     tval     no     sqrt(np.var(val)) ")
      
    
    
    





    
# %%
