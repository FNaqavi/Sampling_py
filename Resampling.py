# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:34:00 2021

@author: naqavi
"""

import numpy as np
import pandas as pd
from Draw_zones import Draw_zones
from SampleFromP import SampleFromP
from approx_methods import approx_methods

times = pd.read_csv('carTimes_peak.csv',';', index_col=False, header=None)
theta_car = -0.06        
Nalt = 1240
hd_lst = []

(home, d) = Draw_zones(times)
print(home, d)

v_od = times*theta_car;
v = v_od[home][0:Nalt] # Utility to homezone. Sampling is done based on this utility
v_target = v_od[d][0:Nalt] # Utility on which log-sum is supposed to be calculated.  
expv = np.exp(v)
ph = expv/np.sum(expv) # Probability from homezone. Used for sampling locations (ph)
ph = ph.values.reshape(-1,1)
#pt = np.exp(v_target)/(np.sum(np.exp(v_target))) # Probability in zone d
nit = 100  # number of iterations? 
Nsamp = 200

# vr = v_target
a = 0.6
vr =  v_target*a + v*(1-a);
pr = np.exp(vr)/(np.sum(np.exp(vr))) # Probabilities used for updating corrections        
expt = np.exp(vr)
 
I, NI, w, c = SampleFromP(ph, Nsamp)
base_approx = np.matmul(w, expt[I].T)
print(f"base_approx={base_approx}")
print(f"sum_exp={sum(expt)}")
method1 = "RejectSampling"
method2 = "SelfNormalized"
method3 = "RegressionEstimatorIS"  ## I need to check this function: expt/pr is constant

#approx1 = logsum_methods(ph, pr, expt, Nsamp, I, NI, w, c, method1) # get approx in one iteration for one method


mask = (NI>0)
selected_zones = np.array(np.where(mask)).flatten()

# for selected_zones
# 0) Intialize EV in end time 

EV = [np.nan] * 11
EV.append( v_od[home][selected_zones])
# update ev in time t
for t in range(10,0,-1):
    approx = [] 
    for i in selected_zones:
        v_r = v_od[i]   
        expv_r = np.exp(v_r)
        pr_n = expv_r/np.sum(expv_r)
        v_n = v_od[i][selected_zones] + EV[t+1]
        expv_n = np.exp(v_n)
        ph_n = expv_n/np.sum(expv_n) # Probability from homezone. Used for sampling locations (ph)
        ph_n = ph_n[selected_zones]
        ph_n = ph_n.values.reshape(-1,1)
        expv_n = expv_n[selected_zones]
        lgsum = approx_methods(ph_n, pr_n[selected_zones], expv_n, Nsamp, I, NI, w[selected_zones], c, method2)
        approx.append(lgsum)
    EV[t] = np.log(approx)
    print(t)

EV.pop(0)

# for all zones 
# EV_all = [np.nan] * 11
# EV_all.append(v_od[home][0:1240])
# for t in range(10,0,-1):
#     approx = [] 
#     for i in range(1240):

#         v_n = v_od[i][0:1240] + EV_all[t+1]
#         expv_n = np.exp(v_n)
#         approx.append(lgsum)
#     EV_all[t] = np.log(approx)
#     print(t)

# EV_all.pop(0)





















# from Variance_reduction_methods import Variance_reduction_methods
# from Round_els import Round_els
# from clean_df import clean_df
# from plots import plots 
# import matplotlib.pyplot as plt



# hd_rng = 1    # number of random home dest zone pick
# rng = 11      # how many alphas ; see Get_a function
# for hd_rand in range(hd_rng):
#     (hd_lst ,approx) = Variance_reduction_methods(times, theta_car, Nalt, hd_lst, rng, home, d)
 


# def plot_weights(weights, wlabel):    
#     y  = np.array(weights)
#     x = range(len(weights))
#     plt.plot(x, y)   
#     m, b = np.polyfit(x, y, 1) 
#     plt.plot(x, m*x + b)
#     plt.title(label = wlabel)
#     return plt.show()

# plot_weights(wr, 'w_reject_sampling')
# plot_weights(ws, 'w_sample_from_p')
# plot_weights(w4s, 'w_self_normalized')
# plot_weights(w_regests, 'w_reg_est_IS')
    
          
# hd_lst = Round_els(hd_lst)
# (df_all, m1, m2, m3, m4) = clean_df(hd_rng, hd_lst, rng)

# for length_a in range(hd_rng):
#     df_seperated = df_all.iloc[:, 0+20*length_a:20+20*length_a] 
#     plots(df_seperated)


  
# u1 = theta_car * times
# p1 = np.exp(u1) / np.sum(np.exp(u1), axis = 1)
# ls = np.sum(np.log(p1), axis = 1)


