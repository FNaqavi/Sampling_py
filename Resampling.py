# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:34:00 2021

@author: naqavi
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Draw_zones import Draw_zones
from SampleFromP import SampleFromP
from approx_methods import approx_methods


times = pd.read_csv('carTimes_peak.csv',';', index_col=False, header=None)
theta_car = -0.06      
theta_trip=-6.12
Nalt = 1240
Tmax = 20
hd_lst = []

(home, d) = Draw_zones(times)
print(home, d)

# home = 1233
# d = 721
# home = 212
# d = 1223

v_od = times*theta_car;
v = v_od.loc[home,0:Nalt] # Utility to homezone. Sampling is done based on this utility. ph is calculated based on this
v_target = v_od.loc[d,0:Nalt] # Utility on which log-sum is supposed to be calculated. pr is calculated based on this
expv = np.exp(v)
ph = expv/np.sum(expv) # Probability from homezone. Used for sampling locations (ph)
ph = ph.values.reshape(-1,1)
#pt = np.exp(v_target)/(np.sum(np.exp(v_target))) # Probability in zone d
nit = 100  # number of iterations? 
Nsamp = 100
 
I, NI, w, c = SampleFromP(ph, Nsamp)

###for selected_zones
mask = (NI>0)
#selected_zones = np.array(np.where(mask)).flatten()
### 0) Intialize EV in end time 
selected_zones = c
#%%

def EV_approx(method):
    print(method)
    EV = [np.nan] * (Tmax+1)
    EV.append(v_od.loc[home,selected_zones])
    ### update ev in time t
    for t in range(Tmax, 0, -1):
        approx = [] 
        for i in selected_zones:
            v_r = v_od.loc[i,:] + v_od.loc[home,:]        # new v
            expv_r = np.exp(v_r)        
            pr_n = expv_r/np.sum(expv_r)
            v_n = v_od.loc[i,selected_zones] + EV[t+1]  + theta_trip    #new vr
            expv_n = np.exp(v_n)
            ph_n = expv_n/np.sum(expv_n) # Probability from homezone. Used for sampling locations (ph)

            exps = approx_methods(ph_n, pr_n[selected_zones], expv_n, Nsamp, I, NI, w[selected_zones], c, method,ph,pr_n,w, expv_n)
            exps += np.exp(v_od.loc[i,home])
            approx.append(exps)
        EV[t] = np.log(approx)
        print(t, sum(EV[t]))
    #print([type(p) for p in EV])
    EV.pop(0)
    return EV

def difference(EV1, EV2):
    diff = []
    zip_object = zip(EV1, EV2)
    meanEVdiff = np.mean(EV1) - np.mean(EV2)
    for EV1_i, EV2_i in zip_object:
        diff.append(EV1_i - EV2_i - 0*meanEVdiff)
    return diff

### for all zones 
EV_all = [np.nan] * (Tmax+1)
EV_all.append(np.array(v_od.loc[home, :]))
print('EV all_zones')
for t in range(Tmax,0,-1):
    lsm = [] 
    for i in range(1240):
        v_n = v_od.loc[i, :] + EV_all[t+1] +theta_trip
        expv_n = sum(np.exp(v_n))
        expv_n += np.exp(v_od.loc[i,home])
        lsm.append(expv_n)
    EV_all[t] = np.log(lsm)
    print(t, sum(EV_all[t][selected_zones]))
EV_all.pop(0)

#%%
method1 = "RejectSampling"
method2 = "SelfNormalized"
method3 = "RegressionEstimatorIS"  ## I need to check this function: expt/pr is constant
method4 = 'base'
method5 = "RegressionEstimatorIS_scaper"

EV_m1 = EV_approx(method1)
EV_m2 = EV_approx(method2)
EV_m3 = EV_approx(method3)
EV_m4 = EV_approx(method4)
EV_m5 = EV_approx(method5)

#%%
plt.plot([np.mean(np.abs(EV_m1[t]-EV_m1[t-1])) for t in range(1,Tmax)], 'r')
plt.plot([np.mean(np.abs(EV_m2[t]-EV_m2[t-1])) for t in range(1,Tmax)], 'b')
plt.plot([np.mean(np.abs(EV_m3[t]-EV_m3[t-1])) for t in range(1,Tmax)], 'g')
plt.plot([np.mean(np.abs(EV_m4[t]-EV_m4[t-1])) for t in range(1,Tmax)], 'm')
plt.plot([np.mean(np.abs(EV_m5[t]-EV_m5[t-1])) for t in range(1,Tmax)], 'k')
plt.plot([np.mean(np.abs(EV_all[t]-EV_all[t-1])) for t in range(1,Tmax)], 'c')

plt.show()

diff1 = [np.mean(np.abs(x)) for x in difference([y[selected_zones] for y in EV_all], EV_m1)]
diff2 = [np.mean(np.abs(x)) for x in difference([y[selected_zones] for y in EV_all], EV_m2)]
diff3 = [np.mean(np.abs(x)) for x in difference([y[selected_zones] for y in EV_all], EV_m3)]
diff4 = [np.mean(np.abs(x)) for x in difference([y[selected_zones] for y in EV_all], EV_m4)]
diff5 = [np.mean(np.abs(x)) for x in difference([y[selected_zones] for y in EV_all], EV_m5)]
plt.show()
print('print diff1', diff1)
print('print diff2', diff2)
print('print diff3', diff3)
print('print diff4', diff4)
print('print diff5', diff5)

xx = pd.DataFrame([diff1, diff2, diff3, diff4, diff5], index = ['diff1', 'diff2', 'diff3', 'diff4', 'diff5']).T
xx.plot.line(legend=True, color = {"diff1": "r", "diff2": "b", 'diff3':'g','diff4':'m', 'diff5':'k'})
plt.show()


