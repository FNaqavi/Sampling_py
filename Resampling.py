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
from Get_a import Get_a
#from Round_els import Round_els
#from clean_df import clean_df
#from plots import plots

times = pd.read_csv('carTimes_peak.csv',';', index_col=False, header=None)
theta_car = -0.06      
theta_trip= -6.12
Nalt = 1240
Tmax = 10
hd_lst = []

(home, d) = Draw_zones(times)
print(home, d)


# home = 704 
# d = 568


v_od = times*theta_car;
v = v_od.loc[home,0:Nalt] # Utility from homezone. Sampling is done based on this utility. ph is calculated based on this
v_target = v_od.loc[d,0:Nalt] # Utility on which log-sum is supposed to be calculated. pr is calculated based on this
expv = np.exp(v)
ph = expv/np.sum(expv) # Probability from homezone. Used for sampling locations (ph)
ph = ph.values.reshape(-1,1)
#pt = np.exp(v_target)/(np.sum(np.exp(v_target))) # Probability in zone d
nit = 100  # number of iterations? 
Nsamp = 90  # number of zones to be sampled
 
I, NI, w, c = SampleFromP(ph, Nsamp)

###for selected_zones
mask = (NI>0)
#selected_zones = np.array(np.where(mask)).flatten()
### 0) Intialize EV in end time 
selected_zones = c

#%% Recursive logit with sampled 
corr = []
def EV_approx(method):
    print(method)
    EV = [np.nan] * (Tmax+1)
    EV.append(v_od.loc[home,selected_zones])
    ### update ev in time t
    for t in range(Tmax, 0, -1):
        approx = [] 
        for i in selected_zones:
            v_r = v_od.loc[i,:] + v_od.loc[home,:]        # new v (1240 values)
            expv_r = np.exp(v_r)        
            pr_n = expv_r/np.sum(expv_r)
            v_n = v_od.loc[i,selected_zones] + EV[t+1] + theta_trip    # new vr (Nsamp number of values)
            expv_n = np.exp(v_n)
            ph_n = expv_n/np.sum(expv_n) # Probability from homezone (any selected zone). # Used for sampling locations (ph)   
            exps = approx_methods(ph_n, pr_n[selected_zones], expv_n, Nsamp, I, NI, w[selected_zones], c, method, ph, pr_n,w, expv_n)
            exps += np.exp(v_od.loc[i,home])
            approx.append(exps)
            corr.append(np.corrcoef(pr_n[selected_zones], ph_n))
        EV[t] = np.log(approx)
        print(t, sum(EV[t]))
    #print([type(p) for p in EV])
    EV.pop(0)
    return EV




#%% Recursive logit when all 1240 zones are in the choice set 
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

#%% Plotting the recurive logits with variance reduction method

def difference(EV1, EV2):
    diff = []
    zip_object = zip(EV1, EV2)
    # meanEVdiff = np.mean(EV1) - np.mean(EV2)
    for EV1_i, EV2_i in zip_object:
        # diff.append(EV1_i - EV2_i - 0*meanEVdiff)
        diff.append(np.sqrt((EV1_i - EV2_i)**2))

    return diff
method1 = "RejectSampling"
method2 = "SelfNormalized"  # ratio method
method3 = "RegressionEstimatorIS"  ## I need to check this function: expt/pr is constant  # control variate method
method4 = 'base'
method5 = "RegressionEstimatorIS_scaper"
# EV_m1 = EV_approx(method1)
EV_m2 = EV_approx(method2)
EV_m3 = EV_approx(method3)
EV_m4 = EV_approx(method4)
EV_m5 = EV_approx(method5)

  #%%
#plt.plot([np.mean(np.abs(EV_m1[t]-EV_m1[t-1])) for t in range(1,Tmax)], 'r')
plt.plot([np.mean(np.abs(EV_m2[t]-EV_m2[t-1])) for t in range(1,Tmax)], 'b')
plt.plot([np.mean(np.abs(EV_m3[t]-EV_m3[t-1])) for t in range(1,Tmax)], 'g')
plt.plot([np.mean(np.abs(EV_m4[t]-EV_m4[t-1])) for t in range(1,Tmax)], 'm')
plt.plot([np.mean(np.abs(EV_m5[t]-EV_m5[t-1])) for t in range(1,Tmax)], 'k')
plt.plot([np.mean(np.abs(EV_all[t]-EV_all[t-1])) for t in range(1,Tmax)], 'c')
plt.show()

#diff1 = [np.mean(np.abs(x)) for x in difference([y[selected_zones] for y in EV_all], EV_m1)]
diff2 = [np.mean(np.abs(x)) for x in difference([y[selected_zones] for y in EV_all], EV_m2)]
diff3 = [np.mean(np.abs(x)) for x in difference([y[selected_zones] for y in EV_all], EV_m3)]
diff4 = [np.mean(np.abs(x)) for x in difference([y[selected_zones] for y in EV_all], EV_m4)]
diff5 = [np.mean(np.abs(x)) for x in difference([y[selected_zones] for y in EV_all], EV_m5)]
plt.show()
#print('print diff1', diff1)
print('print diff2', diff2)
print('print diff3', diff3)
print('print diff4', diff4)
print('print diff5', diff5)

# xx = pd.DataFrame([diff1, diff2, diff3, diff4, diff5], index = ['diff1', 'diff2', 'diff3', 'diff4', 'diff5']).T
# ax = xx.plot.line(legend=True, color = {"diff1": "r", "diff2": "b", 'diff3':'g','diff4':'m', 'diff5':'k'})

home_zone = str(home)
destination = str(d)

# xx = pd.DataFrame([diff2, diff3, diff4, diff5], index = ['diff2', 'diff3', 'diff4','diff5']).T
# xx.rename({'diff2':'SelfNormalized', 'diff3':'RegressionEstimatorIS', 'diff4':'base', 'diff5':'RegressionEstimatorIS_scaper'},inplace= True, axis = 1)
# ax = xx.plot.line(legend=True, color = {"SelfNormalized": "b",  'RegressionEstimatorIS':'k', 'base':'m', 'RegressionEstimatorIS_scaper': 'yellow'} , title='Nested, home = '+ home_zone,  alpha=0.5)
# ax.set(xlabel='time steps', ylabel='variance from EV of all zones, Nested')
# ax.legend(loc='upper left')
# ax.invert_xaxis()
# plt.show()


xx = pd.DataFrame([diff2, diff3, diff4], index = ['diff2', 'diff3', 'diff4']).T
xx.rename({'diff2':'SelfNormalized', 'diff3':'RegressionEstimatorIS', 'diff4':'base'},inplace= True, axis = 1)
ax = xx.plot.line(legend=True, color = {"SelfNormalized": "b",  'RegressionEstimatorIS':'k', 'base':'m'} , title='Nested, home = '+ home_zone,  alpha=0.5)
ax.set(xlabel='time steps', ylabel='variance from EV of all zones, Nested')
ax.legend(loc='upper left')
ax.invert_xaxis()
plt.show()

corr1 = [corr[i][1,0] for i in range(len(corr))]
SN, RegIS, bas, RegIS_scaper = np.split(np.array(corr1),4)
t10,t9,t8,t7,t6,t5,t4,t3,t2,t1 = np.split(RegIS,10)
tt10,tt9,tt8,tt7,tt6,tt5,tt4,tt3,tt2,tt1 = np.split(SN,10)
ttt10,ttt9,ttt8,ttt7,ttt6,ttt5,ttt4,ttt3,ttt2,ttt1 = np.split(bas,10)
X = []
X.append(np.mean(t10))
X.append(np.mean(t9))
X.append(np.mean(t8))
X.append(np.mean(t7))
X.append(np.mean(t6))
X.append(np.mean(t5))
X.append(np.mean(t4))
X.append(np.mean(t3))
X.append(np.mean(t2))
X.append(np.mean(t1))

X1 = []
X1.append(np.mean(tt10))
X1.append(np.mean(tt9))
X1.append(np.mean(tt8))
X1.append(np.mean(tt7))
X1.append(np.mean(tt6))
X1.append(np.mean(tt5))
X1.append(np.mean(tt4))
X1.append(np.mean(tt3))
X1.append(np.mean(tt2))
X1.append(np.mean(tt1))

X2 = []
X2.append(np.mean(ttt10))
X2.append(np.mean(ttt9))
X2.append(np.mean(ttt8))
X2.append(np.mean(ttt7))
X2.append(np.mean(ttt6))
X2.append(np.mean(ttt5))
X2.append(np.mean(ttt4))
X2.append(np.mean(ttt3))
X2.append(np.mean(ttt2))
X2.append(np.mean(ttt1))

X = np.array(X)
X1 = np.array(X1)
X2 = np.array(X2)

XX = pd.DataFrame([X,X1,X2], index = ['RegIS', 'SN','basicIS']).T

ax1 = XX.plot(legend=True, color = {"SN": "b",  'RegIS':'k', 'basicIS':'m'} , title='covariance', alpha=0.5)
ax1.set(xlabel='time steps', ylabel='correlation with No-IS EV')
ax1.legend(loc='upper right')
plt.show()

#%% MNL 
"""
expt = np.exp(v_target)
real = np.sum(expt)

def EV_approx_MNL(method,a): 
    v_r = v_target*a + v*(1-a)   # new v (1240 values)
    expv_r = np.exp(v_r)        
    pr_n = expv_r/np.sum(expv_r)
    v_n = v_od.loc[d,selected_zones]   # new vr (Nsamp number of values)
    expv_n = np.exp(v_n)
    ph_n = expv_n/np.sum(expv_n) # Probability from homezone (any selected zone). # Used for sampling locations (ph)   
    exps = approx_methods(ph_n, pr_n[selected_zones], expv_n, Nsamp, I, NI, w[selected_zones], c, method, ph, pr_n,w, expv_n)
    return exps 

def avg_df(expss,k, mthd):
    xx = pd.DataFrame()
    xx['me'] = expss.mean(axis=1)
    xx['error'] = real - xx['me']
    xx['no'+mthd] = np.sqrt(np.mean((real-expss)**2,axis =1))
    xx['dev'] = np.sqrt(np.var(expss,axis = 1)/(k))       
    xx['tval'] = np.abs(xx['error']/xx['dev'])
    xx['a'] = As
    return xx

rng = 11
hd_rng = 1 
As = np.linspace(0.1,1,rng)
exps2 = pd.DataFrame()
exps3 = pd.DataFrame()
exps4 = pd.DataFrame()

for k in range(nit):
    I, NI, w, c = SampleFromP(ph, Nsamp)
    selected_zones = c
    print(k)
    expss2 = []
    expss3 = []
    expss4 = []
    for i in range(rng):
        a = Get_a(i, rng)
        lst2 = EV_approx_MNL(method2,a)
        expss2.append(lst2)
        lst3 = EV_approx_MNL(method3,a)
        expss3.append(lst3)
        lst4 = EV_approx_MNL(method4,a)
        expss4.append(lst4)
    exps2[k] = pd.Series(expss2) 
    exps3[k] = pd.Series(expss3) 
    exps4[k] = pd.Series(expss4) 

hd_lst2 = []            
hd_lst3 = []            
hd_lst4 = []            

hd_lst2 = avg_df(exps2, k, str(2))
hd_lst3 = avg_df(exps3, k, str(3))
hd_lst4 = avg_df(exps4, k, str(4))



lse = pd.DataFrame((hd_lst2['no2'], hd_lst3['no3'], hd_lst4['no4'])).T
lse.rename({'no2':'SelfNormalized','no3':'RegressionEstimatorIS', 'no4':'base'},inplace= True, axis = 1)
# home_zone = str(home)
# destination = str(d)
ax = lse.plot(colormap='jet', title='MNL, home = '+ home_zone +' d = '+ destination)
ax.set(xlabel='a (a0 = 0, a10 = 1)', ylabel='variance from EV of all zones')
ax.legend(loc='upper right')
plt.show()

"""
