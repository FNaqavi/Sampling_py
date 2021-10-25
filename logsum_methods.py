# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:21:39 2021

@author: naqavi
"""

import numpy as np
import pandas as pd
from SampleFromP import SampleFromP

def logsum_methods(prob, u_samp, NI, method, N, i):
    p = prob
    approx = np.zeros((i,4))   
    
    if method == 'SampleFromP':
        w = NI/p / N   # N: number of samples , p(prob from any selected zone to all selected zones)
        approx[i,0] = np.matmul(w, np.exp(u_samp).T)  # should be done only from one zone at a time (u_samp is a matrix right now)
    
    elif method == 'w4':
        w = NI/p / N
        pr1 = np.exp(u_samp) / np.sum(np.exp(u_samp), axis = 0)
        pr_n = pd.DataFrame(0, index = np.arange(1240), columns = np.arange(1240))
        pr1 = pr1.drop_duplicates()   # drop duplicate rows
        pr1 = pr1.loc[:,~pr1.columns.duplicated()]  # drop duplicate columns
        
        # M = np.sum(pd.Series.multiply(w[np.flatnonzero(w)], pr1[np.flatnonzero(w)]))
        # M = w.multiply(pr1)
        # #w4 = w[np.flatnonzero(w)]/M
        # div_wm = w / M
        # w4 = div_wm.dropna(axis = 1)
        # #approx[k,2] = np.matmul(w4, expt[Ir].values.reshape(len(expt[Ir]),1))
   
        w.multiply(pr1)
    else:
        #RegressionEstimatorIS(p, q, f, c)
        #pr,p_sample,expt/pr,c
        f = f[c]   
        w = p[c]/q[c]    # sampling weight, i.e., p/q where p is target distribution and q is
                         # distribution from which samples are drawn.                         
        n = len(w)
        mw = np.mean(w)         
        b = np.sum ( (w-mw) * w *f ) / np.sum((w-mw)**2)       
        mu_1 =  1/n *  np.sum ( w * f)          
        mu_2 = 1/n *  np.sum (  b * (w - 1))         
        mu = mu_1 - mu_2


    return approx