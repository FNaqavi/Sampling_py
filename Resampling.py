# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:34:00 2021

@author: naqavi
"""

import pandas as pd
import numpy as np
from Variance_reduction_methods import Variance_reduction_methods
from Round_els import Round_els
from clean_df import clean_df
from plots import plots 
from nests_logsums import nests_logsums

times = pd.read_csv('carTimes_peak.csv',';', index_col=False, header=None)
theta_car = -0.06        
Nalt = 1240
hd_lst = []


#%%  plot

# hd_rng = 1    # number of random home dest zone pick
# rng = 5      # how many alphas ; see Get_a function
# for hd_rand in range(hd_rng):
#     (hd_lst ,approx ) = Variance_reduction_methods(times, theta_car, Nalt, hd_lst, rng)
          
# hd_lst = Round_els(hd_lst)
# (df_all, m1, m2, m3, m4) = clean_df(hd_rng, hd_lst, rng)

# for length_a in range(hd_rng):
#     df_seperated = df_all.iloc[:, 0+20*length_a:20+20*length_a] 
#     plots(df_seperated)


#%% nested

hd_rng1 = 1  
rng1 =  5
for hd_rand in range(hd_rng1):
    (hd_lst ,approx) = nests_logsums(times, theta_car, Nalt, hd_lst, rng1)
    
    
# u1 = theta_car * times
# p1 = np.exp(u1) / np.sum(np.exp(u1), axis = 1)
# ls = np.sum(np.log(p1), axis = 1)


