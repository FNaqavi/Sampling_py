# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:34:00 2021

@author: naqavi
"""

import pandas as pd
from Variance_reduction_methods import Variance_reduction_methods
from Round_els import Round_els
from clean_df import clean_df
from plots import plots 
from SampleFromP import SampleFromP
import matplotlib.pyplot as plt

times = pd.read_csv('carTimes_peak.csv',';', index_col=False, header=None)
theta_car = -0.1        # what is this?
Nalt = 1240
hd_lst = []
hd_rng = 1    # number of random home dest zone pick

for hd_rand in range(hd_rng):
    hd_lst = Variance_reduction_methods(times, theta_car, Nalt, hd_lst)
          
hd_lst = Round_els(hd_lst)
(df_all, m1, m2, m3, m4) = clean_df(hd_rng, hd_lst)


for length_a in range(hd_rng):
    df_seperated = df_all.iloc[:, 0+20*length_a:20+20*length_a] 
    no = plots(df_seperated)

no.rename({'no1':'BasicIS', 'no2':'RejectSampling','no3':'SelfNormalized', 'no4':'ControlVariate'},inplace= True, axis = 1)
no.plot()
no.set_ylabel("Variance")
plt.show()
    
#plt.plot(pd.DataFrame((df_all['no1']))) 
    


        










