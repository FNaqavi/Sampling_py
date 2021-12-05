# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:16:13 2021

@author: naqavi
"""
import pandas as pd
from column_names import column_names

def clean_df(hd_rng, hd_lst, rng):
    df_all = pd.DataFrame()
    df_all1 = pd.DataFrame()
    for i in range(hd_rng):
        lst = hd_lst[i]
        df = pd.DataFrame(lst ,columns = ['me', 'error', 'no', 'dev', 'tval', 'a'])    
        l = [1, 2, 3, 4] * rng
        df['i'] = l[i]       
        (m1, m2, m3, m4) = column_names(df)
        df_all1 = pd.concat([m1, m2, m3, m4], axis = 1)
        df_all = pd.concat([df_all1,df_all], axis = 1)
        
    return (df_all, m1, m2, m3, m4)