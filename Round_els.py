# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:03:12 2021

@author: naqavi
"""
import numpy as np 


def Round_els(df):
    for i in range(len(df)):
        lst = df[i]
        df[i] = [np.round(elem, 2) for elem in lst ]   
        
    return df