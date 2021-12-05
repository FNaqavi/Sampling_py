# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:57:08 2021

@author: naqavi
"""
import pandas as pd

def plots(df): 
    no = pd.DataFrame((df['no1'], df['no2'], df['no3'], df['no4'])).T
    return (no.plot())