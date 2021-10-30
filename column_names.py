# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:31:30 2021

@author: naqavi
"""


def column_names(df):
    m1 = df.groupby(["i"]).get_group(1).drop(columns = ['a', 'i']).reset_index(drop = True).rename(columns = { 'me': 'me1', 'error': 'error1', 'no':'no1', 'dev':'dev1', 'tval':'tval1'})  # 11 runs with different a for method 1 
    m2 = df.groupby(["i"]).get_group(2).drop(columns = ['a', 'i']).reset_index(drop = True).rename(columns = { 'me': 'me2', 'error': 'error2', 'no':'no2', 'dev':'dev2', 'tval':'tval2'})  # 11 runs with different a for method 2
    m3 = df.groupby(["i"]).get_group(3).drop(columns = ['a', 'i']).reset_index(drop = True).rename(columns = { 'me': 'me3', 'error': 'error3', 'no':'no3', 'dev':'dev3', 'tval':'tval3'})  # 11 runs with different a for method 3
    m4 = df.groupby(["i"]).get_group(4).drop(columns = ['a', 'i']).reset_index(drop = True).rename(columns = { 'me': 'me4', 'error': 'error4', 'no':'no4', 'dev':'dev4', 'tval':'tval4'})  # 11 runs with different a for method 4
    
    return (m1, m2, m3, m4)