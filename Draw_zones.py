# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:59:05 2021

@author: naqavi
"""


def Draw_zones(times):
    rand_hd_zones = times.sample().sample(axis=1)
    col = rand_hd_zones.columns.values.astype(int)[-1]        # get idx for column
    row = rand_hd_zones.index.values.astype(int)[-1]         # get idx for rows
    
    home = col       # Zone used for sampling locations
    d = row          # Zone where we want to calculate log-sum
    return (home, d)