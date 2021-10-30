# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:08:14 2021

@author: naqavi
"""
import numpy as np


def Get_a(i, rng):
        
    """
         If a is 1, we use the probabilites in d. Should give very good approximation of logsums
         If a is 0, we don't gain any information. Should give no improvement
    """
    if rng>1:
        rnge = np.linspace(0.1,.9,rng)
    elif (rng<2):
        rnge = np.linspace(0.6,1,rng)
    a = rnge[i]
    return a

