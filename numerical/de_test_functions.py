# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:07:58 2019

@author: bwinters
"""

import numpy as np     
        
def mf(t: float, x: float) -> float:
    """A benchmark function"""
    return (t * x - x**2) / t**2

def mf2(t: float, x: float) -> float:
    """A benchmark function"""
    return x * (1 - np.e**t) / (1 + np.e**t)

def ansf(t: float) -> float:
    """This is the analytic answer to mf"""
    return t / (0.5 + np.log(t))
    
def ansf2(t: float) -> float:
    """This is the analytic answer to mf2"""
    return 12 * np.e**t / (np.e**t + 1)**2

def finv(t,x):
    return 1/t

def fexp(t,x):
    return x

def test1(t: float, x: float) -> float:
    """A benchmark function"""    
    return 2 * t
def true1(t: float) -> float:
    """This is the analytic answer to test1"""    
    return(t**2)
    
def norm_p(t,x):
    return (1/np.sqrt(2*np.pi))*np.e**(-t*t/2)