# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:59:52 2017

@author: bwinters
"""
import numpy as np
from typing import Callable, Tuple

def secant_method(a: float, b: float, M: int, 
                  delta: float, epsilon: float, 
                  function: Callable) -> Tuple[int,float,float]:
    x0: float = a
    x1: float = b
    fa: float = function(x0)
    fb: float = function(x1)
    for i in range(M):
        x2 = x1 - fb * (x1 - x0) / (fb - fa)
        fc = function(x2)
        if (abs(x2 - x1) < delta) or (abs(fc) < epsilon):
            return(i,x2,fb)
        x0 = x1
        x1 = x2
        fa = function(x0)
        fb = function(x1)
        
    return(i, x2, fb)
    
def kep(x):
    return x - 0.016 * np.sin(x) - 1.5
    
def root2(x):
    return(x**2 - 2)
    
def funA(x):
    return(np.sin( x / 2) - 1)
    
def funB(x):
    return(np.e**x - np.tan(x))
    
def funC(x):
    return(x**3 - 12 * x**2 + 3 * x + 1)