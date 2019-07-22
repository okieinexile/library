# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:06:41 2017

@author: bwinters
"""
import numpy as np
import itertools
from typing import Callable

def bisection(a: float, b: float, M: int, 
              delta: float, epsilon: float, function: Callable):
    """
    This will find a zero of function in [a,b]
    
    """
    u: float = function(a)
    v: float = function(b)
    e: float = b - a
    if np.sign(u) == np.sign(v):
        raise ValueError(f'{function.__name__} has the same sign on {a} and {b}. ')
    count = itertools.count()
    while next(count) < M:
        e = e / 2
        c = a + e
        w = function(c)
        if (abs(e) < delta) or (abs(w) < epsilon):
            return(c, e, next(count) - 1)
        if np.sign(w) != np.sign(u):
            b = c
            v = w
        else:
            a = c
            u = w        
    return(c, e, next(count)-1)
    
def mFun(x):
    return 1/x-2**x

def sqrt2(x):
    return x**2-2
    
def kep(x):
    return x-0.016*np.sin(x)-1.5
def prb1(x):
    return x-2*np.sin(x)