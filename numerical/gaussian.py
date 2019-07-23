# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:41:28 2017

@author: bwinters
"""
from numpy import sqrt
from typing import Callable

def gauss1(a: float, b: float, function: Callable) -> float:
    alpha=1/sqrt(3)
    lam = lambda t: (a / 2) * (1 - t) + (b / 2) * (1 + t)
    return ((b - a) / 2) * (function(lam(-alpha)) + function(lam(alpha)))

def gauss4(a: float, b: float, function: Callable) -> float:

    x = [0,0.538469310105683, 0.906179845938664]
    A = [0.568888888888889, 0.478628670499366, 0.236926885056189]
    u = (a + b) / 2
    S = A[0] * function(u)
    for i in range(1,3):
        u = ((b - a) * x[i] + a + b) / 2
        v = ((a - b) * x[i] + a + b) / 2
        S = S + A[i] * (function(u) + function(v))
    S = (b - a) * S / 2
    return S    
    
