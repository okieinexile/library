# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:58:14 2017

@author: bwinters
"""
from typing import Callable

def trapezoid(a: float, b: float, n: int, function: Callable) -> float:
    h = (b - a) / n
    terms = map(lambda i: (function(a + i * h) + function(a + (i + 1) * h)), 
                range(n))
    return (h / 2) * sum(terms) 
    
def simpson(a: float, b: float, n: int, function: Callable) -> float:
    h = (b - a) / n
    terms = map(lambda i: (function(a + i * h) + 4 * function(a + (i + 0.5) * h) 
            + function( a + (i + 1) * h)),
            range(n))
    return (h / 6) * sum(terms)

def newton_cotes3(a: float, b: float, n: int, function: Callable) -> float:
    h = (b - a) / n
    terms = map(lambda i: (function(a + i * h) + 3 * function(a + i * h + h / 3)
                + 3 * function(a + i * h + 2 * h / 3) + function(a + i * h + h)), 
                range(n))
    return (h / 8) * sum(terms)   
    
