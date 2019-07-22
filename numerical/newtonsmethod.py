# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 08:25:16 2017

@author: bwinters
"""
from typing import Tuple, Callable

def newtons_method(x0: float, M: int, 
                   delta: float, epsilon: float) -> Tuple[int,float,float]:
    
    # Put in function and its derivative.
    f: Callable = lambda x: x**2 - 2
    fp: Callable = lambda x: 2 * x
    
    # Begin iteration process.
    y: float = f(x0)
    for k in range(M):
        x1 = x0 - y / fp(x0)
        y = f(x1)
        if (abs(x1 - x0) < delta) or (abs(y) < epsilon):
           break
        x0 = x1
        
    return(k, x1, y)
        
        
        
    

    