# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:07:54 2017

@author: bwinters
"""
from typing import Callable, List, Tuple
from runge_kutta import runge_kutta5
import pylab as plt




def pred_corr(M: int, h: float, t: float, x: float, 
              f: Callable) -> List[Tuple[float,float]]:

    """
    This solves the initial value problem
    x'=f(t,x)
    using 5th order method predictor-corrector method
    M is the number of steps
    h is the step size
    t is the initial value of the parameter
    x is the initival value of the function
    it returns a dictionary containing the knots and values
    that can be put into a spline
    """           
    start = runge_kutta5(4, h, t, x, f)
    X = list(map(lambda p: p[1], start))
    T = list(map(lambda p: p[0], start))
    t1, t2, t3, t4, t5 = T
    x1, x2, x3, x4, x5 = X
    answer = list(zip(T, X))
    for k in range(M - 4):
        f1, f2, f3, f4, f5 = f(t1, x1), f(t2, x2), f(t3, x3), f(t4, x4), f(t5, x5)
        x6 = x5 + (h / 720) * (1901 * f5 - 2774 * f4 + 2616 * f3 - 1274 * f2 +251 * f1)
        f6 = f(t5 + h, x6)
        x6 = x5 + (h / 720) * (251 * f6 + 646 * f5 - 264 * f4 + 106 * f3 - 19 * f2)
        x1, x2, x3, x4, x5 = x2, x3, x4, x5, x6
        t1, t2, t3, t4, t5 = t2, t3, t4, t5, t5 + h
        answer.append((t5,x5))        
    return answer

def adams_bashforth(M: int, h: float, t: float, x: float, 
                    f: Callable) ->List[Tuple[float,float]]:
    """
    This solves the initial value problem
    x'=f(t,x)
    using Adams-Bashforth 5th order method
    M is the number of steps
    h is the step size
    t is the initial value of the parameter
    x is the initival value of the function
    it returns a list of M + 5 ordered pairs
    """       
    
    start = runge_kutta5(4, h, t, x, f)

    T = list(map(lambda p: p[0], start))
    X = list(map(lambda p: p[1], start))
    t1, t2, t3, t4, t5 = T
    x1, x2, x3, x4, x5 = X
    answer = list(zip(T, X))
    for k in range(M - 4):
        f1, f2, f3, f4, f5 = f(t1, x1), f(t2, x2), f(t3, x3), f(t4, x4), f(t5, x5)
        x6 = x5 + (h / 720) * (1901 * f5 - 2774 * f4 + 2616 * f3 - 1274 * f2 + 251 * f1)
        x1, x2, x3, x4, x5 = x2, x3, x4, x5, x6
        t1, t2, t3, t4, t5 = t2, t3, t4, t5, t5 + h
        answer.append((t5, x5))
    return answer
        

    

    
def graph(output):
    """
    This graphs a spline with given knots and values
    """

    knots=output['t']
    values=output['x']
    plt.plot(knots,values)
    return None    
    
