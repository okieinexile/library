# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:50:02 2017

@author: bwinters
"""
from typing import Callable, List, Tuple

def euler(M: int, h: float, t: float, x: float, 
          f: Callable) -> List[Tuple[float,float]]:
    """
    This solves the initial value problem
    x'=f(t,x)
    using Euler's method
    M is the number of steps
    h is the step size
    t is the initial value of the parameter
    x is the initival value of the function
    it returns a dictionary containing the knots and values
    that can be put into a spline
    """    
    interval = [t]
    X = [x]
    answer =[(t, x)]
    for k in range(M):
        x = x + h * f(t,x)
        t = t + h
        interval.append(t)
        X.append(x)
        answer.append((t,x))
    return answer

def heun(M: int, h: float, t: float, x: float, 
         f: Callable) -> List[Tuple[float,float]]:
    """
    This solves the initial value problem
    x'=f(t,x)
    using Heun's method
    M is the number of steps
    h is the step size
    t is the initial value of the parameter
    x is the initival value of the function
    it returns a dictionary containing the knots and values
    that can be put into a spline
    """
    answer = [(t, x)]
    for k in range(M):
        F1 = h * f(t,x)
        F2 = h * f(t + h, x + F1)
        x = x + 0.5 * (F1 + F2)
        t = t + h
        answer.append((t, x))
    return(answer)

def runge_kutta4(M: int, h: float, t: float, x: float, 
                 f: Callable) -> List[Tuple[float,float]]:
    """
    This solves the initial value problem
    x'=f(t,x)
    using the 4th order Runge-Kutta method
    M is the number of steps
    h is the step size
    t is the initial value of the parameter
    x is the initival value of the function
    it returns a dictionary containing the knots and values
    that can be put into a spline
    """
    answer = [(t, x)]
    for k in range(M):
        F1 = h * f(t,x)
        F2 = h * f(t + h / 2, x + F1 / 2)
        F3 = h * f(t + h / 2, x + F2 / 2)
        F4 = h * f(t + h, x + F3)
        x = x + (F1 + 2 * F2 + 2 * F3 + F4) / 6    
        t = t + h
        answer.append((t, x))
    return(answer)        

def runge_kutta5(M: int, h: float, t: float, x: float, f: Callable):
    """
    This solves the initial value problem
    x'=f(t,x)
    using Runge-Kutta's 5th order method
    M is the number of steps
    h is the step size
    t is the initial value of the parameter
    x is the initival value of the function
    it returns a dictionary containing the knots and values
    that can be put into a spline
    """       

    answer = [(t, x)]
    for k in range(M):
        F1 = h * f(t, x)
        F2 = h * f(t + h / 2, x + F1 / 2)
        F3 = h * f(t + h / 2 ,x + F1 / 4 + F2 / 4)
        F4 = h * f(t + h, x - F2 + 2 * F3)
        F5 = h * f(t + 2 * h / 3, x + 7 * F1 / 27 + 10 * F2 / 27 + F4 / 27)
        F6 = h * f(t +h / 5, x + 28 * F1 / 625 
                   - F2 / 5 + 546 * F3 / 625 + 54 * F4 / 625 - 378 * F5 / 625)
        x = x + (F1 / 24 + 5 * F4 / 48 + 27 * F5 / 56 + 125 * F6 / 336)    
        t = t + h
        answer.append((t, x))
    return answer

