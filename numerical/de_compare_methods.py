# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:08:59 2019

@author: bwinters
"""
from typing import Callable, List, Tuple
from runge_kutta import runge_kutta4, runge_kutta5
from predictor_corrector import pred_corr, adams_bashforth
from de_test_functions import mf, ansf
import pandas as pd


def compare(answer: List[Tuple[float,float]], real_function: Callable) -> List:
    """
    This compares a numerical solution in 
    data={'t':, 'x':} to the function
    real_function
    """
    err = list(map(lambda a: abs(a[1] - real_function(a[0])), answer))
    return err   

def meth_compare(M: int,h: float, t: float, x: float, 
                 f: Callable, true_answer: Callable) -> None:
    """
    This compares the solution to  the initial value 
    problem x'=f(t,x) using
    4th order Runge-Kutta
    Adams-Bashforth
    Predictor-Corrector (Adams-Moulton)
        M is the number of steps
    h is the step size
    t is the initial value of the parameter
    x is the initival value of the function
    It creates the file compare.csv as output
    true_answer is a benchmark test function.
    """
    ab = adams_bashforth(M, h, t, x, f)
    pc = pred_corr(M, h, t, x, f)
    ab_err = compare(ab, true_answer)
    pc_err = compare(pc, true_answer)
    T = list(map(lambda p: p[0], ab))
    data = {
            't': T,
            'adams_bashforth': ab_err,
            'predictor_corrector': pc_err
            }
    df: pd.DataFrame = pd.DataFrame(data)
#    df.to_csv('method_compare.csv')

    return df