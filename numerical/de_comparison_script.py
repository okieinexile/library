# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:01:09 2019

@author: bwinters
"""

from typing import Callable, List, Tuple

from predictor_corrector import pred_corr, adams_bashforth
from runge_kutta import euler, heun, runge_kutta4, runge_kutta5
from de_test_functions import mf, ansf, mf2, ansf2, test1, true1, finv, ansfinv, fexp, ansfexp
from de_compare_methods import meth_compare

METHODS = [euler, heun, runge_kutta4, runge_kutta5, adams_bashforth, pred_corr]
BENCHMARKS = [(mf, ansf), (mf2, ansf2), (test1, true1), (finv, ansfinv), 
              (fexp, ansfexp)]

def test_script(benchmarks: Tuple[Callable,Callable], 
                methods: List[Callable], t: float, 
                M: int = 100, h: float = 0.01, 
                filename: str = 'compare.csv') -> None:
    f_prime, true_answer = benchmarks
    x: float = true_answer(t)
    meth_compare(methods, M, h, t, x, f_prime, true_answer, filename)
    return None
    
    