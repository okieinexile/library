# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:08:59 2019

@author: bwinters
"""
from typing import Callable, List, Tuple, Dict
import pandas as pd


def compare_to_known(answer: List[Tuple[float,float]], known_function: Callable) -> List:
    """
    This compares a numerical solution in 
    answer to the benchmark function known_function
    """
    err = list(map(lambda a: abs(a[1] - known_function(a[0])), answer))
    domain = list(map(lambda a: a[0], answer))
    return domain, err   

def meth_compare(methods: List[Callable], M: int,h: float, t: float, x: float, 
                 f: Callable, true_answer: Callable, 
                 filename: str = 'method_compare.csv') -> None:
    """

    """
    answer_dict: Dict = dict()
    for function in methods:
        T, answer_dict[function.__name__] = compare_to_known(
                function(M, h, t, x, f), true_answer)
    answer_dict['t'] = T
    try:
        df: pd.DataFrame = pd.DataFrame(answer_dict)
    except:
        for key in answer_dict.keys():
            print(f'{key} has a length of {len(answer_dict[key])}')
    df.set_index('t', inplace =True)
    df.to_csv(filename)

    return df