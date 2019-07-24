# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:22:44 2017

@author: bwinters
"""
from typing import Callable, List

def taylor_series(M: int, h: float, t: float, x: float, 
                  derivatives: List[Callable]):
    answer = [(t, x)]
    for k in range(M):
        xp = derivatives[0](t)
        xpp = derivatives[1](t)
        xppp = derivatives[2](t)
        xpppp = derivatives[3](t)
        x = x + h * (xp + (h / 2) * (xpp + (h / 3) * (xppp + (h / 4) * xpppp)))
        t = t + h
        answer.append((t,x))
    return answer
    
