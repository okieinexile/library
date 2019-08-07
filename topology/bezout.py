# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:58:34 2017

@author: bwinters
"""

from typing import Tuple

def bezout(p: int,q: int) -> Tuple[int,int,int]:
    """
    This will return (d, s, t) where d is the greatest common denominator
    of p,q and d = s * p + t * q.
    """
    a, r = div(p, q)
    if r == 0:
        return (q, 0, 1)
    if a == 0:
        return (p, 1, 0)
    A = list()
    while r != 0:
        A.append(a)        
        p, q = q, r
        a, r = div(p, q)
    s0 = 1
    t0 = -A.pop()
    while len(A) > 0:        
        s1 = t0
        t1 = s0 - t0 * A.pop()
        s0, t0 =s1, t1
    return q, s0 ,t0
    
def div(p: int, q: int) -> Tuple[int,int]: 
    """
    This performs the division algorithm and returns Q and R where
    p = q * Q + R, 0 < q.
    """
    return(p // q, p % q)

def divides(p: int,q: int) ->bool:
    """
    Returns True if p divides q and false otherwise.
    """
    r = abs(q) % abs(p)
    return r==0
    