# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:08:47 2019

@author: bwinters
"""
import numpy as np
from typing import List, Dict, Callable
from math import factorial

class InterpolatingPolynomial:
    
    
    def __init__(self, data: Dict) -> None:

        self.knots: List[float] = sorted(list(data.keys()))
        self.values: List[float] = [ data[key] for key in self.knots]
        self.newton_coefs: List[float] = self._newton_coefs()
        return None
    
    @classmethod
    def sample_function(cls, f: Callable, a: float, b: float,
                        number_of_knots: int) ->'InterpolatingPolynomial':
        data: Dict = dict( (t, f(t)) for t in np.linspace(a,b, number_of_knots))
        return cls(data)
    
    def _newton_coefs(self) -> List[float]:
        c = [self.values[0]]
        for k in range(1,len(self.knots)):
            d = self.knots[k] - self.knots[k - 1]
            u = c[k - 1]
            for i in range(k - 2, -1, -1):
                u = u * (self.knots[k] - self.knots[i]) + c[i]
                d = d * (self.knots[k] - self.knots[i])
            c.append((self.values[k]-u)/d)
        return c
    
    def newton_eval(self, v: float) -> float:
        p = 1
        SUM = 0
        for i in range(len(self.newton_coefs)):
            if i > 0:
                p = p * (v - self.knots[i - 1])
            SUM = SUM + p * self.newton_coefs[i]
        return(SUM)
        
    def divided_difference_table(self) -> np.matrix:

        #Intialize c
        c=np.matrix(np.zeros((len(self.knots),len(self.knots))))
        for i in range(len(self.values)):
            c[i,0]=self.values[i]
        # Populate c
        for j in range(1, len(self.values)):
            for i in range(len(self.values) - j):
                c[i,j] = ((c[i + 1, j - 1] - c[i, j - 1]) 
                / (self.knots[i + j] - self.knots[i]))
        return(c)   
        
    def dd_eval(self, v: float) -> float:
  
        # Get coefficients.
        c: np.matrix = self.divided_difference_table()
        coeffs: List[float] = [c[0,i] for i in range(len(self.values))]
        y = coeffs[0]
        p = 1
        for i in range(1, len(self.knots)):
           p = p * (v - self.knots[i - 1]) 
           y = y + coeffs[i] * p
        return(y)
    
    @staticmethod    
    def chebyshev_nodes(a: float, b: float, n: int) -> List[float]:
        nodes: List[float] = list()
        for k in range(1,n + 1):
            mNode = (0.5 * (a + b) 
            + 0.5 * (b - a) * np.cos(np.pi * (2 * k - 1) / ( 2 * n)) )
            nodes.append(mNode)
        nodes.reverse()   
        return(nodes)
        
class HermitePolynomial:
    
    def __init__(self, data: Dict) -> None:
        self.data: Dict = data
        self.knots: List[float] = list(data.keys())
        return None
    
    def eval(self, v: float) -> float:
        coefs: List[float] = self.coefs()
        knots: List[float] = self._X()
        values: List[float] = [self.fn(0, x) for x in knots]
        return self._eval(knots, values, coefs, v)
    
    def fn(self, order:int, key: float) -> float:
        if key not in self.data.keys():
            raise ValueError(f'{key} is not a knot.')
        if order not in range(len(self.data[key])):
            raise ValueError(f'No data on {order} derivative at {key}.')
        return self.data[key][order]/factorial(order)
    
    def f(self, X: List[float]) -> float:
        if X[0] == X[-1]:
            return self.fn(len(X) - 1, X[0])
        else:
            return (self.f(X[1:])-self.f(X[0:-1]))/(X[-1] - X[0])
    
    def _X(self) -> List[float]:
        out_list: List[float] = list()
        for k in self.data.keys():
            out_list += [k] * len(self.data[k])
        return out_list
    
    def coefs(self) -> List[float]:
        X: List[List[float]] = self._X()
        L = [ X[0:i] for i in range(1,len(X)) ]
        L.append(X)
        return [self.f(item) for item in L]
    
    @staticmethod
    def _eval(knots: List[float], values: List[float], 
              coefficients: List[float], v: float) -> float:
        # Number of knots must equal number of values.
        if len(knots) != len(values):
            raise ValueError('knots and values of different lengths')
            
        # Number of coefficients must equal number of knots and values.
        if len(coefficients) != len(knots):
            raise ValueError(f'Number of coefficents not equal to number of knots.')
        
        # Evaluate!
        y: float = coefficients[0]
        p: float = 1
        for i in range(1, len(knots)):
           p = p * (v - knots[i - 1]) 
           y = y + coefficients[i] * p
        return(y)        
        
        
        
        