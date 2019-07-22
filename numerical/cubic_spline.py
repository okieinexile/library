# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 08:31:15 2019

@author: bwinters
"""

import numpy as np
import pylab as plt
import pandas as pd
from typing import Dict, List, Tuple, Callable



class CubicSpline:
    
    
    def __init__(self, data: Dict):
        self.knots: List[float] = sorted(list(data.keys()))
        self.values: List[float] = [data[key] for key in self.knots]
        self.spline_data: Dict = self._spline_data()
        self.function: Callable = self._eval
        return None
    
    def _eval(self, x):
        #This evaluates a cubic spline with given knots
        #and corresponding values at the point
        i: int = self._get_interval_index(x)
        A: float = self.spline_data['A'][i]
        B: float = self.spline_data['B'][i]
        C: float = self.spline_data['C'][i]
        D: float = self.spline_data['D'][i]    
        S =(A * (self.knots[i + 1] - x)**3 + B * (x - self.knots[i])**3 
            + C * (x - self.knots[i]) + D * (self.knots[i + 1] - x))
        return S

    def _spline_data(self):
        #This gets the cubic spline coefficients
        #for given knots and values
        #It is used in spline_eval
        myData: Dict = dict()
        z, h, u, v = self._get_z()
    
        A, B, C, D = list(), list(), list(), list()
    
        for i in range(len(self.knots)-1):
            A.append(z[i]/(6*h[i]))
            B.append(z[i+1]/(6*h[i]))
            C.append(self.values[i+1]/h[i]-z[i+1]*h[i]/6)
            D.append(self.values[i]/h[i]-z[i]*h[i]/6)
        myData = {
                'z':z,
                'h':h,
                'u':u,
                'v':v,
                'A':A,
                'B':B,
                'C':C,
                'D':D            
                }        
        return(myData)
        
    def _get_z(self) -> Tuple[float,float,float,float]:
        #This gets the z-values for a spline curve
        #It is used in cubic_spline_coeffs

        h, b, u, v, z = list(), list(), [0], [0], (len(self.knots))*[0]

        for i in range(len(self.knots) - 1):
            h.append(self.knots[i + 1] - self.knots[i])
            b.append(6 * (self.values[i + 1] - self.values[i]) / h[i])
        u.append(2 * (h[0] + h[1]))
        v.append(b[1] - b[0])
        
        for i in range(2,len(self.knots) - 1):
            u.append(2 * (h[i] + h[i - 1]) -h[i - 1]**2 / u[i - 1])
            v.append(b[i] - b[i - 1] - h[i - 1] * v[i - 1] / u[i - 1])
        z[len(self.knots) - 1] = 0
        
        for i in range(len(self.knots) - 2, 0, -1):
            z[i] = (v[i] - h[i] * z[i + 1]) / u[i]
    
        return(z,h,u,v)        
    
    def _get_interval_index(self, x: float) -> int:
        #This returns the index of the interval that
        #contains x. It is used in spline_eval
    
        if (x < self.knots[0]) or (x > self.knots[-1]):
            raise ValueError(f'{x} not in Range.')
        for i in range(len(self.knots) - 1):
            if (x >= self.knots[i]) and (x <= self.knots[i+1]):
                return i
        raise ValueError('Something has gone horribly arry.')
    
    def graph(self, number_of_points: int) -> None:
        x = np.linspace(min(self.knots), max(self.knots), number_of_points)
        y = [self._eval(t) for t in x]
        plt.axis('equal')
        plt.plot(x,y)
        return None
        
    @classmethod
    def sample_function(cls, f: Callable, 
                        a: float, b: float, sample_size: int) -> 'CubicSpline':
        out_dict: Dict = dict( (t,f(t)) for t in np.linspace(a, b, sample_size))
        return cls(out_dict)
    
    @classmethod
    def read_excel(cls, filename: str) -> 'CubicSpline':
        df: pd.DataFrame = pd.read_excel(filename)
        pairs = zip(df['x'], df['y'])
        data = dict( p for p in pairs)
        return cls(data)
        