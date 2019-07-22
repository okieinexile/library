# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:50:46 2019

@author: bwinters
"""

import pandas as pd
import numpy as np
 
kep_con = 133318.48254272068

def kepler_constant():
    df = pd.read_csv('data/cleaned_data.csv')
    df.set_index('name', inplace = True)
    KC = list()
    for _, name in enumerate(df.index):
        T = df.loc[name,'period']
        a = df.loc[name, 'a']
        KC.append(T**2/a**3)
        
    KC.pop(0)
    
    mu, sigma = np.mean(KC), np.std(KC)
    for name in df.index:
        print(name, df.loc[name, 'period'], np.sqrt(mu * df.loc[name, 'a']**3))        
    return mu, sigma

def make_period(x) -> float:
    try:
        return np.sqrt(kep_con * float(x)**3)
    except:
        return None

def asteroids() -> pd.DataFrame:
    df = pd.read_csv('data/asteroid_frame.csv', 
                     usecols = ['Name', 'Epoch', 'a', 'e', 'i', 'omega', 'Omega', 'M','H', 'G'],
                     low_memory = False)
    columns = list(df.columns)
    columns[0] = 'name'
    df.columns = columns
    name_filter = list(filter(lambda x: not x.isdigit(), df['name']))
    named_df = df.loc[df['name'].isin(name_filter)]
    period = map( make_period, named_df['a'])
    named_df['period'] = list(period)
    return named_df

