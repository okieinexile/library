# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 08:28:06 2019

@author: bwinters
"""

from typing import List, Generic
import tkinter as tk

class SelectDate(tk.Tk):
    
    
    def __init__(self, parent: tk.Frame, year: int, **options) -> None:
        tk.Tk.__init__(self, parent, **options)
        self.title(string = 'Set Date')
        self.year: int = year
        self.day: int = None
        self.month: int = None
        self.set_widgets()      
        button = tk.Button(self, 
                           text = 'Enter Date',
                           command = self.button_press)
        button.pack(side = tk.BOTTOM)
        return None
    
    def set_widgets(self) -> None:
        month_frame = tk.Frame(self)
        month_label = tk.Label(month_frame, text = 'Month')
        month_label.pack(side = tk.TOP)
        self.month_scale = tk.Scale(month_frame,
                               from_ = 1,
                               to = 12)
        self.month_scale.pack(side = tk.LEFT)
        month_frame.pack(side = tk.LEFT)
        
        day_frame = tk.Frame(self)
        day_label = tk.Label(day_frame, text = 'Day')
        day_label.pack(side = tk.TOP)
        self.day_scale = tk.Scale(day_frame,
                               from_ = 1,
                               to = 31)
        self.day_scale.pack(side = tk.LEFT)
        day_frame.pack(side = tk.LEFT)
        
        year_label = tk.Label(self, text = self.year)
        year_label.pack(side = tk.LEFT)
             
        return None
    
    def button_press(self) -> None:
        self.day = self.day_scale.get()
        self.month = self.month_scale.get()
        print(f'{self.month}/{self.day}/{self.year}')
        return None
        

