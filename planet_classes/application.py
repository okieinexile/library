#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 09:05:48 2019

@author: bobby
"""
from typing import List
import tkinter as tk
from PIL import Image, ImageTk
import planet_classes as pc

class Application(tk.Tk):

    def __init__(self, master: tk.Tk = None, year: int = 2019, **options) -> None:
        tk.Tk.__init__(self, master, **options)
        self.geometry('500x250')
        self.year: int = year
        self.month: int = None
        self.day: int = None
        self.date = None
        self.title(f'Pick a date in {self.year}')
        self.image_name = 'images/image.jpg'
        self._load_planets()
        self.setup_widgets()
        return None
    
    def button_command(self) -> None:
        self.month = self.month_slider.get()
        self.day = self.day_slider.get()
        self.date = f'{self.month}/{self.day}/{self.year}'
        self.picture_label.config(text = self.date)
        self._choose_planets()
        sky_tools = pc.SkyTools()
        sky_tools.set_system(self.chosen_planets,'Earth', self.date)
        sky_tools.plot(self.date)
        self.image = ImageTk.PhotoImage(Image.open(self.image_name))
        self.image_holder.config(image = self.image)
        return None    
    
    def setup_widgets(self) -> None:
        self._set_up_button()
        self._set_up_month_slider()      
        self._set_up_day_slider()
        self._set_up_checkboxes()
        self._set_up_picture()
        return None
    
    def _choose_planets(self) -> None:
        chooser = zip(self.checkbox_variables, self.planet_names)
        self.chosen_planets = list()
        for chk, planet in chooser:
            if chk.get() == 1:
                self.chosen_planets.append(planet)
    
    def _load_planets(self) -> None:
        self.planetary_system = pc.PlanetarySystem.read_csv()
        self.planet_names: List = self.planetary_system.names 
        return None
    
    def _set_up_button(self) -> None:
        button = tk.Button(self, text = 'Press',
                           command = self.button_command)
        button.pack(side = tk.TOP)        
        return None
    
    def _set_up_checkboxes(self) -> None:
        checkbox_frame = tk.Frame(self)
        self.checkbox_variables: List = list()
        for planet in self.planet_names:
            var = tk.IntVar()
            checkbox = tk.Checkbutton(checkbox_frame,
                                   text = planet,
                                   onvalue = 1,
                                   offvalue = 0,
                                   variable = var)
            checkbox.pack( side = tk.TOP)
            self.checkbox_variables.append(var)
        checkbox_frame.pack(side = tk.LEFT)
                           
            
        return None

    def _set_up_day_slider(self) -> None:
        day_frame = tk.Frame(self)
        day_label = tk.Label(day_frame, text = 'Day')
        day_label.pack()
        self.day_slider = tk.Scale(day_frame,
                                from_ = 1,
                                to = 31)
        self.day_slider.pack()
        day_frame.pack(side = tk.LEFT)        
        return None    
    
    def _set_up_month_slider(self) -> None:
        month_frame = tk.Frame(self)
        month_label = tk.Label(month_frame, text = 'Month')
        month_label.pack()
        self.month_slider = tk.Scale(month_frame,
                                from_ = 1,
                                to = 12)
        self.month_slider.pack()
        month_frame.pack(side = tk.LEFT)        
        return None
    
    def _set_up_picture(self) -> None:
        picture_frame = tk.Frame(self)
        self.image = tk.PhotoImage( file = 'images/dog.gif')
        self.image_holder = tk.Label(picture_frame, image = self.image)
        self.image_holder.pack(side = tk.RIGHT)      
        self.picture_label = tk.Label(picture_frame, text = self.date)
        self.picture_label.pack(side = tk.BOTTOM)
        picture_frame.pack(side = tk.RIGHT)
        return None
