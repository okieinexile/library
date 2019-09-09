# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 09:58:07 2019

@author: bwinters
"""

from tkinter import Menu, Tk, Canvas, NW
from tkinter.messagebox import showerror
from PIL import ImageTk, Image 

import single_planet as sp



def not_done() -> None:
    showerror('Not implemented','Not yet available')
    return None

def single_planet() -> None:
    sp.do_window()
    return None
    

def make_menu(window: Tk) -> None:
    top_menu = Menu(window)
    window.config(menu = top_menu)
    orbit_menu = Menu(top_menu)
    orbit_menu.add_command(label = 'Single Planet', command = sp.do_window)
    orbit_menu.add_command(label = 'Multiple Planets', command = not_done)
    orbit_menu.add_command(label = 'Inner System', command = not_done)
    orbit_menu.add_command(label = 'Outer System', command = not_done)
    top_menu.add_cascade(label = 'Orbits', menu = orbit_menu, underline = 0)
    
    position_menu = Menu(top_menu)
    position_menu.add_command(label = 'Particular Date')
    position_menu.add_command(label = 'Particular Time')
    top_menu.add_cascade(label = 'Position', menu = position_menu, underline = 0)
    
    planetary_data_menu = Menu(top_menu)
    planetary_data_menu.add_command(label = 'Orbital Elements', command = not_done)
    top_menu.add_cascade(label = 'Planetary Data', menu = planetary_data_menu, underline = 0)  
    
    return None
    

if __name__ == '__main__':
    main_window = Tk()
    main_window.title('The Planets in Their Courses')
    make_menu(main_window)
    image_canvas = Canvas(main_window, width = 500, height = 500)
    image_canvas.pack()
    image = ImageTk.PhotoImage(Image.open('mars.png'))  
    image_canvas.create_image(20,20, anchor=NW, image=image)    
    main_window.mainloop()