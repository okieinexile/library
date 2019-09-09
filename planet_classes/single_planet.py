# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:44:33 2019

@author: bwinters
"""
import my_widgets as mw

import tkinter  as tk
from tkinter.messagebox import showerror
from PIL import ImageTk, Image 

from planet_classes import PlanetarySystem

from typing import List, Tuple

planetary_system: PlanetarySystem = PlanetarySystem.read_csv()
MONTHS =['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']

month: str = 'Not set'
day: int = -1

def not_done() -> None:
    showerror('Not implemented','Not yet available')
    return None
def grab_values() -> None:
    global month
    month = month_radio.report()
    day = day_slider.var.get()
    print(f'{month} {day}')
    return 

if __name__ == '__main__':
    main_window = tk.Tk()
    main_window.title(string = 'Planets')
    #sd = mw.SelectDate(main_window, 2019)

    

    main_window.mainloop()



