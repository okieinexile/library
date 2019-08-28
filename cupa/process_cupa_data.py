# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:46:49 2019

@author: bwinters
"""

from cupa_data import FacultyData
import tkinter


def first_merger(rank_file: str, salary_file:str, 
                 preliminiary_file: str) -> FacultyData:
    FD: FacultyData = FacultyData(rank_file, salary_file)
    FD.create_preliminary_file(preliminiary_file)
    print("""
          You will now need to create the spec file manually to take 
          multidisciplinary departments into consideration.  In particular,
          EMLL has to be split into ENG and ML.
          You may also use this as an opportunity to do whatever other 
          fixing that needs to be done.
          """)
    return FD

def second_merger(faculty_data: FacultyData, cupa_norm_file: str, 
                  specialty_file: str, median_file: str ) -> None:
    faculty_data.update_with_median_salary(cupa_norm_file, specialty_file, 
                                           median_file)
    
    return None

def test():
    FD = FacultyData('rank_data.xlsx','raw_salary.xlsx')
    FD.create_preliminary_file('prelim.xlsx')
    FD.update_with_median_salary('cupa_norms.xlsx', 'spec_data.xlsx','median_data.xlsx')
    return FD.create_target_salaries('target_salaries.xlsx')
    