# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:33:35 2019

@author: bwinters
"""
from datetime import datetime
import pandas as pd
from numpy import mean, exp
from typing import Dict, List, Tuple, Generic
from difflib import get_close_matches

class CupaDictionary:
    
    
    def __init__(self, filename: str) -> None:
        self.dataframe: pd.DataFrame = pd.read_excel(filename)
        self.filename: str = filename
        return None
    
    def median_lookup(self, rank: str, specialty: str) -> float:
        if specialty == 'CAS':
            return -1
        specialty_frame: pd.DataFrame = self.dataframe.loc[self.dataframe.loc[:,'SPEC'] == specialty]
        specialty_frame.set_index('RANK', inplace = True)
        return specialty_frame.loc[rank.upper(),'MEDIAN']
    
class FacultyMember:
    
    
    def __init__(self, first_name: str, last_name: str, data: Dict = dict()) -> None:
        self.first_name: str = first_name
        self.last_name: str = last_name
        self.data: Dict = data
        return None
    
    def __repr__(self) -> str:
        out: str = f'{self.last_name}, {self.first_name} '
        for key, value in self.data.items():
            out += f'{key}: {value} '
        return out
    
    def id(self) -> str:
        return '{:07}'.format(int(self.get_attribute('id')))
    
    def rank(self) -> str:
        return self.get_attribute('rank/attained')[0]
    
    def years_in_rank(self) -> int:
        return datetime.now().year - self.year_promoted_to_current_rank()
    
    def year_promoted_to_current_rank(self) -> int:
        return self.get_attribute('rank/attained')[1]
    
    def get_attribute(self, attribute: str) -> str:
        if attribute in self.data.keys():
            return(self.data[attribute])
        raise ValueError(f'{attribute} not found in {self.data.keys()}')
    
    def set_attribute(self, attribute: str, value: Generic) -> None:
        self.data[attribute]: Generic = value
        return None
    
class FacultyList:
    
    def __init__(self, faculty_list: List[FacultyMember]) -> None:
        self.faculty_list: List[FacultyMember] = faculty_list
        return None
        
    
    def DataFrame(self) -> pd.DataFrame:
        last_name: List[str] = [teacher.last_name for teacher in self.faculty_list]
        first_name: List[str] = [teacher.first_name for teacher in self.faculty_list]
        ID: List[str] = [teacher.id() for teacher in self.faculty_list]
        rank: List[str] = [teacher.rank() for teacher in self.faculty_list]
        years_in_rank: List[str] = [teacher.years_in_rank() for teacher in self.faculty_list]
        dept_code: List[str] = [teacher.get_attribute('dept_code') for teacher in self.faculty_list]
        return pd.DataFrame({
                'last_name': last_name,
                'first_name': first_name,
                'id': ID,
                'rank': rank,
                'years_in_rank': years_in_rank,
                'dept_code': dept_code
                })
    
    @classmethod
    def read_excel(cls, filename: str) -> 'FacultyList':
        df: pd.DataFrame = pd.read_excel(filename)
        faculty_list: List[FacultyMember] = list()
        for i in df.index:
            if (cls.get_rank(df, i) is not None) and cls.is_artsandscience(df, i):
                faculty_list.append(FacultyMember(df.loc[i,'first'], 
                                                  df.loc[i,'last'],
                                                  {'rank/attained':cls.get_rank(df, i),
                                                   'id': df.loc[i, 'Person Number'],
                                                   'college': cls.get_college(df, i),
                                                   'dept_code': cls.get_dept_code(df, i),
                                                   'first_name': df.loc[i,'first'],
                                                   'last_name': df.loc[i,'last'],
                                                   } ) )
        return cls(faculty_list)
    
    def retrieve_faculty(self, last_name: str, first_name: str) -> FacultyMember:
        # scores: List[Tuple[FacultyMember,float]] = list()
        try:
            last_best = get_close_matches(last_name, self.last_names())[0]
        except:
            raise ValueError(f'{last_name} is not in our faculty list.')
            
        try:
            first_best = get_close_matches(first_name, self.first_names())[0]
        except:
            raise ValueError(f'{first_name} is not in our faculty list')
            
            
        for teacher in self.faculty_list:
            if (teacher.last_name == last_best) and (teacher.first_name == first_best):
                return teacher
        raise ValueError(f'{last_name}, {first_name} not found')
    
    def in_list(self, teacher: FacultyMember) -> bool:
        last_flag: bool = teacher.last_name in self.last_names()
        first_flag: bool = teacher.first_name in self.first_names()
        return last_flag and first_flag
    
    def add_salary_data(self, salary_file) -> None:
        df: pd.DataFrame = pd.read_excel(salary_file)
        for i in df.index:
            last_name, first_name = self.parse_name(df, 'name', i)
            try:
                teacher = self.retrieve_faculty(last_name, first_name)
            except:
                continue
            teacher.set_attribute('salary', round(df.loc[i, 'base'], 2))
            teacher.set_attribute('years_in_rank',teacher.years_in_rank())
            teacher.set_attribute('rank', teacher.rank())
        return None
    
    def last_names(self) -> List[str]:
        return list(map(lambda faculty: faculty.last_name, self.faculty_list))
    
    def first_names(self) -> List[str]:
        return list(map(lambda faculty: faculty.first_name, self.faculty_list))    
    
    def faculty(self) -> List[str]:
        full_name = zip(self.last_names(),self.first_names())
        return list(map(lambda name: '{}, {}'.format(name[0],name[1]), full_name))
    
    def ranks(self) -> List[str]:
        return [teacher.rank() for teacher in self.faculty_list]
    
    @staticmethod
    def get_rank(data_frame: pd.DataFrame, row_number: int) -> Tuple[str,str]:
        out: str = None
        for rank in ['inst', 'asst_prof', 'assoc_prof', 'prof']:
            if isinstance(data_frame.loc[row_number, rank], pd.Timestamp):
                out = (rank, data_frame.loc[row_number, rank].year)
        return out
    
    @staticmethod
    def get_college(data_frame: pd.DataFrame, row_number: int) -> str:
        try:
            out = data_frame.loc[row_number, 'Division'].split('-')[1].strip()
        except:
            out = None
        return out
    
    @staticmethod
    def get_dept_code(data_frame: pd.DataFrame, row_number: int) -> str:
        unit: str = data_frame.loc[row_number, 'unit']
        translation: Dict ={
                'Nursing, Irene Ransom Bradley School of': 'NURSING',
                'Biology': 'BIOL',
                'Mathematics': 'MATH',
                'Music': 'MUSIC',
                'Arts and Sciences, College of': 'CAS',
                'English and Modern Languages': 'EML',
                'Art': 'ART',
                'History, Philosophy and Social Sciences':  'HPSS',
                'Chemistry': 'CHEM',
                'Communication': 'COMM',
                'Physics': 'PHYSICS',
                'Family and Consumer Sciences': 'FCS'
                }
        if not (unit in translation.keys()):
            raise ValueError(f'{unit} not in Arts and Sciences')
        return translation[unit]
    
    @staticmethod
    def parse_name(data_frame: pd.DataFrame, name_field: str, row_number) -> Tuple[str,str]:
        entry: str = data_frame.loc[row_number, name_field]
        if entry =='Vacant':
            return 'Position', 'Vacant'
        try:
            last_name, first_name = data_frame.loc[row_number, name_field].split(',')
        except:
            print('Something went wrong.')
            return entry, ''
        return last_name, first_name
        
        
    @classmethod
    def is_artsandscience(cls, data_frame: pd.DataFrame, row_number: int) -> bool:
        return cls.get_college(data_frame, row_number ) == 'College of Arts & Sciences'
        
        
class FacultyData:
    
    
    def __init__(self, rank_file: str, salary_file: str) -> None:
        self.FacultyList: FacultyList = FacultyList.read_excel(rank_file)
        self.faculty: List[FacultyMember] = self.FacultyList.faculty_list
        self.FacultyList.add_salary_data(salary_file)
        self.preliminary_file: str = None        
        self.median_file: str = None
        self.dataframe: pd.DataFrame = None
        return None
    
    def data(self) -> List[float]:
        dictionaries: List[Dict] = list()
        for teacher in self.faculty:
            dictionaries.append(teacher.data)
        return dictionaries
    
    def to_excel(self, filename: str) -> None:
        self.dataframe.sort_values(by = ['dept_code'], inplace = True)
        self.dataframe.to_excel(filename)
        return None
    
    def create_preliminary_file(self, preliminary_file: str) -> None:
        df: pd.DataFrame = self.DataFrame()
        df.sort_values(['dept_code'], inplace = True)
        df.to_excel(preliminary_file)
        self.preliminary_file: str = preliminary_file
        return None
    
    def update_with_median_salary(self, cupa_file: str, specialty_file:str,
                                  median_file: str) -> None:
        if self.preliminary_file is None:
            raise ValueError('You need to create a preliminary file.')
        cupa_dictionary: CupaDictionary = CupaDictionary(cupa_file)
        self.dataframe: pd.DataFrame = pd.read_excel(specialty_file)
        median: List[float] = list()
        ratio: List[float] = list()
        for i in self.dataframe.index:
            rank = self.dataframe.loc[i,'rank'].upper()
            specialty = self.dataframe.loc[i,'spec'].upper()
            median.append(cupa_dictionary.median_lookup(rank, specialty))
            ratio.append(self.dataframe.loc[i,'salary'] 
                        / cupa_dictionary.median_lookup(rank, specialty))
        self.dataframe['median'] = median
        self.dataframe['ratio'] = ratio
        self.dataframe.to_excel(median_file)
        self.median_file: str = median_file
        return None
    
    def create_target_salaries(self, target_salaries: str) -> None:
        if self.median_file is None:
            raise ValueError('You need to update the date with median salaries.')
        sigmoid = lambda x: 1 / (1 + exp(-x))
        target_ratio: Dict = dict()
        for rank in self.distinct_ranks():
            sub_frame =self.dataframe[self.dataframe['rank'] == rank]
            base, delta_ratio = self.span(sub_frame['ratio'])
            average_years_in_rank = mean(sub_frame['years_in_rank'])
            slope = delta_ratio / average_years_in_rank
            target_ratio[rank] = (slope, base)
        
        adjustment_score: List[float] = list()
        for row_num in self.dataframe.index:
            rank = self.dataframe.loc[row_num, 'rank']           
            adjustment_score.append(
                    sigmoid(
                    target_ratio[rank][0] * self.dataframe.loc[row_num,'years_in_rank']  
                                 + target_ratio[rank][1]
                                 )
                    )
        self.dataframe['adjustment_score'] = adjustment_score
        self.dataframe.sort_values(['rank', 'adjustment_score'], 
                                   inplace = True,
                                   ascending = False)
        self.dataframe.to_excel(target_salaries)    
        return None
     
    def DataFrame(self) -> pd.DataFrame:
        key_list: List = self.data()[0].keys()
        include = ['dept_code','last_name', 'first_name','salary','years_in_rank', 'rank']
        out_dict: Dict = dict((key, list()) for key in key_list if (key in include))
        for i, dictionary in enumerate(self.data()):
            for key in key_list:
                if not (key in include):
                    continue
                out_dict[key].append(self.data()[i][key])
        return pd.DataFrame(out_dict)


    def distinct_ranks(self) -> List[str]:
        return set(self.dataframe['rank'])
    
    @staticmethod
    def span(data: iter) -> Tuple[float,float]:
        return min(data), max(data) - min(data)
            