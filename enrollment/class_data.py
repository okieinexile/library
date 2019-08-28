# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 08:53:24 2019

@author: bwinters
"""
import pandas as pd
from bs4 import BeautifulSoup
import requests 
from itertools import chain
from os import getcwd, listdir
from os.path import isfile, join
import csv




HEADERS = {'User-Agent': 'Mozilla/5.0'}
BASE_URL = 'https://go.pittstate.edu/_sched/getsched?TERM='
url_str = '{}{}{}&DEPT={}'
CLEAN_PATH = 'clean_data/{}_clean_{}.csv'
REPORTS = 'reports/{}.xlsx'
UNCLEAN_DATA = 'unclean_data/'
CLEAN_DATA = 'clean_data/'


class Section:
    """
    Section is intended to store and manipulation the data from the section of
    a class.  It has been designed to be used within the class SemesterSchedule,
    which is defined below.
    """
    def __init__(self, data_dict: dict) -> None:
        """
        Section is initiated by a dictionary, data_dict, which contains the
        data for a particular class section.
        """
        
        # Set up the attributes of the class.
        self.id: str = data_dict['course_id']
        self.title: str = data_dict['title']
        self.credit: str = data_dict['credit']
        self.instructor: str = data_dict['instructor']
        self.maximum: str = data_dict['max_cap']
        self.enrollment: str = data_dict['enrollment']
        
        self.term: str = None
        
        return None
    
    def section_enrollment(self) -> int:
        """
        If self.enrollment can be converted to an integer, then return that as
        the enrollment. Otherwise assume it is a comment string of some sort
        and return a 0.
        """
        try:
            out = int(self.enrollment)
        except:
            out = 0
        return out
    
    def course_number(self) -> int:
        """
        Return the course number of the class section.
        """
        return int(self.id.split('*')[1])
    
    def course_code(self) -> str:
        """
        Return the course code associated to a section. This will be in the form
        PREFIX*nnn  .
        """
        return self.id.split('*')[0] + '*' + self.id.split('*')[1]
    
    def course_prefix(self) -> str:
        """
        Returns the course PREFIX of the section.
        """
        return self.id.split('*')[0]
    
    def section_credit(self) -> int:
        """
        If the number of credit hours associated with a section can be converted
        to an integer, return that integer.  If it is a code, return 0.
        """
        try:
            out = int(self.credit)
        except:
            out = 0
        return out
    
    def student_credit_hours(self):
        """
        Return the number of student credit hours generated by a section.
        """
        return self.section_credit() * self.enrollment()


    
class SemesterSchedule:
    """
    The purpose of this class is to hold and manipulate the data of all the 
    sections in the semester schedule of a department for one semester.
    
    """

    def __init__(self, dept_code: str, term_code: str) -> None:
        """
        Uses the department code and the term code to call up the particular
        schedule from the collected data.
        """
        
        ### Use pandas to read the dats from a clean file into a dataframe.
        self.data_frame = pd.read_csv(CLEAN_PATH.format(dept_code, term_code))
        
        # Set the class attributes.
        self.term: str = term_code
        self.list_of_sections: list = self.data_frame['course_id']
        self.sections: list = self._set_sections()
        self.course_codes: set = set( s.course_code() for s in self.sections)
        self.course_numbers: set = set( s.course_number() for s in self.sections)
        self.course_prefixes: set = set(s.course_prefix() for s in self.sections)
        return
    
    def _set_sections(self) -> list:
        """
        This will return a list of sections for the semester to be used in the 
        initialization of the class.
        """
        out_list: list = list()
        self.data_frame.set_index('course_id')
        for key in self.data_frame.index:
            sec_dict: dict = dict()
            sec_dict['course_id']=self.data_frame.loc[key,'course_id']
            sec_dict['instructor']=self.data_frame.loc[key,'instructor']
            sec_dict['title']=self.data_frame.loc[key,'title']
            sec_dict['credit']=self.data_frame.loc[key,'credit']
            sec_dict['max_cap']=self.data_frame.loc[key,'max_cap']
            sec_dict['enrollment']=self.data_frame.loc[key,'enrollment']
            out_list.append(Section(sec_dict) )         
        return out_list
    
    def year(self) -> int:
        """
        This will take the data from the term_code, stored as self.term,
        and return the year of the term as yyyy in integer.
        """
        return 2000 + int(self.term[0:2])
    
    def semester(self) -> str:
        """
        This will take data from the term_code, stored as self.term, and
        return the name of the semester as a word in English.
        """
        my_table: dict ={
                'SU': 'Summer',
                'SP': 'Spring',
                'WF': 'Fall'
                }
        return my_table[self.term[2:]]
        
    def semester_courses(self):
        """
        This returns the sections from a semester an their enrollment as a 
        pandas dataframe.
        """
        
        # Make the set of course codes into an ordered list.
        course_list: list = sorted(list(self.course_codes))
        
        # Calculate enrollment
        enrollment: list =list( map( self.course_enrollment , course_list))
        
        # Put into a dictionary and from there into a pandas dataframe.
        out: dict = {
                    'courses': course_list, 
                    'enrollment': enrollment
                    }
        df: pd.DataFrame = pd.DataFrame(out)
        df.set_index('courses', inplace = True)
        return df
    
    def course_enrollment(self, course_code: str) -> int:
        """
        If the course with course_code was not offered returns 0; otherwise
        returns the total enrollment of all the sections.
        """
        if not (course_code in self.course_codes):
            return 0
        flt = filter(lambda s: s.course_code() == course_code, self.sections)
        enrollments = [s.section_enrollment() for s in flt]
        return sum(enrollments)
    
    def total_enrollment(self) -> int:
        """
        Returns the total enrollment of all the sections of the semester.
        """
        enrollment: list  = [s.enrollment for s in self.sections]
        return sum(enrollment)
    
    def total_student_credit_hours(self) -> int:
        """
        Returns the total number of student credit hours generated during the
        semester.
        """
        sch: list = [s.enrollment * s.section_credit() for s in self.sections]
        return sum(sch)
    
    def semester_report(self) -> dict:
        """
        Returns a dictionary of statistics for the semester.
        """
        return {
                'total_enrollment': self.total_enrollment(),
                'student_credit_hours': self.total_student_credit_hours()              
                }
    
      


class EnrollmentData:
    """
    The purpose of this class is to hold and manipulate the enrollment data 
    for all of the semesters between first_term and last_term, inclusive.
    """

    def __init__(self, dept_code: str, first_term: str, last_term: str ) -> None:
        
        # Create an inclusive list of terms from first_term to last_term
        tg = EnrollmentData.term_gen(first_term)
        self.terms: list = list()
        for t in tg:
            if EnrollmentData.term_precedes(t, last_term):
                self.terms.append(t)
            else:
                break
        
        # set up attributes    
        self.dept_code: str = dept_code
        self.schedules: dict = dict( (term, SemesterSchedule(dept_code, term))
                                for term in self.terms)    
        self.course_codes: set = set(
                chain.from_iterable(
                [schedule.course_codes 
                 for schedule in self.schedules.values()]))
    
        return None
    
    def ay_report(self, first_year: int, last_year: int, makefile: bool = True) -> None:
        """
        This will generated a report for the class enrollment for academic years
        first_year through last_year, inclusive.  As a default the results
        are stored in an excel file.
        """
        # If the years are typed in as yy instead of yyyy we correct it.
        if first_year < 100:
            first_year = first_year + 2000
        if last_year < 100:
            last_year = last_year + 2000
        
        # Put the course codes in an ordered list.
        courses: list = sorted(list(self.course_codes))
        for y in range(first_year, last_year + 1):
            if not self._ay_data_present(y):
                raise ValueError(
                      'Not all of the terms of academic year ' + \
                      '{} are present in the data'.format(y)
                        )
        
        
        # Calculate the enrollments and put the results in a dictionary.
        enrollment_dict: dict = dict()
        for y in range(first_year, last_year + 1):
            enrollment_dict[y] = list()
            for course in courses:                
                enrollment_dict[y].append(self.ay_course_enrollment(course, y))
        enrollment_dict['course'] = courses
        
        # Put results in a dataframe and save it.
        df: pd.DataFrame = pd.DataFrame(enrollment_dict)
        df.set_index('course', inplace = True)
        if makefile:
            filename ='{} from {} to {}'.format(self.dept_code, 
                                                first_year, last_year)
            df.to_excel(REPORTS.format(filename))
        return df
        
    
    def course_enrollment(self, course_code: str, term_code: str) -> int:
        """
        This will return the total enrollment of all of the sections in a 
        course in a particular term.
        """
        if not (course_code in self.course_codes):
            raise ValueError(
                    '{} was not offered in range listed.'.format(course_code)
                    )
        if not (term_code in self.terms):
            raise ValueError( 
                    '{} is not within the range of the Data.'.format(term_code)
                    )
        return self.schedules[term_code].course_enrollment(course_code)
    
    def ay_course_enrollment(self, course_code: str, year: int) -> int:
        """
        This will give the total number of students enrolled in a course
        in the academic year: year = yyyy.  Academic year consists of the WF, 
        SP, and SU terms.
        """
        
        terms: list = self.ay_term_codes( year )
        for t in terms:
            if not (t in self.terms):
                raise ValueError(
                        '{} not in range of data.'.format(t)
                        )
        if not (course_code in self.course_codes):
            raise ValueError(
                    '{} not offered in range of data.'.format(course_code)
                    )
        enrollment: int = sum(
                list(
                    map(lambda t: self.course_enrollment(course_code,t), terms)
                        ))
        return enrollment
    
    
    def ay_term_codes(self, year: int) -> list:
        """
        This generates all of the term codes in a given academic year.
        """
        if year > 100:
            year = year % 100
        return [str(year)+'WF', str(year + 1)+'SP', str(year + 1)+'SU']
    
    def _ay_data_present(self, year: int) -> bool:
        """
        This determines whether or not all of the terms of the given 
        academic year are conntained within the current EnrollmentData 
        object.
        """
        terms: list = self.ay_term_codes(year)
        for t in terms:
            if not (t in self.terms):
                return False
        return True
    
    @staticmethod
    def term_precedes( term1: str, term2: str) -> bool:
        """
        This determines where term1 precedes or equals term term2.
        """
        if term1 == term2:
            return True
        if int(term1[0:2]) < int(term2[0:2]):
            return True
        if int(term1[0:2]) > int(term2[0:2]):
            return False
        if term1[2:] == 'SP':
            return True
        if term1[2:] == 'SU' and term1[2:] == 'WF' :
            return True
        if term1[2:] == 'SU' and term1[2:] == 'SP' :
            return False
        return False
    
    @staticmethod
    def term_gen( term: str) ->str:
        """
        This generator function begins at term and generates all subsequent
        academic terms.
        """
        yr: int = int(term[0:2])
        sm: str = term[2:]
        yield term
        while True:
            if sm =='WF':
                term = str(yr + 1) + 'SP'
            if sm =='SP':
                term = str(yr) + 'SU'
            if sm =='SU':
                term = str(yr) + 'WF'
            yield term
            yr: int = int(term[0:2])
            sm: str = term[2:]
            


class Scraper:
    """
    The purpose of this class is to scrape schedule data from the Web.
    """
    
    def __init__(self, dept_code: str, term_code: str) -> None:
        self.dept_code: str = dept_code
        self.term_code: str = term_code
        return None
    
    
    def scrape(self) -> None:
        """
        Once the Scraper object is created <object>.scrape() will capture the
        data and put it in a subfolder UNCLEAN_DATA which can be set in the 
        preamble of this file.
        """
        letters: list = self._get_dept_soup()
        sched_list = self.get_dept_data(letters)
        data_table = [['course_id','title','credit','instructor',
                     'max_cap','enrollment']]
        for row in sched_list:
            new_row = []
            new_row.append(row['course_id'])
            new_row.append(row['title'])
            new_row.append(row['credit'])
            new_row.append(row['instructor'])
            new_row.append(row['max_cap'])
            new_row.append(row['enrollment'])
            data_table.append(new_row.copy())
        fmt_str = '{}{}_dept_data_{}.csv'
        filename = fmt_str.format(UNCLEAN_DATA,
                                  self.dept_code, self.term_code)
        self.sv_data(data_table,filename)
        return None

    def _get_dept_soup(self) ->list:
        """
        This uses BeautifulSoup to get departmental data.
        """
        sched_url=url_str.format(
            BASE_URL,
            self.term_code[0:2],
            self.term_code[2:],
            self.dept_code
            )
        response=requests.get(
            sched_url,
            headers=HEADERS)
        soup=BeautifulSoup(response.content,"lxml")
        table=soup.find_all('table',{'class':"center90 bgsilver"})
        return table   
    
            
    def get_dept_data(self, letters: list) -> list:
        sections: list = list()
        for let in letters:
            sections = sections + self._get_schedule_data(let)
        return sections

    def _get_schedule_data(self, table) -> list:
        sections=[]
        for row in table.select('tr')[1:-1]:
            course_id=row.select('td')[0].text
            title=row.select('td')[1].text
            credit=row.select('td')[2].text
            try:
                instructor=row.select('td')[7].text
            except:
                instructor=None
            try:        
                max_cap=row.select('td')[8].text
            except:
                max_cap=None
            try:
                enrollment=row.select('td')[9].text
            except:
                enrollment=None
            sections.append({
                'course_id':course_id,
                'title':title,
                'credit':credit,
                'instructor':instructor,
                'max_cap':max_cap,
                'enrollment':enrollment
                }
            )
        return sections
    
    def sv_data(self, list_of_rows: list, filename: str) -> None:
        with open(filename, 'w') as pFile:
            csv_writer = csv.writer(pFile, delimiter = ',')
            for row in list_of_rows:
                csv_writer.writerow(row)
        return None



class DataCleaner:
    """
    When data is scraped from the website it is saved to a CSV file.  However
    these files are not necessarily in a form which can be take to a DateFrame
    by pandas.  DataCleaner provides methods to clean the files that have been
    scraped from the Web to transform them to a pandas friendly format.
    """ 
    def __init__(self, dept_code: str, dept_prefixes: list = None) -> None:
        """
        We initialize a DataCleaner with the department code and the course
        prefixes that are used by that department.
        """        
        self.dept: list = dept_code
        self.clean_data = CLEAN_DATA
        self.unclean_data = UNCLEAN_DATA
        self.prefixes: list = dept_prefixes
        self.prefixes.append(dept_code)
        return None
    
    def clean_csv(self) -> None:
        """
        This takes the files from the unclean directory, cleans them, and saves
        the result in the clean directory.
        """        
        directory_files: list = DataCleaner.files_in_directory(self.unclean_data)
        my_files = filter(lambda f: self.is_dept(f), directory_files)
        for filename in my_files:
            newfile = DataCleaner.new_name(filename)
            with open(self.clean_data + newfile,'w') as out:
                with open(self.unclean_data + filename,'r') as f:
                    out.writelines(next(f))
                    for line in f:
                        if self.is_dept(line):
                            out.writelines(line)
        return None    
    
    @staticmethod
    def files_in_directory(subdirectory: str = None) -> list:
        """
        This method generates a list of files in a particular subdirectory of
        the current working directory.
        """
        
        # If no subdirectory is specified, we simple use the current working
        # directory.      
        if subdirectory is None:
            mypath:str = getcwd()        
        else:
            mypath: str = getcwd() + '/' + subdirectory
        return [f for f in listdir(mypath) if isfile(join(mypath, f))] 
       
    def is_dept(self, string: str) -> bool:
        if self.prefixes is None:
            return (string[0:len(self.dept)] == self.dept)
        else:
            for pre in self.prefixes:
                if pre == string[0: len(pre)]:
                    return True
        return False
    
    @staticmethod 
    def new_name(filename: str) -> str:
        """
        This takes the name of an unclean file and transforms it to the name
        for the corresponding clean file.
        """        
        fmt: str = '{}_clean_{:02d}{}.{}'
        stub, ext = filename.split('.')
        dept, _, _, sem = stub.split('_')
        if len(sem) == 3:
            dt = sem[0:1]
            rm =sem[1:]
        else:
            dt = sem[0:2]
            rm = sem[2:]
        return fmt.format(dept, int(dt),rm, ext)
    
