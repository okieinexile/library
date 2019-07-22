# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:54:06 2019

@author: bwinters
"""
import pandas as pd
import numpy as np
import pylab as plt


# =============================================================================
# The following path.file must be set in order for these to work at all.
# The file must be in pandas.read_csv accessable format with the column names
# name,	m,	e,	a,	i,	Omega,	omega,	perihelion, 	period,	tilt
# 
# name is a string	
# m is mass but is not accessed except for informational purposes in kilograms
# e is eccentricity
# a  is semi-major axis in astronomical units
# i is inclination	in degrees
# Omega is the longitude of the ascending node	in degrees
# omega is the argument of the 	perihelion	 in degrees
# period is the orbital period in days	
# tilt the tilt of the axis of rotatin in degrees; it only really works for 
#     earth at this point
# 
# =============================================================================





# This is the default location of the data file.
PLANETARY_DATA='data/cleaned_data.csv'
UNCLEAN_DATA = 'data/planetary_data_2019.csv'

def clean_data() -> None:
    """This is to be used whenever the data in UNCLEAN_DATA is
    updated.
    """
    df: pd.DataFrame = pd.read_csv(UNCLEAN_DATA)
    st = SkyTools((0,0,0,1), (0,0,0,1), '12/31/2800')
    new_list: list = list()
    for item in df['perihelion']:
        new_list.append(st._ord_day(item)) 
    df['perihelion'] = new_list
    df.set_index('name', inplace = True)
    df.to_csv(PLANETARY_DATA)
    return None



# =============================================================================
# Here we define the Planet class
# =============================================================================

class Planet():
    
    def __init__(self, name: str) -> None:
        self._data_path: str = PLANETARY_DATA
        self.name: str = name
        self._data: dict = self._read_csv()
        self.matrix = self._get_matrix()
        self.tilt : np.matrix = self._rotation(self._data['tilt'],[1,0,0])          
        return None
# Exposing data
        
    def perihelion(self) -> float:
        return self._data['perihelion']
    
    def period(self) -> float:
        return self._data['period']
    
    def semi_major_axis(self) -> float:
        return self._data['a']    
    
    def eccentricity(self) -> float:
        return self._data['e']    
    
    def Omega(self) -> float:
        return self._data['Omega']
    
    def omega(self) -> float:
        return self._data['omega']  
    
    def inclination(self) -> float:
        return self._data['i']

    def data_path(self) -> str:
        return self._data_path

# Positional functions

    def position(self, day: int) -> tuple:
        """
        day is the number of days since perihelion
        """
        # Find the position in the ecliptic
        day_angle: float = 2 * np.pi / self.period()
        e: float = self.eccentricity()
        a: float = self.semi_major_axis()
        M: float = day_angle * day
        nu: float = self._get_nu(M)
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
        x, y = r * np.cos(nu), r * np.sin(nu)
        
        # Move the planet to its place in space
        w = self.matrix * (np.matrix([x, y, 0]).T)
        return (w[0,0], w[1,0], w[2,0])
        
    def _orbit(self, final_day: int) -> list:
        """
        Will calculate the elliptical orbit of the planet from its perihelion 
        to the 
        
        final_day, 
        
        which is the integer value of the day with day 0 at our base
        day (1/1/1800) established in the passed.
        """
        e: float = self.eccentricity()
        a: float  =self.semi_major_axis()
        perihelion: int = int(self.perihelion())
        n: float = 2 * np.pi / self.period()
        orbit_data: list = list()   
        for i in range(perihelion,final_day+1):
            M = (i - perihelion ) * n
            nu = self._get_nu(M)
            r = a * (1 - e ** 2) / ( 1 + e * np.cos(nu) )
            orbit_data.append((i ,
                               r*np.cos(nu) ,
                               r*np.sin(nu) ))
        return orbit_data    

    def _put_in_position(self, final_day: int) -> iter:
        """
        This puts the orbit of a planet in 3D position in the solar system.
        """
        perihelion: int = int(self.perihelion())
        if final_day < perihelion:
            raise ValueError('The final day is earlier than the perihelion')            
        orbit_data: iter = zip( range(perihelion, final_day + 1),
                map(self.position, range(0, final_day - perihelion + 1)))  
        return orbit_data  
    
    def plot(self, particular_day: int = None) -> None:
        """
        particular_day is an integer which is the number of days since
        the base day of 1/1/1800.
        """
        final_day: int = int(self.perihelion() + self.period())
        position_data: list = self._put_in_position(final_day)
        X: list = list(map(lambda x: x[1], position_data))
        Y: list = list(map(lambda x: x[2], position_data))
        plt.axis('equal')
        # plot only total orbit
        if particular_day is None:
            plt.plot(X,Y,'k-')
            return None
        
        # plot the total orbit with a particular day distinguished
        if not (particular_day is None):
            x, y, z = self.position(particular_day)
            plt.plot(X, Y, 'k-', x, y, 'g^')
            return None
        raise ValueError('{} is invalid for {}'.format(particular_day, self.name))
        return None

# Mathematical Functions
      
    def _get_nu(self, M: float)->float:
        """This uses Newton's Method to solve Kepler's Equation
        to obtain nu. It uses 
        
        M the mean anomaly and 
        e the eccentricity
        
        """
        #set the seed for E with M
        E: float = M
        e: float = self.eccentricity()
        tol: float = 1e-14
        testFunction: float = E - e * np.sin(E) - M
        
        #iterate until within tolerance
        while abs(testFunction) > tol:
            der = 1 - e * np.cos(E)
            E = E - testFunction / der
            testFunction = E - e * np.sin(E) - M
        tnu2: float = np.sqrt((1 + e) / ( 1 - e )) * np.tan(E / 2)
        nu: float = 2 * np.arctan( tnu2 )
        
        # the answer should be between 0 and 2pi
        if nu < 0:
            nu = nu + 2 * np.pi
        return nu

    def _get_matrix(self) -> np.matrix:
        """ 
        This creates a position matrix out of a planets orbital elements.
        """
        
        # obtain the various orbital elements
        Omega: float = self.Omega()
        omega: float = self.omega()
        i: float = self.inclination()
        
        # set up the matrices for the various orbit elements
        R1: np.matrix = self._rotation(Omega,[0,0,1])
        R2: np.matrix = self._rotation(i,[np.cos(Omega),np.sin(Omega),0])
        R3: np.matrix = self._rotation(omega,[R2[0,2],R2[1,2],R2[2,2]])
        R: np.matrix = R3 * R2 * R1
        return R    
       
    def _rotation(self,theta: float, axisList: iter) -> np.matrix:
        """
        This produces the rotation of matrix of angle theta about the 
        axis whose vector is given in the axisList
        """
        axis: np.matrix = np.matrix(axisList)
        phi: float = theta * np.pi / 180
        u:float = (1 / (axis * axis.T)[0,0])  *axis
        v: float = np.sin(phi / 2) * u
        r: float = np.cos(phi / 2)
        v1: float = v[0,0]
        v2: float = v[0,1]
        v3: float = v[0,2]
        p: float = r**2 - (v * v.T)[0,0]
        M: np.matrix = np.matrix( ( (p + 2 * v1**2,  2 * (v1 * v2-r * v3), 2 * (v1 * v3 + r * v2)),
                                  (2 * (v1 * v2 + r * v3), p - 2 * v2**2, 2 * (v2 * v3 - r * v1)),
                                  (2 * (v1 * v3 - r * v2),  2 * (v2 * v3 + r * v1), p + 2 * v3**2)))     
        return(M) 

# File handling function

    def _read_csv(self):
        # Planetary data is the file of a pandas dataframe that includes
        # our data
        df: 'dataframe' = pd.read_csv(self._data_path)
        df.set_index('name',inplace = True)
        if self.name not in df.index:
            raise ValueError('Planet Name {} in not in our list.'.format(self.name))
        my_data: dict = dict()
        my_data['m']: float = df.loc[self.name, 'm']
        my_data['e']: float = df.loc[self.name, 'e']
        my_data['a']: float = df.loc[self.name, 'a']
        my_data['i']: float = df.loc[self.name, 'i']
        my_data['Omega']: float = df.loc[self.name, 'Omega']
        my_data['omega']: float = df.loc[self.name, 'omega']
        my_data['perihelion']: float = df.loc[self.name, 'perihelion']
        my_data['period']: float = df.loc[self.name, 'period']
        my_data['tilt']: float = df.loc[self.name, 'tilt']
        return my_data
    
    def set_path(self, user_data: str ) -> None:
        """
        This will reset the data path from the default
        """
        self._data_path: str = user_data
        return None

# =============================================================================
# Here we define the PlanetarySytem class
# =============================================================================
    
class PlanetarySystem:
    
    def __init__(self, list_of_names: list = None, base_planet: str = None) -> None:
        self.filename: str = PLANETARY_DATA
        if list_of_names is None:
            self.names: list = None
            self.base: str = base_planet
            self.planet: dict = None
            self.base_planet: Planet = None
            self._recent_perihelion: float = None
            return None
        if base_planet not in list_of_names:
            raise ValueError('{} is not in the list of names.'.format(base_planet))
            
        self.names: list = list_of_names
        self.base: str = base_planet
        self.planet: dict = self._get_planets()
        self.base_planet: Planet = self.planet[self.base]
        self._recent_perihelion: float = max(self.planet[key].perihelion() 
                                             for key in self.names)
        return None
    
    def read_csv(self, filename: str = PLANETARY_DATA, 
                 base_planet: str = 'Earth') -> None:
        df: pd.DataFrame = pd.read_csv(filename)
        df.set_index('name',  inplace = True)
        self.names: list = list(df.index)
        if not(base_planet in self.names):
            raise ValueError('{} not in file list'.format(base_planet))
        self.base: str = base_planet
        self.planet: dict = self._get_planets()
        self.base_planet: Planet = self.planet[self.base]
        self._recent_perihelion: float = max(self.planet[key].perihelion() 
                                             for key in self.names)
        return None
        
    def _get_planets(self) -> dict():
        """
        This creates a dictionary of planets for the planetary system.
        """
        my_planets: dict = dict((key, Planet(key)) for key in self.names)
        return my_planets
    
    def _get_relative_coordinates(self, final_day: int) -> dict:
        """
        This computes the coordinates of each planet (excluding the base)
        with respect to the base.  The output is a dictionary of dictionary. 
        The outer dictionary is keyed to planets; the inner dictionary is keyed 
        to (absolute) day of the orbit.
        """
        base_orbit: list = self.planet[self.base]._put_in_position(final_day)
        first_day_base:float = min(map(lambda x: x[0], base_orbit))
        base_data: dict = dict((x[0],(x[1], x[2], x[3])) for x in base_orbit)
        relative_position: dict = dict()
        for name in self.names:
            if name == self.base:
                continue
            relative_position[name]:dict = dict()
            planet_orbit = self.planet[name]._put_in_position(final_day)
            first_day_planet = min(map(lambda x: x[0], planet_orbit))
            planet_data = dict((x[0],(x[1], x[2], x[3])) for x in planet_orbit)
            for day in range(max(first_day_base,first_day_planet), final_day+1):
                relative_position[name][day]= (
                        planet_data[day][0] - base_data[day][0],
                        planet_data[day][1] - base_data[day][1],
                        planet_data[day][2] - base_data[day][2])                                             
        return relative_position
    
    def position_of(self, planet_name: str, day: int) -> tuple:
        """
        planet_name is a name of a planet in the system
        day is the number of a day in an absolute system
        """
        if planet_name == self.base:
            print('Base Planet')
            return (0.0, 0.0, 0.0)
        planet: Planet = self.planet[planet_name]
        base_planet: Planet =self.base_planet
        days_since_perihelion: int = day - planet.perihelion()
        x1, y1, z1 = planet.position(days_since_perihelion)
        x0, y0, z0 = base_planet.position(day - base_planet.perihelion())       
        return (x1 - x0, y1 - y0, z1 - z0)
    
    def plot(self, particular_day: int = None) -> None:
        if particular_day < self._recent_perihelion:
            raise ValueError('Date must be later that {}'.format(
                    self._recent_perihelion()))
        for name, P in self.planet.items():
            P.plot(particular_day)
        return None
            
        
# =============================================================================
# Here we define the SkyTools class
# =============================================================================
    
class SkyTools:
    
    def __init__(self, latitude: tuple, 
                 longitude: tuple, final_day: str) -> None:
        self._BASE_DAY: str = '1/1/1800'
        self._D_M_S: str = '{}d{}m{}s'
        self._D_M_S_s: str = '{}d{}m{}s{}'
        self._H_M_S: str = '{}h{}m{}s'
        self._TIME: str = '{:02d}:{:02d}:{:02d}'
        self._FILES: str = 'data/{}_position_{}.csv'
        self._final_day: int = self._ord_day(final_day)
        self._system_set: bool = False
        self._view_sky: bool = False
        # Change latitude and longitude from the form
        # (degrees, minutes, seconds, sign) to degrees in decimal.
        # The tuple input form is meant for web applications
        self.latitude = latitude[3] * ( latitude[0] +
                                latitude[1] / 60 + 
                                latitude[2] / 3600) / 180*np.pi
        self.longitude = longitude[3] * (longitude[0] + 
                                  longitude[1] / 60 + 
                                  longitude[2] / 3600) / 180*np.pi
        return None
    
    def set_system(self, list_of_names: list, base_planet: str) -> None:
        self._system_set: bool = True        
        self.planetary_system: PlanetarySystem = PlanetarySystem(list_of_names, base_planet)
        self.base_planet: Planet = self.planetary_system.base_planet
        RC, SC = self._celestial_coords()
        self.rectangular_coords: dict = RC
        self.celestial_coords: dict = SC

        return None
    
    def view_sky(self, day: str, time: str) -> None:
        if self._system_set == False:
            raise ValueError('Must set system before using view_sky')
        
        self._day_number = self._ord_day(day)
        sidereal_times = self._side_day() 
        position_data = dict()
        for item in self.celestial_coords:
            position_data[item] = dict()
            x, y, z = self.rectangular_coords[item][self._day_number]
            rho = np.sqrt( x**2 + y**2 +z**2)
            x, y, z = x / rho, y / rho, z / rho
            vector = np.matrix((x,y,z)) 
            for gmt, side in sidereal_times:
                lam = side / 12*np.pi + self.longitude
                A, a = self._horizontal_coords(vector, lam)
                position_data[item][gmt] = A * 180 / np.pi, a * 180 / np.pi
        out_data = dict()
        
        for item in position_data:
            df_data = dict()
            df_data['times'] = list(position_data[item].keys())
            df_data['azimuth'] = list(map( lambda key: position_data[item][key][0], 
                   position_data[item].keys()))
            df_data['altitude'] = list(map( lambda key: position_data[item][key][1], 
                   position_data[item].keys()))
            df = pd.DataFrame(df_data)
            df.to_csv(self._FILES.format(item, day.replace('/','-')))            
            out_data[item]= position_data[item][time]
        self._view_sky: bool = True
        return out_data
    
    def plot(self, date: str) -> None:
        particular_day: int = self._ord_day(date)
        self.planetary_system.plot(particular_day)
        return None
    
    def position_of(self, planet_name: str, date: str, time: str ='00:00:00'):
        x, y, z = self.planetary_system.position_of(planet_name,
                                                    self._ord_day(date))
        data: dict = dict()
        data['name'] = planet_name
        decimal_hour: float = (float(time.split(':')[0]) +
                               float(time.split(':')[1]) / 60 + 
                               float(time.split(':')[2]) / 3600 )
        # Get the equatoral coordinates
        M: np.matrix = self.planetary_system.base_planet.tilt
        v: np.matrix = M * (np.matrix([x, y , z])).T
        x, y, z = v[0,0], v[1,0], v[2, 0]
        rho: float = np.sqrt(x**2 + y**2 + z**2)
        x, y, z = x / rho, y / rho, z / rho
        r: float = np.sqrt(x**2 + y**2)
        delta: float = np.arcsin(z)
        alpha: float = 2 * np.arctan( (1 - x/r) / (y / r))
        if alpha < 0 :
            alpha = alpha + 2 * np.pi
        data['declination']: str = self._rad_to_dms(delta)
        data['right_asension']: str = self._rad_to_hms( alpha )
        
        # Get the horizontal coordinates
        celestial_vector = np.matrix([x, y, z])        
        day: int = self._ord_day(date)
        lam = (self._greenwich_mean_sidereal_time(decimal_hour, day) / 12 * np.pi + 
               self.longitude )
        P = self._transition_matrix(lam ,self.latitude)
        hor_vector = celestial_vector*P
        x=hor_vector[0,0]
        y=hor_vector[0,1]
        z=hor_vector[0,2]
        azimuth, altitude = self._to_horiz(x,y,z)    
        data['azimuth'] = self._rad_to_dms(azimuth)
        data['altitude'] = self._rad_to_dms(altitude)
        
        return data
        
    
    

# =============================================================================
# Geometric functions
# =============================================================================

    def _celestial_coords(self) -> tuple:
        """
        returns a tuple of dictionaries of the positions of the system of
        planets as a tuple (dict, dict) where each dict is a dictionary of 
        dictionaries, the dictionary at the bottom being indexed by absolute 
        days
        """
        if self._system_set == False:
            raise ValueError('The system has not been set.')
        M = self.planetary_system.base_planet.tilt
        # rect_coords is a dictionary of dictionary; upper dictionary is
        # indexed by planets in the system; lower dictionary is planetary
        # coordinates indexed by days.
        rect_coords: dict = self.planetary_system._get_relative_coordinates(self._final_day)
        spherical_coords: dict = dict()
        for name, coord_list in rect_coords.items():
            spherical_coords[name] = dict()
            for day, coords in coord_list.items(): 
                x, y, z = coords
                x, y, z = M * (np.matrix([x, y , z])).T
                rho = np.sqrt(x**2 + y**2 + z**2)
                x, y, z = x / rho, y / rho, z / rho
                r = np.sqrt(x**2 + y**2)
                delta = np.arcsin(z)
                alpha = 2 * np.arctan( (1 - x/r) / (y / r))
                if alpha < 0 :
                    alpha = alpha + 2 * np.pi
                    
                spherical_coords[name][day] = (
                                self._rad_to_hms(alpha), 
                                self._rad_to_dms(delta)    )
        return rect_coords, spherical_coords
    def _horizontal_coords(self, celestial_vector: np.matrix , lam: float)-> tuple:
        """This takes the position of a planet at a position in 
        equatorial coordinates at a particular 
        lam: sidereal time
        and at
        lat: latitude  
        and return its azimuth and altitude.
        """
        P = self._transition_matrix(lam ,self.latitude)
        hor_vector = celestial_vector*P
        x=hor_vector[0,0]
        y=hor_vector[0,1]
        z=hor_vector[0,2]
        azimuth, altitude = self._to_horiz(x,y,z)
        return(azimuth, altitude)
    
    def _transition_matrix(self, lam: float, phi: float) -> np.matrix:
        """Given lam and phi this returns a transition matrix that
        will change equatorial coordiantes to horizontal coordinates and
        vice versa."""
        P=np.matrix([[-np.sin(lam), -np.sin(phi) * np.cos(lam), np.cos(phi)  *np.cos(lam)],
                     [ np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi) * np.sin(lam)],
                     [0,             np.cos(phi),               np.sin(phi)]])              
        return(P)
        
    def _to_horiz(self, x: float, y: float, z: float) -> tuple:
        """this changes the rectangular coordinates of x,y,z to spherical
        coordinates A and a, azimuth and altitude."""
        a = np.arcsin(z)
        r=np.cos(a)
        xp,yp=x/r,y/r
        if xp==0:
            return(np.pi/2,a)
        tn=(1-yp)/xp
        A=2*np.arctan(tn)
        if A<0:
            A=2*np.pi-abs(A)
        return(A,a)      

    
# =============================================================================
# Time and Date functions    
# =============================================================================
    
    def _ord_day(self, date: str) -> int:
        """ 
        mdy is a string of the form 'mm/dd/yyyy' of the day in question
        basemdy (below) is a string of the form 'mm/dd/yyyy' of a base day in the past
        which is after the change to Gregorian Time. We set it as BASE_DAY
        from the class.
        """
        # ds is the number of days that have passed since the BASE_DAY
        days_since = 0         
        base_month, base_day, base_year = [ int(x) for x in self._BASE_DAY.split('/')]
        month, day, year = [ int(x) for x in date.split('/')]
        if year < base_year:
            raise ValueError('Error: Date too early')
        # Add the days for every year
        for i in range(base_year, year):
            if self._is_leap(i):
                days_since += 366
            else:
                days_since += 365
        # Add the days for every month
        for i in range(1, month):
            days_since += self._month_length(i, year)
        # Add the days of the month
        days_since += day       
        return int(days_since)
    
    def _is_leap(self, year: int) -> bool:
        """
        year is the year in question. This returns true or false depending upon
        whether year is a leap year
        """
        if ((year % 4) != 0):
            return False
        if (year % 100 )!=0:
            return True
        if (year % 400)!=0:
            return False
        return True
    
    def _month_length(self, m: int, y: int) -> int:
        """
        Returns the number of days in the month m of year y
        """
        months = [31,28,31,30,31,30,31,31,30,31,30,31] 
        if (m < 1) or (m > 12):
            raise ValueError('Month number must be between 1 and 12')
        if self._is_leap(y) and m == 2:
            return 29
        return months[m-1] 

    def _side_day(self) -> tuple:
        """
        This will take 
        day which is an integer given the day in order
        and return (gmt,gmst) for that day
        """
        out: list = list()
        day: int = self._day_number
        for minute in range(1440):
            time = minute / 60       
            GMST = self._greenwich_mean_sidereal_time(time, day)
            out.append((self._TIME.format(minute // 60, minute % 60, 0), 
                        GMST))
        return out
    def _greenwich_mean_sidereal_time(self, time: float, day: int ) -> float:
        """
        This takes time, greenwich mean time, in the format decimal hours
        and returns greenwich mean sidereal in decimal hours.
        """
        days: int = day - self._ord_day('1/1/2000')
        D: float = days - 0.5 + time / 24
        b: float = 18.697374558
        m: float = 24.06570982441908 
        greenwich: float = (b + m * D) %24
        return(greenwich)        

# =============================================================================
# Formatting Functions 
# =============================================================================

    def _rad_to_hms(self, rad: float) -> str:
        """This changes decimal radian measure to hours, minutes, and seconds
        in a string format."""
        hours = 12 * rad / np.pi
        h = int(hours -hours % 1)
        hours = (hours - h) * 60
        m = int(hours - hours % 1)
        hours = (hours - m) * 60
        s = int(hours - hours % 1)
        return self._H_M_S.format(h, m, s)
    
    def _hms_to_dec(self, ra: str) -> float:
        h, ra = ra.split('h')
        m, ra = ra.split('m')
        s = ra[0:-1]
        out=float(h)+float(m)/60+float(s)/3600
        return out
    
    def _rad_to_dms(self, rad: float) -> str:
        """This changes decimal radian measure to degrees, minutes, and seconds
        in a string format."""    
        sign: int = int(np.sign(rad))
        rad: float = np.abs(rad)
        degrees: float = 180 * rad / np.pi
        d: int = int(degrees - degrees % 1)
        degrees:int = (degrees - d) * 60
        m: int = int(degrees - degrees % 1)
        degrees = (degrees - m) * 60
        s = int(degrees - degrees % 1)
        out: str = self._D_M_S.format(d, m, s)
        if sign > 0:
            out = ' + '+ out # + ' North'
        if sign < 0:
            out = ' - ' + out #+' South'
        return out 
    
    def angle_to_tuple(self, y: float ) -> tuple:
        x = abs(y)
        s: int = int( ((x*3600) % 3600) % 60 )
        m: int = int( ((x*3600-s) // 60) % 60 )
        try:
            sign: int = int( x / y )
        except:
            sign: int = 0
        return (int(x), m, s, sign)