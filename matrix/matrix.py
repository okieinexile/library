# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 08:21:55 2018

@author: bwinters
"""

from typing import List, TypeVar, Tuple
import csv
from itertools import count
import pandas as pd
import array

Matrix = TypeVar('Matrix')

#==============================================================================
# This is a matrix class with row and column operations    
# It is designed to do integral arithmetic
#==============================================================================

class matrix():
    
    
    def __init__(self, entries: List[int], rows: int, cols: int) -> None:
        if rows * cols != len(entries):
            raise ValueError('rows times columns does not equal number of elements')
        if any(int(e) != e for e in entries):
            raise ValueError('Non Integer Entry')
        self.list: List[int] = entries
        self.rows: int = rows
        self.cols: int = cols

        return None
    
    @classmethod
    def read_csv(cls, filename) -> Matrix:
        rows_iter = count(1)
        List=[]
        with open(filename,'r') as mfile:
            csv_reader = csv.reader(mfile)
            next(csv_reader) # goes passed header row
            for row in csv_reader:
                List += [int(entry) for entry in row[1:]]
                rows = next(rows_iter)   
        return cls(List,rows, len(List) // rows)
    
    def to_csv(self, filename: str) -> None:
        col_dict: dict = dict((c,self._get_column(c)) for c in range(self.cols))
        sv_df: pd.DataFrame = pd.DataFrame(col_dict)
        sv_df.to_csv(filename)
        return None
    
    def _get_row(self, row: int) -> List[int] :
        return [entry for entry in self.list[row * self.cols: (row + 1) * self.cols]]
    
    def _get_column(self, column: int) -> List[int]: 
        return [e for i, e in enumerate(self.list) if ((i - column) % self.cols) == 0]
    
    @classmethod
    def copy(cls, M: Matrix) -> Matrix:
        entries: List[int] = M.list.copy()
        N: Matrix = matrix(entries, M.rows, M.cols)
        return N
                
    def get_entry(self, i: int, j: int) -> int:
        k: int = i * self.cols + j
        return self.list[k]
        
    def set_entry(self, i: int, j: int, entry: int) -> None:
        if i > self.rows:
            raise ValueError('not in row range')
        if j > self.cols:
            raise ValueError('not in column range')
        k: int = i * self.cols + j
        self.list[k] = entry
        return None
    
    def invert(self) -> Matrix:
        M = matrix(self.list.copy(),self.rows,self.cols)
        if M.rows != M.cols:
            raise ValueError('Must be a square matrix')
        I = M.id_pre()
        for start in range(M.rows):
            i, j = M.find_pivot(start)
            if j > start:
                raise ValueError('Matrix is not invertable')
            M.swap_rows(start, i)
            I.swap_rows(start, i)
            entry = M.get_entry(start, start)
            if entry != 0:
                M.row_mult(start, 1/entry)
                I.row_mult(start, 1/entry)
            else:
                raise ValueError('Matrix is not invertable')
            for k in range(start + 1, M.rows):
                entry = M.get_entry(k, start)
                M.add_rows(start, k, -entry)
                I.add_rows(start,k, -entry)
            for pivot in range(M.rows-1,0,-1):
                for r in range(pivot):
                    entry = M.get_entry(r,pivot)
                    M.add_rows(pivot, r, -entry)
                    I.add_rows(pivot, r, -entry)
        return I
        
    def swap_cols(self, i: int , j: int) -> None:
        for k in range(self.rows):
            a = self.get_entry(k, i)
            b = self.get_entry(k, j)
            self.set_entry(k, i ,b)
            self.set_entry(k, j, a)
        return None
        
    def swap_rows(self, i: int, j: int) -> None:
        for k in range(self.cols):
            a=self.get_entry(i,k)
            b=self.get_entry(j,k)
            self.set_entry(i,k,b)
            self.set_entry(j,k,a)
        return None
        
    def add_cols(self, i: int, j: int, scalar: float) -> None:
        for k in range(self.rows):
            a=self.get_entry(k,i)
            b=self.get_entry(k,j)
            self.set_entry(k,j,scalar*a+b)
        return None
    
    def row_mult(self, r: int, scalar: float) -> None:
        if r not in range(self.rows):
            raise ValueError('Row out of range')
        for i in range(self.cols):
            entry = self.get_entry(r, i)
            self.set_entry(r, i, scalar * entry)
        return None

    def col_mult(self, c: int, scalar: float) -> None:
        if c not in range(self.cols):
            raise ValueError('Column out of range')
        for i in range(self.rows):
            entry = self.get_entry(i, c)
            self.set_entry(i, c, scalar * entry)
        return None        
    
    def add_rows(self,i: int, j: int, scalar: float):
        for k in range(self.cols):
            a = self.get_entry(i, k)
            b = self.get_entry(j, k)
            self.set_entry(j, k, scalar * a + b)
        return None        

    def combine_rows(self, i: int, j: int, a: int, b: int, c: int, d: int):
        """replaces rows i,j with linear combinations
        of row i and row j"""
        if i not in range(self.rows):
            raise ValueError('row number out of range')
        if j not in range(self.rows):
            raise ValueError('row number out of range')
        for k in range(self.cols):
            entry_i = a * self.get_entry(i, k) + b * self.get_entry(j, k)
            entry_j = c * self.get_entry(i, k) + d * self.get_entry(j, k)
            self.set_entry(i, k, entry_i)
            self.set_entry(j, k, entry_j)
        return None
        
    def combine_columns(self, i: int, j: int, a: float, b: float, c: float, d: float):
        """Replaces column i,j with a linear combinations 
        of column i and column j"""
        if i not in range(self.cols):
            raise ValueError('column number out of range')
        if j not in range(self.cols):
            raise ValueError('column number out of range')
        for k in range(self.rows):
            entry_i=a*self.get_entry(k,i)+b*self.get_entry(k,j)
            entry_j=c*self.get_entry(k,i)+d*self.get_entry(k,j)
            self.set_entry(k,i,entry_i)
            self.set_entry(k,j,entry_j)
        return None

        
    def show(self):
        out=[self.list[i * self.cols : (i + 1) * self.cols] for i in range(self.rows)]
        for i in range(self.rows):
            out.append(self.list[i * self.cols : (i + 1) * self.cols])
        return out
    
    def __repr__(self) -> str:
        out_string = str('')
        for i in range(self.rows):
            out_string = (out_string 
                          + str(self.list[i * self.cols : (i + 1) * self.cols]) ) + '\n'
        return out_string
    
    @staticmethod
    def multiply(M: 'matrix', N: 'matrix') -> 'matrix':
        List: list = (M.rows * N.cols) * [0]
        if M.cols != N.rows:
            raise ValueError('cols of first must equal rows of second')
        for i in range(M.rows):
            for j in range(N.cols):
                S = 0
                for k in range(M.cols):
                   S = S + M.get_entry(i, k) * N.get_entry(k, j) 
                List[i * N.cols + j] = S
        P=matrix(List, M.rows, N.cols)
        return P
        
    def id_post(self) -> Matrix:
        List: list = list()
        for i in range(self.cols):
            for j in range(self.cols):
                if i == j:
                    List.append(1)
                else:
                    List.append(0)
        P: Matrix = matrix(List, self.cols, self.cols)
        return P
    
    def id_pre(self) -> Matrix:
        List: list = list()
        for i in range(self.rows):
            for j in range(self.rows):
                if i==j:
                    List.append(1)
                else:
                    List.append(0)
        P: Matrix = matrix(List, self.rows, self.rows)
        return P

    def sn_compact(self):
        M=matrix(self.list.copy(), self.rows, self.cols)
        crank, rrank, brank, z_col=M.ranks()
        T: List[int] = []
        ones = 0
        entry: int = M.get_entry(ones,ones)
        while entry == 1:
            ones += 1
            if (ones < M.rows) and (ones < M.cols):
                entry = M.get_entry(ones, ones)
            else:
                return (ones, T,crank,rrank,brank,ones+len(T),z_col)
        count=ones
        while entry != 0:
            count += 1
            T.append(entry)
            if (count < M.rows) and (count < M.cols):
                entry = M.get_entry(count, count)
            else:
                break
        return (ones, T,crank,rrank,brank,ones+len(T),z_col)
        
    def smith_normal(self, reminder: bool = False) -> None:
        top = 0
        self.U = self.id_pre()
        self.V = self.id_post()
        K, L = self.find_pivot(top)
        if K == -1:
            return None
        while K >= 0:
            if reminder and (top % 20 == 0):
                print(f'Now doing row {top} of {self.rows}' )
            self.swap_rows(K, top)
            self.U.swap_rows(K, top)
            self.swap_cols(L, top)
            self.V.swap_cols(L, top)
            while not self.pivot_zeroed(top):
                self.improve_pivot_down(top)
                self.improve_pivot_right(top)
            entry = self.get_entry(top, top)
            if entry < 0:
                self.row_mult(top,-1)
                self.U.row_mult(top,-1)
                #self.V.col_mult(top,-1)
            top += 1
            if top < self.rows:
                K, L = self.find_pivot(top)
            else:
                return None   
        return None
    
    @staticmethod
    def smith(N: Matrix, reminder: bool = False) -> Matrix:
        M: Matrix = matrix.copy(N)
        top = 0
        M.U = M.id_pre()
        M.V = M.id_post()
        K, L = M.find_pivot(top)
        if K == -1:
            return None
        while K >= 0:
            if reminder and (top % 20 == 0):
                print('Now doing row ',top, ' of ', M.rows )
            M.swap_rows(K,top)
            M.U.swap_rows(K,top)
            M.swap_cols(L,top)
            M.V.swap_cols(L,top)
            while not M.pivot_zeroed(top):
                M.improve_pivot_down(top)
                M.improve_pivot_right(top)
            entry=M.get_entry(top,top)
            if entry < 0:
                M.row_mult(top,-1)
                M.U.row_mult(top,-1)
                M.V.col_mult(top,-1)
            top+=1
            if top < M.rows:
                K, L = M.find_pivot(top)
            else:
                return None   
        return M        
    
    def find_pivot(self, i: int) -> Tuple[int,int]:
        K: int = i
        L: int = i
        entry: float = self.get_entry(K, L)        
        if entry != 0:
            return (K, L)
        while entry == 0 and L < self.cols:
            while entry == 0 and K < self.rows:
                K += 1
                if K < self.rows:
                    entry = self.get_entry(K, L)
                    if entry != 0:
                        return (K, L)
            K = i
            L += 1
            if L < self.rows:
                entry = self.get_entry(K, L)
                if entry != 0:
                    return (K, L)
        return (-1,-1)
    
    def pivot_zeroed(self,i: int) -> bool:
        for k in range(i + 1, self.rows):
            if self.get_entry(k, i) != 0:
                return False
        for k in range(i + 1, self.cols):
            if self.get_entry(i, k) != 0:
                return False
        return True
    
    def improve_pivot_down(self, i: int) -> None:
        if i not in range(self.rows):
            raise ValueError(f'{i} is not in row range')
        for j in range(i + 1, self.rows):
            a: int = self.get_entry(i, i)
            b: int = self.get_entry(j, i)
            if not matrix.divides(a, b):
                if matrix.divides(b, a):
                    self.swap_rows(i, j)
                    self.U.swap_rows(i, j)
                else:
                    self.refine_row(i, j)
            a = self.get_entry(i, i)
            b = self.get_entry(j, i)
            scalar = (-1) * (b // a)
            self.add_rows(i, j, scalar)            
            self.U.add_rows(i, j, scalar)
        return None
    
    def improve_pivot_right(self, i: int) -> None:
        for j in range(i + 1, self.cols):
            a = self.get_entry(i, i)
            b = self.get_entry(i, j)
            if not self.divides(a, b):
                if self.divides(b, a):
                    self.swap_cols(i, j)
                    self.V.swap_cols(i, j)
                else:
                    self.refine_col(i, j)
            a = self.get_entry(i, i)
            b = self.get_entry(i, j)
            scalar = (-1) * (b // a)
            self.add_cols(i, j, scalar)            
            self.V.add_cols(i, j, scalar)
        return None       
        
        
    def refine_row(self,i: int, j: int) -> None:
        a = self.get_entry(i, i)
        b = self.get_entry(j, i)
        beta, sigma, tau=self.gcd(a, b)
        alpha = a // beta
        gamma = b // beta
        self.combine_rows(i, j, sigma, tau , -gamma, alpha)
        self.U.combine_rows(i, j, sigma, tau, -gamma, alpha)
        return None
    
        
    def refine_col(self,i: int, j: int) -> None:
        a = self.get_entry(i, i)
        b = self.get_entry(i,j)
        beta, sigma, tau = self.gcd(a, b)
        alpha = a // beta
        gamma = b // beta
        self.combine_columns(i, j, sigma, tau, -gamma, alpha)
        self.V.combine_columns(i, j, sigma, tau, -gamma, alpha)
        return None
        
    def refine_col_mat(self,i,j,sigma,tau) -> 'matrix':
        P = self.id_post()
        P.set_entry(i, i, sigma)
        P.set_entry(j, i, tau)
        return P
        
    def sweep_down(self, i: int) -> None:
        for k in range(i+1,self.rows):
            if self.get_entry(k,i):
                self.add_rows_2(i,k)
        return None
    
    def sweep_right(self,i) -> None:
        for k in range(i+1,self.cols):
            if self.get_entry(i,k)!=0:
                self.add_cols_2(i,k)
        return None
        
    def zero_cols(self):
        my_count = count()
        for k in range(self.cols):
            flag=True
            for i in range(self.rows):
                entry = self.get_entry(i,k)
                if entry != 0:
                    flag = False
                    break
            if flag:
                next(my_count)
        return next(my_count)
           
    @staticmethod    
    def ranks(M: 'matrix') -> Tuple[int,int,int,int]:
        #This returns the rank of Zp and of B_{p-1}
        N = matrix.smith(M)
        crank = N.cols
        rrank = N.rows
        brank = N.last_nonzero_diag()+1
        z_col = N.zero_cols()
        return (crank,rrank,brank,z_col)
        
    def last_nonzero_diag(self) -> int:
        i: int = 0
        entry: int = self.get_entry(i, i)
        if entry == 0:
            return -1
        while entry != 0:
            i += 1
            if i < self.rows and i < self.cols:
                entry = self.get_entry(i, i)
                if entry == 0:
                    return i - 1
            else:
                return i - 1
        return i - 1
    
    @staticmethod
    def div(p: int, q: int) -> Tuple[int,int]:
        a = p // q
        r = p % q
        return(a, r)
    
    @staticmethod    
    def divides(p: int, q: int) -> bool:
        r = abs(q) % abs(p)
        if r == 0:
            return True
        else:
            return False
        return None 
           
    @staticmethod
    def gcd(m: int, n: int) -> Tuple[int,int,int]:
        if matrix.divides(m,n):
            return (abs(m), m // abs(m), 0)
        if matrix.divides(n,m):
            return (abs(n), 0, n // abs(n))
        p, q = m, n
        a,r=matrix.div(p,q)
        A=[a]
        while r!=0:
            p,q=q,r
            a,r=matrix.div(p,q)
            A.append(a)
        A.pop()
        s0=1
        t0=-A.pop()
        while A:
            s1=t0
            t1=s0-t0*A.pop()
            s0,t0=s1,t1
        return q,s0,t0    
        

        
#==============================================================================
# These deal with the prime order function
#==============================================================================

primes_cache=dict()

def prime_order(n):
    """This will give the order of a prime in the increasing list
    of primes."""
    if n >= 50_000:
        raise ValueError(f'{n} is too large.')
    if n in primes_cache:
        return primes_cache[n]
    with open('primes_50000.bin','rb') as f:
        primes = array.array('i')
        primes.fromfile(f,50_000)
        
    for i in range(50_000):
        primes_cache[i] = primes[i]
    return primes_cache[n]

def prime() -> int:
    yield 2
    cache: List[int] = [2]
    my_count = count(3)
    candidate = next(my_count)
    divisible = False
    while True:
        for p in cache:
            if candidate % p == 0:
                divisible = True
                break
        if not divisible:
            cache.append(candidate)
            yield candidate
        divisible = False
        candidate = next(my_count)

def make_prime_list(number_of_primes: int) -> array:
    A = array.array('i')
    my_count = count()
    my_primes = prime()
    while next(my_count) < number_of_primes:
        A.append(next(my_primes))
    with open(f'primes_{number_of_primes}.bin','wb') as f:
        A.tofile(f)
    
    return None
    