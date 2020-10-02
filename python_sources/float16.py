# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:58:17 2019

@author: PRANTO DEV
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import math
import itertools
import matplotlib.pyplot as plt



class Float16():
   

    def __init__(self, bitstring):
      


        assert(len(bitstring)==16)


        self.sign = bitstring[0]


        self.exponent = bitstring[1:5]

        self.mantissa = bitstring[5:]

        self.val = self.calculate_value()
        
    def __str__(self):
        return f'Sign bit value: {self.sign}\n' + \
            f'Exponent value: {self.exponent}\n' + \
            f'Mantissa value: {self.mantissa}\n' + \
            f'Floating value: {self.val}\n'
    
    def tobitstring(self):
        return self.sign + self.exponent + self.mantissa
    
    def toformattedstring(self):
        return ' '.join([self.sign, self.exponent, self.mantissa])
    
    def calculate_value(self):
        val = 0.0
        if self.exponent == '1111':
            if self.mantissa =='00000000000':
                val=math.inf
            else :
                val= math.nan
          #subnormal      
        elif self.exponent =='0000' :
            expo = 2**(-6) 
            for exp, bit in enumerate(self.mantissa):
                if bit == '1':
                    val+=2**(-(exp+1))
            val*=expo  
        else:
            val = 1.0
            expo =0.0

            for exp, bit in enumerate(self.mantissa):
                if bit == '1':
                    val+=2**(-(exp+1)) 
            

            for e, bit in enumerate(reversed(self.exponent)):
                if bit=='1':
                    expo+=2**e

            val *= 2**(expo-7)
        if(self.sign=='1'):
            val*=(-1)
        return val
        


def test1a():
    count = 0
    data = [ '0011100000000010', '0100000000000000', '1100000000000000', '0100010000000000',
             '1100010000000000', '0100100000000000', '1100100000000000', '0100101000000000',
             '1100101000000000', '0100110000000000', '1100110000000000', '0101101110000000',
             '0010010000000000', '0000000000000001', '0000011111111111', '0000100000000000',
             '0111011111111111', '0000000000000000', '1000000000000000', '0111100000000000',
             '1111100000000000', '0111100000000001', '0111110000000001', '0111111111111111',
             '0010101010101011', '0100010010010001', '0011100000000000', '0011100000000001']
    result = ['(1025, 1024)', '(2, 1)', '(-2, 1)', '(3, 1)', '(-3, 1)', '(4, 1)', '(-4, 1)',
               '(5, 1)', '(-5, 1)', '(6, 1)', '(-6, 1)', '(23, 1)', '(3, 16)', '(1, 131072)',
               '(2047, 131072)', '(1, 64)', '(4095, 16)', '(0, 1)', '(0, 1)', 'inf', '-inf',
               'nan', 'nan', 'nan', '(2731, 8192)', '(3217, 1024)', '(1, 1)', '(2049, 2048)']
    
    test = [Float16(x).val for x in data]
    for index in range(len(test)):
        d = test[index]
        try:
            test[index] = str(d.as_integer_ratio())
        except Exception:
            test[index] = str(d)
        if test[index] == result[index]:
            count += 1
        else:
            print(data[index], result[index], test[index])
    print(count, 'out of 28')

def histogram():
    combinations = itertools.product('01', repeat=16)
    bitstrings = [''.join(x) for x in combinations]
    numbers = list(map(Float16, bitstrings))
    values = [x.val for x in numbers]
    positive = values[0:30720]
    negative = values[32768:63488]
    plt.hist(positive, 64)
    plt.hist(negative, 64)
test1a()
histogram()

        