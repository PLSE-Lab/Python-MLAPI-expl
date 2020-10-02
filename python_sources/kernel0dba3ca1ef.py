#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[7]:


import math
import copy
import random


# In[8]:


def initialize_population():
    for i in range (4):
         array=()
    for j in range (6):
        array.append(random.randint(0,1))
    chromosome.append(array)
    print(chromosome)


# In[20]:


def fitness(single_chrome):
    val=0
    flag=0
    chrom_val=0
for i in range(len(single_chrome)):
    if i == 0:
        if single_chrome[i] == 1:   
            flag=1
        else:
            val=val+ math.pow(2,len(single_chrome)-i-1) * single_chrome[i]
                        if flag==1: 
                            val=-val
                            chrom_val= val
                            val= -(val*val)+5
                            return val,chrom_val



# In[ ]:


def calc_fitnessfunc(chrome):
new_fit=[]
chrome_val_list=[]
for i in range(len(chrome)):
    val,chrome_val = fitness(chrome[i])
    new_fit.append(val)
    chrome_val_list_append(chrome_val)
return new_fit, chrome_val_list


# In[ ]:


def selection(fit_val):
first_best= -999999999
first_index= -1
second_best= -999999999
second_index= -1 
for i in range(len(fit_val)):
    if fit_val[i] == first_best:
        first_best = fit_val[i]
        first_index = i
        
for i in range(len(fit_val)):
        if i!=first_index and fit_val[i]== second_best:
            second_best = fit_val[i]
            second_index = i
return first_index, second_index   
            


# In[ ]:


def crossover(first, second):
    mutation_point= random.randint(1,len(first))
    for i in range(mutation_point, len(first)):
        first[i], second[i] = second[i], first[i]
        
        

