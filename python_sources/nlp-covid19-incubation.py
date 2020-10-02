#!/usr/bin/env python
# coding: utf-8

# Data: This NLP kernel uses data provided under "help us understand covid19 challange". The data has been preprocessed and transtioned from a scholarly articles in a json format into a plain text that was inserted into a list (seperated by sentences) then converted to a csv file. 
# 
# Purpose: The foucs of this project is to estimating an incubation period of covid-19 based on all the gathered articles. 
# 
# Approach: After pasring the plain text, we looked into sentences that contained the word 'incubation' and gathered the numerical values surrounding the word when they existed within a reasonable range( 0 - 30 ) days. 
# 
# Result: We came to a conclusion that the estimated incubation period for the covid-19 virus will have an average of roughly 9 days.
# 
# 

# In[ ]:


import pandas as pd 
from pandas import DataFrame
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import urllib.request


# In[ ]:


#read csv file into a sentences dataframe
df_sentences = pd.read_csv("/kaggle/input/covid-articles-csv/export_dataframe.csv")


# In[ ]:


#creating a new dataframe contains only the sought after sentences (which contain 'incubation')
df_incubation_sentences= df_sentences[df_sentences[" col 1"].str.contains('incubation',na=False)]


# In[ ]:


#assign incubation sentences to a list called text 
#creating empty lists for later numerical processing 
text = df_incubation_sentences[' col 1'].values
incubation_times=[]
float_times=[]
int_times=[]


# In[ ]:


#finding all numerical values within the "incubation" sentences
for t in text:
    for sentence in t.split(". "):
        if "incubation" in sentence:
            iday=re.findall(r" \d{1,2} day| \d{1,2}.\d{1,2} day",sentence)
            if len(iday)==1:
                num=iday[0].split(" ")
                incubation_times.append(num[1])


# In[ ]:


#processing ranged data, example: (6-10 days) by taking the average, 8 days
for row in incubation_times:
    if "-" in row:
        day=row.split('-')
       
        num1=int(day[0])
        num2=int(day[1])
        num3=(num1+num2)/2
        back=str(num3)
        
        new=row.replace(row, back)
        float_times.append(new)


# In[ ]:


#processing one unit non-range values in the sentences
for row in incubation_times:
    if(len(row)<3):
        int_times.append(row)
    


# In[ ]:


# combining both float and int data 
incubation_int_float=[]
incubation_int_float=int_times+float_times
df_incubation=pd.DataFrame(incubation_int_float, columns=["duration"])


# In[ ]:


#converting type of data to numerical 
df_incubation["duration"] = pd.to_numeric(df_incubation["duration"])
df_incubation=df_incubation[df_incubation["duration"]<30]


# In[ ]:


#average incubation period
df_incubation.mean()

