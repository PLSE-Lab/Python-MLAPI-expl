#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# **It is a dataset including information on amazon job opening around the world from June 2011 to March 2018. This dataset is collected using Selenium and BeautifulSoup by scraping all of the jobs for Amazon job site.**
# By analysing this dataset, I have found out many interesting insights such as -
# * - number of job openings in a specific location
# * - jobs available for a specific qualification
# * - Most popular skillset 
# * - and many other things

# Following section of code gives the number of job openings in Bangalore, India and in seattle, US.

# In[ ]:


import csv
with open('../input/amazon_jobs_dataset.csv') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    
    dic={}
    dic['IN, KA, Bangalore']=0
    dic['US, WA, Seattle ']=0
    for row in file_data:
        if row['location']=='IN, KA, Bangalore ':
            dic['IN, KA, Bangalore']+=1
        elif row['location']=='US, WA, Seattle ':
            dic['US, WA, Seattle ']+=1
    for i in dic:
        print(dic[i], end=' ')


# Now i will write a code for the no. of job openings related to computer vision.

# In[ ]:


with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    lst=list(file_data)
    count=0
    for row in lst:
        a=row['Title'].split()
        if 'Vision' in a:
            count+=1
print(count)


# Following code is going to be for job openings in Canada.

# In[ ]:


with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    lst=list(file_data)
    count=0
    for row in lst:
        a=row['location'].split()
        if not('US,' in a and 'CA,' in a) and 'CA,' in  a:
            count+=1
print(count)


# Following code is for the month having most job openings in Year 2018 and the count of its job openings.

# In[ ]:


with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    lst=list(file_data)
    l=[]
    for row in lst:
        a=row['Posting_date'].split()
        if '2018' in a:
            l.append(a)
dic={}
for i in l:
    if i[0] in dic.keys():
        dic[i[0]]+=1
    else:
        dic[i[0]]=1
freq=0
maxmonth='January'
for i in dic:
    if dic[i]>freq:
        freq=dic[i]
        maxmonth=i
print(maxmonth, freq)


# Now I am going to write a code for the number of job openings are present if applicant have Bachelor degree.
# Here I will use the BASIC QUALIFICATIONS feature to find out whether bachelor degree for Job is required or not. Keywords that I am using are 'Bachelor', 'BS' and 'BA'.

# In[ ]:


with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    lst=list(file_data)
count=0
for row in lst:
    a=row['BASIC QUALIFICATIONS']
    if ("Bachelor" in a or "BA" in a or "BS" in a):
        count+=1
print(count)


# Among Java, C++ and Python, following code will reveal that which  language has more job openings in India for Bachelor Degree Holder?
# I will print the language and the number of job openings as an integer value.
# > **Here I will use the BASIC QUALIFICATIONS feature to find out whether bachelor degree for Job is required or not. Keywords that I will be useing are 'Bachelor', 'BS' and 'BA' and I will use the BASIC QUALIFICATIONS feature to find out whether Language is required for the job or not. Keywords that I'll used for language searching are 'Java','C++' or 'Python'.**

# In[ ]:


with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    lst=list(file_data)
count=0
dic={}
dic['Java']=0
dic['C++']=0
dic['Python']=0
for row in lst:
    loc=row['location'].strip().split(",")
    a=row['BASIC QUALIFICATIONS']
    if 'Java' in a and ("Bachelor" in a or "BA" in a or "BS" in a) and (loc[0]=='IN'):
        dic['Java']+=1
    elif 'C++' in a and ("Bachelor" in a or "BA" in a or "BS" in a) and (loc[0]=='IN'):
        dic['C++']+=1
    elif 'Python' in a and ("Bachelor" in a or "BA" in a or "BS" in a) and (loc[0]=='IN'):
        dic['Python']+=1
count=0
language=""
for i in dic.keys():
    if count<dic[i]:
        count=dic[i]
        language=i
print(language, count)


# Now lets find out the country from where amazon need the most Java Developers. I will print the country and the number of job openings as integer value.
# **Here I have used the BASIC QUALIFICATIONS feature to find out whether Java is required for the job or not.Keyword is used is 'Java'.**

# In[ ]:


with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    lst=list(file_data)
    dic={}
    for row in lst:
        a=row['BASIC QUALIFICATIONS']
        b=row['location'].split(",")[0]
        if "Java" in a:
            if b in dic.keys():
                dic[b]+=1
            else:
                dic[b]=1
                
count=0
country=""
for i in dic:
    if dic[i]>count:
        count=dic[i]
        country=i
print(country, count)


# **Analysis using Matplpotlib**

# ** Line graph between no. of Job postings with respect to year**

# In[ ]:


#importing important libraries
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    year=[]
    for row in file_data:
        year.append(row['Posting_date'].split()[2])
    np_year=np.array(year, dtype='int')
    dic=dict()
    for i in np_year:
        if i in dic.keys():
            dic[i]+=1
        else:
            dic[i]=1
    xaxis=[]
    yaxis=[]
    for i in dic.keys():
        xaxis.append(i)
        yaxis.append(dic[i])
    plt.plot(xaxis, yaxis, color='blue')
    plt.show()
    for i in range(len(xaxis)):
        print(xaxis[len(xaxis)-i-1], yaxis[len(xaxis)-i-1])
        
    


# **Bar graph between Month vs Job Openings**

# In[ ]:


with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    month=[]
    for row in file_data:
        month.append(row['Posting_date'].split()[0])
    np_month=np.array(month)
    dic=dict()
    for i in np_month:
        if i in dic:
            dic[i]+=1
        else:
            dic[i]=1
    xaxis=[]
    yaxis=[]
    for i in dic.keys():
        xaxis.append(i)
        yaxis.append(dic[i])
    
    plt.bar(xaxis, yaxis, color='orange')
    plt.xticks(rotation=45)
    plt.show()
    for i in range(len(xaxis)):
        print(xaxis[i], yaxis[i])


# **Pie chart between Indian cities vs No. of jobs opening**
# > Here I have printed the Indian cities and percentage of Job distribution in India upto 2 decimal places. and I have printed the percentage of job distribution in Descinding order.

# In[ ]:


with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    dic=dict()
    city=[]
    for row in file_data:
        if 'IN' in row['location'] and not 'IE' in row['location']:
            city.append(row['location'].split()[2])
    for i in city:
        if i in dic:
            dic[i]+=1
        else:
            dic[i]=1
            
    xaxis=[]
    yaxis=[]
    for i in dic:
        xaxis.append(i)
        yaxis.append(dic[i])
    np_xaxis=np.array(xaxis)
    np_yaxis=np.array(yaxis, dtype='int')
    
    np_xaxis=np_xaxis[np.argsort(np_yaxis)]
    np_yaxis=np.sort(np_yaxis)
    
    np_xaxis=np_xaxis[::-1]
    np_yaxis=np_yaxis[::-1]
    
    plt.pie(np_yaxis, labels=np_xaxis, autopct='%.2f%%', radius=2, explode=[0.1, 0.1, 0.1, 0.1, 0.8])
    plt.show()
    
    for i in range(len(np_xaxis)):
        print(np_xaxis[i], format((np_yaxis[i]*100)/sum(dic.values()), '.2f'))


# **The scatter graph between year vs No. of jobs opening related to Java**
# > I have printed the year and number of Jobs opening in Java Profile.
# Note: I have used the Keyword 'Java' or 'java' in Basic Qualification feature for finding the job opening related to Java Profile.  and i have printed the year in ascending order.

# In[ ]:


with open('../input/amazon_jobs_dataset.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    year=[]
    for row in file_data:
        if 'java' in row['BASIC QUALIFICATIONS'] or 'Java' in row['BASIC QUALIFICATIONS']:
            year.append(row['Posting_date'].split()[2])
    np_year=np.array(year, dtype='int')
    dic=dict()
    for i in np_year:
        if i in dic.keys():
            dic[i]+=1
        else:
            dic[i]=1
    xaxis=list(dic.keys())
    yaxis=list(dic.values())
    
    np_xaxis=np.array(xaxis)
    np_yaxis=np.array(yaxis)
    
    np_xaxis=np_xaxis[::-1]
    np_yaxis=np_yaxis[::-1]
    
    plt.scatter(np_xaxis, np_yaxis)
    plt.show()
    
    for i in range(len(np_xaxis)):
        print(np_xaxis[i], np_yaxis[i])

