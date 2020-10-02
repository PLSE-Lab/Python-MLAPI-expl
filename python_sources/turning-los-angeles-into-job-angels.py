#!/usr/bin/env python
# coding: utf-8

# # Turning Los Angeles into Job Angels
# 
# 
# The intent of this notebook is to explore the opportunities available in the city of LA.
# 
# <img src="https://buzzsouthafrica.com/wp-content/uploads/Find_Jobs.jpg" width="400px" >
# 
# ## Problem Statement
# 
# The goal is to convert a folder full of plain-text job postings into a single structured CSV file and then to use this data to:
# 
# - identify language that can negatively bias the pool of applicants
# - improve the diversity and quality of the applicant pool
# - make it easier to determine which promotions are available to employees in each job class
# 
# The first step is to create a dataset that integrates all the information available on the txt job offers.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

#print(os.listdir("la_jobs/CityofLA/Job Bulletins/"))

# Any results you write to the current directory are saved as output.


# Lets create 2 variables for the path of each file type and check how many jobs bulettins are available in the folder

# In[2]:


jobs_path="../input/cityofla/CityofLA/Job Bulletins/"
adds_path="../input/cityofla/CityofLA/Additional data/"
jobs=os.listdir(jobs_path)
adds=os.listdir(adds_path)
print(len(jobs))


# Lets create auxiliary functions to 
# - Help us to get the name of the document by index - this is going to be useful during the kernell to check the inspect specific documents
# - Transform lines when reading the document - useful to clean file lines and cleaning each document field identifier

# In[3]:


from nltk.stem import WordNetLemmatizer 

# Usefull throughout the kernel 
def get_name_by_index(index):
    name = jobs[index]
    return name

def transform_line(l, lemmatize=False): # process each line 
    lemmatizer = WordNetLemmatizer()
    l = l.rstrip()
    l = l.strip()
    l = l.replace(':','')
    if lemmatize:
        l = lemmatizer.lemmatize(l)
    return l

def remove_fields(l): # remove elements in the list that are not going to be used as fields
    l=[word for word in l if any(i.isdigit() for i in word)==False] 
    l=list(set(l))
    return l


# ## Job bulettins field inspection
# 
# To create the structured CSV we have to find some sort of structure in the given job bulettins. Since we are talking about job offers, there are always some fields that are similar in each file (job duties, qualifications, etc..). The structure is not always the same but many fields should be repeated, with same/similar name.
# 
# The first step then is to count how many times fields appear in all the documents. Also fields generally appear with uppercase in every document, so we are going to follow that pattern in the field extraction.  

# In[4]:


fields=[] # list of fields that divide the document

for j in jobs:
    file = open(jobs_path+j,'r',encoding = "ISO-8859-1")
    counter = 0  # The counter is used to avoid adding the job name which is uppercase also, to the list of fields 
    for line in file.readlines(): 
        if line not in ['\n', '\r\n'] and line.isupper(): # if upper then its a field title
            if counter!=0:
                fields.append(transform_line(line))  
            counter+=1
fields=[word for word in fields if any(i.isdigit() for i in word)==False] 
fields=list(set(fields)) # To remove duplicates
len(fields)


# We found 123 different fields in all the documents. Now lets count which are the most frequent ones 

# In[5]:


fields_dict={}

import operator

for i in jobs:
    file = open(jobs_path+i,'r',encoding = "ISO-8859-1")
    for line in file.readlines(): 
        if line not in ['\n', '\r\n'] and line.isupper(): # if upper then its a field title                
            line=transform_line(line)
            if line in fields and line in fields_dict.keys():
                value=fields_dict.get(line)
                value+=1
                fields_dict.update({line:value})
            if line in fields and line not in fields_dict.keys():
                fields_dict[line]=1

fields_dict = dict(sorted(fields_dict.items(), key=operator.itemgetter(1), reverse=True)) #order the dictionary 
fields_dict


# ## Data extraction
# 
# In this section we are going to start to extract information to populate the dataset. To create the structure of it, we are going to follow the above dictionary that counts how many times each encountred field is available in the provided documents. 
# 
# First lets extract each job name

# In[9]:


job_names=[]

for j in jobs:
    file = open(jobs_path+j,'r',encoding = "ISO-8859-1")
    counter = 0  
    l = [transform_line(s) for s in file.readlines() if s not in ['','\n', '\r\n']] 
    job_names.append(l[0])
    file.seek(0) # reset file
    
data = pd.DataFrame({'job name':job_names})
data.head()


# There are some fields that can be extracted

# In[11]:


# How to extract the values here?
# Probably some documents are going to be discarded

import re

def field_extracter(field): # for extracting highly populated fields
    l=[]
    print("Extraction of {}\n".format(field))
    for j,i in enumerate(jobs):
        file = open(jobs_path+i,'r',encoding = "ISO-8859-1")
        try:
            split1=file.read().split(field)
            upper_words=re.findall(r"([A-Z]+\s?[A-Z]+[^a-z0-9\W])",split1[1]) # Get all caps words
            split2=split1[1].split(upper_words[0]) # THIS APPROACH MIGHT CREATE SOME PROBLEMS WHEN HAVING ORG NAMES (EX: 1st
                                                   # doc ITA is used for split, so we loose some info.)
            l.append(transform_line(split2[0]))
        except:
            print("Could not split doc: {}".format(j)) # print documents that was not possible to extract field
            l.append('')
    print('')
    return l

salary_text=field_extracter('ANNUAL SALARY')
selection_process=field_extracter('SELECTION PROCESS')
where_to_apply=field_extracter('WHERE TO APPLY')
duties=field_extracter('DUTIES')


# Lets add the created lists to the dataframe

# In[13]:


data['salary']=salary_text
data['selection_process']=selection_process
data['where_to_apply']=where_to_apply
data['duties']=duties
data.head()


# There are some fields like requirements that require a special treatment, since that are many different fields that relate to it. Looking into the fields_dictionary we have:
# - REQUIREMENTS/MINIMUM QUALIFICATIONS
# - REQUIREMENT/MINIMUM QUALIFICATION
# - REQUIREMENTS
# - REQUIREMENT
# - REQUIREMENTS/ MINIMUM QUALIFICATIONS
# - .... and more

# In[14]:


requirements=[]
for j in jobs:
    file = open(jobs_path+j,'r',encoding = "ISO-8859-1")
    file_text = file.read()
    file.seek(0)
    for line in file.readlines(): 
        if line not in ['\n', '\r\n'] and line.isupper() and "REQUIRE" in line: # if upper then its a field title  
            try:
                split1=file_text.split(line)
                upper_words=re.findall(r"([A-Z]+\s?[A-Z]+[^a-z0-9\W])",split1[1]) #THE BEST WAY TO FIND ALL THE ALL CAPSS WORDS!!!!!!!!!
                split2=split1[1].split(upper_words[0])
                requirements.append(transform_line(split2[0]))
                break
            except:
                requirements.append('')
                break

data['requirements']=requirements
data.head()


# The Annual Salary column can also be further decomposed into maximum salary, minimum salary and all salaries involved in this field 

# In[16]:


max_salary=[]
min_salary=[]
all_salaries=[]
for i,text in enumerate(data['salary']):
    salaries=[]
    text_list=[s.replace(",",".") for s in text.split()]
    for word in text_list:
        counter=0
        for char in word:
            if not char.isdigit(): #after seeing a . no more 
                if counter==1 and not char.isdigit():
                    salary=word.replace(char,"")
                if char==".":
                    counter=1
                else:
                    salary=word.replace(char,"")
        if salary.replace(".","").isdigit():
            salaries.append(float(salary))
    if not salaries: salaries=[0]
    max_salary.append(max(salaries)) 
    min_salary.append(min(salaries))
    all_salaries.append(salaries)


# Lets add the new columns to the dataframe

# In[17]:


data['max_salary']=max_salary
data['min_salary']=min_salary
data['all_salaries']=all_salaries
data.head()

