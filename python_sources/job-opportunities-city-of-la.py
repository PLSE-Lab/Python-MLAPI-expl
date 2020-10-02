#!/usr/bin/env python
# coding: utf-8

# # Importing the required packages

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

import nltk
import re

import glob
import io
# Any results you write to the current directory are saved as output.

import seaborn as sns
from decimal import Decimal
import locale
from datetime import datetime
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# # Looking into CSV files
# Additional data folder contains 3 CSV files:
#     1. job_titles.CSV : This file contains Job class titles of the available jobs.
#     2. sample job class export template.CSV : This file contains sample submission of csv.
#     3. kaggle_data_dictionary.CSV: This file contains the name and details of the columns to include in the exported csv file.

# In[ ]:


job_titles = pd.read_csv("../input/cityofla/CityofLA/Additional data/job_titles.csv", header = None)
job_titles.head()


# In[ ]:


sample_job_export = pd.read_csv('../input/cityofla/CityofLA/Additional data/sample job class export template.csv')
sample_job_export


# In[ ]:


kaggle_data_dictionary = pd.read_csv('../input/cityofla/CityofLA/Additional data/kaggle_data_dictionary.csv')
kaggle_data_dictionary


# # Feature Extraction

# In this section sevaral functions are defined to get various featueres. Major features are extracted from each file and appended to the data frame, while some additional features are extracted after the data frame is made with few features.

# In[ ]:


code = r'Class\s{1,2}Code:\s*(\d*)'
open_d = r'Open Date:\s*(\d\d-\d\d-\d\d)'
sal = r'(\$(\d+,\d+))((\s(to|and|-)\s)(\$\d+,\d+))?' # Taken from kaggle kernel
sal_dwp = r'Power\sis\s((\$(\d+,\d+))((\s(to|and|-)\s)(\$\d+,\d+))?)'
dut = r'DUTIES\W+(.*\n)' # Duties
req = r'REQUIREMENT(S)?(/MINIMUM\sQUALIFICATION)?\W+(.*\n)' #Requirements
end_d = r'(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)\W+(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)(.*\d)'
exp_len = r'(one|two|three|four|five|six|seven|eight|nine)\s(?=(year|years|month|months)\sof\s(?=(full-time|part-time)))' 
sch_typ = r'(college|university|high school|apprentice)' # School Type
edu_y = r'(\b\w+(-year))(?=\D+(college|university))' #Education Year
edu_major = r'(college or university|college|university|apprentice)\D+(in|as a)\s(\D+);' # Education Major
course_l = r'(\d+\ssemester.+quarter(\sunits)?)'
#open_d = r'Open Date: (.*)\n' 


# The above section containd some of the regex used to extract features. These Regular Expressions are used to extract various details needed to be included in the columns. 
# The regex for salary (sal) is taken from kernel https://www.kaggle.com/shahules/discovering-opportunities-at-la

# In[ ]:


def doextraction(glob_text):
    col = ['FILE_NAME', 'JOB_CLASS_TITLE', 'JOB_CLASS_NO', 'REQUIREMENT_SET_ID', 'REQUIREMENT_SUBSET_ID', 'JOB_DUTIES',           'EDUCATION_YEARS', 'SCHOOL_TYPE', 'EDUCATION_MAJOR', 'EXPERIENCE_LENGTH', 'FULL_TIME_PART_TIME',            'EXP_JOB_CLASS_TITLE', 'EXP_JOB_CLASS_ALT_RESP', 'EXP_JOB_CLASS_FUNCTION', 'COURSE_COUNT', 'COURSE_LENGTH',           'COURSE_SUBJECT', 'MISC_COURSE_DETAILS', 'DRIVERS_LICENSE_REQ', 'DRIV_LIC_TYPE', 'ADDTL_LIC', 'EXAM_TYPE',            'SALARY_START', 'SALARY_END', 'ENTRY_SALARY_DWP', 'REQUIREMENTS', 'APPLICATION_DEADLINE', 'OPEN_DATE']
    
    df = pd.DataFrame(columns = col) # Initializing a data frame with all the column names mentioned above.
    """Get all the files from the given glob and pass them to the extractor."""
    for thefile in glob.glob(glob_text)[:200]:
        with io.open(thefile, 'r', errors = 'replace') as fyl:
            text = fyl.read()
            df = get_features(text, thefile, df)
            
    return df 


# In the doextraction function a dataFrame is initialized with all the column names. This function takes in path for the text files, read each file and send it to the get_features function. The dataframe with details from the text is updated after reading each file. 

# In[ ]:


with open('../input/cityofla/CityofLA/Job Bulletins/ADVANCE PRACTICE PROVIDER CORRECTIONAL CARE 2325 020808 REV 111214.txt', encoding = 'utf-8') as f:
    print(f.read())


# From the above output we can see the general pattern of the text files.

# In[ ]:


def get_headings(text): # Returns all the headings in the text file as a list 
    headings_list = []
    for line in text.split('\n'):
        if line.isupper():
            headings_list.append(line.strip())
    
    return headings_list


# In the pattern of the text files one can observe that the headings in the file are UpperCase letters. So, Using above function I am extracting headings. Also, it is observed that the first heading is the job title.

# In[ ]:


def get_job_class(text): 
    # This function returns the title and the class code of the job. 
    job_title = text.strip().splitlines()[0].strip()
    class_code = re.findall(code, text)[0]
    
    return job_title, class_code 


# In[ ]:


def get_open_date(text):
    try:
        open_date = re.findall(open_d, text)[0]
    except:
        open_date = np.NaN
        
    return open_date


# In[ ]:


def get_salary(text):
    salary_range = re.search(sal, text)
    
    try:
        salary_start = salary_range.group(1)
    except:
        salary_start = np.NaN
        
    try:
        salary_end = salary_range.group(6)
    except:
        salary_end = np.NaN
        
    salary_dwp = re.search(sal_dwp, text).group(1) if re.search(sal_dwp, text) is not None else np.NaN
    
    return salary_start, salary_end, salary_dwp


# In the above function text is taken as input and finds the lower and upper limit of the salary mentioned. Also, the first DWP-specific salary range is found.
# If the salary is flat-rated then the one amount is returned.

# In[ ]:


def get_requirement(headings, text):
    # All the text mentioned in the requirement section is returned.
    x = headings.index([elm for elm in headings if elm.startswith('REQUIREMENT')][0])
    m = re.search(headings[x], text).end()
    n = re.search(headings[x+1], text).start()
    requirement = text[m:n].strip()
    
    return requirement


# Requirements mentioned in the text are extracted by finding the character position after the Requirement heading and the position previous to next heading. After finding the characters range, requirements are extracted from string.

# In[ ]:


def get_duties(headings, text):
    try:
        x = headings.index([elm for elm in headings if elm.startswith('DUT')][0])
        m = re.search(headings[x], text).end()
        n = re.search(headings[x+1], text).start()
        duties = text[m:n].strip()
    except:
        duties = np.NaN
        
    return duties


# The Duties mentioned in the Bulletin are extracted finding the range of the characters similar to the requirements

# In[ ]:


def get_req_id(requirement):
    # This function returns requirement_set_id and requirement_subset_id.
    req_id = re.search('(\d)\.', requirement.strip()[:2]).group(1) if re.search('(\d\.)', requirement.strip()[:2]) is             not None else np.NaN
    req_sub_id = re.search('\n(a)\.', requirement).group(1) if re.search('\n(a)\.', requirement) is not None             else np.NaN
    
    return req_id, req_sub_id


# In[ ]:


def get_exp_job(requirement):
    exp_id = re.findall(r'(?<=Los Angeles as a )(\w+\s\w+)', requirement)
    if len(exp_id) == 1:
        k1 = exp_id[0]
        k1b = np.NaN
    elif len(exp_id) > 1:
        k1 = exp_id[0]
        k1b = exp_id[1]
    else:
        k1 = k1b = np.NaN
    
    k2 = re.search('(?<=experience)(.+)(;|\.|or)', requirement).group(1).strip() if re.search('(?<=experience)(.+)(;|\.|or)', requirement)            is not None else np.NaN
    
    return k1, k1b, k2


# In this function k1 is the job title of the job one must hold to satisfy the requirement. In this case I am taking only first two letters of the job title.
# k1b is the alternate class of k1 and k2 is the field in which experience is required to satisfy this job requirements.

# In[ ]:


def get_exam_type(headings):
    heads_text = ' '.join(headings)

    if re.search('\sINTERDEPART\w+ PROMOT', heads_text) and re.search('\sOPEN COMPETIT', heads_text):
        e_type ='OPEN_INT_PROM'
    elif re.search('\sINTERDEPART\w+ PROMOT', heads_text):
        e_type = 'INT_DEPT_PROM'
    elif re.search('\sDEPARTMENT\w+ PROMOT', heads_text):
        e_type = 'T_PROM'
    else:
        e_type = 'OPEN'
        
    return e_type


# The information for the exam type is extracted in the headings list as this whole line is UpperCase letters. In this the exam type is categorized based on the description given in kaggle_data_dictionary.CSV file for exam type. Firstly, we are checking for the interdepartmental promotion and open competition followed by Interdedartmental promotion, departmental promotion and open type exam. These are labeled as described inthe description of CSV file.

# In[ ]:


def get_features(text, filename, df):
    filename = filename.replace('../input/cityofla/CityofLA/Job Bulletins/', '')
    headings = get_headings(text)
    
    job_title, class_code = get_job_class(text)
    
    open_date = get_open_date(text)
    
    salary_start, salary_end, salary_dwp = get_salary(text)
    
    requirement = get_requirement(headings, text)
    
    req_id, req_sub_id = get_req_id(requirement)
    
    duties = get_duties(headings, text)
    
    # e1 is the conjunction used in the requirements
    e1 = requirement.splitlines()[0][-3:].strip() if len(requirement.splitlines()) >1 else np.NaN
    if e1 != 'and' and e1 != 'or':
        e1 = np.NaN
        
    k1, k1b, k2 =  get_exp_job(requirement)   
    
    # p1 and p2 are 'DRIVERS_LICENSE_REQ' and 'DRIV_LIC_TYPE'
    if re.search('(positions may require a valid California driver\'s license)', text, re.I) is not None:
        p1 = 'P'
        
    elif re.search('(driver\'s license is required)', text, re.I) is not None:
        p1 = 'R' 
        
    else:
        p1 = np.NaN
    
    p2 = re.search("((?<=Class)\s\w\s)(?=\D+driver's)", text, re.I).group(0).strip() if             re.search("((?<=Class)\s\w\s)(?=\D+driver's)", text, re.I) is not None else np.NaN
    
    try:
        x = re.findall(end_d, text)[0]
        deadline = ''.join(x).strip()
    except:
        deadline = np.NaN
     
    exam_type = get_exam_type(headings)
    
    df = df.append({'FILE_NAME': filename, 'JOB_CLASS_TITLE': job_title, 'JOB_CLASS_NO': class_code,                   'REQUIREMENT_SET_ID': req_id, 'REQUIREMENT_SUBSET_ID': req_sub_id, 'JOB_DUTIES': duties,                   'EXP_JOB_CLASS_TITLE': k1, 'EXP_JOB_CLASS_ALT_RESP': k1b, 'EXP_JOB_CLASS_FUNCTION': k2,                     'DRIVERS_LICENSE_REQ': p1, 'DRIV_LIC_TYPE': p2, 'EXAM_TYPE': exam_type, 'SALARY_START': salary_start, 'SALARY_END': salary_end,                     'ENTRY_SALARY_DWP': salary_dwp, 'REQUIREMENTS': requirement, 'APPLICATION_DEADLINE': deadline,                     'OPEN_DATE': open_date}, ignore_index = True)
    
    # Append the features extracted from each text file.
    return df


# get_features function takes in each text file at a time and extract features from that and appends the extracted features to a data frame.
# Also, in the above function whether driver's license required or not and the class of divers license required are extracted from text.

# In[ ]:


data_df = doextraction('../input/cityofla/CityofLA/Job Bulletins/*.txt')


# In[ ]:


def get_extra_features(data_df):
    data_df['EXPERIENCE_LENGTH'] = data_df.REQUIREMENTS.apply(lambda x: re.search(exp_len,x, re.IGNORECASE).group(1)+ ' ' + re.search(exp_len,x, re.IGNORECASE).group(2)                                                               if re.search(exp_len,x, re.IGNORECASE) is not None else np.NaN)
    data_df['FULL_TIME_PART_TIME'] = data_df.REQUIREMENTS.apply(lambda x: re.search(exp_len,x,re.I).group(3) if re.search(exp_len,x,re.I) is not None else np.NaN)
    data_df['SCHOOL_TYPE'] = data_df.REQUIREMENTS.apply(lambda x: re.search(sch_typ, x, re.I).group(1)                                                        if re.search(sch_typ, x, re.I) is not None else np.NaN)

    data_df['EDUCATION_YEARS'] = data_df.REQUIREMENTS.apply(lambda x: re.search(edu_y, x, re.I).group(1)                                                           if re.search(edu_y, x, re.I) is not None else np.NaN)


    data_df['EDUCATION_MAJOR'] = data_df.REQUIREMENTS.apply(lambda x: re.search(edu_major, x).group(3)                                                           if re.search(edu_major, x) is not None else np.NaN)
    
    
    data_df.COURSE_LENGTH = data_df.REQUIREMENTS.apply(lambda x: re.search(course_l, x, re.I).group(0) if re.search(course_l, x, re.I)                                                  is not None else np.NaN)
    return data_df


# In the above function get_extra_features I am adding the features that can be extracted from requirements.

# In[ ]:


data_df = get_extra_features(data_df)
data_df.head(15)


# Let us check the percentage of missing values in each column. Five columns are completely misssing as they are not given any values.

# In[ ]:


percent_missing = data_df.isna().mean().round(4) * 100
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', ascending=False, inplace=True)
missing_value_df.head(15)


# # More to be added soon

# If you like my kernel or think it's helpful, please upvote. Thank You.
