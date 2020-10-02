#!/usr/bin/env python
# coding: utf-8

# # Importing the required packages

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import nltk
import re

import glob
import io

import seaborn as sns
from decimal import Decimal
import locale
from datetime import datetime
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from gensim.summarization import summarizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy import sparse


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

# In this section sevaral functions are defined to get various features. Major features are extracted from each file and appended to the data frame, while some additional features are extracted after the data frame is made with few features.

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


# The above section contains some of the regex used to extract features. These Regular Expressions are used to extract various details needed to be included in the columns. 
# The regex for salary (sal) is taken from kernel https://www.kaggle.com/shahules/discovering-opportunities-at-la

# In[ ]:


def doextraction(glob_text):
    col = ['FILE_NAME', 'JOB_CLASS_TITLE', 'JOB_CLASS_NO', 'REQUIREMENT_SET_ID', 'REQUIREMENT_SUBSET_ID', 'JOB_DUTIES',           'EDUCATION_YEARS', 'SCHOOL_TYPE', 'EDUCATION_MAJOR', 'EXPERIENCE_LENGTH', 'FULL_TIME_PART_TIME',            'EXP_JOB_CLASS_TITLE', 'EXP_JOB_CLASS_ALT_RESP', 'EXP_JOB_CLASS_FUNCTION', 'COURSE_COUNT', 'COURSE_LENGTH',           'COURSE_SUBJECT', 'MISC_COURSE_DETAILS', 'DRIVERS_LICENSE_REQ', 'DRIV_LIC_TYPE', 'ADDTL_LIC', 'EXAM_TYPE',            'SALARY_START', 'SALARY_END', 'ENTRY_SALARY_DWP', 'REQUIREMENTS', 'APPLICATION_DEADLINE', 'OPEN_DATE']
    
    df = pd.DataFrame(columns = col) # Initializing a data frame with all the column names mentioned above.
    """Get all the files from the given glob and pass them to the extractor."""
    for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', errors = 'replace') as fyl:
            if 'Vocational' in thefile:
                continue
            else:
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


# # Explore through Visualizations

# For the purpose of visualizations, the numerical type of salary variables are created. Also, new columns with the Job open year and month are extracted. 

# In[ ]:


data_df['SALARY_START_NUM'] = data_df.SALARY_START.apply(lambda x: Decimal(re.sub(r'[^\d.\-]', '', x)) if x is not np.NaN else np.NaN)
data_df['SALARY_END_NUM'] = data_df.SALARY_END.apply(lambda x: Decimal(re.sub(r'[^\d.\-]', '', x)) if type(x) is str else np.NaN)
data_df.SALARY_START_NUM = data_df.SALARY_START_NUM.astype(np.float32)
data_df.SALARY_END_NUM = data_df.SALARY_END_NUM.astype(np.float32)


# In[ ]:


data_df['OPEN_YEAR'] = [str(datetime.strptime(x, '%m-%d-%y').year) if type(x) is str else np.NaN for x in data_df.OPEN_DATE]
data_df['OPEN_MONTH'] = [str(datetime.strptime(x, '%m-%d-%y').month) if type(x) is str else np.NaN for x in data_df.OPEN_DATE]


# Salary end is taken as None in my code when it is flat-rated. So, we will assign numerical value of salary end as the salary start. Also, salary end is changed to string 'flat-rated' where it is None

# In[ ]:


data_df.SALARY_END_NUM[data_df.SALARY_END.isnull()] = data_df.SALARY_START_NUM[data_df.SALARY_END.isnull()]
data_df.SALARY_END[data_df.SALARY_END.isnull()] = 'flat-rated'


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))
g = sns.distplot(data_df.SALARY_START_NUM[data_df.SALARY_START_NUM.notnull()])
ax.set(xlabel='Salary Start', title='Distribution Plot of the Salary Start')


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))
g = sns.distplot(data_df.SALARY_END_NUM[data_df.SALARY_END_NUM.notnull()])
ax.set(xlabel='Salary End', title='Distribution Plot of the Salary End')


# From the above two plots we can infer that most of the starting salaries for jobs in the City of LA are in the range of $50,000 - $100,000 and most of the Salary End is distributed in the range $75,000 - $125,000.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))
g = sns.countplot(x = "OPEN_YEAR", data = data_df, palette="Blues_d", order = ['1999', '2002', '2005', '2006', '2008',                        '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'])
ax.set(xlabel='Year', ylabel='Number of Jobs open in year', title='Count of Jobs open in each year')
plt.close(2)


# From the above plots we can observe that most of the jobs are in the years 2016 - 2018. There are some job postings from past years. Let us check their salary distribution to see if they are affecting mena salaries.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
g = sns.catplot(x = "OPEN_YEAR", y = "SALARY_START_NUM", data = data_df, kind = "box", ax= ax, palette="Blues_d",                order = ['1999', '2002', '2005', '2006', '2008',                        '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'])
ax.set(xlabel='Job Open Year', ylabel='Salary Start', title='Distribution of Salary Start over years')
g.set_xticklabels(rotation = 30)
plt.close(2)


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
g = sns.catplot(x = "OPEN_YEAR", y = "SALARY_END_NUM", data = data_df, kind = "box", ax= ax, palette="Blues_d",                order = ['1999', '2002', '2005', '2006', '2008',                        '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'])
ax.set(xlabel='Year', ylabel='Salary End', title='Distribution of Salary End over years')
g.set_xticklabels(rotation = 30)
plt.close(2)


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
g = sns.barplot(x='OPEN_YEAR', y='SALARY_START_NUM', data=data_df, estimator=np.median, ci = None, palette="Blues_d",               order = ['1999', '2002', '2005', '2006', '2008',                        '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'])
ax.set(xlabel='Year', ylabel='Salary Start Median per year', title='Salary Start Median over years')
plt.close(2)


# In[ ]:


fig, ax = plt.subplots(figsize = (10, 8))
g = sns.barplot(x='OPEN_YEAR', y='SALARY_END_NUM', data=data_df, estimator=np.median, ci = None, palette="Blues_d",                order = ['1999', '2002', '2005', '2006', '2008',                        '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'])
ax.set(xlabel='Year', ylabel='Salary End Median per year', title='Salary End Median over years')
plt.close(2)


# From the above plots we can understand that the salaries for jobs in 2019 are less compared to other years. Jobs over the years 2014 - 2018 are almost similar. So, let us check the jobs in 2019

# In[ ]:


data_df[data_df.OPEN_YEAR == '2019']


# These jobs have less salaries, that is the reason we have low salary in 2019 compared top other years

# Now, we'll look into the jobs as per Open Months

# In[ ]:


fig, ax = plt.subplots()
g = sns.countplot(x = "OPEN_MONTH", data = data_df, palette="Blues_d",                  order = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
ax.set(xlabel='Month', ylabel='Number of Jobs open in Month', title='Count of Jobs open in each Month')
plt.close(2)


# From the above plot we can understand that most jobs are opened in the months March, April, October and December.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
g = sns.catplot(x = "OPEN_MONTH", y = "SALARY_START_NUM", data = data_df, kind = "box", ax= ax, palette="Blues_d",                order = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
ax.set(xlabel='Month', ylabel='Salary Start', title='Distribution of Salary Start over months')
g.set_xticklabels(rotation = 30)
plt.close(2)


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 6))
g = sns.barplot(x='OPEN_MONTH', y='SALARY_START_NUM', data=data_df, estimator=np.median, ci = None, palette="Blues_d",                order = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
ax.set(xlabel='Month', ylabel='Salary End Median per Month', title='Salary End Median over months')
plt.close(2)


# From the above plots we can observe that the jobs opened in all months have similar salary distributions.

# In[ ]:


exp = []
for elm in data_df.EXPERIENCE_LENGTH:
    if type(elm) == str:
        if elm.lower().strip().startswith('one'):
            exp.append('1')
        elif elm.lower().strip().startswith('two'):
            exp.append('2')
        elif elm.lower().strip().startswith('three'):
            exp.append('3')
        elif elm.lower().strip().startswith('four'):
            exp.append('4')
        elif elm.lower().strip().startswith('five'):
            exp.append('5')
        elif elm.lower().strip().startswith('six year'):
            exp.append('6')
        elif elm.lower().strip().startswith('six mon'):
            exp.append('0.5')
        else:
            exp.append('oth')
            
    else:
        exp.append('0')
        
data_df['EXP_NUM'] = exp


# In[ ]:


exp = []
for elm in data_df.EXPERIENCE_LENGTH:
    if type(elm) == str:
        if elm.lower().strip().startswith('one'):
            exp.append('1')
        elif elm.lower().strip().startswith('two'):
            exp.append('2')
        elif elm.lower().strip().startswith('three'):
            exp.append('3')
        elif elm.lower().strip().startswith('four'):
            exp.append('4')
        elif elm.lower().strip().startswith('five'):
            exp.append('5')
        elif elm.lower().strip().startswith('six year'):
            exp.append('6')
        elif elm.lower().strip().startswith('six mon'):
            exp.append('0.5')
        else:
            exp.append('oth')
            
    else:
        exp.append('0')
        
data_df['EXP_NUM'] = exp


# In[ ]:


data_df.EXP_NUM.unique()


# In[ ]:


experience=data_df['EXP_NUM'].value_counts().reset_index()
labels=experience['index']
sizes=experience['EXP_NUM']
plt.figure(figsize=(5,7))
plt.pie(sizes,explode=(0.05, 0, 0, 0,0.1,0,0.1,0.3, 0),labels=labels, autopct='%1.1f%%', shadow = True)
plt.gca().axis('equal')
plt.title('Experience value count')
plt.show()


# In the above pie plot we can observew that maximum number of jobs require 2 years of experience, followed by jobs for freshers.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
g = sns.catplot(x = "EXP_NUM", y = "SALARY_START_NUM", data = data_df, kind = "box", ax= ax, palette="Blues_d",                order = ['0', '0.5', '1', '2', '3', '4', '5', '6', 'oth'])
ax.set(xlabel='Number of years of experience', ylabel='Salary Start', title='Distribution of Salary Start over number of years of experience required')
g.set_xticklabels(rotation = 30)
plt.close(2)


# In[ ]:


fig, ax = plt.subplots()
g = sns.barplot(x='EXP_NUM', y='SALARY_START_NUM', data=data_df, estimator=np.median, ci = None, palette="Blues_d",                order = ['0', '0.5', '1', '2', '3', '4', '5', '6', 'oth'])
ax.set(xlabel='Number of years of experience', ylabel='Salary Start Median per Month', title='Salary Start Median over experienxe required')
plt.close(2)


# The above plots says that the jobs taht require 0 - 1 year experience have low salary compared to others.

# In[ ]:


fig, ax = plt.subplots()
g = sns.countplot(x = "EXAM_TYPE", data = data_df, palette="Blues_d")
ax.set(xlabel='Exam Type', ylabel='Number of jobs', title='Count of jobs as per Exam Type')
plt.close(2)


# From the above plot we can observe that there are very few jobs in the Departmental Promotions. Also, less jobs of Open type are present.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
g = sns.catplot(x = "EXAM_TYPE", y = "SALARY_START_NUM", data = data_df, kind = "box", ax= ax, palette="Blues_d")
ax.set(xlabel='Exam Type', ylabel='Salary Start', title='Distribution of Salary Start over Exam Type')
g.set_xticklabels(rotation = 30)
plt.close(2)


# In[ ]:


fig, ax = plt.subplots()
g = sns.barplot(x='EXAM_TYPE', y='SALARY_START_NUM', data=data_df, estimator=np.median, ci = None, palette="Blues_d")
ax.set(xlabel='Exam Type', ylabel='Salary Start Median per Month', title='Salary Start Median over months')
plt.close(2)


# From the above two plots we can observe that with Departmental Promotions have high salary compared to other jobs. And jobs with Open exam type have less salary.

# # Most influential words in Requirements

# In[ ]:


req_all = ' '.join(req for req in data_df.REQUIREMENTS)


# In[ ]:


stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()
lem_req = []
for word in word_tokenize(req_all):
    if word not in stop_words:
        lem_req.append(lem.lemmatize(word))
        
vect = TfidfVectorizer(ngram_range=(1,3), max_features=100)
tfidf_score = vect.fit_transform(lem_req)


# In[ ]:


vect.vocabulary_.keys()


# Most influentuial words in Requirements are degree, accredited, four year, college, university, etc. 

# # Most influential words in Duties

# In[ ]:


duties_all = ' '.join(str(dut) for dut in data_df.JOB_DUTIES)


# In[ ]:


lem_dut = []
for word in word_tokenize(duties_all):
    if word not in stop_words and len(word) > 3:
        lem_dut.append(lem.lemmatize(word))
        
vect = TfidfVectorizer(ngram_range=(1,3), max_features=100)
tfidf_score = vect.fit_transform(lem_dut)

vect.vocabulary_.keys()


# Most influentuial words in Duties are Performs, Professional, work, public, etc.

# # Comparing Extraction v/s Abstraction based summary 

# ## Extraction Based Summary

# In[ ]:


with open('../input/cityofla/CityofLA/Job Bulletins/311 DIRECTOR  9206 041814.txt') as f:
    p = f.read()
    summ = gensim.summarization.summarize(p, ratio = 0.1)
    print(summ)


# As we can see the summary of th bullentin above, the summarizer is taking lines from the first line of the paragraph or taking lines with line numbers. SO, this summarization will not be more for someone looking for a job.

# ## Abstraction based summary

# In[ ]:


sent = nltk.sent_tokenize(p)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(sent)

km = KMeans(n_clusters = 10,init='k-means++', max_iter=100, n_init=1)
km.fit(X)


# For this example, I am taking kMeans algorithm and dividing the whole bulletin into ten segments. Later distance of the sentence from their cluster center is found and 10 sentences close to their respective cluster centers are taken as summary.

# In[ ]:


avg = []
closest = []
for j in range(10):
    idx = np.where(km.labels_ == j)[0]
    avg.append(np.mean(idx))
closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
ordering = sorted(range(10), key=lambda k: avg[k])
summary = ' '.join([sent[closest[idx]] for idx in ordering])
summary


# The above summary seems to be taking text from the distributed topics. Maybe this summary will help someone looking for a job decide easily if he can look more into the job.

# 

# In the below code I am deleting the columns for which the information is not gathered.

# # Post Processing Data

# In[ ]:


data_df = data_df.drop(columns=['COURSE_SUBJECT', 'MISC_COURSE_DETAILS', 'ADDTL_LIC', 'COURSE_COUNT'])


# In[ ]:


data_df


# In[ ]:


kaggle_data_dictionary = kaggle_data_dictionary.drop(kaggle_data_dictionary.index[22])


# In[ ]:


kaggle_data_dictionary = kaggle_data_dictionary.drop(kaggle_data_dictionary.index[20])


# In[ ]:


kaggle_data_dictionary = kaggle_data_dictionary.drop(kaggle_data_dictionary.index[17])


# In[ ]:


kaggle_data_dictionary = kaggle_data_dictionary.drop(kaggle_data_dictionary.index[14])


# In[ ]:


kaggle_data_dictionary['Allowable Values'][19] = 'A,B,C,I'


# In[ ]:


kaggle_data_dictionary = kaggle_data_dictionary.append({'Field Name': 'SALARY_START', 'Annotation Letter':'S1a', 'Description': 'Lower bound of the Salary', 'Data Type': 'String',                               'Allowable Values': '$#####', 'Accepts Null Values?': 'Yes','Additional Notes': np.NaN}, ignore_index = True)


# In[ ]:


kaggle_data_dictionary = kaggle_data_dictionary.append({'Field Name': 'SALARY_END', 'Annotation Letter':'S1b', 'Description': 'Upper bound of the Salary', 'Data Type': 'String',                               'Allowable Values': '$##### , flat-rated', 'Accepts Null Values?': 'Yes','Additional Notes': np.NaN}, ignore_index = True)


# In[ ]:


kaggle_data_dictionary = kaggle_data_dictionary.append({'Field Name': 'REQUIREMENTS', 'Annotation Letter':'D', 'Description': 'Reqiorement of job', 'Data Type': 'String',                               'Allowable Values': np.NaN, 'Accepts Null Values?': 'Yes','Additional Notes': np.NaN}, ignore_index = True)


# In[ ]:


kaggle_data_dictionary = kaggle_data_dictionary.append({'Field Name': 'APPLICATION_DEADLINE', 'Annotation Letter':'U', 'Description': 'Deadline to apply for job', 'Data Type': 'String',                               'Allowable Values': np.NaN, 'Accepts Null Values?': 'Yes','Additional Notes': np.NaN}, ignore_index = True)


# In[ ]:


kaggle_data_dictionary = kaggle_data_dictionary.append({'Field Name': 'OPEN_YEAR', 'Annotation Letter':'T1', 'Description': 'Year in which the Bulleetin os opened',                                                        'Data Type': 'String', 'Allowable Values': np.NaN, 'Accepts Null Values?': 'Yes','Additional Notes': np.NaN},                                                       ignore_index = True)


# In[ ]:


kaggle_data_dictionary = kaggle_data_dictionary.append({'Field Name': 'OPEN_MONTH', 'Annotation Letter':'T2', 'Description': 'Month in which the Bulleetin os opened',                                                        'Data Type': 'String', 'Allowable Values': np.NaN, 'Accepts Null Values?': 'Yes','Additional Notes': np.NaN},                                                       ignore_index = True)


# In[ ]:


data_df.to_csv('Bulletins Data.csv', index = False)
kaggle_data_dictionary.to_csv('kaggle_data_dictionary.csv', index = False)


# # Job Recommendations

# The job bulletins available donot communicate well to get patterns and also not providing any information about the organization. 
# All bulletins if managed to be released in same format in future will help in extracting information precisely. 
# The text should be small but giving all information required, this way more people will go through the job information. 
# Also, it would be great to create a job recommendation column for every job, that suggests silmilar jobs opened. This will help applicant to look at more jobs. 
# Information regarding one bulletin should be in one row so the applicant will look at one place and not be confused.
# General requirements or Duties need not be mentioned in all Bulletins.

# Thank You, for viewing my kernel. Please, upvote if you like it.!
