

# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:21:29 2020

@author: rkbra
"""

import pandas as pd
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/glassdoor-data-science-jobs/glassdoor_jobs.csv')

# salary parsing

df = df[df['Salary Estimate'] != '-1']

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kd = salary.apply(lambda x: x.replace('K', '').replace('$',''))

min_hr = minus_kd.apply(lambda x: x.lower().replace('per hour', ''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary + df.max_salary) / 2


# company name text only
df['company_name'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis=1)

# state field
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[-1] if len(x.split(',')) > 1 else -1)

df['job_in_HQ'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)

# age of company
df['age'] = df['Founded'].apply(lambda x: x if x < 1 else 2020 - x)

# parsing of job description (python, etc.)
# python
df['python_reqd'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

# r studio
df['r_reqd'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r.studio' in x.lower() else 0)

# SQL
df['sql_reqd'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)

# Spark
df['spark_reqd'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

# aws
df['aws_reqd'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

# Excel 
df['excel_reqd'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)


df.to_csv('salary_data_cleaned.csv', index=False)





