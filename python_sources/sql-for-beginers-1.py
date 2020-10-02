#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

database = "../input/database.sqlite"  #Insert path here

# Any results you write to the current directory are saved as output.


# **1. show tables**

# In[ ]:


conn = sqlite3.connect(database)

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table'
                     """, conn)
tables


# **2. select 'Salaries' table **

# In[ ]:


salaries = pd.read_sql('''SELECT *
                         FROM Salaries
                      ''', conn ) 
salaries.head()


# 3. employees whose base pay is more than 300.000

# In[ ]:


rich_employees = pd.read_sql(''' SELECT EmployeeName, BasePay
                                 FROM Salaries
                                 WHERE BasePay > 300000''', conn)
rich_employees.head()


# 4. employees whose base pay is betwee 200.000 and 300.000

# In[ ]:


medium_employees = pd.read_sql(''' SELECT EmployeeName, BasePay
                                   FROM Salaries
                                   WHERE BasePay BETWEEN 200000 AND 300000 ''', conn)
medium_employees.head()


# 5. Select employees with specific BasePay

# In[ ]:


not_ford = pd.read_sql(''' SELECT EmployeeName, BasePay
                           FROM Salaries
                           WHERE BasePay IN (167411.18,155966.02,212739.13) 
                           LIMIT 10
                        ''', conn)
not_ford


# 6. Select Employees From Specific Year

# In[ ]:


employee_2011 = pd.read_sql(''' SELECT EmployeeName, TotalPay, Year  
                                FROM Salaries
                                WHERE Year = '2011'
                                LIMIT 10
                                       ''', conn)
employee_2011


# In[ ]:


employee_2011_2012 = pd.read_sql(''' SELECT EmployeeName, TotalPay, Year  
                                     FROM Salaries
                                     WHERE Year BETWEEN '2011' AND '2012'
                                       ''', conn)
employee_2011_2012.tail()


# 7. Choose employees with last name Carajan
# 
#     % means the rest of the letters

# In[ ]:


last_name_Carajan = pd.read_sql(''' SELECT EmployeeName, JobTitle, BasePay 
                                    FROM Salaries
                                    WHERE EmployeeName LIKE '%Carajan' 
                                    ''', conn)
last_name_Carajan


# 8. Find

# In[ ]:


start_with_j = pd.read_sql(''' SELECT EmployeeName, JobTitle 
                               FROM Salaries
                               WHERE EmployeeName LIKE 'j%'
                               LIMIT 10
                           ''', conn)
start_with_j


# In[ ]:


second_letter_j = pd.read_sql(''' SELECT EmployeeName, JobTitle 
                                  FROM Salaries
                                  WHERE EmployeeName LIKE '_j%'
                                  LIMIT 10
                              ''', conn)
second_letter_j


# In[ ]:


third_letter_j = pd.read_sql(''' SELECT EmployeeName, JobTitle 
                                 FROM Salaries
                                 WHERE EmployeeName LIKE '__j%'
                                 LIMIT 10
                             ''', conn)
third_letter_j


# In[ ]:


last_letter_j = pd.read_sql(''' SELECT EmployeeName, JobTitle 
                                FROM Salaries
                                WHERE EmployeeName LIKE '%j'
                                LIMIT 10
                             ''', conn)
last_letter_j


# In[ ]:


start_with_j_end_with_j = pd.read_sql(''' SELECT EmployeeName, JobTitle 
                                          FROM Salaries
                                          WHERE EmployeeName LIKE 'j%j'
                                          LIMIT 10
                                      ''', conn)
start_with_j_end_with_j


# In[ ]:


job_start_with_captain = pd.read_sql(''' SELECT EmployeeName, JobTitle
                                         FROM Salaries
                                         WHERE JobTitle LIKE 'captain%'
                                         LIMIT 10
                                         ''', conn)
job_start_with_captain


# In[ ]:


other_pay_end_16 = pd.read_sql(''' SELECT EmployeeName, OtherPay
                                   FROM Salaries
                                   WHERE OtherPay LIKE '%16'
                                   LIMIT 10
                                ''', conn)
other_pay_end_16


# In[ ]:


other_pay_start_with_20_end_16 = pd.read_sql(''' SELECT EmployeeName, OtherPay
                                                 FROM Salaries
                                                 WHERE OtherPay LIKE '20%16'
                                                 LIMIT 10
                                              ''', conn)
other_pay_start_with_20_end_16


# In[ ]:


other_pay_start_with_20_end_16 = pd.read_sql(''' SELECT EmployeeName, OtherPay
                                                 FROM Salaries
                                                 WHERE OtherPay LIKE '20%16'
                                                 LIMIT 10
                                              ''', conn)
other_pay_start_with_20_end_16


# In[ ]:


employee_name_contain_er_then_one_letter_kan = pd.read_sql(''' SELECT EmployeeName, OtherPay
                                                               FROM Salaries
                                                               WHERE EmployeeName LIKE '%er_kan%'
                                                               LIMIT 10
                                                           ''', conn)
employee_name_contain_er_then_one_letter_kan


# In[ ]:


emplyee_name_albert_and_job_title_captain = pd.read_sql(''' SELECT EmployeeName, JobTitle
                                                            FROM Salaries
                                                            WHERE EmployeeName LIKE '%albert%' AND JobTitle LIKE '%captain%'
                                                            LIMIT 10
                                                        ''', conn)
emplyee_name_albert_and_job_title_captain


# In[ ]:


emplyee_name_albert_or_jennifer = pd.read_sql(''' SELECT EmployeeName, JobTitle
                                                  FROM Salaries
                                                  WHERE EmployeeName in ('ALBERT HOLT JR', 'Jennifer Kane')
                                                  LIMIT 10
                                              ''', conn)
emplyee_name_albert_or_jennifer


# 9. How we can excape Caps ?
# 
# LIKE  is already case insensitive. But in PostgreSQL, it is used ILIKE instead

# In[ ]:


# this is valid for PostgreSQL
# emplyee_name_jennifer = pd.read_sql(''' SELECT EmployeeName, JobTitle
#                                         FROM Salaries
#                                         WHERE EmployeeName iLIKE '%jennifer%'
#                                         LIMIT 10
#                                     ''', conn)
# emplyee_name_jennifer


# 10. using AND & OR

# In[ ]:


emplyee_hired_before_2011_or_after_2013 = pd.read_sql(''' SELECT EmployeeName, JobTitle, Year
                                                          FROM Salaries
                                                          WHERE Year < 2012 OR Year > 2013
                                                          LIMIT 10
                                                      ''', conn)
emplyee_hired_before_2011_or_after_2013


# In[ ]:


rich_employees_2011 = pd.read_sql(''' SELECT *
                                      FROM Salaries
                                      WHERE (BasePay > 300000 OR TotalPay > 500000) AND Year = 2011
                                      LIMIT 10
                                  ''', conn)
rich_employees_2011


# In[ ]:




