#!/usr/bin/env python
# coding: utf-8

# # Salary Analysis - plenty of data cleaning and methods for EDA

# **Import libraries**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir('../input'))


# **Read csv**

# In[ ]:


df = pd.read_csv('../input/Salaries.csv')
df.head()


# **Columns(3,4,5,6,12) have mixed type, lets check that using info**

# In[ ]:


df.info()


# **For describe() it won't get any details on those columns**

# In[ ]:


df.describe()


# **See what data type these columns has**

# In[ ]:


toCorrect = list(df.columns[3:7])
# status columns has no float type, lets skip it
# toCorrect.append(df.columns[12])
toCorrect


# **Start with what different types are in each columns**

# In[ ]:


ldict = {}
for i in toCorrect:
    ldict[i] = set()
    for j in df[i].unique():
        ldict[i].add(type(j))
ldict


# **Float is ok. Check what values of str (String) are there**

# In[ ]:


ldict = {}
temp = {}
for i in toCorrect:
    ldict[i] = set()
    count = 0
    temp[i] = []
    for j in df[i].unique():
        if(type(j) == str):
            ldict[i].add(j)
            # limit as list would go long as in version 3
            if count < 5:
                temp[i].append(j)
                count += 1
temp


# **Most of them show numbers as string, but there are some string (alphabetic) values. Check what values they are?**

# In[ ]:


import re
patt = re.compile("[A-Z]+.*",re.IGNORECASE)
for i in list(ldict.keys()):
    print(i,end=" = [")
    for j in ldict[i]:
        if patt.match(j):
            print(j,end=", ")
    print("]")


# **These are string values that we can't directly convert to string**

# In[ ]:


df[(df['BasePay'] == 'Not Provided') | (df['OvertimePay'] == 'Not Provided') | (df['OtherPay'] == 'Not Provided') | 
   (df['Benefits'] == 'Not Provided')]


# **These are only 4 columns, and there is no information provided by these columns, so we can delete them**

# In[ ]:


df = df[~df['Id'].isin([148647,148651,148652,148653])]
df.head()


# **Lets bring them (numeric string) in one type**

# In[ ]:


for i in toCorrect:
    df[i] = pd.Series(map(lambda l:np.float64(l), df[i]))
# running previous piece of code to check type
ldict = {}
for i in toCorrect:
    ldict[i] = set()
    for j in df[i].unique():
        ldict[i].add(type(j))
ldict


# **Check for NaN values**

# In[ ]:


df.isnull().sum()


# **Notes and Status have lot of NaN, lets remove them** 

# In[ ]:


del df['Notes']
del df['Status']


# **Examine few examples of NaN values**

# In[ ]:


df[df['BasePay'].isnull()].head(2)


# In[ ]:


df[df['Benefits'].isnull()].head(2)


# **For BasePay and Benefits, it is clear that NaN values are zeroes (from TotalPay and TotalPayBenefits)**<br>
# **So, simply fill NaN values with 0**

# In[ ]:


df.fillna(value=0,inplace=True)


# **Looks good, lets run describe method**

# In[ ]:


df.describe()


# **min value for payment is negative, make it zero**

# In[ ]:


toCorrect.extend(['TotalPay', 'TotalPayBenefits'])
for i in toCorrect:
    df[i] = df[i].apply(lambda l: np.float64(0) if l < 0 else l)
df.describe()


# **There are some duplicate entries in JobTitle**<br>
# **For example, POLICE OFFICER 3 and POLICE OFFICER III**

# In[ ]:


df[df['JobTitle'].apply(lambda l: ((l.upper().find('POLICE DEPARTMENT') != -1)) | (l.upper().find('POLICE OFFICER') != -1) | (l.upper() == 'CHIEF OF POLICE'))]['JobTitle'].unique()


# **Also some entries like Transit Operator appear twice as Complete UPPERCASE and Capitalized**

# In[ ]:


df[df['JobTitle'].apply(lambda l: l.upper() == 'TRANSIT OPERATOR')]['JobTitle'].unique()


# **Normalizing above cases for JobTitle**

# In[ ]:


# RE to match string ending with <any text><space><numbers> 
patt = re.compile(".* [0-9]+$")

# replace numbers with roman equivalent
def i2r(n):
    roman = ''
    d = {1000 : 'M', 900 : 'CM', 500 : 'D', 400 : 'CD', 100 : 'C', 90 : 'XC', 50 : 'L', 40 : 'XL', 10 : 'X', 9 : 'IX', 5 : 'V', 4 : 'IV', 1 : 'I'}
    while n > 0:
        for i in d.keys():
            while n >= i:
                roman += d[i]
                n -= i
    return roman

def norm(l):
    # convert to uppercase
    l = l.upper()
    # to convert to roman
    if patt.match(l):
        i = 1
        while True:
            if l[-i:].isdecimal():
                i += 1
            else:
                break
        l = l[:-i] + ' ' + i2r(int(l[-i:]))
    return l 

print(norm('Transit Operator 12'))


# **Apply above function**

# In[ ]:


df['JobTitle'] = df['JobTitle'].apply(norm)
# check for previous duplication
df[df['JobTitle'].apply(lambda l: ((l.upper().find('POLICE DEPARTMENT') != -1)) | (l.upper().find('POLICE OFFICER') != -1) | (l.upper() == 'CHIEF OF POLICE'))]['JobTitle'].unique()


# **Also apply str.upper for EmployeeName**

# In[ ]:


df['EmployeeName'] = df['EmployeeName'].apply(str.upper)
df.head()


# **Data transformed and cleaned, time for EDA**<br>
# **Check with number of records yearwise**

# In[ ]:


sns.countplot(df['Year'], palette='magma')


# **Lets check number of job title records**

# In[ ]:


jobcount = df['JobTitle'].value_counts()[:20]
sns.barplot(x=jobcount, y=jobcount.keys())


# **Check it yearwise**

# In[ ]:


fig, ax = plt.subplots(4, figsize = (8, 13))
for i in range(4):
    jcount = df[df['Year'] == (2011 + i)]['JobTitle'].value_counts()[:10]
    sns.barplot(x=jcount, y = jcount.keys(),ax = ax[i])
    ax[i].set_title(str(2011+i))
    ax[i].set_xlabel(' ')
    ax[i].set_xlim(0,2500)


# **Write a simple method that would accept a list of JobTitle and plot multiple barplots for 'BasePay', 'Benefits' and 'TotalPay' yearwise for each element in that list**

# In[ ]:


param = ['BasePay', 'Benefits', 'TotalPay']
def by_year(emp_list):
    d = df[df['JobTitle'].isin(emp_list)].groupby(['JobTitle', 'Year']).mean().reset_index()
    for i in range(3):
        splot = sns.factorplot(data = d, x = param[i], y = 'JobTitle', hue = 'Year', kind = 'bar', size = len(emp_list) * 2).set(title = param[i])
        #splot = sns.catplot(data = d, x = param[i], y = 'JobTitle', hue = 'Year', kind = 'bar', aspect = len(emp_list) / 2.5, height = len(emp_list) * 1.5).set(title = param[i])


# **Do it for JobTitle for which most record are available**

# In[ ]:


top5s = df['JobTitle'].value_counts().keys()[:5]


# In[ ]:


by_year(top5s)


# **Write a method to plot distribution of a 'BasePay', 'Benefits' and 'TotalPay' JobTitle by year using violin-plot**

# In[ ]:


def dist_by_year(emp):
    fig, ax = plt.subplots(3, 1, figsize = (15,13))
    for i in range(3):
        sns.violinplot(data = df[df['JobTitle'] == emp], x = 'Year', y = param[i], ax = ax[i]).set(title = param[i])


# In[ ]:


dist_by_year('TRANSIT OPERATOR')


# **Write a method to above plots for comparison between number of JobTitles**

# In[ ]:


def dist_among_job(emp_list):
    fig1, ax1 = plt.subplots(4, 1, figsize = (16,13))
    fig2, ax2 = plt.subplots(4, 1, figsize = (16,13))
    fig3, ax3 = plt.subplots(4, 1, figsize = (16,13))
    for i in range(4):
        sns.violinplot(data = df[(df['JobTitle'].isin(emp_list)) & (df['Year'] == (2011 + i))], x = 'JobTitle', y = 'BasePay', ax = ax1[i])
    for i in range(4):
        sns.violinplot(data = df[(df['JobTitle'].isin(emp_list)) & (df['Year'] == (2011 + i))], x = 'JobTitle', y = 'Benefits', ax = ax2[i])
    for i in range(4):
        sns.violinplot(data = df[(df['JobTitle'].isin(emp_list)) & (df['Year'] == (2011 + i))], x = 'JobTitle', y = 'TotalPay', ax = ax3[i])
    ax1[0].set(title='BasePay - 2011-14')
    ax2[0].set(title='Benefits - 2011-14')
    ax3[0].set(title='TotalPay - 2011-14')
    


# In[ ]:


dist_among_job(top5s)


# **Write a method for same for large number of JobTitle**

# In[ ]:


def large_dist_among_job(emp_list):
    fig1, ax1 = plt.subplots(4, 1, figsize = (16,13))
    fig2, ax2 = plt.subplots(4, 1, figsize = (16,13))
    fig3, ax3 = plt.subplots(4, 1, figsize = (16,13))
    
    for i in range(4):
        for j in range(len(emp_list)):
            sns.distplot(df[df['JobTitle'] == emp_list[j]]['BasePay'], hist = False, label = emp_list[j], ax = ax1[i])
            
    for i in range(4):
        for j in range(len(emp_list)):
            sns.distplot(df[df['JobTitle'] == emp_list[j]]['Benefits'], hist = False, label = emp_list[j], ax = ax2[i])
            
    for i in range(4):
        for j in range(len(emp_list)):
            sns.distplot(df[df['JobTitle'] == emp_list[j]]['TotalPay'], hist = False, label = emp_list[j], ax = ax3[i])
            
    ax1[0].set(title='BasePay - 2011-14')
    ax2[0].set(title='Benefits - 2011-14')
    ax3[0].set(title='TotalPay - 2011-14')
    


# **Check for records of POLICE OFFICER**

# In[ ]:


large_dist_among_job(df[df['JobTitle'].apply(lambda l: ((l.upper().find('POLICE OFFICER') != -1)) | (l.upper().find('CHIEF OF POLICE') != -1))]['JobTitle'].unique()[:10])


# In[ ]:




