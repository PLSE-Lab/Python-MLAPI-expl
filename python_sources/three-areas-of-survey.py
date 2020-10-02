#!/usr/bin/env python
# coding: utf-8

# * Learning
# * Earning
# * Hiring
# > are the major areas (other than patriots and feminist) in which these surveys will have greater impact (according to my little amount of knowledge).
# let's see some of the insights to make impacts in those fields.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


questions=pd.read_csv('../input/kaggle-survey-2019/questions_only.csv')


# In[ ]:


cnt=0
for i in questions.loc[0]:
    cnt+=1
    print(cnt,i)


# In[ ]:


text_responses=pd.read_csv('../input/kaggle-survey-2019/other_text_responses.csv')


# In[ ]:


text_responses.head()


# In[ ]:


text_responses.columns


# In[ ]:


choice_responses=pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')


# In[ ]:


choice_responses.head()


# In[ ]:


choice_questions=choice_responses.loc[0]
choice_responses=choice_responses.loc[1:].reset_index()


# In[ ]:


del choice_responses['index']


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:





# **Hiring**

# The primary hiring focus on hiring the fresher right. let's see the students and their skills in different perspective to hire them.

# In[ ]:


role_count=choice_responses['Q5'].value_counts()


# In[ ]:


plt.figure(figsize=(12,10))
plt.title('Roles of Respondants')
plt.xlabel('Percentage')
plt.ylabel('Roles')
plt.barh(role_count.index,role_count/sum(role_count))


# The above graph shows that there are more than 20 percentage of people who did the survey are students who are interested in data science.

# In[ ]:


student_count=pd.crosstab(choice_responses['Q3'],choice_responses['Q5'])['Student'].sort_values(ascending=False)


# In[ ]:


top_student_count=student_count[student_count>=50]


# In[ ]:


for i in student_count:
    if i <50:
        top_student_count['Other']+=i


# In[ ]:


plt.figure(figsize=(12,10))
plt.title('Students in each country')
plt.xlabel('Percentage')
plt.ylabel('Country')
plt.barh(top_student_count.index,top_student_count/sum(top_student_count))


# This shows that there are many students who are interested in data science belongs to India and US. So HR people can focus on these countries to hire the fresh data enthusiasts. But this is not enough for hiring right. So we need to analyze the skills of these people.

# In[ ]:


students=choice_responses.loc[choice_responses['Q5']=='Student']


# In[ ]:


student_exp=[]
for i in ['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']:
    student_exp.append(students[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('IDEs used by students')
plt.xlabel('Percentage')
plt.ylabel('IDE')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# This shows the Jupyter notebook is mostly used by the students. VS code, Pycharm, Rstudio and spyder comes next.

# In[ ]:


ide_cnt=12-students[['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.figure(figsize=(12,10))
plt.title('Number of IDEs used by each students')
plt.ylabel('Percentage')
plt.xlabel('Number of IDEs')
plt.bar(ide_cnt.index,ide_cnt/sum(ide_cnt))


# This shows most of the students can work in more than 2 IDEs which industry people using nowadays. That means students can work in industry projects directly once they have been hired. But nearly 20 percentage of people not using any IDEs which shows they need to update themself.

# In[ ]:


student_ide=[]
for i in ['Q17_Part_1','Q17_Part_2','Q17_Part_3','Q17_Part_4','Q17_Part_5','Q17_Part_6','Q17_Part_7','Q17_Part_8','Q17_Part_9','Q17_Part_10','Q17_Part_11','Q17_Part_12']:
    student_ide.append(students[i].value_counts())
student_ide=pd.concat(student_ide)
plt.figure(figsize=(12,10))
plt.title('Notebooks used by Students')
plt.xlabel('Percentage')
plt.ylabel('Notebook')
plt.barh(student_ide.index,student_ide/sum(student_ide))


# Google colab and kaggle notebooks are the major notebook products used by the students. Most of the students are not using any notebook products itself. This shows they prefer offline IDEs to work with.

# In[ ]:


student_exp=[]
for i in ['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']:
    student_exp.append(students[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('Languages known by students')
plt.xlabel('Percentage')
plt.ylabel('Language')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# It's not a surprise. Python is the dominating language. SQL, R, C++ are some of the languages also used by the students.

# In[ ]:


ide_cnt=12-students[['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.figure(figsize=(12,10))
plt.title('Number of Languages known by Students')
plt.ylabel('Percentage')
plt.xlabel('Number of Languages')
plt.bar(ide_cnt.index,ide_cnt/sum(ide_cnt))


# Oh! 25 percentage of students does know any language at all. It shows they are thinking that programming is not needed for the data science. They need to change their minds.

# In[ ]:


student_exp=[]
for i in ['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']:
    student_exp.append(students[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('Vizualization tools known by the Students')
plt.xlabel('Percentage')
plt.ylabel('Tools')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# Matplotlib and seaborn are widely used by the students. Because they mostly work in python. ggplot2 and plotly comes next in this row.

# In[ ]:


ide_cnt=12-students[['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.figure(figsize=(12,10))
plt.title('Number Of vizualization tools known by each students')
plt.ylabel('Percentage')
plt.xlabel('Number of tools')
plt.bar(ide_cnt.index,ide_cnt/sum(ide_cnt))


# 

# In[ ]:


ml_exp_years=students['Q23'].value_counts()
plt.figure(figsize=(12,10))
plt.title('Students ML experience')
plt.xlabel('Percentage')
plt.ylabel('Experience')
plt.barh(ml_exp_years.index,ml_exp_years/sum(ml_exp_years))


# In[ ]:


students.loc[students['Q23']=='20+ years']['Q1']


# I was really shocked to see this that a student(so called) has 20+ years experience in ML but his age is from 25 to 29. I think that legend has taken his own biological NN into account. Apart from that the students are predominantly working in ML for less than one years.

# In[ ]:


student_exp=[]
for i in ['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']:
    student_exp.append(students[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('ML Models used by students')
plt.xlabel('Percentage')
plt.ylabel('Models')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=12-students[['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.figure(figsize=(12,10))
plt.title('Number Of ML models used by each students')
plt.ylabel('Percentage')
plt.xlabel('Number of models')
plt.bar(ide_cnt.index,ide_cnt/sum(ide_cnt))


# In[ ]:


student_exp=[]
for i in ['Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8']:
    student_exp.append(students[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('AutoML tools used by by the Students')
plt.xlabel('Percentage')
plt.ylabel('Tools')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=8-students[['Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.figure(figsize=(12,10))
plt.title('Number Of AutoML frameworks used by each students')
plt.ylabel('Percentage')
plt.xlabel('Number of Frameworks')
plt.bar(ide_cnt.index,ide_cnt/sum(ide_cnt))


# In[ ]:


student_exp=[]
for i in ['Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7']:
    student_exp.append(students[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('Image frameworks used by the Students')
plt.xlabel('Percentage')
plt.ylabel('Framework')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=7-students[['Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.figure(figsize=(12,10))
plt.title('Number Of Image frameworks used by each students')
plt.ylabel('Percentage')
plt.xlabel('Number of Frameworks')
plt.bar(ide_cnt.index,ide_cnt/sum(ide_cnt))


# In[ ]:


student_exp=[]
for i in ['Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6']:
    student_exp.append(students[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('NLP frameworks used by the Students')
plt.xlabel('Percentage')
plt.ylabel('Framework')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=6-students[['Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.figure(figsize=(12,10))
plt.title('Number Of NLP frameworks used by each students')
plt.ylabel('Percentage')
plt.xlabel('Number of Frameworks')
plt.bar(ide_cnt.index,ide_cnt/sum(ide_cnt))


# In[ ]:


student_exp=[]
for i in ['Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12']:
    student_exp.append(students[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('ML frameworks used by the Students')
plt.xlabel('Percentage')
plt.ylabel('Framework')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=12-students[['Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.figure(figsize=(12,10))
plt.title('Number Of ML frameworks used by each students')
plt.ylabel('Percentage')
plt.xlabel('Number of Frameworks')
plt.bar(ide_cnt.index,ide_cnt/sum(ide_cnt))


# The above graphs shows that the students need to improve to meet industry standards. But more than half are almost ready to face the world of data science. They will be hired easily.

# But Hiring data scientists or experts are very difficult. We can't say anything from this survey about that.

# **Earning**

# It's human nature to think about higher pay right. Let's see how we can make it.

# I grouped the salaries into 5 groups just for similicity.

# In[ ]:


salary_map={'30,000-39,999':'medium', '5,000-7,499':'low','250,000-299,999':'very_high',
       '4,000-4,999':'low', '60,000-69,999':'medium', '10,000-14,999':'low', '80,000-89,999':'medium',
       '$0-999':'very_low', '2,000-2,999':'very_low', '70,000-79,999':'medium', '90,000-99,999':'high',
       '125,000-149,999':'high', '40,000-49,999':'medium', '20,000-24,999':'medium',
       '15,000-19,999':'medium', '100,000-124,999':'high', '7,500-9,999':'low',
       '150,000-199,999':'high', '25,000-29,999':'medium', '3,000-3,999':'low', '1,000-1,999':'low',
       '200,000-249,999':'very_high', '50,000-59,999':'medium', '> $500,000':'very_high',
       '300,000-500,000':'very_high'}


# In[ ]:


for i in salary_map:
    choice_responses.loc[choice_responses['Q10']==i,'Q10']=salary_map[i]


# In[ ]:


salary_range=choice_responses['Q10'].value_counts()
salary_range.plot(kind='bar',figsize=(10,10),title='Salary Range')


# In[ ]:


india_responses=choice_responses.loc[choice_responses['Q3']=='India']
us_responses=choice_responses.loc[choice_responses['Q3']=='United States of America']


# In[ ]:


salary_group=pd.crosstab(choice_responses['Q3'],choice_responses['Q10'])
salary_group.plot(kind='bar',stacked=True,figsize=(20,10),title='Salary range in each countries')


# In[ ]:


salary_group=india_responses['Q10'].value_counts()
salary_group.plot(kind='bar',figsize=(10,10),title='Salary Range in India')


# In[ ]:


salary_group=us_responses['Q10'].value_counts()
salary_group.plot(kind='bar',figsize=(10,10),title='Salary Range in US')


# This shows lot of very high earners are from US. India has different payscale, Other countries also. So to earn more we need to go to US. Skills of High earners are follows:

# In[ ]:


high_salary_responses=choice_responses.loc[choice_responses['Q10'].isin(['high','very_high'])]


# In[ ]:


india_high_responses=india_responses.loc[india_responses['Q10'].isin(['medium','high','very_high'])]


# In[ ]:


us_high_responses=us_responses.loc[us_responses['Q10'].isin(['high','very_high'])]


# In[ ]:


student_exp=[]
for i in ['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']:
    student_exp.append(high_salary_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('IDEs used by the High earners')
plt.xlabel('Percentage')
plt.ylabel('IDEs')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=12-high_salary_responses[['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_ide=[]
for i in ['Q17_Part_1','Q17_Part_2','Q17_Part_3','Q17_Part_4','Q17_Part_5','Q17_Part_6','Q17_Part_7','Q17_Part_8','Q17_Part_9','Q17_Part_10','Q17_Part_11','Q17_Part_12']:
    student_ide.append(high_salary_responses[i].value_counts())
student_ide=pd.concat(student_ide)
plt.figure(figsize=(12,10))
plt.title('Notebooks used by the High earners')
plt.xlabel('Percentage')
plt.ylabel('Notebooks')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


student_exp=[]
for i in ['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']:
    student_exp.append(high_salary_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('Languagesknown by the High earners')
plt.xlabel('Percentage')
plt.ylabel('Languages')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=12-high_salary_responses[['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_exp=[]
for i in ['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']:
    student_exp.append(high_salary_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('Vizualization tools used by the High earners')
plt.xlabel('Percentage')
plt.ylabel('Tools')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=12-high_salary_responses[['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


ml_exp_years=high_salary_responses['Q23'].value_counts()
plt.figure(figsize=(12,10))
plt.title('ML experience the High earners')
plt.xlabel('Percentage')
plt.ylabel('Experience')
plt.barh(ml_exp_years.index,ml_exp_years)


# In[ ]:


student_exp=[]
for i in ['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']:
    student_exp.append(high_salary_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('ML Models used by the High earners')
plt.xlabel('Percentage')
plt.ylabel('Models')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=12-high_salary_responses[['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_exp=[]
for i in ['Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8']:
    student_exp.append(high_salary_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('AutoML framework used by the High earners')
plt.xlabel('Percentage')
plt.ylabel('Framework')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=8-high_salary_responses[['Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_exp=[]
for i in ['Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7']:
    student_exp.append(high_salary_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('Image framework used by the High earners')
plt.xlabel('Percentage')
plt.ylabel('Framework')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=7-high_salary_responses[['Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_exp=[]
for i in ['Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6']:
    student_exp.append(high_salary_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('NLP framework used by the High earners')
plt.xlabel('Percentage')
plt.ylabel('Framework')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=6-high_salary_responses[['Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_exp=[]
for i in ['Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12']:
    student_exp.append(high_salary_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.figure(figsize=(12,10))
plt.title('ML framework used by the High earners')
plt.xlabel('Percentage')
plt.ylabel('Framework')
plt.barh(student_exp.index,student_exp/sum(student_exp))


# In[ ]:


ide_cnt=12-high_salary_responses[['Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# The above graphs shows python and deep learning skills make them to get more pay. learn it and earn more.

# **Learning**

# Learning from other data scientists and experts. But it is different from earning. Skills to become good data scientists.

# In[ ]:


learning_responses=choice_responses.loc[choice_responses['Q5'].isin(['Data Scientist','Data Analyst','Research Scientist','Business Analyst','Data Engineer','Statistician','Product/Project Manager'])]


# In[ ]:


student_exp=[]
for i in ['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']:
    student_exp.append(learning_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.barh(student_exp.index,student_exp)


# In[ ]:


ide_cnt=12-learning_responses[['Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8','Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_ide=[]
for i in ['Q17_Part_1','Q17_Part_2','Q17_Part_3','Q17_Part_4','Q17_Part_5','Q17_Part_6','Q17_Part_7','Q17_Part_8','Q17_Part_9','Q17_Part_10','Q17_Part_11','Q17_Part_12']:
    student_ide.append(learning_responses[i].value_counts())
student_ide=pd.concat(student_ide)
plt.figure(figsize=(10,10))
plt.barh(student_ide.index,student_ide)


# In[ ]:


student_exp=[]
for i in ['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']:
    student_exp.append(learning_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.barh(student_exp.index,student_exp)


# In[ ]:


ide_cnt=12-learning_responses[['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8','Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_exp=[]
for i in ['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']:
    student_exp.append(learning_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.barh(student_exp.index,student_exp)


# In[ ]:


ide_cnt=12-learning_responses[['Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8','Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


ml_exp_years=learning_responses['Q23'].value_counts()
plt.barh(ml_exp_years.index,ml_exp_years)


# In[ ]:


student_exp=[]
for i in ['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']:
    student_exp.append(learning_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.barh(student_exp.index,student_exp)


# In[ ]:


ide_cnt=12-learning_responses[['Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8','Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_exp=[]
for i in ['Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8']:
    student_exp.append(learning_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.barh(student_exp.index,student_exp)


# In[ ]:


ide_cnt=8-learning_responses[['Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_exp=[]
for i in ['Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7']:
    student_exp.append(learning_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.barh(student_exp.index,student_exp)


# In[ ]:


ide_cnt=7-learning_responses[['Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_exp=[]
for i in ['Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6']:
    student_exp.append(learning_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.barh(student_exp.index,student_exp)


# In[ ]:


ide_cnt=6-learning_responses[['Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# In[ ]:


student_exp=[]
for i in ['Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12']:
    student_exp.append(learning_responses[i].value_counts())
student_exp=pd.concat(student_exp)
plt.barh(student_exp.index,student_exp)


# In[ ]:


ide_cnt=12-learning_responses[['Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8','Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12']].isnull().sum(axis=1)
ide_cnt=ide_cnt.value_counts()
ide_cnt.sort_index(ascending=False)
plt.bar(ide_cnt.index,ide_cnt)


# The above graph shows deep learning is the bigger difference between earner and learner. So Deep learning is to be learned by the students to become a good data scientists. Thank you guys. Insights will be updated soon. 