#!/usr/bin/env python
# coding: utf-8

# As an Indian, I am interested to have an insight on the Indian population who are interested in the data science field. 
# Let's explore the dataset, to find out more, on where does India stands with respect to other countries, in the data science world!!!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


MCR=pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv")
questions=pd.read_csv("/kaggle/input/kaggle-survey-2019/questions_only.csv")
survey_schema=pd.read_csv("/kaggle/input/kaggle-survey-2019/survey_schema.csv")
other_text_response=pd.read_csv("/kaggle/input/kaggle-survey-2019/other_text_responses.csv")


# In[ ]:


MCR.head()


# In[ ]:


MCR.shape


# In[ ]:


MCR_description=MCR.loc[0]
MCR_columns=MCR.columns


# In[ ]:


for _ in range(0,len(MCR_description)-1):
    print(MCR_columns[_],"==>",MCR_description[_])


# In[ ]:


MCR.rename(columns = {"Q1": "Age", 
                     "Q2":"Gender",
                     "Q3":"Country",
                     "Q4":"Educational Qualification",
                     "Q5":"Current Role",
                     "Q6":"Size of Company",
                      "Q7":"Persons responsible for Data Science",
                      "Q8":"Does your current employer incorporate machine learning methods into their business?",
                      "Q10":"Salary",
                      "Q11": "Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?",
            
                     }, 
                                 inplace = True) 


# In[ ]:


MCR.drop(0,inplace=True)
MCR.head()


# In[ ]:


MCR.drop(['Q2_OTHER_TEXT','Q5_OTHER_TEXT'],axis=1,inplace=True)


# In[ ]:


MCR.drop('Time from Start to Finish (seconds)',axis=1,inplace=True)
MCR.head()


# In[ ]:


MCR.Age.value_counts()


# In[ ]:


sns.countplot(MCR.Age)


# A very interesting observation is , we can find that there are 100 people in the list who is more than 70 years of age, and is interested in machine learning

# In[ ]:


MCR[MCR['Age']=='25-29'].Country.value_counts().head(10).plot(kind='barh')


# An interesting observation is that the maximum number of candidates within the age of 25-29 are from India, followed by USA

# In[ ]:


MCR['Educational Qualification'].value_counts()


# In[ ]:


MCR[MCR['Country']=='India']['Educational Qualification'].value_counts().plot(kind='barh')


# Maximum of the candidates from India who are interested in Data Science have Bachelor's degree. Whereas in the total population, the maximum number of candidates holds a Master degree

# In[ ]:


MCR.Gender.value_counts()


# In the total population, the male candidates interested in data science is almost double of the female population.

# In[ ]:


MCR[MCR['Country']=='India'].Gender.value_counts().plot(kind='barh')


# In India, there is a significantly less number of female candidates(almost 1/5th), who are interested in data science

# In[ ]:


MCR['Current Role'].value_counts()


# In[ ]:


MCR[MCR['Country']=='India']['Current Role'].value_counts().plot(kind='barh')


# Here, again we can see another interesting observation.
# In the world population, the maximum number of people interested in data science are working as'data scientist', followed by students and then 'software engineer'.
# Whereas, in India, the maximum number of candidates interested in data science are students, which is way more than the 'data scientist' and 'software engineer' role.

# In[ ]:


MCR['Size of Company'].value_counts()


# In[ ]:


MCR[MCR['Country']=='India']['Size of Company'].value_counts().plot(kind='barh')


# If we observe carefully, we can see that in both the cases(world population and India), the companies having a strength of 0-49 employees have a significant number of candidates who are interested in data science. Which means that startups are mostly interested in using data science in their day to day activities.
# Again companies having a strength more than 10k are also using data science considerably.

# To be continued...
