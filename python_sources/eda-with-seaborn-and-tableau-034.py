#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import pandas as pd
coursera_course_data = pd.read_csv("../input/coursera-course-data/coursera-course-data.csv")
coursera_course_detail_data = pd.read_csv("../input/coursera-course-data/coursera-course-detail-data.csv")


# In[ ]:


#Exploratory Data Analysis


# In[ ]:


#looking for null objects
course_data = coursera_course_detail_data
course_data.info()


# In[ ]:


Tags = course_data.Tags
New_tags = Tags.str.rsplit(pat='[',expand=True)
temp = (New_tags[1].str.split(pat=']',expand=True))
Final = temp[0].str.split(pat=',',expand=True)
Subj_cnt = pd.concat([Final[0].value_counts(),Final[1].value_counts()])
Subj_cnt_df = pd.DataFrame(Subj_cnt)
Subj_cnt_df.sort_values(by=0,ascending=True)


# In[ ]:


#Dropping Duplicates
Subj_cnt_df.drop_duplicates(inplace=True)


# In[ ]:


#Creating Ranks
Subj_cnt_df['Rank'] = Subj_cnt_df[0].rank(ascending=False)


# In[ ]:


Top_10_cat = Subj_cnt_df[Subj_cnt_df['Rank'] <= 10]


# In[ ]:


Bot_10_cat = Subj_cnt_df[(Subj_cnt_df['Rank'] <= Subj_cnt_df.shape[0]) & (Subj_cnt_df['Rank'] > (Subj_cnt_df.shape[0]-10))]


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x='index',y=0,data= Bot_10_cat.reset_index(),palette='Greens_d')
print('Least Number of Courses for the Following domains')


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(x='index',y=0,data=Top_10_cat.reset_index(),palette="rocket")
print('Highest Number of Courses for the Following domains')


# In[ ]:


Diff_count = course_data.Difficulty.value_counts()
Diff_count = Diff_count.reset_index()


# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x='index',y='Difficulty',data=Diff_count,palette='Blues_d')


# In[ ]:


https://public.tableau.com/views/GlobalTerrorismExploratoryDataAnalysis-Kaggle/TargetDashboard?:embed=y&:display_count=yes


# In[ ]:


#Import Tableau Visualisation
from IPython.display import IFrame
IFrame('https://public.tableau.com/views/EDAonCourseradata/EDA?:embed=y&:display_count=yes', width=1000, height=830)


# In[ ]:




