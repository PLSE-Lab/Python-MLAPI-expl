#!/usr/bin/env python
# coding: utf-8

# <H1> Quick analysis of Data  </H1>

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data_2017 = pd.read_csv("/kaggle/input/kaggle-survey/multipleChoiceResponses_2017.csv",encoding='ISO-8859-1')
data_2018 = pd.read_csv("/kaggle/input/kaggle-survey/multipleChoiceResponses_2018.csv")
data_2019 = pd.read_csv("/kaggle/input/kaggle-survey/multiple_choice_responses_2019.csv")


# In[ ]:


data_2019.shape,data_2018.shape,data_2017.shape


# In[ ]:


data_2017.drop(0,inplace=True)
data_2018.drop(0,inplace=True)
data_2019.drop(0,inplace=True)


# In[ ]:


data_year = data_2017.shape[0],data_2018.shape[0],data_2019.shape[0]
year=["2017","2018","2019"]


# <h3> Less Particpation in 2019 Survey ?</h3>
# Comparatively to 2018 there is a decrease in the number of participation in 2019 Survey , however its better than 2017 
# 1. In 2017 number of people who attended the survey is around 16716     
# 2. In 2018 number of people who attended the survey is around 23860  
# 3. In 2019 number of people who attended the survey is around 19718     
# 

# In[ ]:


plt.plot(year,data_year)


# In[ ]:


data_2017["GenderSelect"].value_counts(),data_2018["Q1"].value_counts(),data_2019["Q2"].value_counts()


# In[ ]:


gender_per_2017=((data_2017["GenderSelect"].value_counts())/data_2017.shape[0])*100
gender_per_2018=((data_2018["Q1"].value_counts())/data_2018.shape[0])*100
gender_per_2019=((data_2019["Q2"].value_counts())/data_2019.shape[0])*100
gender_per_2017,gender_per_2018,gender_per_2019


# <H3> Counts of Male And Female </h3>
# 1. In 2017,2018 and 2019 Percenatge of Male Data scientist  who partcipated in the survey is around 81% and Female is 16% 
# 

# In[ ]:


fig =plt.figure(figsize=(20,10))

ax1 = plt.subplot2grid((1,3),(0,0))
plt.title("2017 Data",weight="bold",size=15)
sns.countplot(data_2017["GenderSelect"],order=data_2017["GenderSelect"].value_counts().index)
plt.xticks(weight='bold',rotation=45)

ax1 = plt.subplot2grid((1,3),(0,1))
plt.title("2018 Data",weight="bold",size=15)
sns.countplot(data_2018["Q1"],order=data_2018["Q1"].value_counts().index)
plt.xticks(weight='bold',rotation=45)

ax1 = plt.subplot2grid((1,3),(0,2))
plt.title("2019 Data",weight="bold",size=15)
sns.countplot(data_2019["Q2"],order=data_2019["Q2"].value_counts().index)
plt.xticks(weight='bold',rotation=45)
plt.show()


# <h3>Do we need a Masters to become Data scientis?  </h3>
# 1. People with Master Degree are in higest number in both the year 2018 and 2019. In 2019 there are 8549 and in 2018 there are 10855 who completed the Masters Degree are particpated in Survey .   
# 2. There are around 6000 bachelosrs degree in 2019 and 7000 in 2018.  
# Accoring to the Survey it shows the survey 45% people are having Master degree as there highest education, 30% with bachelors degree and 14 % as Doctoral degree 

# In[ ]:


degree_per_2018=((data_2018["Q4"].value_counts())/data_2018.shape[0])*100
degree_per_2019=((data_2019["Q4"].value_counts())/data_2019.shape[0])*100
degree_per_2018,degree_per_2019


# In[ ]:


fig = plt.figure(figsize=(20,10))
ax1 = plt.subplot2grid((1,2),(0,0))
plt.title("Higest Level of Education",weight='bold',size=27)
sns.countplot(y=data_2019["Q4"],order=data_2019['Q4'].value_counts().index)
plt.xlabel("2019 Data",weight='bold',size=25)
plt.yticks(rotation=45,weight='bold',size=12)

ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(y=data_2018["Q4"],order=data_2018['Q4'].value_counts().index)
plt.xlabel("2018 Data",weight="bold",size=25)
plt.yticks(rotation=45,weight='bold',size=12)
plt.subplots_adjust(top=0.85)
plt.show()


# <H3> Is Students are showing more intrest towards Data Science ?</H3>
# 1. Compare to 2017,In 2018 and 2019 we can see Drastic increase of Students who are pursuing  Data Science    
# 2. Software Engineers are also Showing more intrest towards data Science because of the application in all the industry  and Huge oppurtinty towards Jobs. 

# In[ ]:


fig = plt.figure(figsize=(20,10))
ax1=plt.subplot2grid((1,4),(0,1))
plt.title("2019_data",weight='bold')
sns.countplot(y=data_2019["Q5"],order=data_2019["Q5"].value_counts().index)


ax1=plt.subplot2grid((1,4),(0,2))
plt.title("2018_data",weight='bold')
sns.countplot(y=data_2018["Q6"],order=data_2018["Q6"].value_counts().index)

ax1=plt.subplot2grid((1,4),(0,3))
plt.title("2017_data",weight='bold')
sns.countplot(y=data_2017["CurrentJobTitleSelect"],order=data_2017["CurrentJobTitleSelect"].value_counts().index)


plt.show()


# In[ ]:





# In[ ]:




