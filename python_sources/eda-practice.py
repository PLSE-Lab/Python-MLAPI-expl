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


#Lets import important libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Fetching file into a dataframe
df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")


# In[ ]:


#Lets understand data
df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# Salary column has 67 null values .Need to fix this.Lets replace null values by 0.

# In[ ]:


df['salary'].fillna(0,inplace = True)


# In[ ]:


df.isnull().sum()


# Now our data is preprocessed and ready for exploration.

# In[ ]:


#Count of males and females in data
df_gender_analysis =  df[['gender','status']].groupby(['gender'], as_index = False).count()
df_gender_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['gender'],as_index = False).count()
df_gender_analysis['Placed'] = df_gender_analysis1['status']
df_gender_analysis['Placed_Percent'] = df_gender_analysis['Placed']/df_gender_analysis['status']*100
df_gender_analysis.rename(columns = {'gender':'Gender' , 'status':'Total_Students'})


# Percentage of Male students is more than that of Females .
# Male students got placed more as compared to females.

# In[ ]:


#Salary analysis by gender
df_gender_analysis2 = df[['gender','salary']].groupby(['gender'],as_index = False).mean()
df_gender_analysis2


# Here we can see that male students have more mean salary as compared to females 

# In[ ]:


#Lets see salary distribution  
plt.figure(figsize=(12,9))
df_male = df.loc[df['gender'] == 'M']
ax = sns.distplot(df_male['salary'].loc[df['salary']!=0])
ax.ticklabel_format(style = 'plain')
plt.show()


# Mostly people got salaries in between 200k to 400k.

# In[ ]:


#Lets see salary distribution by gender.
fig, ax =plt.subplots(1,2 , figsize = (14,7))
df_male = df.loc[df['gender'] == 'M']
df_female = df.loc[df['gender'] == 'F']
sns.distplot(df_male['salary'].loc[df['salary']!=0] , ax = ax[0])
sns.distplot(df_female['salary'].loc[df['salary']!=0] , ax = ax[1])
ax[0].ticklabel_format(style = 'plain')
ax[1].ticklabel_format(style = 'plain')
ax[0].set_title("Male")
ax[1].set_title("Female")
plt.show()


# Deviation is less for males than compared to females which means most people get salary close to mean salary.

# In[ ]:


sns.countplot(x = 'workex' , data =df , hue = 'status')


# From the graph there is huge difference between placed count and unplaced for people having work expereince.That means with work experience you are more likely to get placed as compared to wth no work experience.Work Experience is an imortant factor for your placement.It increases your chances of getting a placement.Now lets see the impact of work experience on salary.

# In[ ]:


#Lets see mean salaries for workex and no workex
df[['workex','salary']].groupby(['workex']).mean()


# Work ex mean salary is higher as expected

# In[ ]:


sns.boxplot(x = 'workex' , y = 'salary' , data=df.loc[df['status'] == 'Placed'])


# In[ ]:


figure , ax = plt.subplots(1,2 , figsize = (15,9))
df1 = df.loc[df['status'] == 'Placed']
sns.distplot(df1.loc[df['workex'] == 'Yes']['salary'] ,hist = False,ax = ax[0])
ax[0].set_title("Work Experience")
sns.distplot(df1.loc[df['workex'] == 'No']['salary']  , hist = False , ax = ax[1])
ax[1].set_title("No Work Experience")
ax[0].ticklabel_format(style = 'plain')
ax[1].ticklabel_format(style = 'plain')


# Workex people have higher salaries as compared to no work experience.Hence we can say with work experience you can get a placement with more salary than with no work experience.

# Lets see placement numbers for people from different ssc boards

# In[ ]:


df_boards_analysis =  df[['ssc_b','status']].groupby(['ssc_b'], as_index = False).count()
df_boards_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['ssc_b'],as_index = False).count()
df_boards_analysis['Placed'] = df_boards_analysis1['status']
df_boards_analysis['Placed_Percent'] = df_boards_analysis['Placed']/df_boards_analysis['status']*100
df_boards_analysis.rename(columns = {'ssc_b':'SSC_Board' , 'status':'Total_Students'})


# There is not much difference in placed percent so sscboard does not play an important role in deciding placement status.

# In[ ]:


sns.catplot(x = 'status' , y = 'ssc_p' , data = df )


# Students who have more ssc percentage got placement .People below 50% were unplaced .

# In[ ]:


sns.boxplot(x = 'status' , y = 'ssc_p' , data = df )


# We can see the same thing in box plot more clearly that ssc percentage influences placement

# In[ ]:


sns.boxplot(x = 'ssc_b' , y = 'salary' , data = df.loc[df['status']=='Placed'])


# Salaries are not that impacted from your ssc board . Although there are some outliers in central higher than others

# In[ ]:


df_hscboards_analysis =  df[['hsc_b','status']].groupby(['hsc_b'], as_index = False).count()
df_hscboards_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['hsc_b'],as_index = False).count()
df_hscboards_analysis['Placed'] = df_hscboards_analysis1['status']
df_hscboards_analysis['Placed_Percent'] = df_hscboards_analysis['Placed']/df_hscboards_analysis['status']*100
df_hscboards_analysis.rename(columns = {'hsc_b':'HSC_Board' , 'status':'Total_Students'})


# Same for hsc board it doesnt have much impact on placement as it is almost same for both boards.

# In[ ]:


sns.boxplot(x = 'hsc_b' , y = 'salary' , data = df.loc[df['status']== 'Placed'] )


# In[ ]:


sns.boxplot(x = 'status' , y = 'hsc_p' , data = df )


# HSc percentage is also impacting placement . More the value more likely the student is getting a placement.

# In[ ]:


df_hscsubject_analysis =  df[['hsc_s','status']].groupby(['hsc_s'], as_index = False).count()
df_hscsubject_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['hsc_s'],as_index = False).count()
df_hscsubject_analysis['Placed'] = df_hscsubject_analysis1['status']
df_hscsubject_analysis['Placed_Percent'] = df_hscsubject_analysis['Placed']/df_hscsubject_analysis['status']*100
df_hscsubject_analysis.rename(columns = {'hsc_s':'HSC_subject' , 'status':'Total_Students'})


# Students from commerce and sccience background got more placements than arts.We can say HSC subject also has somewhat impact on placement

# In[ ]:


sns.boxplot(x = 'hsc_s' , y = 'salary' , data = df.loc[df['status']== 'Placed'] )


# In[ ]:


sns.scatterplot(x = 'ssc_p', y = 'hsc_p' , data =df , hue = 'status')


# In[ ]:


sns.scatterplot(x = 'ssc_p', y = 'degree_p' , data =df , hue = 'status')


# In[ ]:


sns.scatterplot(x = 'degree_p', y = 'hsc_p' , data =df , hue = 'status')


# people having more percentage tend to get placed than compared to unplaced.Be it in ssc , hsc or degree.Percentage matters for placement.

# In[ ]:


sns.countplot(x='degree_t' , data =df )


# If you see there is majority of people from commerce background .

# In[ ]:


df_degree_analysis =  df[['degree_t','status']].groupby(['degree_t'], as_index = False).count()
df_degree_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['degree_t'],as_index = False).count()
df_degree_analysis['Placed'] = df_degree_analysis1['status']
df_degree_analysis['Placed_Percent'] = df_degree_analysis['Placed']/df_degree_analysis['status']*100
df_degree_analysis.rename(columns = {'degree_t':'Degree_Stream' , 'status':'Total_Students'})


# People from Science and Commerce have got the highest placements as compared to others.So Degree  Stream does affect the placement

# In[ ]:


sns.boxplot(x = 'status' , y = 'degree_p' , data = df )


# More the degree percentage more is the probability of placement.

# In[ ]:


sns.scatterplot(x = 'degree_p' , y= 'salary' , data = df.loc[df['salary'] != 0])


# We don't see any strong relation in degree_p and salary !!
# 

# In[ ]:


sns.catplot(x = 'degree_t' , y = 'salary' , data = df.loc[df['salary'] != 0])


# In[ ]:


sns.boxplot(x = 'degree_t' , y = 'salary' , data = df.loc[df['salary'] != 0])


# Salaries for Sci&tech are higher than the other two streams.

# In[ ]:


sns.countplot(x = 'degree_t' , data =df , hue = 'gender')

Gender distribution is pretty good in others category as compared to sci and commerce
# In[ ]:


sns.catplot(x = 'degree_t' , y = 'salary' , data =df.loc[df['status']== 'Placed'] , hue = 'gender')


# In[ ]:


sns.boxplot(x = 'degree_t' , y = 'salary', hue='gender' , data = df.loc[df['salary'] != 0])


# For Science the salary range is bigger  than other categories.Salaries are more in science as compared to commerce.

# In[ ]:


sns.regplot(x = 'mba_p', y = 'salary' , data = df)


# There is no significant relation seen for mba percentage and salary.

# In[ ]:


sns.scatterplot(x = 'mba_p' , y = 'degree_p' , data = df , hue = 'status' )


# In[ ]:


sns.catplot(x = 'status' , y = 'etest_p' , data = df )


# Etest scores are not that different for placed and unplaced students.We can check mean score though.

# In[ ]:


df[['status','etest_p']].groupby('status').mean()


# As we can see mean is also almost same so Placement is not that affected by etest score

# In[ ]:


sns.regplot(x = 'etest_p' , y = 'salary' , data = df.loc[df['salary'] != 0])


# The relation is not that significant for etest score and salary.

# In[ ]:


df_specialisation_analysis =  df[['specialisation','status']].groupby(['specialisation'], as_index = False).count()
df_specialisation_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['specialisation'],as_index = False).count()
df_specialisation_analysis['Placed'] = df_specialisation_analysis1['status']
df_specialisation_analysis['Placed_Percent'] = df_specialisation_analysis['Placed']/df_specialisation_analysis['status']*100
df_specialisation_analysis.rename(columns = {'specialisation':'MBA_specialisation' , 'status':'Total_Students'})


# More students from Marketting and finance got placed as compared to  Marketting and HR.There is more demand for the former specialisation.

# In[ ]:


sns.barplot(x = 'specialisation' , y = 'salary' , data =df.loc[df['status']== 'Placed'] , hue = 'gender')


# Mean Salary difference is not that significant for different specialisation in MBA and even among genders.Thogh salaries are higher in finance as compared to HR and it also supports our observation above that demand is more in finance.Male employees earn more than females.

# In[ ]:


sns.catplot(x = 'specialisation' , y = 'salary' , data =df.loc[df['status']== 'Placed'] , hue = 'gender')


# Lets Do Label Encoding :
# 

# In[ ]:


df = pd.get_dummies(df,columns = ['status'])
df = pd.get_dummies(df,columns = ['specialisation'])
df = pd.get_dummies(df,columns = ['gender'])
#df['status'] = le.fit_transform(df['status'])

#df_dummy.rename()


# In[ ]:



df.rename(columns = {'status_Not Placed':'Not Placed' , 'status_Placed':'Placed' ,
                    'specialisation_Mkt&Fin':'Marketting and Finance' , 'specialisation_Mkt&HR':'Marketting and HR',
                    'gender_F':'Female' , 'gender_M':'Male'}, inplace =True)


# In[ ]:


df.corrwith(df['Placed'])


# SSC_percentage matters the most for getting a placement as it is highly correlated with Placement.

# As we can see from above analysis , the  factors affecting placement are :
# MBA specialisation , Degree Stream ,HSC_subject , Work Experience ,Degree percentage , ssc percentage ,Gender and hsc percentage
# 
# Males get more placement and higher salaries than females.
# 
# * MBA Specialization: Finance Department has higher placements and higher salaries
# * Work Experience : People having workexperience are more likely of getting placed with higer salaries.
# * Degree Percentage : Having more percentage can increase chances of placement but won't impact salary much
# * Degree Stream : Science and commerce students get placed more as compared to arts .Salaries are higher for science than other streams.
# * Having Higher HSC and SSC percentage can increase your chances of getting placed although boards don't play an important role in getting a placement 
# * Males are getting more placements and higher salaries than females.
# 
# 
# Percentage matters for getting a placement.SSC percentage has greatest influence on placement
# 
# 
# 
# 
