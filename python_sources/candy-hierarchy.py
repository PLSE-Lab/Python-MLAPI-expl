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


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df= pd.read_excel('/kaggle/input/candy-data/candyhierarchy2017.xlsx')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.iloc[:,0:50].info()


# In[ ]:


df.iloc[:,50:].info()


# In[ ]:


Q6=df.iloc[:,6:109]


# In[ ]:


df.drop(df.iloc[:,7:109],axis=1,inplace=True)


# In[ ]:


df


# In[ ]:


df.drop('Internal ID',axis=1,inplace=True)
df.drop('Unnamed: 113',axis=1,inplace=True)


# In[ ]:


df.info()


# In[ ]:


df.drop(['Q12: MEDIA [Yahoo]','Q12: MEDIA [ESPN]','Q12: MEDIA [Daily Dish]','Q9: OTHER COMMENTS','Q7: JOY OTHER','Q8: DESPAIR OTHER'],axis=1,inplace=True)


# In[ ]:


df.shape


# In[ ]:


df['Q3: AGE'].unique()


# In[ ]:


s=pd.to_numeric(df['Q3: AGE'],downcast='float',errors='ignore')


# In[ ]:


s=pd.to_numeric(s,downcast='float',errors='coerce')


# In[ ]:


df.info()


# In[ ]:


df['Q3: AGE'].unique()


# In[ ]:


df.replace(df['Q3: AGE'],s,inplace=True)


# In[ ]:


df['Q3: AGE'].replace(['old enough','45-55','24-50','?','no','Many','hahahahaha','older than dirt','Enough','See question 2','old','ancient','old enough'],np.nan,inplace=True)


# In[ ]:


df['Q3: AGE'].unique()


# In[ ]:


df['Q3: AGE'].replace(['5u','46 Halloweens.','sixty-nine','Over 50','OLD','MY NAME JEFF','59 on the day after Halloween','your mom'
                      'I can remember when Java was a cool new language', '60+'],np.nan,inplace=True)


# In[ ]:


df['Q3: AGE'].unique()


# In[ ]:


df['Q3: AGE'].replace([312,1000,'Old enough','your mom','I can remember when Java was a cool new language'],np.nan,inplace=True)


# In[ ]:


pd.to_numeric(df['Q3: AGE']).head()


# In[ ]:


df.info()


# In[ ]:


#df[['Click Coordinate(x)', 'Click Coordinate(y)']]=df['Click Coordinates (x, y)'].str.split(',', expand = True)


# In[ ]:


new = df['Click Coordinates (x, y)'].str.split(',', expand = True)


# In[ ]:


newx=new[0].str.split('(', expand = True)


# In[ ]:


newx.head()


# In[ ]:


newy=new[1].str.split(')', expand = True)


# In[ ]:


newy.head()


# In[ ]:


df.head(2)


# In[ ]:


df['Click Coordinate X']=newx[1]


# In[ ]:


df['Click Coordinate Y']=newy[0]


# In[ ]:


df.drop(columns=['Click Coordinates (x, y)'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.columns.tolist()


# In[ ]:


nam=df.columns.str.split(': ').str[1]


# In[ ]:


nam


# In[ ]:


col_name = ['GOING OUT?','GENDER','AGE','COUNTRY','ADMINISTRATIVE DEFINITION','100 Grand Bar','DRESS','DAY',
            'MEDIA [Science]','Click Coordinate X','Click Coordinate Y']


# In[ ]:


df.columns=col_name


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df['COUNTRY']= df['COUNTRY'].str.upper()
df['ADMINISTRATIVE DEFINITION']= df['ADMINISTRATIVE DEFINITION'].str.upper()
df['DRESS']= df['DRESS'].str.upper()
df['DAY']= df['DAY'].str.upper()


# In[ ]:


df.head()


# In[ ]:


df['GENDER'].unique()


# In[ ]:


df['GENDER'].replace("I'd rather not say",'Other',inplace=True)


# In[ ]:


df['COUNTRY'].unique()


# In[ ]:


df['100 Grand Bar'].unique()


# In[ ]:


df['GOING OUT?'].unique()


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.drop(index=[0],axis=0,inplace=True)


# In[ ]:


df.reset_index(drop=True,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df['Click Coordinate X']=df['Click Coordinate X'].astype(float)


# In[ ]:


df['Click Coordinate Y']=df['Click Coordinate X'].astype(float)


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df['GOING OUT?'].value_counts()


# In[ ]:


df['GOING OUT?'].fillna('No',inplace=True)


# In[ ]:


df['GENDER'].value_counts()


# In[ ]:


df['GENDER'].fillna('Male',inplace=True)


# In[ ]:


df['COUNTRY'].value_counts().head()


# In[ ]:


df['COUNTRY'].fillna('USA',inplace=True)


# In[ ]:


df['ADMINISTRATIVE DEFINITION'].value_counts().head(6)


# In[ ]:


df['ADMINISTRATIVE DEFINITION'].fillna('CALIFORNIA',inplace=True)


# In[ ]:


df['AGE'].mode()


# In[ ]:


df['AGE'].fillna(40.0,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['100 Grand Bar'].value_counts()


# In[ ]:


df['100 Grand Bar'].fillna('JOY',inplace=True)


# In[ ]:


df['DRESS'].value_counts()


# In[ ]:


df['DRESS'].fillna('WHITE AND GOLD',inplace=True)


# In[ ]:


df['DAY'].mode()


# In[ ]:


df['DAY'].fillna('FRIDAY',inplace=True)


# In[ ]:


(df['Click Coordinate X'].mode())


# In[ ]:


df['Click Coordinate X'].fillna(76.0,inplace=True)


# In[ ]:


(df['Click Coordinate Y'].mode())


# In[ ]:


df['Click Coordinate Y'].fillna(76.0,inplace=True)


# In[ ]:


df.drop(columns='MEDIA [Science]',inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


num=Q6.columns.str.split('|').str[1]


# In[ ]:


num


# In[ ]:


Q6.columns=num


# In[ ]:


df.shape


# In[ ]:


Q6.head()


# In[ ]:


Q6.shape


# In[ ]:


Q6.drop_duplicates(inplace=True)


# In[ ]:


#Q6.fillna(Q6.mode,inplace=True)


# In[ ]:


Q6.isnull().sum().head()


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.reset_index(drop=True,inplace=True)


# In[ ]:


df.head()


# In[ ]:


sns.pairplot(df,hue='GENDER')


# In[ ]:


sns.pairplot(df,hue='GOING OUT?')


# In[ ]:


sns.pairplot(df,hue='100 Grand Bar')


# In[ ]:


df.head(1)


# In[ ]:


sns.countplot('GENDER',data=df)


# In[ ]:


sns.countplot('GOING OUT?',data=df)


# In[ ]:


sns.pairplot(df,hue='DRESS')


# In[ ]:


sns.pairplot(df,hue='DAY')


# In[ ]:


df.head(2)


# In[ ]:


sns.distplot(df['AGE'])


# In[ ]:


sns.distplot(df['Click Coordinate X'])


# In[ ]:


sns.distplot(df['Click Coordinate Y'])


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='COUNTRY',data=df)


# In[ ]:


df['COUNTRY'].value_counts()


# In[ ]:


df['COUNTRY'].replace(['UNITED STATES','USA','UNITED STATES OF AMERICA','US','US OF A','U.S.A.','U S A'],'USA',inplace=True)


# In[ ]:


df['COUNTRY'].value_counts().head(1)


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='COUNTRY',data=df)


# In[ ]:


df.head()


# In[ ]:


df['ADMINISTRATIVE DEFINITION'].value_counts().head(10)


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='ADMINISTRATIVE DEFINITION',data=df)


# In[ ]:


sns.stripplot(x="DAY", y="AGE",hue='GENDER',split=True, data=df)


# In[ ]:


sns.stripplot(x="100 Grand Bar", y="AGE",hue='GENDER',split=True, data=df)


# In[ ]:


sns.boxenplot(x="100 Grand Bar", y="AGE",hue='GENDER', data=df)


# In[ ]:


sns.countplot(x="100 Grand Bar",data=df)


# In[ ]:


df.shape


# In[ ]:


sns.countplot('100 Grand Bar',hue='GENDER',data=df)


# In[ ]:


df.head()


# In[ ]:


sns.countplot('100 Grand Bar',hue='GOING OUT?',data=df)


# In[ ]:


sns.countplot('100 Grand Bar',hue='DRESS',data=df)


# In[ ]:


sns.countplot('100 Grand Bar',hue='DAY',data=df)


# In[ ]:


g = sns.PairGrid(df,hue='GENDER')
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)


# In[ ]:


def plott(x,y,df):
    sns.boxplot(x,y,data=df)


# In[ ]:


df.head(2)


# In[ ]:


plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plott('GENDER','AGE',df)
plt.subplot(2,2,2)
plott('GENDER','Click Coordinate X',df)
plt.subplot(2,2,3)
plott('GENDER','Click Coordinate Y',df)


# In[ ]:


plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plott('100 Grand Bar','AGE',df)
plt.subplot(2,2,2)
plott('100 Grand Bar','Click Coordinate X',df)
plt.subplot(2,2,3)
plott('100 Grand Bar','Click Coordinate Y',df)


# In[ ]:


plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plott('GOING OUT?','AGE',df)
plt.subplot(2,2,2)
plott('GOING OUT?','Click Coordinate X',df)
plt.subplot(2,2,3)
plott('GOING OUT?','Click Coordinate Y',df)


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plott('DRESS','AGE',df)
plt.subplot(2,2,2)
plott('DRESS','Click Coordinate X',df)
plt.subplot(2,2,3)
plott('DRESS','Click Coordinate Y',df)


# In[ ]:


sns.heatmap(df.corr(),cmap='coolwarm',annot=True,linecolor='white',linewidths=1)


# In[ ]:


df.drop(columns='Click Coordinate Y',inplace=True)


# In[ ]:


df.head()


# In[ ]:


def plott(x,y,h,df):
    sns.violinplot(x,y,hue=h,data=df)


# In[ ]:


plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plott('GOING OUT?','AGE','GENDER',df)
plt.subplot(2,2,2)
plott('GOING OUT?','Click Coordinate X','GENDER',df)


# In[ ]:


plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plott('GOING OUT?','AGE','100 Grand Bar',df)
plt.subplot(2,2,2)
plott('GOING OUT?','Click Coordinate X','100 Grand Bar',df)


# In[ ]:


plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plott('GOING OUT?','AGE','GOING OUT?',df)
plt.subplot(2,2,2)
plott('GOING OUT?','Click Coordinate X','GOING OUT?',df)


# In[ ]:


plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plott('GOING OUT?','AGE','DRESS',df)
plt.subplot(2,2,2)
plott('GOING OUT?','Click Coordinate X','DRESS',df)


# In[ ]:


plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plott('GOING OUT?','AGE','DAY',df)
plt.subplot(2,2,2)
plott('GOING OUT?','Click Coordinate X','DAY',df)


# In[ ]:


df['ADMINISTRATIVE DEFINITION'].value_counts().head()


# # This is a short story that tells us some information about this data
# 
This data is about Halloween, where males and females and some who did not want to reveal its gender participated, but the largest number of males was followed by females, and the vast majority of them participated with the aim of treating themselves as they ranged from 18 to 80 years, where the majority of the participants were from 21 To 62. Participants between latitude and longitude 70 to 80 where more than two countries participated, namely the United States of America and then Canada, where we observed that 62% of them got 100 Grand Bar, while 31% felt desperate (analyzes are still not performed here On the rest of the sweets) most of the happy people were men
# In[ ]:




