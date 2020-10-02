#!/usr/bin/env python
# coding: utf-8

# ![Imgur](https://i.imgur.com/1vHslWZ.png)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np #linear algebra
import pandas as pd #data processing
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# *                                                         **JOURNEY TO THE CENTER OF THE DATA**
#  **Ok** ,This is my first kernel I do on kaggle here . I took a look to over all kernels here and I am here to  note some unique ideas using aggressive intuition about this dataset directed to the android developers community in order to help them developing their apps and increase reviews and also number of downloads.LETS GO 
#  
# Steps to our tiny journey:
# Take a look to our data
# 
# **1.Univariate Analysis**   
#     
# **2.Bivariate Analysis**

# In[ ]:


#lets read our data
db=pd.read_csv('../input/googleplaystore.csv')


# In[ ]:


'''lets descover our all data'''

#tail
db.tail()


# In[ ]:


#head of data
db.head()


# In[ ]:


db.info()


# In[ ]:


db.shape


# In[ ]:


db.describe()


# In[ ]:


db.dtypes


# we have found that all our features data types are objects ecept Rating
# now we will convert some of them which we are interested in into numeric
# Features that we are interested in are:
# Reviews ,Rating,Category ,Installs,Type,Size, Contet Rating,Current version and android versions. I think these are the most important features for Univariate analysis.

# **Univariate : **  
#     1.Description
#     2.Transformation
#     3.Analysis

# **Reviews:**
# 
# 1.Description
# 
# 2.Transformation

# In[ ]:


'''Description'''
db['Reviews'].describe()


# In[ ]:


db.Reviews.isna().sum()


# In[ ]:


'''Transformation'''
db['Reviews']=db['Reviews'].astype(int)


# **ERROR** : ValueError: invalid literal for int() with base 10: '3.0M'  !!
# one M in 10841 ? is it millions? even if. I considered it as outlier . so It will affect my data aggressively
# lets get rid of it
# 1. determine position of M row
#     a.**Manually** : crazy method . I am crazy .So what if you have 2 millions raws>>>BooooM
#     b.**Automatically**
# 2. remove it
# 

# In[ ]:


#get the position of the M value
db[db['Reviews'].str.contains('M')]


# Position of **10472** . 
# now remove it

# In[ ]:


#drop series containing M
db.drop(db.index[10472],inplace=True)


# In[ ]:


#convert column into integer values
db['Reviews']=db['Reviews'].astype(int)


# In[ ]:


#check for conversion again
db.dtypes


# THATS IT , 
# it is now numeric.

# Befor Continuing , Lets make a function that makes our analysis instead of repeating codes:
# Illustrating  absolute frequency and plotting using barplot

# In[ ]:


def univariateAnalysis(featureName):
    mDB=pd.DataFrame({'Absolute_Frequency':featureName.value_counts()})    
    #get index of data as x axis and convert to an array
    x=mDB.index.values
    #get y and convert to an array
    y=mDB.Absolute_Frequency.values
    #scaling
    plt.figure(figsize=(8,8))
    #colors = sns.color_palette("CMRmap", len(db))
    #plotting
    BarplotDB=sns.barplot(x,y,palette="CMRmap")
    #get ticks and make it 90 degree to be visible clearly
    BarplotDB.set_xticklabels(BarplotDB.get_xticklabels(), rotation=90)


# **Type**:
# 1.Discription
# 2.Transformation
# 3.Analysis
# 

# In[ ]:


db.Type.value_counts()


# In[ ]:


db.Type.isna().sum()


# In[ ]:


'''Transformation'''
db['Type'].fillna('Free',inplace=True)


# In[ ]:


db.Type.isna().sum()


# In[ ]:


'''Analysis'''
univariateAnalysis(db.Type)


# 1. As we thought by filling with free value, free apps is more predominant than paid.
# 2.difference is too much; a customer want a free rather than paid app.
# 3.developers should take care if they design paid apps.

# **Size:**
#  1. Description
# 
#  2.Transformation
# 
# 

# In[ ]:


'''Description'''
db.Size.value_counts()


# In[ ]:


db.Size.isna().sum()


# We have alot of problems here, So what we will do is:
# 1 .  replace Varies with device by the Cenral tendency measure:
#     I will use the trimmed mean which is more accurate than mean(I think) which is the same as the mean but              removing outliers by 10% from both sides of our ordered data.also calculating other measures.
# 2 . remove M (Megabytes)
# 3 .  replace K(kilobytes) by .5 M 

# In[ ]:


'''Transformation'''
#remove M  from values of size 
db['Size'] = db['Size'].map(lambda x: x.rstrip('M'))
#extract numeric values from size column and put it in new series
newSize=db[db[['Size']].apply(lambda x: x[0].isdigit(), axis=1)]
'''here some rows were removed but we will fix it latter dont worry'''
#make new dataframe
newSizeData=newSize.Size
#convert numeric values into floats 
newSizeData=newSizeData.astype(float)
'''removing fractions will not affect size at all'''


# In[ ]:


#mean
newSizeData.mean()


# In[ ]:


#median
newSizeData.median()


# In[ ]:


#mode
newSizeData.mode()


# In[ ]:


#trimmed mean(I love using it )
sp.stats.trim_mean(newSizeData,.1)


# In[ ]:


30.768015322001435


# lets continue

# In[ ]:


db.Size.str.contains("k").value_counts()
#replace all values containg K by .5
db.loc[db['Size'].str.contains('k'), 'Size'] = '.5M'
'''NOW WE WILL MAKE A THIng replace varies with 0 and get'''
db.loc[db['Size'].str.contains('Varies with device'), 'Size'] = '31M'
db['Size'] = db['Size'].map(lambda x: x.rstrip('M'))


# In[ ]:


db.Size.value_counts()


# As we see it is fixed now

# **Content Rating
# **1.Description
# 2.Analysis

# In[ ]:


'''CONTENT RATING'''
  #count categories values                  
db['Content Rating'].value_counts() 


# In[ ]:


db['Content Rating'].isna().sum()


# In[ ]:


univariateAnalysis( db['Content Rating'])


# Developers tend to  develope everyone apps more than others.this is because the overall population that have variations in interests and needs.
# 

# **Current version**
# 1.Description.
# 2.Transformation.
# 3.Analysis.

# In[ ]:


'''Description'''
 db['Current Ver'].value_counts()


# In[ ]:


db['Current Ver'].isna().sum()


# In[ ]:


'''Transformation'''
db['Current Ver']=db['Current Ver'].fillna(value=4.1)


# Why do we fillna by 4.1 ?
# thats because 99. of android evices uses ths version or above of it
# so guessing any version will be in the range of this version

# In[ ]:


'''Analysis'''
univariateAnalysis( db['Android Ver'])


# 1- As we thought, filling na  by 4.1 (API=16)is totally true
# a prief  search will show you that this percentage is standard.
# 
# 2- Most of apps having 4.1 version or above . this versions allow most of apps features  to work. so as android developer be carefull of using features that include minimum API >16  .if the app doesn't work ,customer will remove it immediately.
# 
# 3- latest versions which are 6,7,8 . People don't care about updating their versions.

# **Installs:**
# 1. Description
# 
# 2.Tranformation
# 
# 3.Analysis

# In[ ]:


db['Installs'].value_counts()


# In[ ]:


db['Installs'].isnull().sum()


# In[ ]:


'''Transformation'''
#remove + from installs
db['Installs'] = db['Installs'].map(lambda x: x.rstrip('+'))
#remove commas from installs
db['Installs']  = db['Installs'] .str.replace(',', '')
db['Installs'] =db['Installs'] .astype(int)
db['Installs'].value_counts()


# In[ ]:


'''Analysis'''
univariateAnalysis(db['Installs'])


# **what we have here??**
# ups and downs ! . For all apps I think that is because users often don't update their mobile versions so  when a new version or update of the app releases, when they open their app again ,crashes occurs. This leades to unInstalling applications .****

# **Rating :**
# 
# 1 . Description
# 
# 
# 2 . Analysis

# In[ ]:


'''Description'''
db['Rating'].value_counts() 


# In[ ]:


db['Rating'].isna().sum()


# Transformation of Nan values  **is not prefered here ** 
# 
# due to probability  of violation in rating apps. It appears as if you try to improve the image of an application to get more reviews or more downloads.
# 
# 

# In[ ]:


'''Analysis'''
univariateAnalysis(db['Rating'])


# Size
# 1 . Description
# 2 . 

# Most of apps have rating up to 4.3 

# **Bivariate Analysis**

# In[ ]:



def barplotAnalysis(x,y):
    
    plt.figure(figsize=(10,8))
    plt.xticks(rotation=90)
    #plotting
    sns.barplot(x,y,palette="CMRmap")


# In[ ]:


barplotAnalysis(db['Content Rating'],db['Installs'])


# developers tend to develope apps for every one and they succeed to achieve higher rates of downloads

# In[ ]:





# In[ ]:


barplotAnalysis(db['Category'],db['Installs'])


# communications- social apps have higher rates of installs 

# In[ ]:


barplotAnalysis(db['Rating'],db['Installs'])


# In[ ]:



#convert to float
#bins
x = [0, 10,20,50,100,1000]
db['Size']=db['Size'].astype(float)

db['Size'] = pd.cut(db['Size'],x)
db.dtypes
#coding bins groups
#db.Size.astype("category").cat.codes
barplotAnalysis(db['Size'],db['Installs'])
db.dtypes


# Lower sized apps have lower  rate of downloads than higher sized apps. that is because most of downloads is for social apps (Communication) .which usually have high sizes .

# this is for now and thank you for watching
# **TO BE CONTINUED.....**

# In[ ]:




