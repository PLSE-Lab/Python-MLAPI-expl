#!/usr/bin/env python
# coding: utf-8

# #                              Google PlayStore Dataset Analysis

# ## In this Notebook, we are going to analyse the dataset (taken from Kaggle) of all the Apps in the Google Play Store
# 
# ### The series of steps followed are :

# > #### 1. Importing Packages
# > #### 2. Reading Data
# > #### 3. Data Preprocessing
#     > #### -  3.1 Handling NULL Values
#     > #### -  3.2 Handling Data Types and Values
# > #### 4. Analyzing Features
# > #### 5. Furthur Analysis.

# <a id='1'></a>
# ## 1. Importing the required packages. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#To hide Warning messages.
import warnings
warnings.filterwarnings('ignore')


# ***

# <a id='2'></a>
# ## 2.  Reading Data

# In[ ]:


df = pd.read_csv("../input/googleplaystore.csv")

df.head()


# ***

# <a id='3.1'></a>
# 
# ## 3. Data Preprocessing
# 
# 
# ### 3.1 Handling NULL Values

# #### This is a very crucial step in every analysis and model, which on doing, improves the accuracy of insights and predictions.

# In[ ]:


print(df.isnull().sum())

df.dropna(inplace=True) #Dropping Rows with Null values


# > #### There are many NULL values in Rating, and few in Type,Content Rating and Versions.

# In[ ]:


df.drop_duplicates(inplace=True)


# > #### Removing Duplicate entries.

# In[ ]:


df.shape


# >  ####   After removing the rows with Null values and the duplicate entries, We have got 8886 apps to analyze for their ratings and performance.
# 
# ***

# <a id='3.2'></a>
# ### 3.2 Handling Data Types of each Feature
# 
# #### The data types of each feature must be changed to a proper format that can be used for analysis.

# In[ ]:


df.dtypes  # Displaying Data types of each feature.


# > #### The feature *Reviews* must be of numerical type. So we should change it.

# In[ ]:


df.Reviews = df.Reviews.astype('int64') #Changing to int type.


# >  #### Other Features like *Size*, *Installs*, *Price* and *Android Vers* also must be of numeric type.
# >  #### The values they are holding must be changed to a proper format so that we can use them for analysis and plots. 
# >  #### Example : '10000+' to 10000, '$100' to 100 

# ### Changing the Feature : Installs

# In[ ]:


newInstalls = []

for row in df.Installs:
    
    row = row[:-1]
    newRow = row.replace(",", "")
    newInstalls.append(float(newRow))
    

df.Installs = newInstalls

df.Installs.head()


# ### Changing the feature : Size

# In[ ]:


newSize = []

for row in df.Size:
    newrow = row[:-1]
    try:
        newSize.append(float(newrow))
    except:
        newSize.append(0) #When it says - Size Varies.
    
df.Size = newSize

df.Size.head()


# ### Changing the feature, Price

# In[ ]:


newPrice = []

for row in df.Price:
    if row!= "0":
        newrow = float(row[1:])
    else:
        newrow = 0 
        
    newPrice.append(newrow)
        
df.Price = newPrice

df.Price.head()
    


# ### Changing the feature, Android Ver

# In[ ]:


newVer = []

for row in df['Android Ver']:
    try:
        newrow = float(row[:2])
    except:
        newrow = 0  # When the value is - Varies with device
    
    newVer.append(newrow)
    
df['Android Ver'] =  newVer

df['Android Ver'].value_counts()


# ***

# <a id='4'></a>
# ## 4. Analyzing Features :

# ### 4.1 Categories

# > ##### Displaying all the categories and their counts.

# In[ ]:


df.Category.value_counts() 


# In[ ]:


df.Category.value_counts().plot(kind='barh',figsize= (12,8))


# > #### **Insight** : Maximum Number of Apps belong to the Family and Game Category.
# 
# ***

# ### 4.2  Rating

# In[ ]:


df.Rating.describe()


# > #### Distribution Plot of 'Rating'

# In[ ]:


sns.distplot(df.Rating)


# > #### Insight : Most of the apps, clearly hold a rating above 4.0 ! And surprisingly a lot seem to have 5.0 rating.
# 
# ***

# In[ ]:


print("No. of Apps with full ratings: ",df.Rating[df['Rating'] == 5 ].count())


# > #### There are 271 Apps in the store which hold 5.0 Ratings. Do all of these actually deserve it? Or are these spammed ratings? Lets analyze furthur.
# 
# ***

# ### 4.3 Consider the Reviews:

# > ##### Distribution Plot of the feature 'Reviews'

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df.Reviews)


# > #### Let's look into those apps which have a good amount of Reviews.

# In[ ]:


df[df.Reviews>40000000]


# > #### Insight : The most famous Apps like WhatsApp, Facebook and Clash of Clans are the most reviewed Apps as shown above.
# 
# > Note : And I still have to figure out, how to remove the duplicate entries. My Apologies.
# 
# ***

# ### 4.4  Type:

# In[ ]:


plt.pie(df.Type.value_counts(), labels=['Free', 'Paid'], autopct='%1.1f%%')


# > #### Insight: 93% of the Apps are Free in the Play Store.
# 
# ***

# ### 4.5 Price

# In[ ]:


df[df.Price == df.Price.max()]


# > #### Insight : The most costly App in the Store is: *I'm Rich - Trump Edition* costing $400
# 
# ***

# ### 4.6 Android Version

# In[ ]:


df['Android Ver'].value_counts()


# > #### Count Plot of the various Versions

# In[ ]:


sns.countplot(df['Android Ver'])


# > #### Insight : Most of the apps support Android 4.0 and above.
# 
# 
# ***

# <a id='5'></a>
# ### 5. Furthur Analysis
# 
# ### Looking at the Apps with 5.0 ratings:

# In[ ]:


df_full = df[df.Rating == 5]

df_full.head()


# > #### Distribution plot of 'Installs' of Apps with 5.0 Ratings

# In[ ]:


sns.distplot(df_full.Installs)


# In[ ]:


df_full.Installs.value_counts().sort_index()


# > #### Insight : There are many Apps that have full ratings but less downloads/installs. So we can't really consider those apps as the best ones.
# 
# ***

# ### Consider the Apps with 5.0 Ratings and Maximum Installs :

# In[ ]:


df_full_maxinstalls = df_full[df.Installs > 1000]

df_full_maxinstalls[['App', 'Category', 'Installs']]


# > ### Checking the No. of Reviews of 5.0 Rating Apps

# In[ ]:


sns.distplot(df_full.Reviews)


# > #### The above distribution is clearly skewed. Apps with very few reviews easily managed to get 5.0 ratings which can be misleading.
# > #### So let's filter out the ones with more than 30 reviews. These filtered ones are the apps that really stand for 5.0 rating.

# In[ ]:


df_full = df_full[df.Reviews > 30]


# In[ ]:


print("No. of Apps having 5.0 Rating with sufficient Reviews: ",df_full.App.count())


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(df_full.Genres)


# > #### Insight : Apps related to Education, LifeStyle and Tools seem to fetch full Ratings with sufficient number of reviews.
# 
# ***

# In[ ]:


sns.countplot(df_full.Price)


# > #### Insight : All the Apps with 5.0 ratings are Free to install.
# 
# ***
