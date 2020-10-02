#!/usr/bin/env python
# coding: utf-8

# #### Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Reading Data for a csv file

# In[ ]:


df = pd.read_csv('../input/flavors_of_cacao.csv')


# #### Data Exploration

# In[ ]:


df.head()


# #### Data Metrics

# In[ ]:


df.info()


# In[ ]:


df.describe()


# #### Checking for NaN Attributes

# In[ ]:


df.isnull()


# #### Heat Map for better Visualization

# In[ ]:


sns.heatmap(df.isnull(), cbar = False, cmap='coolwarm')


# #### Checking different column names

# In[ ]:


df.columns


# #### Different Bean type and their counts

# In[ ]:


df['Bean\nType'].value_counts()


# #### Total number of Beans 

# In[ ]:


df['Bean\nType'].nunique()


# #### Checking for correlation

# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


df.columns


# #### Getting Unique Values for every text related column 

# In[ ]:


print('Unique Values:')
print('Company (Maker-if known): ',df['Company\xa0\n(Maker-if known)'].nunique())
print('Specific Bean Origin or Bar Name: ', df['Specific Bean Origin\nor Bar Name'].nunique())
print('Company Location: ',df['Company\nLocation'].nunique())
print('Bean Type: ', df['Bean\nType'].nunique())
print('Broad Bean Origin', df['Broad Bean\nOrigin'].nunique())
print('Review Date: ', df['Review\nDate'].nunique())
print('Cocoa Percent: ', df['Cocoa\nPercent'].nunique())


# #### Data Visualization
# 

# #### Rating Distribution

# In[ ]:


sns.countplot(x = df['Rating'])


# About 370 ratings below to 3.5 followed by 3.0 

# #### Year-wise distribution

# In[ ]:


sns.countplot(x = df['Review\nDate'])


# #### Rating and Review Date Concentrations

# In[ ]:


sns.jointplot(x = 'Rating', y= 'Review\nDate', data = df, kind='kde', color = 'brown')


# #### Converting String into Integers for better classification

# In[ ]:


df['Cocoa\nPercent'] = df['Cocoa\nPercent'].str.replace('%', '')
df['Cocoa\nPercent'] = df['Cocoa\nPercent'].str.replace('.', '')
df['Cocoa\nPercent'] = df['Cocoa\nPercent'].astype(int)


# #### Corrections: Cocoa Percent cannot be above 100 %
# * 75.5% --->   75.5
# * 75.5 --->   755

# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(x= 'Cocoa\nPercent', data = df, color = 'brown')


# #### To fix the above error

# In[ ]:


def normalizeIt(percent):
    if percent > 100:
        percent = int(str(percent)[:2])
    return percent


# In[ ]:


df['Cocoa\nPercent'] = df['Cocoa\nPercent'].apply(normalizeIt)


# #### Let's Plot it again

# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(x= 'Cocoa\nPercent', data = df, color = 'brown')


# It worked!!

# #### Converting Rating

# In[ ]:


df['Rating'] = (df['Rating']* 100).astype(int)
df['Rating'].head(5)


# In[ ]:


df.columns


# #### Featurizing Text

# In[ ]:


company = pd.get_dummies(df['Company\xa0\n(Maker-if known)'],drop_first=True)
sbOrigin = pd.get_dummies(df['Specific Bean Origin\nor Bar Name'],drop_first=True)
companyLocation = pd.get_dummies(df['Company\nLocation'],drop_first=True)
bType = pd.get_dummies(df['Bean\nType'],drop_first=True)
bbOrigin = pd.get_dummies(df['Broad Bean\nOrigin'],drop_first=True)


# In[ ]:


df = pd.concat([df, company, sbOrigin, companyLocation, bType, bbOrigin], axis = 1)


# #### Dropping Columns which have been Featurized

# In[ ]:


df.drop(['Company\xa0\n(Maker-if known)', 'Specific Bean Origin\nor Bar Name','Company\nLocation', 'Bean\nType', 
         'Broad Bean\nOrigin'], axis = 1, inplace = True )


# #### Removing Duplicate Column 
# Added due to featurization.
# 
# [StackOverFlow link](https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns)

# In[ ]:


df = df.loc[:,~df.columns.duplicated()]


# #### Splitting Into Training and Testing data sets

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop('Rating', axis = 1) #Features
y = df['Rating']   # Target Variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)


# #### Importing Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)


# In[ ]:


df.columns


# #### Checking for Duplicate Columns 

# In[ ]:


df['Venezuela'].head(5)


# In[ ]:


rfc_pred = rfc.predict(X_test)


# #### Let's Compare how the model performed

# In[ ]:


from sklearn.metrics import classification_report, accuracy_score


# In[ ]:


print(classification_report(y_test,rfc_pred))


# In[ ]:


print(accuracy_score(y_test,rfc_pred)*100)


# Hmm, Let's try something else 

# #### So most ratings are between to 3.0 to 3.5

# In[ ]:


sns.countplot(x = 'Rating', data=df)


# #### Let's Group them as if they were stars
# *You can use a much more complex logic for this.*

# In[ ]:


def rating_to_stars(rating):
    
    rating = int(rating)
    
    if (rating == 0.0 ):
        return 0.0
    elif (rating > 0 ) and (rating <= 199 ):
        return 1.0
    elif (rating >= 200 ) and (rating <= 299 ):
        return 2.0
    elif (rating >= 300 ) and (rating <= 399 ):
        return 3.0
    else:
        return 4.0


# #### Let's apply it

# In[ ]:


df['Rating'] = df['Rating'].apply(rating_to_stars)


# #### Did it Work ?

# In[ ]:


sns.countplot(x = 'Rating', data=df)


# * Most ratings are in 3
# * Makes sense as many of them were between 3 to 3.5

# #### Splitting Again

# In[ ]:


X = df.drop('Rating', axis = 1)
y = df['Rating']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)


# In[ ]:


rfc = RandomForestClassifier(n_estimators=5000, min_weight_fraction_leaf= 0)
rfc.fit(X_train, y_train)


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,rfc_pred))


# In[ ]:


print(accuracy_score(y_test,rfc_pred)*100)


# 
