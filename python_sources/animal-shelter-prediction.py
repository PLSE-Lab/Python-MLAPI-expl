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


# %pwd
get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', 'input')
get_ipython().run_line_magic('cd', 'shelter-animal-outcomes')


# In[ ]:


# data is the training and validation set
data = pd.read_csv('train.csv.gz',compression='gzip')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.AnimalType.unique()


# In[ ]:


data.SexuponOutcome.unique()
# in this case we can actually replace 'nan' values with 'unknown'


# In[ ]:


data.OutcomeType.unique()


# In[ ]:


data.OutcomeSubtype.unique()
# this can play a key feature role in predicting the 'outcometype'
# I'shall replace 'nan' value with either most frequent one or least freq one and check the results
# else I'll create new value: or after encoding it onehotencoder way we will drop nan column


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data['OutcomeType'].value_counts().plot.bar(figsize=(10,5),fontsize=14,color='red')
plt.title('Outcome Type',color='white',fontsize=18)
plt.xlabel('Category',color='yellow',fontsize=16)
plt.ylabel('Frequency',color='yellow',fontsize=16)
plt.show()


# In[ ]:


data['OutcomeSubtype'].value_counts().plot.bar(figsize=(10,5),fontsize=14,color='green')
plt.title('Outcome Subtype ',color='white',fontsize=18)
plt.xlabel('Category',color='yellow',fontsize=16)
plt.ylabel('Frequency',color='yellow',fontsize=16)
plt.show()


# In[ ]:


data['SexuponOutcome'].value_counts().plot.bar(figsize=(10,5),fontsize=14,color='blue')
plt.title('Sex Upon Outcome',color='white',fontsize=18)
plt.xlabel('Category',color='yellow',fontsize=16)
plt.ylabel('Frequency',color='yellow',fontsize=16)
plt.show()


# In[ ]:


data['AnimalType'].value_counts().plot.bar(figsize=(10,5),fontsize=14,color='orange')
plt.title('Animal Type',color='white',fontsize=18)
plt.xlabel('Category',color='yellow',fontsize=16)
plt.ylabel('Frequency',color='yellow',fontsize=16)
plt.show()


# In[ ]:


data['AgeuponOutcome'].value_counts().plot.bar(figsize=(10,5),fontsize=14,color='grey')
plt.title('Age upon Outcome',color='white',fontsize=18)
plt.xlabel('Category',color='yellow',fontsize=16)
plt.ylabel('Frequency',color='yellow',fontsize=16)
plt.show()


# In[ ]:


data.apply(lambda x: sum(x.isnull()/len(data)))


# 1. OutcomeSubtype: replace null values with 'other'
# 2. SexuponOutcome: replace null value with 'Unknown'
# 3. Name: replace null values with 'noName' and non-null values with 'hasName'
# 4. AgeuponOutcome: replace null values with 'mean value'... Also convert week,month,year into 'days'
# 5. DateTime: apply condition of time convert into 'morning,afterNoon,night,afterMidnight'
#    >We can try to create one extra column using DateTime which will be 'season of the year' using month from date part.
# 6. Convert into dummy variable avoiding the trap!! 
# 7. Drop columns which are converted in suitable ones.
# 8. Convert into onehotencoding
# 9. AtLast,There will be need of scaling the data?
# 

# In[ ]:


df = data.copy()


# In[ ]:


# Step1
df['OutcomeSubtype'] = df[['OutcomeSubtype']].fillna(value='Other')


# In[ ]:


# Step2
df['SexuponOutcome'] = df[['SexuponOutcome']].fillna(value='Unknown')


# In[ ]:


# Step3
df['Name'] = df[['Name']].fillna(value='noName')
df['Name'] = df[['Name']].replace(to_replace=df[df['Name']!='noName'],value='hasName')


# In[ ]:


#  just for assureity I will undo changes to df by running copy block...
# then I'll just run the following two blocks... and just casually check if there are any other values...
df.Name.value_counts()


# In[ ]:


df.AgeuponOutcome


# we did mistake in if condition (Step4) 'days' instead of 'day' so it doesn't changed the value, 
# >also we didn't checked null values/ nan values..

# In[ ]:


# Step4
# In this step you deal with each one at time, i.e, year, month, week, days!
# we  will convert everything to integer(days)
age=[]
for x in df.AgeuponOutcome:
    if 'years' in str(x):
        x = int(x[0:2]) * 365
    if 'year' in str(x):
        x = int(x[0:1]) * 365
    if 'months' in str(x):
        x = int(x[0:2]) * 30
    if 'month' in str(x):
        x = int(x[0:1]) * 30
    if 'weeks' in str(x):
        x = int(x[0:2]) * 7
    if 'week' in str(x):
        x = int(x[0:1]) * 7
    if 'days' in str(x):
        x = int(x.replace('days',''))
    # i have now corrected the mistake
    if 'day' in str(x):
        x = int(x.replace('day',''))
    age.append(x)


# In[ ]:


df['age'] = pd.Series(age).values
df.head(10)


# In[ ]:


# As now we have seen that AgeuponOutcome is correctly converted into age, so now we can drop AgeuponOutcome column
df.drop(columns=['AgeuponOutcome'],inplace=True)


# In[ ]:


# Step5
# first we check if values are in string or datetime format
type(df.DateTime[0])


# In[ ]:


# converting DateTime(str) to datetime(correct format to extract directly the individual parts like year,month, hour,etc.)
from datetime import datetime
df['datetime'] = df['DateTime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))


# In[ ]:


# now we can drop original 'DateTime' column
df.drop(columns=['DateTime'],inplace=True)


# In[ ]:


df.head()


# In[ ]:


timing=[]
# lets create earlymorning(03:00-06:59), morning(07:00-11:59), afternoon(12:00-15:59), evening(16:00-19:59), night(20:00-23:59), latenight(00:00-02:59)
for x in df.datetime:
    if (0 <= x.hour < 3):
        x = 'late_night'
    elif (3 <= x.hour < 7):
        x = 'early_morning'
    elif (7 <= x.hour < 12):
        x = 'morning'
    elif (12 <= x.hour < 16):
        x = 'afternoon'
    elif (16 <= x.hour < 20):
        x = 'evening'
    else:
        x = 'night'
    timing.append(x)


# In[ ]:


pd.Series(timing).shape


# In[ ]:


df.shape


# In[ ]:


df['time'] = pd.Series(timing).values


# In[ ]:


df.time.dtype


# In[ ]:


season=[]
# lets create earlymorning(03:00-06:59), morning(07:00-11:59), afternoon(12:00-15:59), evening(16:00-19:59), night(20:00-23:59), latenight(00:00-02:59)
for x in df.datetime:
    if (3 <= x.month <= 5):
        x = 'Spring'
    elif (6 <= x.month < 8):
        x = 'Summer'
    elif (9 <= x.month < 11):
        x = 'Fall(autumn)'
    else:
        x = 'Winter'
    season.append(x)


# In[ ]:


pd.Series(season).shape


# In[ ]:


df['season'] = pd.Series(season).values
df.head()


# In[ ]:


# as now we have extracted the needed features - time and season
# Now we wih to drop/remove dateitme feature
df.drop(columns=['datetime'],inplace=True)
df.head()


# In[ ]:


df.Breed.value_counts()


# In[ ]:


#  now as we can see there are a whole lot of breeds of both cats and dogs so I will turn them simply into 'mix' and 'pure' breeds.
# for this we will focus on two part of the string values => 'Mix'(for mix) and all other as pure.
df['Breed'] = df['Breed'].apply(lambda x: 'Mix' if 'Mix' in x else 'Pure')
df.head()   


# In[ ]:


# similarly for Color we will focus on few top colors and consider other colors as 'other_color'
# top 17 colors are having frequency more than 500. so, we'll have total 18 colors including other_color
df.Color.value_counts()[0:17]


# In[ ]:


other_colors = []
for x in df.Color.unique()[17:]:
    other_colors.append(x)

df['colors'] = df['Color'].replace(list(other_colors),'other_colors')
df.colors.value_counts()


# In[ ]:


df.drop(columns=['Color'],inplace=True)
df.tail(20)


# In[ ]:


# Step6 Converting all the categorical values into dummy variables
# ALSO WE AVOID TOUCHING 'OutcomeType' WE'LL DEAL WITH IT LATER ON
df.head()


# In[ ]:


df.columns


# # Before moving forward with feature engineering part involving conversion of categorical variables to dummy variables
# ## Just check what are the unique values for each feature (column) so as to be sure for nan/null/other non-entertaining values for features.

# In[ ]:


# check for name
df.Name.unique()


# In[ ]:


# check for OutcomeType
df.OutcomeType.unique()


# In[ ]:


# check for OutcomeSubtype
df.OutcomeSubtype.unique()


# In[ ]:


# check for AnimalType
df.AnimalType.unique()


# In[ ]:


# check for SexuponOutcome
df.SexuponOutcome.unique()


# In[ ]:


# check for Breed
df.Breed.unique()


# In[ ]:


# check for age
df.age.unique()


# > we can see age is having 'nan' value....we need to replace it by mean? yes we can...

# In[ ]:


# check for time
df.time.unique()


# In[ ]:


# check for season
df.season.unique()


# In[ ]:


# check for colors
df.colors.unique()


# ## so we need to deal with 'nan' in age... let's replace it with mean value for age!

# In[ ]:


# changing float dtype to str stype for easiness in computation 
df.age.astype(str)

# replacing nan values with mean values...
df['age'].fillna(df.age.mean(), inplace = True)

# again convert it into int
df.age.astype(int)


# > also I earlier thought that it is'nan' (string) but it was empty or null value...

# In[ ]:


df.age.unique()


# In[ ]:


df.head(3)


# In[ ]:


df.dtypes


# In[ ]:


df1 = pd.get_dummies(df['Name'])
df = pd.concat([df, df1], axis=1)

df1 = pd.get_dummies(df['AnimalType'])
df = pd.concat([df, df1], axis=1)

df1 = pd.get_dummies(df['SexuponOutcome'])
df = pd.concat([df, df1], axis=1)

df1 = pd.get_dummies(df['OutcomeSubtype'])
df = pd.concat([df, df1], axis=1)

df1 = pd.get_dummies(df['Breed'])
df = pd.concat([df, df1], axis=1)

df1 = pd.get_dummies(df['colors'])
df = pd.concat([df, df1], axis=1)

df1 = pd.get_dummies(df['time'])
df = pd.concat([df, df1], axis=1)

df1 = pd.get_dummies(df['season'])
df = pd.concat([df, df1], axis=1)


# In[ ]:


df.columns


# In[ ]:


# dropping the categorical features now, as they are now converted into numerical oneHotEncoder features
# setting the index as 'AnimalID'
df.drop(columns=['Name', 'OutcomeSubtype', 'AnimalType', 'SexuponOutcome', 'Breed','time', 'season', 'colors'], inplace=True)
df = df.set_index('AnimalID',drop=True)
df.head()


# In[ ]:


df1 = pd.get_dummies(df['OutcomeType'])
df = pd.concat([df, df1], axis=1)


# In[ ]:


df.drop(columns=['OutcomeType'],inplace=True)
df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.shape


# In[ ]:


# just confirming which slicing of columns into x and y is correct
# 0 to 61 total > we have 62 columns
# 61,60,59,58,57 - Y(last 5 columns)

# x=df[0:57]
# y=df[57:62]


# # splitting data for training and testing 
# # also importing log_loss

# ## Note: We can actually reduce dimensionallity reduction technique to reduce the number of dimensions, but for starting we are just using them as it is... 
# #### One hot encoder comes with the curse of increasing vast dimensions/ features.

# In[ ]:


# splitting the datasets
X = df.iloc[:,0:57].values
Y = df.iloc[:,57:62].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)


# In[ ]:


from sklearn.metrics import log_loss


# # Applying Ensemble ExtraTreeClassifier

# In[ ]:


# 1
from sklearn.ensemble import ExtraTreesClassifier
extra_clf = ExtraTreesClassifier(n_estimators=100,criterion='gini',max_depth=3,min_samples_split=5,oob_score=False,
                                 random_state=1,warm_start=False,class_weight=None)
extra_clf.fit(X_train,y_train)
y_pred = extra_clf.predict(X_test)
y_probs = extra_clf.predict_proba(X_test)
log_loss(y_test,y_pred)


# In[ ]:


yb = pd.Series(y_probs)
len(yb[0])


# In[ ]:


len(y_test)


# ### we need to change the shape of yb/y_probs... it is a transpose...row and column exchanged!

# In[ ]:


# 2
from sklearn.ensemble import ExtraTreesClassifier
extra_clf = ExtraTreesClassifier(n_estimators=60,criterion='gini',max_depth=9,
                                 random_state=1,warm_start=True,class_weight=None)
extra_clf.fit(X_train,y_train)
y_pred = extra_clf.predict(X_test)
y_probs = extra_clf.predict_proba(X_test)
log_loss(y_test,y_pred)


# # Applying RandomForestClassifier

# # Applying XGBoostClassifier

# # Applying best of the above classifiers after dimensionality reduction!

# In[ ]:





# In[ ]:





# In[ ]:




