#!/usr/bin/env python
# coding: utf-8

# ## Rain in Australia - Info

# Predict whether or not it will rain tomorrow by training a binary classification model on target RainTomorrow

# **Content**
# 
# This dataset contains daily weather observations from numerous Australian weather stations.
# 
# The target variable RainTomorrow means: Did it rain the next day? Yes or No.
# 
# Note: You should exclude the variable Risk-MM when training a binary classification model. Not excluding it will leak the answers to your model and reduce its predictability. https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
# 

# ##### Kernel Target

# In this kernel i am going to do some basic Feature Engineering i learned from reading around the web and some udemy tutorials. I was looking for a classification problem and this dataset looked promising and interesting. I am not going to do Feature Selection Engineering and I' ll go all in. Feature Selection will be done on 2nd part.

# ---

# ---

# #### IMPORTS

# In[1]:


# basic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# split data
from sklearn.model_selection import train_test_split

# scale data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# ML
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB


# score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics


pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')


# ----

# ##### LOAD DATASET

# In[2]:


data = pd.read_csv('../input/weatherAUS.csv')
data.head(3)


# ----

# ---

# #### DATA MANIPULATION

# We are not going to use Location and RISK_MM.
# 
# 
# **Note from uploader:** You should exclude the variable Risk-MM when training a binary classification model. Not excluding it will leak the answers to your model and reduce its predictability
# 
# **About Location:** In my opinion, location is an important feature  but i will avoid it for the moment just so it will be simplier for me.
# 

# #### DROP THE COLUMNS
# 

# In[3]:


data.drop(['Location','RISK_MM'], axis=1, inplace=True)


# ##### TAKE CARE OF RainTomorrow COLUMN

# In[4]:


data['RainTomorrow'] = data['RainTomorrow'].map( {'No': 0, 'Yes': 1} ).astype(int)


# ---

# #### DATA INFO
# 

# Type of variables

# In[5]:


data.dtypes


# Number of variables

# In[6]:


len(data.dtypes)


# ##### SOME MORE DATA INFO

# In[7]:


data.describe()


# ----

# ----

# ##### SOME MORE DATA MANIPULATION

# Take advantage of date column

# In[8]:


data['DateNew']= pd.to_datetime(data.Date)


# In[9]:


data['Month'] = data['DateNew'].dt.month
data['Day'] = data['DateNew'].dt.day
data['Year'] = data['DateNew'].dt.year


# In[10]:


data.drop(['Date','DateNew'], axis=1, inplace=True)


# ----

# ---

# ##### VARIABLE TYPES

# **Categorical:** There are four categorical variables

# In[11]:


categorical = [var for var in data.columns if data[var].dtype=='O']
#list(set(categorical))
categorical


# **Numerical:** There are 20 numerical variables

# In[12]:


numerical = [var for var in data.columns if data[var].dtype!='O']
#list(set(categorical))
numerical


# ----

# ----

# ## STEP ONE - GET TO KNOW OUR PROBLEMS

# At this point i am not doing any manipulation at all. I am just dealing with pre processing problems of our data.

# ---

# ##### FEATURE ENGINEERING

# **Missing Values:** There are many missing values on our dataset.

# In[13]:


data.isnull().mean()


# ---

# **Outliers:** Checking for outliers on all numerical values. I will use boxplots.I am using multiple boxplots so it will be more clear.

# In[14]:


plt.figure(figsize=(12,8))
data.boxplot(column=['MinTemp','MaxTemp','Evaporation','Sunshine'])


# In[15]:


plt.figure(figsize=(12,8))
data.boxplot(column=['WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm'])


# In[16]:


plt.figure(figsize=(12,8))
data.boxplot(column=['Pressure9am','Pressure3pm'])


# In[17]:


plt.figure(figsize=(12,8))
data.boxplot(column=['Cloud9am','Temp3pm','Temp9am'])


# In[18]:


plt.figure(figsize=(12,8))
data.boxplot(column=['Rainfall'])


# We can see that MinTemp, MaxTemp, Rainfall, Evaporation, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Pressure9am, Pressure3pm, Temp9am, Temp3pm have outliers. Let'e check each one of theese.

# **Skew or Gaussian:** In order to take care of outliers we need to know about being skew or gaussian.

# In[19]:


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.MinTemp.hist(bins=20)
fig.set_ylabel('Temp')
fig.set_xlabel('MinTemp')

plt.subplot(1, 2, 2)
fig = data.MaxTemp.hist(bins=20)
fig.set_ylabel('Temp')
fig.set_xlabel('MaxTemp')


# In[20]:


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.Rainfall.hist(bins=20)
fig.set_ylabel('Rainfall')
fig.set_xlabel('mm')

plt.subplot(1, 2, 2)
fig = data.Evaporation.hist(bins=20)
fig.set_ylabel('Evaporation')
fig.set_xlabel('mm')


# In[21]:


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.WindSpeed9am.hist(bins=20)
fig.set_ylabel('WindSpeed9am')
fig.set_xlabel('WindSpeed9am')

plt.subplot(1, 2, 2)
fig = data.WindSpeed3pm.hist(bins=20)
fig.set_ylabel('WindSpeed3pm')
fig.set_xlabel('WindSpeed3pm')


# In[22]:


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.WindGustSpeed.hist(bins=20)
fig.set_ylabel('WindGustSpeed')
fig.set_xlabel('WindGustSpeed')

plt.subplot(1, 2, 2)
fig = data.Humidity9am.hist(bins=20)
fig.set_ylabel('Humidity9am')
fig.set_xlabel('Humidity9am')


# In[23]:


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.Pressure9am.hist(bins=20)
fig.set_ylabel('Pressure9am')
fig.set_xlabel('Pressure9am')

plt.subplot(1, 2, 2)
fig = data.Pressure3pm.hist(bins=20)
fig.set_ylabel('Pressure3pm')
fig.set_xlabel('Pressure3pm')


# In[24]:


plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.Temp9am.hist(bins=20)
fig.set_ylabel('Temp9am')
fig.set_xlabel('Temp9am')

plt.subplot(1, 2, 2)
fig = data.Temp3pm.hist(bins=20)
fig.set_ylabel('Temp3pm')
fig.set_xlabel('Temp3pm')


# #### CONCLUSION

# **Gaussian**: MinTemp, MaxTemp, WindSpeed3pm, WindGustSpeed, Pressure9am, Pressure3pm, Temp9am, Temp3pm 

# **Skewed:** Rainfall, Evaporation, WindSpeed9am, Humidity9am

# ----

# ##### FIND OUTLIERS - GAUSSIAN

# ##### MinTemp Outliers

# In[25]:


Upper_boundary = data.MinTemp.mean() + 3* data.MinTemp.std()
Lower_boundary = data.MinTemp.mean() - 3* data.MinTemp.std()
print('MinTemp outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))


# ##### MaxTemp Outliers

# In[26]:


Upper_boundary = data.MaxTemp.mean() + 3* data.MaxTemp.std()
Lower_boundary = data.MaxTemp.mean() - 3* data.MaxTemp.std()
print('MaxTemp outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))


# ##### WindSpeed3pm Outliers

# In[27]:


Upper_boundary = data.WindSpeed3pm.mean() + 3* data.WindSpeed3pm.std()
Lower_boundary = data.WindSpeed3pm.mean() - 3* data.WindSpeed3pm.std()
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))


# ###### WindGustSpeed Outliers

# In[28]:


Upper_boundary = data.WindGustSpeed.mean() + 3* data.WindGustSpeed.std()
Lower_boundary = data.WindGustSpeed.mean() - 3* data.WindGustSpeed.std()
print('WindGustSpeed outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))


# ##### Pressure9am Outliers

# In[29]:


Upper_boundary = data.Pressure9am.mean() + 3* data.Pressure9am.std()
Lower_boundary = data.Pressure9am.mean() - 3* data.Pressure9am.std()
print('Pressure9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))


# ###### Pressure3pm Outliers

# In[30]:


Upper_boundary = data.Pressure3pm.mean() + 3* data.Pressure3pm.std()
Lower_boundary = data.Pressure3pm.mean() - 3* data.Pressure3pm.std()
print('Pressure3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))


# ##### Temp9am Outliers

# In[31]:


Upper_boundary = data.Temp9am.mean() + 3* data.Temp9am.std()
Lower_boundary = data.Temp9am.mean() - 3* data.Temp9am.std()
print('Temp9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))


# ###### Temp3pm Outliers

# In[32]:


Upper_boundary = data.Temp3pm.mean() + 3* data.Temp3pm.std()
Lower_boundary = data.Temp3pm.mean() - 3* data.Temp3pm.std()
print('Temp3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))


# In[ ]:





# In[ ]:





# ---

# ##### FIND OUTLIERS - SKEWED

# ##### Rainfall Outliers

# In[33]:


IQR = data.Rainfall.quantile(0.75) - data.Rainfall.quantile(0.25)
Lower_fence = data.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = data.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# ##### Evaporation Outliers

# In[34]:


IQR = data.Evaporation.quantile(0.75) - data.Evaporation.quantile(0.25)
Lower_fence = data.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = data.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# ##### WindSpeed9am Outliers

# In[35]:


IQR = data.WindSpeed9am.quantile(0.75) - data.WindSpeed9am.quantile(0.25)
Lower_fence = data.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = data.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# ###### Humidity9am Outliers

# In[36]:


IQR = data.Humidity9am.quantile(0.75) - data.Humidity9am.quantile(0.25)
Lower_fence = data.Humidity9am.quantile(0.25) - (IQR * 3)
Upper_fence = data.Humidity9am.quantile(0.75) + (IQR * 3)
print('Humidity9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# ----

# **Rare Labels:** There are no rare labels. All atributes are seen more than 1%

# In[37]:


for var in ['WindGustDir',  'WindDir9am', 'WindDir3pm']:
    print(data[var].value_counts() / np.float(len(data)))
    print()


# ----

# **Cardinlity:** Nope.

# In[38]:


for var in categorical:
    print(var, ' contains ', len(data[var].unique()), ' labels')


# ----

# ----

# ##### SPLIT DATA TO AVOID OVERFITTING WHILE ENGINEERING

# In[39]:


X = data.drop('RainTomorrow', axis=1)


# In[40]:


y = data[['RainTomorrow']]


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
X_train.shape, X_test.shape


# -----

# ----
# 

# ##### ENGINEER NUMERICAL MISSING VALUES

# In[42]:


numerical = [var for var in X_train.columns if data[var].dtype!='O']


# In[43]:


for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())


# ##### FILL ALL MISSING VALUES WITH MEAN

# In[44]:


for col in numerical:
    X_train[col] = X_train[col].fillna((X_train[col].mean()))


# In[45]:


for col in numerical:
    X_test[col] = X_test[col].fillna((X_test[col].mean()))


# ##### ENGINEER CATECORIGAL MISSING VALUES

# In[46]:


for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())


# In[47]:


for df in [X_train, X_test]:
    df['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)


# ---

# In[48]:


X_train.isnull().sum()


# In[49]:


X_test.isnull().sum()


# ----

# ---

# ##### ENGINEER OUTLIERS

# In[50]:


to_describe = ['MinTemp','MaxTemp','WindSpeed3pm','WindGustSpeed','Pressure9am','Pressure3pm',
               'Temp9am','Temp3pm','Rainfall','Evaporation','WindSpeed9am','Humidity9am']


# In[51]:


X_train[to_describe].describe()


# ### Outliers
# 
# ##### Gaussian
# 
# - MinTemp:            -7.02 to 31.39
# - MaxTemp:             1.87 to 44.57
# - WindSpeed3pm:       -7.77 to 45.04 (!)
# - WindGustSpeed:      -0.78 to 80.75 (!)
# - Pressure9am:         996.33 to 1038.97
# - Pressure3pm:         994.14 to 1036.36
# - Temp9am:            -2.49 to 36.46
# - Temp3pm:             0.87 to 42.50
# 
# ##### Skewed
# 
# - Rainfall:           -2.40 to 3.20
# - Evaporation:        -11.80 to 21.80
# - WindSpeed9am:       -29.0 to 55.00
# - Humidity9am:        -21.00 to 161.00

# In[52]:


def top_code(df, variable, top):
    return np.where(df[variable]>top, top, df[variable])


# In[53]:


def bottom_code(df, variable, bottom):
    return np.where(df[variable]<bottom, bottom, df[variable])


# In[54]:


for df in [X_train, X_test]:
    df['MinTemp'] = top_code(df, 'MinTemp', 31.38)
    df['MinTemp'] = bottom_code(df, 'MinTemp', -7.02)
    df['MaxTemp'] = top_code(df, 'MaxTemp', 44.57)
    df['MaxTemp'] = bottom_code(df, 'MaxTemp', 1.87)
    df['WindSpeed3pm'] = top_code(df, 'WindSpeed3pm', 45.04)
    df['WindGustSpeed'] = top_code(df, 'WindGustSpeed', 80.75)
    df['Pressure9am'] = top_code(df, 'Pressure9am', 1038.97)
    df['Pressure9am'] = bottom_code(df, 'Pressure9am', 996.33)
    df['Pressure3pm'] = top_code(df, 'Pressure3pm', 1036.36)
    df['Pressure3pm'] = bottom_code(df, 'Pressure3pm', 994.14)
    df['Temp9am'] = top_code(df, 'Temp9am', 36.46)
    df['Temp9am'] = bottom_code(df, 'Temp9am', -2.49)
    df['Temp3pm'] = top_code(df, 'Temp3pm', 42.50)
    df['Temp3pm'] = bottom_code(df, 'Temp3pm', 0.87)
    
    df['Rainfall'] = top_code(df, 'Rainfall', 3.20)
    df['Evaporation'] = top_code(df, 'Evaporation', 21.80)
    df['WindSpeed9am'] = top_code(df, 'WindSpeed9am', 55.00)
    df['Humidity9am'] = top_code(df, 'Humidity9am', 161.00)


# In[55]:


X_train[to_describe].describe()


# In[56]:


X_test[to_describe].describe()


# ---

# ---

# ##### ENCODE CATEGORICAL VARIABLES

# In[57]:


categorical


# In[58]:


for df in [X_train, X_test]:
    df['WindGustDir']  = pd.get_dummies(df.WindGustDir, drop_first=False)
    df['WindDir9am']  = pd.get_dummies(df.WindDir9am, drop_first=False)
    df['WindDir3pm']  = pd.get_dummies(df.WindDir3pm, drop_first=False)
    df['RainToday']  = pd.get_dummies(df.RainToday, drop_first=False)


# ---

# ---

# ##### SCALE DATA

# In[59]:


mx = MinMaxScaler()


# In[60]:


X_train_mx = mx.fit_transform(X_train)


# In[61]:


X_test_mx = mx.fit_transform(X_test)


# ---

# ##### PREDICTIONS

# ##### KNN

# In[62]:


knn = KNeighborsClassifier()


# In[63]:


knn.fit(X_train_mx, y_train)


# In[64]:


predictions = knn.predict(X_test_mx)


# In[ ]:





# In[65]:


print(accuracy_score(y_test, predictions))


# In[66]:


print(confusion_matrix(y_test, predictions))


# In[67]:


print(classification_report(y_test, predictions))


# In[ ]:





# In[ ]:





# ##### Logistic Regression

# In[68]:


logreg = LogisticRegression()


# In[69]:


logreg.fit(X_train_mx, y_train)


# In[70]:


predictions = logreg.predict(X_test_mx)


# In[71]:


print(accuracy_score(y_test, predictions))


# In[72]:


print(confusion_matrix(y_test, predictions))


# In[73]:


print(classification_report(y_test, predictions))


# In[ ]:





# ##### Gaussian

# In[74]:


gaussian = GaussianNB()


# In[75]:


gaussian.fit(X_train_mx, y_train)


# In[76]:


predictions = gaussian.predict(X_test_mx)


# In[77]:


print(accuracy_score(y_test, predictions))


# In[78]:


print(confusion_matrix(y_test, predictions))


# In[79]:


print(classification_report(y_test, predictions))


# Theese are some pretty bad results i guess but for now i just want to familiarize on Feature Engineering.
# 
# Any suggestions would be welcome. Have fun xD

# In[ ]:




