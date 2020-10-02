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


# # <span style="color:lightblue">**Will it rain tomorrow?**</span>
# 
# <iframe src="https://giphy.com/embed/KatjlSAMx0K9zdHMR4" width="480" height="347" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/tcm-vintage-turner-classic-movies-gene-kelly-KatjlSAMx0K9zdHMR4">via GIPHY</a></p>
# 
# #### Context
# Predict whether or not it will rain tomorrow by training a binary classification model 
# 
# #### Content
# This dataset contains daily weather observations from numerous Australian weather stations such as Rainfall, Wind and Humidity.

# ### Variable Descriptions:
# 
# * ** RainTomorrow: The target variable. Did it rain the following day? YES/NO**
# 
# * Date: The date of observation
# * Location: The common name of the location of the weather station
# * MinTemp: The minimum temperature in degrees celsius
# * MaxTemp: The maximum temperature in degrees celsius
# * Rainfall: The amount of rainfall recorded for the day in mm
# * Evaporation: The so-called Class A pan evaporation (mm) in the 24 hours to 9am
# * Sunshine: The number of hours of bright sunshine in the day.
# * WindGustDir: The direction of the strongest wind gust in the 24 hours to midnight
# * WindGustSpeed: The speed (km/h) of the strongest wind gust in the 24 hours to midnight
# * WindDir9am: Direction of the wind at 9am
# * WindDir3p: Direction of the wind at 3pm
# * WindSpeed9am: Wind speed (km/hr) averaged over 10 minutes prior to 9am
# * WindSpeed3pm: Wind speed (km/hr) averaged over 10 minutes prior to 3pm
# * Humidity9a: Humidity (percent) at 9am
# * Humidity3pm: Humidity (percent) at 3pm
# * Pressure9am: Atmospheric pressure (hpa) reduced to mean sea level at 9am
# * Pressure3pm: Atmospheric pressure (hpa) reduced to mean sea level at 3pm
# * Cloud9am: Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many eigths of the sky are obscured by cloud. A 0 measure indicates completely clear sky whilst an 8 indicates that it is completely overcast.
# * Cloud3pm: Fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description of the values
# * Temp9am: Temperature (degrees C) at 9am
# * Temp3pm: Temperature (degrees C) at 3pm
# * RainToday: Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0
# * RISK_MM: The amount of next day rain in mm. Used to create response variable RainTomorrow. A kind of measure of the "risk". Will be left out in the model.
# 

# In[ ]:


# importing libraries and magic functions

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('config', "InlineBackend.figure_format ='retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Exploratory Data Analysis

# In[ ]:


# read data
df = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
df.head()


# **Data Cleaning & Manipulation**

# In[ ]:


df.describe()
print("The size of the dataframe is:",df.shape)


# In[ ]:


# check for null values
df_missing = df.isnull().sum()
df_missing

# calculate the % of missing values
perc_missing = round(100*(df_missing/len(df)),2)
perc_missing


# In[ ]:


# dropping columns with large % of missing values - also dropping RISK_MM based on instructions in the description

df_dropped = df.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm','RISK_MM'], axis=1)


# In[ ]:


df_dropped.isna().sum()


# In[ ]:


# Now we are dropping the remaining rows with nan values

df_dropped = df_dropped.dropna()


# In[ ]:


df_dropped.head()
print("The new size of the dataframe is:", df_dropped.shape)
print("We deleted",df.shape[0]-df_dropped.shape[0],"rows and", df.shape[1]-df_dropped.shape[1],"columns.")
df_dropped.dtypes


# In[ ]:


# change date type to datetime

df_dropped['Date'] = pd.to_datetime(df_dropped['Date'])


# In[ ]:


# Adding columns Year and Month

df_dropped['Year'] = pd.to_datetime(df_dropped['Date']).dt.year
df_dropped['Month'] = pd.to_datetime(df_dropped['Date']).dt.month


# In[ ]:


# set Date as index

df_dropped.set_index('Date', inplace=True)
df_dropped.head()


# ## Exploratory Data Analysis

# In[ ]:


# Plotting rainfall during time
plt.figure(figsize=(20,5))
df_dropped['Rainfall'].plot()
plt.box(False)
plt.title ('Rainfall throughout the Years',fontweight="bold", fontsize=15)


# In[ ]:


# plotting Rainfall per Month
plt.figure(figsize=(8,5))
sns.barplot(x = 'Month', y='Rainfall', data=df_dropped, color = 'skyblue')
plt.box(False)
plt.title ('Rainfall throughout Months', fontweight="bold",fontsize=15)


# In[ ]:


# plotting average Rainfall by Location
df_loc = df_dropped.groupby('Location').agg({'Rainfall':'mean'}).sort_values(by='Rainfall', ascending=False) 

df_loc.plot(kind='bar',figsize=(20,5))
plt.box(False)
plt.title ('Average Rainfall by Location', fontsize=15, fontweight="bold")
plt.show()


# In[ ]:


# Plotting Temperature and Rainfall

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.despine(left=True)
sns.scatterplot(x='MinTemp', y='Rainfall', data=df_dropped, ax=ax1)
ax1.set_title("Lowest Temperature and Amount of Rainfall",fontweight="bold")
sns.scatterplot(x='MaxTemp', y='Rainfall', data=df_dropped, color="tomato", ax=ax2)
ax2.set_title("Highest Temperature and Amount of Rainfall",fontweight="bold")


# ## Data Preparation for ML

# In[ ]:


# Renaming Dataframe for the Machine Learning Part
df_ML = df_dropped


# In[ ]:


# Dropping columns that we do not need for the model building part
df_ML = df_ML.drop(['Location','Year'], axis=1)


# In[ ]:


# Adjusting the Target Variables' values: Yes/No with 1/0
df_ML = df_ML.replace({'RainTomorrow':'Yes','RainToday':'Yes'},1)
df_ML = df_ML.replace({'RainTomorrow':'No','RainToday':'No'},0)


# In[ ]:


# Create Dummies for categorical variables
df_ML = pd.get_dummies(df_ML, prefix = ['WindDir3pm','WindDir9am','WindDir3pm'])

df_ML.head()


# In[ ]:


# Correlation
# Create Correlation mask >0.5:
df_ML_corr = df_ML.corr()
condition = abs(df_ML.corr()) > 0.5
#df_ML_corr[condition]


# In[ ]:


# heatmap
# correlation plot
plt.figure(figsize=(20,20))
sns.heatmap(df_ML.corr(), cmap = 'Wistia')


# In[ ]:


# Dropping highly correlated columns

df_ML = df_ML.drop(['WindGustSpeed','Humidity9am',], axis=1)


# ### Feature Scaling

# In[ ]:


# Standardize our Data - Feature Scaling 0-1 scale 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1)) 

#assign scaler to column:
df_scaled = pd.DataFrame(scaler.fit_transform(df_ML), columns=df_ML.columns)

df_scaled.head()


# ### Feature Selection

# In[ ]:


# Selection of the most important features using SelectKBest
from sklearn.feature_selection import SelectKBest, chi2

X = df_scaled.loc[:,df_scaled.columns!='RainTomorrow']
y = df_scaled[['RainTomorrow']]

selector = SelectKBest(chi2, k=5)
selector.fit(X, y)

X_new = selector.transform(X)
print("The 5 most important features are:", X.columns[selector.get_support(indices=True)]) 


# In[ ]:


# Creating a new dataframe with the most important features

df_new = df_scaled[['Rainfall', 'Humidity3pm','WindDir9am_E', 'WindDir9am_N','RainToday','RainTomorrow']]


# ### Checking the Target variables' distribution

# In[ ]:


df_new['RainTomorrow'].value_counts()[0]


# In[ ]:


Percentage_No = df_new['RainTomorrow'].value_counts()[0]/len(df_new['RainTomorrow'])*100
Percentage_Yes = df_new['RainTomorrow'].value_counts()[1]/len(df_new['RainTomorrow'])*100


# In[ ]:


# checking the distribution of our target variable 
print(df_new['RainTomorrow'].value_counts())

print("Percentage Occurences of No Rain on the following day:", round(Percentage_No,2),"%")
print("Percentage Occurences of Rain on the following day:", round(Percentage_Yes,2),"%")

sns.countplot(df_new['RainTomorrow'])
plt.title('Balance target',fontsize=15, fontweight='bold')
plt.box(False)


# **We can see that the distribution between the two outcomes Rain on the following day and No Rain on the following day is unbalanced. This can lead to a biased result. Therefore we will balance the outcome variable in the next step while splitting the training ans testing data using the stratify argument.**
# 

# ### Train-Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

# clarify what is y and what is X
y = df_new['RainTomorrow']
X = df_new.drop(['RainTomorrow'], axis = 1)

# Train-Test Split 80-20
# Note: We use stratify = y here because we have an unbalanced Dataset and we want to sample equal occurences of the target variable outcomes
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,stratify = y)


# ### Choosing the best Model

# In[ ]:


### Model Pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import time

classifiers = [LogisticRegression(),DecisionTreeClassifier(),KNeighborsClassifier(2)]

for classifier in classifiers:
    t0=time.time()
    pipe = Pipeline(steps=[('classifier', classifier)])
    pipe.fit(X_train, y_train)   
    score = pipe.score(X_test, y_test)
    print(f"The accuracy score is: {round(score,2)*100}%")
    print('Time taken to execute:' , time.time()-t0)


# **We get the best accuracy score with the Logistic Regression Model with 84%. The Decision Tree Algorithm is slightly faster but only scores 83%.KNN performs worse and is comparatable slow.**

# In[ ]:




