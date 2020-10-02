#!/usr/bin/env python
# coding: utf-8

# **Heart Disease Dataset EDA**
# 
# This is a very basic EDA of Heart Disease dataset from UCI. This EDA explores all the features from the dataset and discusses the insights from the plots.
# 
# Looking for feedback and suggestions for improvement.
# 

# In[ ]:


# importing libraries.!!

import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# loading data

data = pd.read_csv('../input/heart.csv')


# In[ ]:


# Let's take a look into the data

data.head()


# In[ ]:


print(f'Data have {data.shape[0]} rows and {data.shape[1]} columns')


# In[ ]:


# distribution of the features in the dataset

data.describe().T


# 1.  Categorical Variables
# 
# Dataset contains some categorical variables like 
# * sex
# * cp
# * fbs
# * restecg
# * exang
# * slope
# * ca
# * thal
# 
# and finally our target
# 
# Let's do some EDA on these features first

# In[ ]:


# check the target distribution

plt.rcParams['figure.figsize'] = (8, 6)

sns.countplot(x='target', data=data);


# Let's see how much males and females are there in the dataset who have a heart problem

# In[ ]:


data.groupby(by=['sex', 'target'])['target'].count()


# In[ ]:


pd.crosstab(data['sex'], data['target'])


# In[ ]:


sns.catplot(x='sex', col='target', kind='count', data=data);


# In[ ]:


print("% of women suffering from heart disease: " , data.loc[data.sex == 0].target.sum()/data.loc[data.sex == 0].target.count())
print("% of men suffering from heart disease:   " , data.loc[data.sex == 1].target.sum()/data.loc[data.sex == 1].target.count())


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(16,7))

data.loc[data['sex']==1, 'target'].value_counts().plot.pie(explode=[0,0.10],autopct='%1.1f%%',ax=ax[0],shadow=True)
data.loc[data['sex']==0, 'target'].value_counts().plot.pie(explode=[0,0.10],autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[0].set_title('Patients (male)')
ax[1].set_title('Patients (female)')

plt.show()


# Percentage of Females is more in this dataset who have a heart disease.
# 
# Let's check cp feature - chest pain type

# In[ ]:


data.groupby(by=['cp', 'target'])['target'].count()


# In[ ]:


pd.crosstab(data['cp'], data['target']).style.background_gradient(cmap='autumn_r')


# In[ ]:


sns.catplot(x='cp', col='target', kind='count', data=data);


# Patients who had chest pain type 2 is more in the category of people with disease. Also, chest pain type 0 is not that serious as there are many people (~110) who had chest pain type 0 without heart disease.
# 
# Let's see the fbs feature now, **fasting blood sugar > 120 mg/dl (1 = true; 0 = false)**

# In[ ]:


data.groupby(by=['fbs', 'target'])['target'].count()


# In[ ]:


sns.catplot(x='fbs', col='target', kind='count', data=data);


# Number of people if blood sugar is almost similar in both categories. fbs won't be a good indicator always for determining heart disease from this dataset.
# 
# Let's now see restecg feature 
# 
# **resting electrocardiographic results <br>
# -- Value 0: normal <br>
# -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) <br>
# -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria **

# In[ ]:


data.groupby(by=['restecg', 'target'])['target'].count()


# In[ ]:


sns.catplot(x='restecg', col='target', kind='count', data=data);


# The number of people having ST-T wave abnormality is more in the category with hart diesase. 
# 
# Let's check the exang feature 
# 
# **exercise induced angina (1 = yes; 0 = no) **

# In[ ]:


data.groupby(by=['exang', 'target'])['target'].count()


# In[ ]:


sns.catplot(x='exang', col='target', kind='count', data=data);


# People without **exercise induced angina** is more in the category with disease.
# 
# Let's see what info does slope feature have to give.
# 
# **slope: the slope of the peak exercise ST segment <br>
# -- Value 1: upsloping <br>
# -- Value 2: flat <br>
# -- Value 3: downsloping **

# In[ ]:


data.groupby(by=['slope', 'target'])['target'].count()


# In[ ]:


sns.catplot(x='slope', col='target', kind='count', data=data);


# Numer of people with **downsloping** is more (~105) in the category with diesease. Hmm, that is worth noting.
# 
# Let's check the feature **ca** now.
# 
# **ca: number of major vessels (0-3) colored by flourosopy **

# In[ ]:


data.groupby(by=['ca', 'target'])['target'].count()


# In[ ]:


sns.catplot(x='ca', col='target', kind='count', data=data);


# Most of the people with heart disease have **ca** as **0**. 
# 
# Let's see what story is feature **thal** telling.
# 
# **thal: 3 = normal; 6 = fixed defect; 7 = reversable defect **

# In[ ]:


data.groupby(by=['thal', 'target'])['target'].count()


# In[ ]:


sns.catplot(x='thal', col='target', kind='count', data=data);


# Most of the people with heart disease have **thal** as **2**.  Interesting.
# 

# 2. Continous Features
# 
# Let's now explore the continous features.
# * age
# * trestbps
# * thalach
# * oldpeak
# 
# Let's see how age is distributed in our dataset and how is the distribution among the people with and without disease.

# In[ ]:


sns.distplot(a=data['age'], color='black');


# In[ ]:


sns.boxplot(x=data['target'], y=data['age']);


# Median age of people with disease is less than that of without disease. Age itself can't be a descriptor to predict disease. 
# 
# Let's see how trestbps is distributed
# 
# **trestbps: resting blood pressure (in mm Hg on admission to the hospital) **

# In[ ]:


sns.distplot(data['trestbps']);


# We can see some outliers from the histogram, let's check in which category do those outliers belong.

# In[ ]:


sns.boxplot(x=data['target'], y=data['trestbps']);


# Outliers that are around 200 are in catgory without disease. We can also see the median **testbps** is almost equal is both cases.
# 
# Let's check **thalach** feature. 
# 
# **thalach: maximum heart rate achieved**
# 
# I personally think heart rate will be a good indicator for a heart disease. Let's check what our data have to tell.

# In[ ]:


sns.distplot(data['thalach'], color='black');


# In[ ]:


sns.boxplot(x=data['target'], y=data['thalach']);


# Yeah, my guess was correct. People with higher heart rate is tend to have hear disease. Also there are outliers who have heart rate around 190 but don't have heart disease. 
# But overall, the median heart rate is more for the people with heart disease.
# 
# Let's see **chol** feature.
# 
# ** chol: serum cholestoral in mg/dl **

# In[ ]:


sns.distplot(data['chol']);


# Woah, an outlier with ~550 mg/dl chol. I guess it will fall under the category of people with disease.

# In[ ]:


sns.boxplot(x='target', y='chol', data=data);


# Eventhough the median **chol** is more for **target** 0 we can see very high values of **chol** in **target** 1. Outliers seems a good indicator.

# **3. Relationship among continous variables**
# 
# Let's see how **thalach** and **col** are related.

# In[ ]:


sns.scatterplot(x='chol', y='thalach', data=data, hue='target');


# **chol** v/s **age**

# In[ ]:


sns.scatterplot(x='chol', y='age', data=data, hue='target');


# **chol** v/s **testbps**

# In[ ]:


sns.scatterplot(x='chol', y='trestbps', data=data, hue='target');


# **chol** v/s **oldpeak**

# In[ ]:


sns.scatterplot(x='chol', y='oldpeak', data=data, hue='target');


# Let's go for scatter matrix for the continous variables, rather than plotting each pair

# In[ ]:


sns.pairplot(data[['chol', 'age', 'trestbps', 'thalach', 'target']], hue='target');


# There is no notable relationship among the features that we can find from the scatter. Let's check the corrleation among these features.

# In[ ]:


corr = data[['chol', 'age', 'trestbps', 'thalach', 'target']].corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, linewidths=1.7, linecolor='white');


# As expected, there is no notable correlation among these features.

# This the end of basic EDA on Heart Disease Dataset. 

# In[ ]:




