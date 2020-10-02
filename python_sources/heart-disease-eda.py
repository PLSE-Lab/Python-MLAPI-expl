#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats






#  information about the data types of the individual variables within the dataframe
#  
#  The dataset contains the following features:
# 1. age(in years)
# 2. sex: (1 = male; 0 = female)
# 3. cp: chest pain type
# 4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# 5. chol: serum cholestoral in mg/dl
# 6. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7. restecg: resting electrocardiographic results
# 8. thalach: maximum heart rate achieved
# 9. exang: exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak: ST depression induced by exercise relative to rest
# 11. slope: the slope of the peak exercise ST segment
# 12. ca: number of major vessels (0-3) colored by flourosopy
# 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
#     14. target: 1 or 0 
#  

# In[ ]:


heart = pd.read_csv('../input/heart.csv')
sns.set()
# getting the shape
heart.info()
heart.head(10)
heart.describe()
heart.shape
heart.dtypes


# Checking for missing value

# In[ ]:



heart.isnull().sum()


#     Distribution of survey pool

# In[ ]:


sns.countplot(x="target", data=heart, palette="bwr")
plt.title('Distibution Patient and Non-Patient')
plt.show()


#     Change sex to categorical gender

# In[ ]:


heart.sex[heart.sex == 1] = 'male'
heart.sex[heart.sex == 0] = 'female'


# Age distribution by gender

# In[ ]:


plt.title('Distibution by age: blue=heart disease')
heart[heart.target==1].age.hist(bins=20);
heart[heart.target==0].age.hist(bins=20);


#     Heart disease distribution by gender

# In[ ]:


heart[heart.sex==1].age.hist(bins=10);
heart[heart.sex==0].age.hist(bins=10);


# Exploratory

# In[ ]:


heart.hist(column=["trestbps", "age","oldpeak","chol"],bins=40)  # Plot specific columns


# In[ ]:



fig, ax = plt.subplots(2,2, figsize=(32,32))
sns.distplot(heart.age, bins = 20, ax=ax[0,0]) 
sns.distplot(heart.oldpeak, bins = 20, ax=ax[0,1]) 
sns.distplot(heart.trestbps, bins = 20, ax=ax[1,0]) 
sns.distplot(heart.chol, bins = 20, ax=ax[1,1]) 



 


# In[ ]:


# Add labels
plt.title('Correlation between Oldpeak and Cholesteral')
plt.xlabel('OldPeak')
plt.ylabel('CHOL')
# Custom the histogram:
sns.jointplot(x=heart["oldpeak"], y=heart["chol"], kind='hex', marginal_kws=dict(bins=5, rug=True))


#     Box plot

# In[ ]:


#hset.boxplot()
# Use a color palette
#hset.boxplot( x=hset["target"], y=hset["sex"])
#sns.plt.show()
sns.boxplot(y='trestbps', x='sex', 
                 data=heart, 
                 palette="colorblind",
                 hue='target')


# In[ ]:


sns.violinplot(y='trestbps', x='sex', 
                 data=heart, 
                 palette="colorblind",
                 hue='target',
                 linewidth=1,
                 width=0.5)


# Fasting blood sugar FBS. higher than 120 mg/dl = 1, else 0.

# In[ ]:



sns.countplot(heart['target'],label="Count Distribution")


# Correlations

# In[ ]:


sns.catplot(x="sex", y="thalach", hue="target", inner="quart", kind="violin", split=True, data=heart)

sns.catplot(x="sex", y="age", hue="target", inner="quart", kind="violin", split=True, data=heart)

sns.catplot(x="sex", y="chol", hue="target", inner="quart", kind="violin", split=True, data=heart)

sns.catplot(x="sex", y="trestbps", hue="target", inner="quart", kind="violin", split=True, data=heart)

sns.catplot(x="sex", y="cp", hue="target", inner="quart", kind="violin", split=True, data=heart)


#     splitting dataset

# In[ ]:




from sklearn.model_selection import train_test_split
# Split our data

features = heart[heart.columns[0:13]]
target = heart['target']
#features_train, features_test, target_train, target_test = train_test_split(features,
                                                                           # target, test_size = 0.25, random_state = 10)
train, test = train_test_split(heart, test_size=0.25)


# In[ ]:


# getting the shape
train.head(10)
train.describe()
train.shape
train.dtypes


# In[ ]:


#corelation matrix
plt.figure(figsize=(11,7))
sns.heatmap(cbar=False,annot=True,data=heart.corr()*100,cmap='coolwarm')
plt.title('% Corelation Matrix')
plt.show()


# In[ ]:


#boxplot of 
plt.figure(figsize=(10,6))
sns.boxplot(data=heart,x='slope',y='thalach',palette='viridis')
plt.plot()


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(data=heart,x='cp',y='chol',palette='viridis')
plt.plot()


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(data=heart,x='target',y='chol',palette='viridis')
plt.plot()


# In[ ]:


# basic plot
plt.xlabel(ax.set_xlabel(), rotation=90)
heart.boxplot()


# Draw a graph with pandas and keep what's returned
ax = df.plot(kind='boxplot', x='target', y='chol')

# Set the x scale because otherwise it goes into weird negative numbers
ax.set_xlim((0, 1000))

# Set the x-axis label
ax.set_xlabel("Target")

# Set the y-axis label
ax.set_ylabel("Cholesterol")


# In[ ]:


p.xaxis.major_label_orientation = "vertical"
heart.plot(kind='bar',alpha=0.75)

