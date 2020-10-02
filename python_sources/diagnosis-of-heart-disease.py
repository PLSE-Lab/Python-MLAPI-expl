#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# > The American Heart Association Statistics 2016 Report indicates that heart disease is the leading cause of death for both men and women, responsible for 1 in every 4 deaths. Even modest improvements in prognostic models of heart events and complications could save hundreds of lives and help to significantly reduce the cost of health care services, medications, and lost productivity.

#   ANALYSIS OF DATA
# 1. age: age in years
# 2. sex: (1 = male; 0 = female)
# 3. cp: chest pain type
# >     A. Value 1: typical angina
# >     B. Value 2: atypical angina
# >     C. Value 3: non-anginal pain
# >     D. Value 4: asymptomatic
# 4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# 5. chol: serum cholestoral in mg/dl
# 6. fbs: (fasting blood sugar > 120 mg/dl)
# >     1 = true
# >     0 = false
# 7. restecg: (resting electrocardiographic results)
# >       Value 0: normal
# >       Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# >       Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# 8. thalach: maximum heart rate achieved
# 9. exang: exercise induced angina
# >       1 = yes
# >       0 = no
# 10. oldpeak: ST depression induced by exercise relative to rest
# 11. slope: the slope of the peak exercise ST segment
# >      Value 1: upsloping
# >      Value 2: flat
# >      Value 3: downsloping
# 12. ca: number of major vessels (0-3) colored by flourosopy
# 13. thal: thalium heart scan
# >     3 = normal (no cold spots)
# >     6 = fixed defect (cold spots during rest and exercise)
# >     7 = reversible defect (when cold spots only appear during exercise)
# 14. pred_attribute: (the predicted attribute) diagnosis of heart disease (angiographic disease status)
# >      Value 0: < 50% diameter narrowing
# >      Value 1: > 50% diameter narrowing (in any major vessel: attributes 59 through 68 are vessels)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../input/Heart_Disease_Data.csv",na_values="?")


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data["pred_attribute"].replace(inplace=True, value=[1, 1, 1, 1], to_replace=[1, 2, 3, 4])


# > **Review all features**

# In[ ]:


columns=data.columns[:14]
plt.subplots(figsize=(18,15))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    data[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()


# **Heart disease partipants**

# In[ ]:


dataset_copy=data[data['pred_attribute']==1]
columns=data.columns[:13]
plt.subplots(figsize=(20,15))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    dataset_copy[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()


# Pair plot
# 1. The diagonal shows the distribution of the the dataset with the kernel density plots.
# 2. The scatter-plots shows the relation between each and every attribute or features taken pairwise.
# 3. Looking at the scatter-plots, we can say that no two attributes are able to clearly seperate the two outcome-class instances.

# In[ ]:


features_continuous=["age", "trestbps", "chol", "thalach", "oldpeak", "pred_attribute"]
sns.pairplot(data=data[features_continuous],hue='pred_attribute',diag_kind='kde')
#plt.gcf().set_size_inches(20,15)
plt.show()


# In[ ]:


# Visualization of age and thalach(max heart rate) with different style of seaborn code
# joint kernel density
# Show the joint distribution using kernel density estimation 
g = sns.jointplot(data.age,data.thalach,kind="kde", size=7)


# In[ ]:


g = sns.jointplot(data.age, data.thalach, data=data,size=5, ratio=3, color="r")


# In[ ]:


# Visualization of the predicted attribute and exercise induced angina with different style of seaborn code
# lmplot 
# Show the results of a linear regression within each dataset
g=sns.lmplot(x='pred_attribute', y='exang', data=data)
plt.show()


# In[ ]:


sns.heatmap(data[data.columns[:14]].corr(),annot=True,cmap='RdYlGn')
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.title('Correlation of Features', y=1.05, size=25)
plt.show()


# In[ ]:


# Plot the orbital period with horizontal boxes
sns.boxplot(x=data.sex, y=data.age, hue=data.pred_attribute, data=data, palette="PRGn")
plt.show()


# In[ ]:


ax = sns.violinplot(x="sex", y="chol", data=data)
plt.show()


# In[ ]:


f,ax1 = plt.subplots(figsize =(20,10))
plt.title('The change of age-cholesterol and age-thalach(max heart rate achieved)', fontsize=20, fontweight='bold')
sns.pointplot(x='age',y='chol',data=data,color='red',alpha=0.8)
sns.pointplot(x='age',y='thalach',data=data,color='green',alpha=0.8)
plt.xlabel('Age',fontsize = 20,color='blue')
plt.ylabel('Values',fontsize = 20,color='blue')
plt.grid()


# In[ ]:


sns.swarmplot(x="sex", y="age",hue="pred_attribute", data=data)
plt.show()


# In[ ]:


sns.countplot(x=data.pred_attribute,data=data)
plt.show()


# In[ ]:


sns.countplot(data.sex)
plt.title("gender",color = 'blue',fontsize=15)

