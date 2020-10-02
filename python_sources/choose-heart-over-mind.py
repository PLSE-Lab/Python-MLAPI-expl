#!/usr/bin/env python
# coding: utf-8

# This is a work in process.I will be updating the kernel in the coming days.If you like my work please do vote for me.

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


# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC 
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
from lightgbm import LGBMClassifier
import warnings
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')


# # 1.Importing and Data Exploration data

# In[ ]:


data=pd.read_csv('../input/heart.csv')
data.head()


# Data contains following information
# 
# * age: The person's age in years
# * sex: The person's sex (1 = male, 0 = female)
# * cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# * trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# * chol: The person's cholesterol measurement in mg/dl
# * fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# * restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# * thalach: The person's maximum heart rate achieved
# * exang: Exercise induced angina (1 = yes; 0 = no)
# * oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# * slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# * ca: The number of major vessels (0-3)
# * thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# * target: Heart disease (0 = no, 1 = yes)

# ### Shape of Data

# In[ ]:


data.shape


# So our data has 303 rows and 14 features

# ### Missing Values

# In[ ]:


data.isnull().sum()


# We are very lucky here there are no Null Values in the dataset.But in this data the missing values are present in the form of value ZERO. Our next task would be to find out numbers of ZEROS in each column.

# ### Finding out Zero's

# In[ ]:


cols = data.columns
cols


# In[ ]:


print("# Rows in the dataset {0}".format(len(data)))
print("---------------------------------------------------")
for col in cols:
    print("# Rows in {1} with ZERO value: {0}".format(len(data.loc[data[col] ==0]),col))


# The columns which have categorical values can have ZERO values.But columns like cp,trestbps,chol,fbs,exang,oldpean and Slope should not have value ZERO.The presence of ZERO in this columns indicate the presence of null values.

# In[ ]:


data.dtypes


# As all the columns are either integer or float there is no need to convert categorical values numeric values.

# # 2.Data Vizualization

# ### Corelation Matrix

# In[ ]:


corrmat = data.corr()
fig = plt.figure(figsize = (16,16))
sns.heatmap(corrmat,vmax = 1,square = True,annot = True,vmin = -1)
plt.show()


# We can see that there is not much correlation between the features in the dataset.If corelation was high we can face issue of multicollinearity.In that case we would need to use feature engineering to avoid multi colinearity.

# ### Histogram

# In[ ]:


data.hist(figsize = (12,12))
plt.show()


# Histogram clearly shows us that ca,cp,exang,fbs,restecg,sex,slope,target and thal are categorical data.Other columns are numerical features.

# ### Effect of Sex

# In[ ]:


sns.barplot(x="sex",y ='age',hue ='target',data=data)
pass


# From the above plot we can say that men have heart disease at lower age compared to women.

# ### Pair Plot

# In[ ]:


sns.pairplot(data,hue='target')
pass


# The sepeartion of the target variable is less in the pairplot.So the level of accuracy expected could be low.We can use dimentionality reduction to represnt the results on a 2D graph.

# # TSNE for Dimentionality Reduction

# In[ ]:


X = data.drop('target',axis =1)
from sklearn.manifold import TSNE
import time 
time_start = time.time()
df_tsne = TSNE(random_state =10).fit_transform(X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[ ]:


df_tsne


# In[ ]:




