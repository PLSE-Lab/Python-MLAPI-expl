#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


Covid19_df = pd.read_excel('../input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
Covid19_df


# In[ ]:


# preview the data
Covid19_df.head()


# In[ ]:


Covid19_df.tail()


# In[ ]:


Covid19_df.describe()


# In[ ]:


Covid19_df.describe(include=['O'])


# In[ ]:


Covid19_df.info()


# In[ ]:


Covid19_df[['GENDER', 'ICU']].groupby(['GENDER'], as_index=False).mean().sort_values(by='ICU', ascending=False)


# In[ ]:


Covid19_df[['PATIENT_VISIT_IDENTIFIER', 'ICU']].groupby(['PATIENT_VISIT_IDENTIFIER'], as_index=False).mean().sort_values(by='ICU', ascending=False)


# In[ ]:


Covid19_df[['AGE_PERCENTIL', 'ICU']].groupby(['AGE_PERCENTIL'], as_index=False).mean().sort_values(by='ICU', ascending=False)


# In[ ]:


Covid19_df[['DISEASE GROUPING 1', 'ICU']].groupby(['DISEASE GROUPING 1'], as_index=False).mean().sort_values(by='ICU', ascending=False)


# In[ ]:


Covid19_df[['DISEASE GROUPING 2', 'ICU']].groupby(['DISEASE GROUPING 2'], as_index=False).mean().sort_values(by='ICU', ascending=False)


# In[ ]:


Covid19_df[['DISEASE GROUPING 3', 'ICU']].groupby(['DISEASE GROUPING 3'], as_index=False).mean().sort_values(by='ICU', ascending=False)


# In[ ]:


Covid19_df[['DISEASE GROUPING 4', 'ICU']].groupby(['DISEASE GROUPING 4'], as_index=False).mean().sort_values(by='ICU', ascending=False)


# In[ ]:


g = sns.FacetGrid(Covid19_df, col='ICU')
g.map(plt.hist, 'WINDOW', bins=20)


# In[ ]:


missing_values = Covid19_df.isnull().sum()
print(missing_values)


# In[ ]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
Covid19_df['WINDOW']= label_encoder.fit_transform(Covid19_df['WINDOW']) 
  
Covid19_df['WINDOW'].unique() 
Covid19_df['AGE_PERCENTIL']= label_encoder.fit_transform(Covid19_df['AGE_PERCENTIL']) 
Covid19_df['AGE_PERCENTIL'].unique() 
Covid19_df


# In[ ]:


from sklearn.impute import SimpleImputer
imp_numeric = SimpleImputer(missing_values=np.nan, strategy='mean')
for col in Covid19_df:
  if Covid19_df[col].isnull().sum() > 0:
    Covid19_df[col] = imp_numeric.fit_transform(Covid19_df[[col]])
  else:
    pass
Covid19_df
missing_values = Covid19_df.isnull().sum()
missing_values


# In[ ]:



#define_x
x = Covid19_df.iloc[:,:230]
x


# In[ ]:


#define_y
y = Covid19_df.iloc[:, -1]
y


# 

# In[ ]:


#feature selection
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2 , f_classif 
X = SelectPercentile(score_func = chi2, percentile=50).fit_transform(x, y)
X

