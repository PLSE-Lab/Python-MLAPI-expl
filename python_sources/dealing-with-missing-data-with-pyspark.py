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


pip install pyspark


# In[ ]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *


# In[ ]:


spark = SparkSession.Builder().getOrCreate()


# In[ ]:


df= spark.read.csv('/kaggle/input/pima-indians-diabetes data.xls',header = True,inferSchema=True)


# In[ ]:


df.count()


# In[ ]:


df.limit(5).show()


# Diastolic_BP: blood pressure
# BMI:body mass index 
# Diabetes_Pedigree: degree of diabetes

# In[ ]:


from pyspark.sql.functions import when, count, col
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()


# In[ ]:


pip install missingno


# In[ ]:


import missingno as msno
msno.bar(df.toPandas())


# In[ ]:


msno.matrix(df.toPandas())


# In[ ]:


msno.heatmap(df.toPandas())


# In[ ]:


#describe correlation by groupoing variables
msno.dendrogram(df.toPandas())

here we can drop the Glucose and BMI columns because there is no correlation with other columns and just few values are missing=> MCAR (Missing Completely At Random)
# In[ ]:


noMissing=df.dropna(how='any',subset=['Glucose','BMI','Diastolic_BP'])


# In[ ]:


noMissing.show()


# In[ ]:


from pyspark.sql.functions import when, count, col
noMissing.select([count(when(isnull(c), c)).alias(c) for c in noMissing.columns]).show()


# In[ ]:


noMissing.count()


# # Imputation techniques

# imputing using the mean, median, mode and constant

# In[ ]:


from sklearn.impute import SimpleImputer


# In[ ]:


noMissing_mean = noMissing.toPandas().copy(deep=True)
mean_imputer = SimpleImputer(strategy='mean')
noMissing_mean.iloc[:, :] = mean_imputer.fit_transform(noMissing_mean)


# In[ ]:


noMissing_median = noMissing.toPandas().copy(deep=True)
mean_imputer = SimpleImputer(strategy='median')
noMissing_median.iloc[:, :] = mean_imputer.fit_transform(noMissing_median)


# In[ ]:


noMissing_fq = noMissing.toPandas().copy(deep=True)
mean_imputer = SimpleImputer(strategy='most_frequent')
noMissing_fq.iloc[:, :] = mean_imputer.fit_transform(noMissing_fq)


# In[ ]:


noMissing_cst = noMissing.toPandas().copy(deep=True)
mean_imputer = SimpleImputer(strategy='constant',fill_value=0)
noMissing_cst.iloc[:, :] = mean_imputer.fit_transform(noMissing_cst)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
data=noMissing.toPandas()
nullity = data.Serum_Insulin.isnull()+data.Glucose.isnull()
imputations = {'Mean Imputation':noMissing_mean, 'Median Imputation': noMissing_median, 
               'Most Frequent Imputation': noMissing_fq, 'Constant Imputation': noMissing_cst}
# Loop over flattened axes and imputations
for ax, df_key in zip(axes.flatten(), imputations):
    # Select and also set the title for a DataFrame
    imputations[df_key].plot(x='Serum_Insulin', y='Glucose', kind='scatter', 
                          alpha=0.5, c=nullity, cmap='rainbow', ax=ax, 
                          colorbar=False, title=df_key)
plt.show()


# imputing using KNN and MICE

# In[ ]:



from fancyimpute import KNN
knn_imputed = noMissing.toPandas().copy(deep=True)
knn_imputer = KNN()
knn_imputed.iloc[:, :] = knn_imputer.fit_transform(knn_imputed)


# In[ ]:


from fancyimpute import IterativeImputer
mice_imputed =noMissing.toPandas().copy(deep=True)
mice_imputer = IterativeImputer()
mice_imputed.iloc[:, :] = mice_imputer.fit_transform(mice_imputed)


# Imputing Categorical column

# In[ ]:


data=noMissing.toPandas().dropna(how='any')


# In[ ]:


data.count()


# In[ ]:


import statsmodels.api as sm
X = sm.add_constant(data.iloc[:, :-1])
y = data['Class']
lm = sm.OLS(y, X).fit()
print(lm.summary())


# In[ ]:


import statsmodels.api as sm
X = sm.add_constant(noMissing_mean.iloc[:, :-1])
y = noMissing_mean['Class']
lm_mean = sm.OLS(y, X).fit()
print(lm_mean.summary())


# In[ ]:


X = sm.add_constant(knn_imputed.iloc[:, :-1])
y = knn_imputed['Class']
lm_knn = sm.OLS(y, X).fit()
print(lm_knn.summary())


# In[ ]:


X = sm.add_constant(mice_imputed.iloc[:, :-1])
y =mice_imputed['Class']
lm_mice = sm.OLS(y, X).fit()
print(lm_mice.summary())


# In[ ]:


data['Skin_Fold'].plot(kind='kde', c='red', linewidth=3)
noMissing_mean['Skin_Fold'].plot(kind='kde')
knn_imputed['Skin_Fold'].plot(kind='kde')
mice_imputed['Skin_Fold'].plot(kind='kde')

# Create labels for the four DataFrames
labels = ['Baseline (Complete Case)', 'Mean Imputation', 'KNN Imputation', 'MICE Imputation']
plt.legend(labels)

