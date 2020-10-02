#!/usr/bin/env python
# coding: utf-8

# [Source](https://spark.apache.org/docs/latest/ml-classification-regression.html#binomial-logistic-regression)

# In[ ]:


get_ipython().system(' pip install pyspark')


# In[ ]:


get_ipython().system(' curl  https://raw.githubusercontent.com/apache/spark/master/data/mllib/sample_libsvm_data.txt > sample_libsvm_data.txt')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("."))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system(' wc -l sample_libsvm_data.txt')


# In[ ]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()


# In[ ]:


from pyspark.ml.classification import LogisticRegression


# In[ ]:


training = spark.read.format("libsvm").load("sample_libsvm_data.txt")
pdf = training.toPandas()
pdf.T


# In[ ]:


lr = LogisticRegression(maxIter = 10, regParam = 0.3, elasticNetParam= 0.8 )


# In[ ]:


lr_model = lr.fit(training)


# In[ ]:


lr_model.coefficients


# In[ ]:


lr_model.intercept


# In[ ]:


mlr = LogisticRegression(maxIter = 10, regParam = 0.3, elasticNetParam=0.8, family="multinomial")


# In[ ]:


mlr_model = mlr.fit(training)


# In[ ]:


mlr_model.coefficientMatrix


# In[ ]:


mlr_model.interceptVector


# In[ ]:


train_summary = mlr_model.summary


# In[ ]:


obj_hist = train_summary.objectiveHistory
for obj in  obj_hist:
    print(obj)


# In[ ]:


train_summary.roc.show()


# In[ ]:


train_summary.areaUnderROC


# In[ ]:


f_measure = train_summary.fMeasureByThreshold
f_measure


# In[ ]:




