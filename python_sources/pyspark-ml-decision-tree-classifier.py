#!/usr/bin/env python
# coding: utf-8

# [Source](https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier)

# In[ ]:


get_ipython().system(' pip install pyspark')


# In[ ]:


get_ipython().system(' curl  https://raw.githubusercontent.com/apache/spark/master/data/mllib/sample_libsvm_data.txt > sample_libsvm_data.txt')


# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


import os
print(os.listdir("."))

# Any results you write to the current directory are saved as output.


# In[ ]:


spark = SparkSession.builder.getOrCreate()
spark


# In[ ]:


sdf = spark.read.format("libsvm").load("sample_libsvm_data.txt")
pdf = sdf.toPandas()
pdf.T


# In[ ]:


str_indx = StringIndexer(inputCol = "label", outputCol = "str_indx_label").fit(sdf)


# In[ ]:


vec_indx = VectorIndexer(inputCol = "features", outputCol = "vec_indx_features", maxCategories = 4).fit(sdf)


# In[ ]:


(sdf_training, sdf_test) = sdf.randomSplit([0.70,0.30])


# In[ ]:


dt = DecisionTreeClassifier(labelCol = "str_indx_label", featuresCol = "vec_indx_features")


# In[ ]:


pipeline = Pipeline(stages=[str_indx, vec_indx, dt])


# In[ ]:


model = pipeline.fit(sdf_training)


# In[ ]:


sdf_pred = model.transform(sdf_test)


# In[ ]:


print(sdf_pred.printSchema())
pdf = sdf_pred.toPandas()
pdf.T
# Failed to execute user defined function($anonfun$11: (struct<type:tinyint,size:int,indices:array<int>,values:array<double>>) => struct<type:tinyint,size:int,indices:array<int>,values:array<double>>)


# In[ ]:


evaluator = MulticlassClassificationEvaluator(labelCol = "str_indx_label", predictionCol = "prediction", metricName = "accuracy")


# In[ ]:


accuracy = evaluator.evaluate(sdf_pred)


# In[ ]:



print(accuracy)
print((1.0 - accuracy))


# In[ ]:


print(model.stages[2])

