#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install pyspark')


# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
 
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


spark = SparkSession.builder.getOrCreate()
spark


# # 1. Load data

# In[ ]:


sdf_train = spark.read.csv("../input/train.csv",inferSchema=True,header=True)
print(sdf_train.printSchema())
pdf = sdf_train.limit(5).toPandas()
pdf.T


# In[ ]:


sdf_test = spark.read.csv("../input/test.csv",inferSchema=True,header=True)
# sdf_train.printSchema()
pdf = sdf_test.limit(5).toPandas()
pdf.T


# # 2. Data cleanup

# In[ ]:


sdf_typecast = sdf_train.withColumn('Ticket', sdf_train['Ticket'].cast("double"))
sdf_typecast = sdf_typecast.fillna(0)
# pdf = sdf_typecast.limit(5).toPandas()
# pdf.T


# # 3. Feature engineering

# In[ ]:


numeric_cols = ['PassengerId','Survived', 'Pclass','Age', 'SibSp','Parch','Ticket','Fare'] 
numeric_features = ['Pclass','Age', 'SibSp','Parch','Fare'] 
# string_features = [ 'Cabin', 'Embarked', 'Sex','Ticket']
# 'Name',
sdf_train_subset = sdf_typecast #.select(numeric_cols)    


# In[ ]:


_stages = []


# In[ ]:


from pyspark.ml.feature import VectorAssembler
assemblerInput = numeric_features # [f + '_vect' for f in string_features] + 
print(assemblerInput)
vectAssembler = VectorAssembler(inputCols  = assemblerInput, outputCol = "vect_features") #.fit(sdf_train_subset)  
_stages += [vectAssembler]
# handleInvalid = "keep" or "skip"


# # 4. ML model

# In[ ]:


from pyspark.ml.classification import DecisionTreeClassifier

# dt = DecisionTreeClassifier(labelCol = 'Survived', featuresCol = 'vect_features') # ,maxDepth=1
# _stages += [dt]


# In[ ]:


from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol = 'Survived', featuresCol = 'vect_features', numTrees = 100, maxDepth = 4)
_stages += [rf]


# In[ ]:


_stages


# In[ ]:


from pyspark.ml import Pipeline

pipeline = Pipeline(stages = _stages)


# In[ ]:


model = pipeline.fit(sdf_train_subset)


# In[ ]:


numeric_cols_test = ['PassengerId', 'Pclass','Age', 'SibSp','Parch','Ticket','Fare'] 

sdf_test_subset = sdf_test.withColumn('Ticket', sdf_test['Ticket'].cast("double")).                         fillna(0).                         select(numeric_cols_test)


# In[ ]:


sdf_predict = model.transform(sdf_test_subset)


# In[ ]:


pdf = sdf_predict.limit(10).toPandas()
pdf.T


# In[ ]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="prediction")
print("Test Area Under ROC: " + str(evaluator.evaluate(sdf_predict, {evaluator.metricName: "areaUnderROC"})))


# In[ ]:


sdf_submission = sdf_predict.select('PassengerId','prediction').withColumn('Survived',sdf_predict['prediction'].cast('integer')).select('PassengerId','Survived')
sdf_submission.show()


# In[ ]:


sdf_submission.coalesce(1).write.csv("submission",mode="overwrite",header=True)


# In[ ]:


print(os.listdir('submission'))


# <a href="submission/part-00000-b53a2b2f-1d11-459b-923b-a7231ed9a7d6-c000.csv"> Download File </a>

# Further reading:   
# https://spark.apache.org/docs/latest/mllib-decision-tree.html  
# https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-trees  
# https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier  

# In[ ]:




