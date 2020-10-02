#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install pyspark')


# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql import DataFrame 
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoderEstimator, VectorAssembler, PCA
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession.builder.getOrCreate()
spark


# In[ ]:


sdf_train = spark.read.csv('/kaggle/input/titanic/train.csv', inferSchema = "true", header = "true")
sdf_train.toPandas().T


# In[ ]:


sdf_test = spark.read.csv("/kaggle/input/titanic/test.csv", inferSchema = "true", header = "true")
sdf_test.toPandas().T


# In[ ]:


numeric_cols = ['PassengerId','Survived', 'Pclass','Age', 'SibSp','Parch','Ticket','Fare'] 
numeric_features = ['PassengerId','Pclass','Age', 'SibSp','Parch','Fare'] 
numeric_cols_test = ['PassengerId', 'Pclass','Age', 'SibSp','Parch','Ticket','Fare', 'Sex'] 
string_features = [ 'Embarked', 'Sex']


# In[ ]:


def _make_numeric(sdf: DataFrame, str_col_name: str) -> DataFrame:
    sdf = sdf.withColumn(str_col_name, sdf[str_col_name].cast('double'))
    return sdf
sdf_train = _make_numeric(sdf_train,'Ticket')
sdf_test = _make_numeric(sdf_test,'Ticket')
# sdf_train.toPandas().T


# In[ ]:


def _impute_median(sdf: DataFrame,lst_columns: list) -> DataFrame:
    imputer = Imputer(inputCols = lst_columns, outputCols = lst_columns)
    sdf = imputer.fit(sdf).transform(sdf)
    return sdf
lst_cols_to_impute = ['Fare','Age']
sdf_train_imputed = _impute_median(sdf_train,lst_cols_to_impute)
sdf_test_imputed = _impute_median(sdf_test,lst_cols_to_impute)


# In[ ]:


# sdf_train_imputed.toPandas().T


# In[ ]:


# def _str_indexer():
_stages = []
string_indexer =  [StringIndexer(inputCol = column ,                                  outputCol = column + '_indx', handleInvalid = 'skip') for column in string_features]
_stages += string_indexer

one_hot_encoder = [OneHotEncoderEstimator(inputCols = [column + '_indx' for column in string_features ],                                           outputCols =  [column + '_encoded' for column in string_features ])]
_stages += one_hot_encoder

assemblerInput =  [f  for f in numeric_features]  
assemblerInput += [f + "_encoded" for f in string_features]
vector_assembler = VectorAssembler(inputCols = assemblerInput,                                    outputCol = 'vect_features')
_stages += [vector_assembler]

pca = PCA(inputCol = 'vect_features', outputCol = 'pca_features', k = 5)
_stages += [pca]

rf = RandomForestClassifier(labelCol = 'Survived', featuresCol = 'pca_features', numTrees = 100, maxDepth = 4, maxBins = 1000)
_stages += [rf]


# In[ ]:


pipeline = Pipeline(stages = _stages)


# In[ ]:


model = pipeline.fit(sdf_train_imputed)


# In[ ]:


sdf_predict = model.transform(sdf_test_imputed)


# In[ ]:


sdf_predict.toPandas().T


# In[ ]:


sdf_submission = sdf_predict.select('PassengerId','prediction')                            .withColumn('Survived',sdf_predict['prediction'].cast('integer'))                            .select('PassengerId','Survived')
sdf_submission.toPandas().T


# In[ ]:


sdf_submission.coalesce(1).write.csv("submission",mode="overwrite",header=True)


# In[ ]:


print(os.listdir('submission'))


# <a href='submission/part-00000-f0fdb22b-1f16-4bb9-8345-f434201a00ae-c000.csv'>download result</a>

# TODO:  
# https://spark.apache.org/docs/latest/ml-features.html#vectorindexer  
# https://spark.apache.org/docs/latest/ml-features.html#bucketizer  

# In[ ]:


Todo:

