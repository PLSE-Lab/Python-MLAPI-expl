#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install pyspark')


# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_profiling
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, OneHotEncoderEstimator, VectorIndexer, PCA
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier 
spark = SparkSession.builder.getOrCreate()
spark


# In[ ]:


sdf_train = spark.read.csv('/kaggle/input/titanic/train.csv', inferSchema = True, header = True)
sdf_train.toPandas().T


# In[ ]:


# pandas_profiling(sdf_train.toPandas())
sdf_train.toPandas().profile_report()


# In[ ]:


sdf_test = spark.read.csv('/kaggle/input/titanic/test.csv', inferSchema = True, header = True)
sdf_test.toPandas().T


# In[ ]:


numeric_cols = ['PassengerId','Survived', 'Pclass','Age', 'SibSp','Parch','Ticket','Fare'] 
numeric_features = ['PassengerId','Pclass','Age', 'SibSp','Parch','Fare'] 
numeric_cols_test = ['PassengerId', 'Pclass','Age', 'SibSp','Parch','Ticket','Fare', 'Sex'] 
string_features = [ 'Embarked', 'Sex'] #, 'Cabin'


# In[ ]:


def _clean_dataset(sdf: DataFrame, col_to_convert: list, col_to_impute: list) -> DataFrame:
    for col in col_to_convert:
        sdf = sdf.withColumn(col,sdf[col].cast('double'))
    col_to_impute += col_to_convert
    imputer = Imputer(inputCols = col_to_impute, outputCols = col_to_impute)
    sdf = imputer.fit(sdf).transform(sdf)
    return sdf
sdf_train_cleaned = _clean_dataset(sdf_train,['Ticket','SibSp','Parch'],['Fare','Age'])
sdf_test_cleaned = _clean_dataset(sdf_test,['Ticket','SibSp','Parch'],['Fare','Age'])

# sdf_cleaned.toPandas().T


# In[ ]:


sdf_test_cleaned.toPandas().profile_report()


# In[ ]:


from pyspark.ml.feature import Bucketizer

splits = [-float("inf"), -0.5, 0.0, 0.5, float("inf")]

data = [(-999.9,), (-0.5,), (-0.3,), (0.0,), (0.2,), (999.9,)]
dataFrame = spark.createDataFrame(data, ["features"])

bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bucketedFeatures")

# Transform original data into its bucket index.
bucketedData = bucketizer.transform(dataFrame)

print("Bucketizer output with %d buckets" % (len(bucketizer.getSplits())-1))
bucketedData.show()


# In[ ]:


_stages = []
string_indexer =  [StringIndexer(inputCol = column ,                                  outputCol = column + '_S_indx', handleInvalid = 'skip') for column in string_features]
_stages += string_indexer

one_hot_encoder = [OneHotEncoderEstimator(inputCols = [column + '_S_indx' for column in string_features ],                                           outputCols =  [column + '_encoded' for column in string_features ])]
_stages += one_hot_encoder

vect_indexer = [VectorIndexer(inputCol = column + '_encoded',
                             outputCol = column + '_V_indx', maxCategories=10) for column in string_features]
_stages += vect_indexer

assemblerInput =  [f  for f in numeric_features]  
assemblerInput += [f + "_V_indx" for f in string_features]
vector_assembler = VectorAssembler(inputCols = assemblerInput,                                    outputCol = 'vect_features')
_stages += [vector_assembler]

pca = PCA(inputCol = 'vect_features', outputCol = 'pca_features', k = 5)
_stages += [pca]

rf = RandomForestClassifier(labelCol = 'Survived', featuresCol = 'pca_features', numTrees = 100, maxDepth = 4, maxBins = 1000)
_stages += [rf]

pipeline = Pipeline(stages = _stages)
model = pipeline.fit(sdf_train_cleaned)
sdf_predict = model.transform(sdf_test_cleaned)
sdf_predict.toPandas().T


# In[ ]:


# pipeline = Pipeline(stages = _stages)
# model = pipeline.fit(sdf_train_cleaned)
# sdf_predict = model.transform(sdf_test_cleaned)
# sdf_predict.toPandas().T


# In[ ]:


sdf_submission = sdf_predict.select('PassengerId','prediction')                            .withColumn('Survived',sdf_predict['prediction'].cast('integer'))                            .select('PassengerId','Survived')
sdf_submission.toPandas().T


# In[ ]:


sdf_submission.coalesce(1).write.csv("submission",mode="overwrite",header=True)


# In[ ]:


print(os.listdir('submission'))


# <a href='submission/part-00000-96909928-691c-4e26-8c6d-518f71fa2aa2-c000.csv'>Download</a>
