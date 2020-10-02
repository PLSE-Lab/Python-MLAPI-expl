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


from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.ml.feature import Imputer, StringIndexer, VectorIndexer, VectorAssembler, OneHotEncoderEstimator, PCA, Bucketizer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
spark = SparkSession.builder.getOrCreate()
spark


# In[ ]:


sdf_train = spark.read.csv('/kaggle/input/titanic/train.csv', inferSchema = True, header = True)
sdf_test = spark.read.csv('/kaggle/input/titanic/test.csv', inferSchema = True, header = True)


# In[ ]:


# ref: https://www.kaggle.com/garbamoussa/titatanic-overfitting-underfitting
def _evaluate_initials(sdf: DataFrame) -> DataFrame:
    dizip_initials = {k:v for k,v in (zip(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                                         ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr']))}
    _sdf = sdf.withColumn('Initial',  F.regexp_extract( sdf['Name'], ('([A-Za-z]+)\.'),1 ) )
    _sdf = _sdf.replace(dizip_initials,1,'Initial')
    return _sdf

def _handle_missing_age(sdf: DataFrame) -> DataFrame:
    _sdf = sdf
    _sdf = _sdf.withColumn('Age', F.when((F.isnull(_sdf['Age'])) & (_sdf['Initial'] == 'Mr') , 33 )                            .otherwise(F.when((F.isnull(_sdf['Age'])) & (_sdf['Initial'] == 'Mrs') , 36)                            .otherwise(F.when((F.isnull(_sdf['Age'])) & (_sdf['Initial'] == 'Master') , 5)                            .otherwise(F.when((F.isnull(_sdf['Age'])) & (_sdf['Initial'] == 'Miss') , 22)                            .otherwise(F.when((F.isnull(_sdf['Age'])) & (_sdf['Initial'] == 'Other') , 46)                            .otherwise(_sdf['Age']) )))))
    return _sdf
    
    
# _sdf.select('Age').distinct().toPandas().T                        
# _sdf.select('Age').toPandas().profile_report()


# In[ ]:


# ref : https://www.kaggle.com/kabure/titanic-baseline-eda-pipes-easy-to-starters
def _create_family_size(sdf: DataFrame) -> DataFrame :
#     family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 
#               5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large',
#               11: 'Large'}

    _sdf = sdf.withColumn('FamilySize', sdf['Parch'] + sdf['SibSp'] + 1 )
#     # bucketting
#     family_map = {1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 11: 3}
#     _sdf = _sdf.replace(family_map,1,'FamilySize')
    
    return _sdf

# _create_family_size(sdf_train).toPandas().T


# In[ ]:


def _clean_dataset(sdf: DataFrame, col_to_convert: list, col_to_impute: list) -> DataFrame:
    for col in col_to_convert:
        sdf = sdf.withColumn(col,sdf[col].cast('double'))
    col_to_impute += col_to_convert

    imputer = Imputer(inputCols = col_to_impute, outputCols = col_to_impute)
    sdf = imputer.fit(sdf).transform(sdf)
    return sdf


# In[ ]:


sdf_train_cleaned = _clean_dataset ( 
    _handle_missing_age(
    _evaluate_initials(
    _create_family_size(sdf_train)
    )) 
    ,['Ticket','SibSp','Parch'],['Fare'] 
)

sdf_test_cleaned = _clean_dataset ( 
    _handle_missing_age(
    _evaluate_initials(
    _create_family_size(sdf_test)
    )) 
    ,['Ticket','SibSp','Parch'],['Fare'] 
)


# In[ ]:





# In[ ]:


numeric_cols = ['PassengerId','Survived', 'Pclass','Age', 'SibSp','Parch','Ticket','Fare'] 
numeric_features = ['PassengerId','Pclass','Age', 'SibSp','Parch','Fare'] 
# numeric_cols_test = ['PassengerId', 'Pclass','Age', 'SibSp','Parch','Ticket','Fare', 'Sex'] 
string_features = [ 'Embarked', 'Sex'] #, 'Cabin'
# string_features += ['Initial']


# In[ ]:





# In[ ]:


# sdf_test_cleaned.toPandas().profile_report()
numeric_features


# In[ ]:


_stages = []
string_indexer =  [StringIndexer(inputCol = column ,                                  outputCol = column + '_S_indx', handleInvalid = "skip") for column in string_features]
_stages += string_indexer

one_hot_encoder = [OneHotEncoderEstimator(inputCols = [column + '_S_indx' for column in string_features ],                                           outputCols =  [column + '_encoded' for column in string_features ])]
_stages += one_hot_encoder

vect_indexer = [VectorIndexer(inputCol = column + '_encoded',
                             outputCol = column + '_V_indx', maxCategories=10) for column in string_features]
_stages += vect_indexer

familt_size_splits = [1, 2, 5, 7, 100] #[-float("inf"), 1, 2, 5, float("inf")]
bucketizer = Bucketizer(splits = familt_size_splits, inputCol = 'FamilySize',outputCol = 'bucketized_FamilySize')
_stages += [bucketizer]

numeric_features += ['bucketized_FamilySize']

assemblerInput =  [f  for f in numeric_features]  
assemblerInput += [f + "_V_indx" for f in string_features]
vector_assembler = VectorAssembler(inputCols = assemblerInput,                                    outputCol = 'vect_features')
_stages += [vector_assembler]

# pca = PCA(inputCol = 'vect_features', outputCol = 'pca_features', k = 5)
# _stages += [pca]

rf = RandomForestClassifier(labelCol = 'Survived', featuresCol = 'vect_features', numTrees = 100, maxDepth = 4, maxBins = 1000)
_stages += [rf]

pipeline = Pipeline(stages = _stages)
model = pipeline.fit(sdf_train_cleaned)
sdf_predict = model.transform(sdf_test_cleaned)
sdf_predict.toPandas().profile_report()


# In[ ]:


sdf_submission = sdf_predict.select('PassengerId','prediction')                            .withColumn('Survived',sdf_predict['prediction'].cast('integer'))                            .select('PassengerId','Survived')
# sdf_submission.toPandas().T


# In[ ]:


sdf_submission.coalesce(1).write.csv("submission",mode="overwrite",header=True)


# In[ ]:


print(os.listdir('submission'))


# <a href='submission/part-00000-7ed8afef-ec95-4247-b97a-12b11efcd035-c000.csv'>Download</a>
