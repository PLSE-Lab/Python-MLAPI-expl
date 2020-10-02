#!/usr/bin/env python
# coding: utf-8

# In[133]:


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


# In[134]:


from pyspark.sql import SparkSession
import pyspark.sql as sparksql
spark = SparkSession.builder.appName('stroke').getOrCreate()
train = spark.read.csv('../input/train_2v.csv', inferSchema=True,header=True)
test = spark.read.csv('../input/test_2v.csv', inferSchema=True,header=True)


# ### Exploring the training data

# In[135]:


train.printSchema()


# In[136]:


train.dtypes


# In[137]:


train.head(5)


# In[138]:


train.toPandas().head(5)


# ### Lets also look at test data

# In[139]:


test.describe().show()


# ### Lets look the the target distribution

# In[140]:


train.groupBy('stroke').count().show()


# As can be seen from this observation. This is an Imbalanced dataset, where the number of observations belonging to one class is significantly lower than those belonging to the other classes. In this case, the predictive model could be biased and inaccurate. There are different strategies to handling Imbalanced Datasets, We will look into it later.

# ### Training feature analysis 

# In[141]:


# create DataFrame as a temporary view for SQL queries
train.createOrReplaceTempView('table')


# influence of work type on getting stroke

# In[142]:


# sql query to find the number of people in specific work_type who have had stroke and not
spark.sql("SELECT work_type, COUNT(work_type) as work_type_count FROM table WHERE stroke == 1 GROUP BY work_type ORDER BY COUNT(work_type) DESC").show()
spark.sql("SELECT work_type, COUNT(work_type) as work_type_count FROM table WHERE stroke == 0 GROUP BY work_type ORDER BY COUNT(work_type) DESC").show()


# It is mostly happening to private or self-employed person.

# Is it related to gender !!!

# In[143]:


spark.sql("SELECT gender, COUNT(gender) as gender_count, COUNT(gender)*100/(SELECT COUNT(gender) FROM table WHERE gender == 'Male') as percentage FROM table WHERE stroke== 1 AND gender = 'Male' GROUP BY gender").show()
spark.sql("SELECT gender, COUNT(gender) as gender_count, COUNT(gender)*100/(SELECT COUNT(gender) FROM table WHERE gender == 'Female') as percentage FROM table WHERE stroke== 1 AND gender = 'Female' GROUP BY gender").show()


# 1.68% male and almost 2% male had stroke.

# Now we will see influence of age on stroke

# In[144]:


spark.sql("SELECT COUNT(age)*100/(SELECT COUNT(age) FROM table WHERE stroke ==1) as percentage FROM table WHERE stroke == 1 AND age>=50").show()


# Here we see that 91.5% stroke had occured for person who are more than 50 years old

# ### Cleaning up training data

# In[145]:


train.describe().show()


# 1. Here we see that there are few missing values in *smoking_status* and *bmi* column
# 2. Also there are few categorical data (*gender, ever_married, work_type, Residence_type, smoking_status* which we need to covert into one hot encoding

# In[146]:


# fill in missing values for smoking status
# As this is categorical data, we will add one data type "No Info" for the missing one
train_f = train.na.fill('No Info', subset=['smoking_status'])
test_f = test.na.fill('No Info', subset=['smoking_status'])


# In[147]:


# fill in miss values for bmi 
# as this is numecial data , we will simple fill the missing values with mean
from pyspark.sql.functions import mean
mean = train_f.select(mean(train_f['bmi'])).collect()
mean_bmi = mean[0][0]
train_f = train_f.na.fill(mean_bmi,['bmi'])
test_f = test_f.na.fill(mean_bmi,['bmi'])


# In[148]:


train_f.describe().show()


# In[149]:


test_f.describe().show()


# Now there is no missing values, Lets work on categorical columns now...

# StringIndexer -> OneHotEncoder -> VectorAssembler

# In[150]:


# indexing all categorical columns in the dataset
from pyspark.ml.feature import StringIndexer
indexer1 = StringIndexer(inputCol="gender", outputCol="genderIndex")
indexer2 = StringIndexer(inputCol="ever_married", outputCol="ever_marriedIndex")
indexer3 = StringIndexer(inputCol="work_type", outputCol="work_typeIndex")
indexer4 = StringIndexer(inputCol="Residence_type", outputCol="Residence_typeIndex")
indexer5 = StringIndexer(inputCol="smoking_status", outputCol="smoking_statusIndex")


# In[151]:


# Doing one hot encoding of indexed data
from pyspark.ml.feature import OneHotEncoderEstimator
encoder = OneHotEncoderEstimator(inputCols=["genderIndex","ever_marriedIndex","work_typeIndex","Residence_typeIndex","smoking_statusIndex"],
                                 outputCols=["genderVec","ever_marriedVec","work_typeVec","Residence_typeVec","smoking_statusVec"])


# The next step is to create an assembler, that combines a given list of columns into a single vector column to train ML model. I will use the vector columns, that we got after one_hot_encoding.

# In[152]:


from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['genderVec',
 'age',
 'hypertension',
 'heart_disease',
 'ever_marriedVec',
 'work_typeVec',
 'Residence_typeVec',
 'avg_glucose_level',
 'bmi',
 'smoking_statusVec'],outputCol='features')


# ### Baseline model

# We are using Decision tree classifier for baseline model

# In[153]:


from pyspark.ml.classification import DecisionTreeClassifier
dtc = DecisionTreeClassifier(labelCol='stroke',featuresCol='features')


# So far we have kind of a complex task that contains bunch of stages, that need to be performed to process data. To wrap all of that Spark ML represents such a workflow as a Pipeline, which consists of a sequence of PipelineStages to be run in a specific order.

# In[154]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[indexer1, indexer2, indexer3, indexer4, indexer5, encoder, assembler, dtc])


# The next step is to split dataset to train and test to train the model and make predictions.

# In[155]:


# splitting training and validation data
train_data,val_data = train_f.randomSplit([0.7,0.3])

# training model pipeline with data
model = pipeline.fit(train_data)


# Now we will evaluate the model with validation data

# In[156]:


# making prediction on model with validation data
dtc_predictions = model.transform(val_data)

# Select example rows to display.
dtc_predictions.select("prediction","probability", "stroke", "features").show(5)


# In[157]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Select (prediction, true label) and compute test error
acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy")
dtc_acc = acc_evaluator.evaluate(dtc_predictions)
print('A Decision Tree algorithm had an accuracy of: {0:2.2f}%'.format(dtc_acc*100))


# In[158]:


# now predicting the labels for test data
test_pred = model.transform(test_f)
test_selected = test_pred.select("id", "features", "prediction","probability")
test_selected.limit(5).toPandas()

