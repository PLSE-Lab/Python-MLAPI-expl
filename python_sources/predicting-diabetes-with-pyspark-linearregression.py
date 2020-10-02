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


'''
4> Predicting Diabetes using LinearRegression from MLib (Machine Learning library from Spark) 

This Diabetes dataset downloaded from Sklearn has ten baseline variables, age, sex, body mass index, average blood 
pressure, and six blood serum measurements were obtained for each of n = 442 diabetes patients, as well as the 
response of interest, a quantitative measure of disease progression one year after baseline.

A fasting blood sugar level less than 100 mg/dL (5.6 mmol/L) is normal. A fasting blood sugar level from 100 to 
125 mg/dL (5.6 to 6.9 mmol/L) is considered prediabetes. If it's 126 mg/dL (7 mmol/L) or higher on two separate 
tests, you have diabetes. Oral glucose tolerance test.
'''
import findspark
findspark.init()
import pyspark
import random

from sklearn import datasets
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

# Import and clean data. Pyspark uses its own type system and unfortunately it doesn't deal with numpy well. 
# It works with python types though. So you need to manually convert the numpy.float64 to float.

diabetes = datasets.load_diabetes()
diabetes_features= []

# Spark uses breeze under the hood for high performance Linear Algebra in Scala. In Spark, MLlib and other 
# ML algorithms depends on org.apache.spark.mllib.libalg.Vector type which is rather dense or sparse.

for feature_list in diabetes.data:
    temp= [float(i) for i in feature_list]
    diabetes_features.append(Vectors.dense(temp))
    
diabetes_target = [float(i) for i in diabetes.target]
features_and_predictions = list(zip(diabetes_target, diabetes_features))

sc = pyspark.SparkContext(appName="LinearRegression_Diabetes")
sqlContext = SQLContext(sc)
df = sqlContext.createDataFrame(features_and_predictions, ["label", "features"])

# Only max iterations is set. We will set parameters for the algorithm after ParamGridSearch
lr = LinearRegression(maxIter=10)

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
paramGrid = ParamGridBuilder()    .addGrid(lr.regParam, [0.1, 0.01])     .addGrid(lr.fitIntercept, [False, True])    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])    .build()


# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
LR_model = tvs.fit(df)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.

LR_model.transform(df)    .select("features","label", "prediction").show()

Dataframe = LR_model.transform(df)    .select("label", "prediction")

# Metrics object needs to have an RDD of (prediction, observation) pairs.
# Convert the dataframe object to an RDD

valuesAndPreds = Dataframe.rdd.map(tuple)

# Instantiate metrics object
metrics = RegressionMetrics(valuesAndPreds)

# Squared Error
print("MSE = %s" % metrics.meanSquaredError)
print("RMSE = %s" % metrics.rootMeanSquaredError)

# R-squared
print("R-squared = %s" % metrics.r2)

# Mean absolute error
print("MAE = %s" % metrics.meanAbsoluteError)

# Explained variance
print("Explained variance = %s" % metrics.explainedVariance)

sc.stop()

