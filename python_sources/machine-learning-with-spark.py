#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# ## Machine Learning with Spark

# This is a tutorial of machine learning with PySpark. I will create a classification model and a regression model using Pipelines.

# ### Install Spark
# Uncommet the code below to install PySpark and sparkmagic.

# In[ ]:


#!pip install sparkmagic
#!pip install pyspark


# ### Import Spark SQL and Spark ML Libraries
# 
# We'll train a **LogisticRegression** model with a **Pipleline** preparing the data, a **CrossValidator** to tuene the parameters of the model, and a **BinaryClassificationEvaluator** to evaluate our trained model.

# In[1]:


from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
#from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.master("local[*]").getOrCreate()


# ### Load Source Data
# The data from the flight.csv file data includes specific characteristics (or features) for each flight, as well as a column indicating how many minutes late or early the flight arrived.

# In[2]:


csv = spark.read.csv('../input/flights.csv', inferSchema=True, header=True)
csv.show(10)


# ### Prepare the Data for a Classification Model (Decision Tree Learning Model)
# I select a subset of columns to use as features and create a Boolean label field named *label* with values 1 or 0. Specifically, **1** for flight that arrived late, **0** for flight was early or on-time.

# In[3]:


data = csv.select("DayofMonth", "DayOfWeek", "Carrier", "OriginAirportID", "DestAirportID", "DepDelay", ((col("ArrDelay") > 15).cast("Int").alias("label")))
data.show(10)


# ### Split the Data
# 
# I will use 70% of the data for training, and reserve 30% for testing. In the testing data, the *label* column is renamed to *trueLabel* so I can use it later to compare predicted labels with known actual values.

# In[4]:


splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)


# ### Define the Pipeline
# 
# A pipeline consists of a series of transformer and estimator stages that typically prepare a DataFrame for modeling and then train a predictive model. In this case, you will create a pipeline with seven stages:
# * A **StringIndexer estimator** that converts string values to indexes for categorical features
# * A **VectorAssembler** that combines categorical features into a single vector
# * A **VectorIndexer** that creates indexes for a vector of categorical features
# * A **VectorAssembler** that creates a vector of continuous numeric features
# * A **MinMaxScaler** that normalizes continuous numeric features
# * A **VectorAssembler** that creates a vector of categorical and continuous features
# * A **DecisionTreeClassifier** that trains a classification model.

# In[5]:


strIdx = StringIndexer(inputCol = "Carrier", outputCol = "CarrierIdx")
catVect = VectorAssembler(inputCols = ["CarrierIdx", "DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID"], outputCol="catFeatures")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures")
numVect = VectorAssembler(inputCols = ["DepDelay"], outputCol="numFeatures")
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")
featVect = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures"], outputCol="features")
lr = LogisticRegression(labelCol="label",featuresCol="features",maxIter=10,regParam=0.3)
#dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[strIdx, catVect, catIdx, numVect, minMax, featVect, lr])


# ### Run the Pipeline to train a model
# Run the pipeline as an Estimator on the training data to train a model.

# In[6]:


piplineModel = pipeline.fit(train)


# ### Generate label predictions
# Transform the test data with all of the stages and the trained model in the pipeline to generate label predictions.

# In[7]:


prediction = piplineModel.transform(test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(100, truncate=False)


# Looking into the results, some trueLabel 1s are predicted as 0. Let's evaluate the model.

# ## Evaluating a Classification Model
# We'll calculate a *Confusion Matrix* and the *Area Under ROC* (Receiver Operating Characteristic) to evaluate the model. 
# ### Compute Confusion Matrix
# Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
# - True Positives
# - True Negatives
# - False Positives
# - False Negatives
# 
# From these core measures, other evaluation metrics such as *precision*, *recall* and *F1* can be calculated.

# In[8]:


tp = float(predicted.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND truelabel == 1").count())
pr = tp / (tp + fp)
re = tp / (tp + fn)
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", pr),
 ("Recall", re),
 ("F1", 2*pr*re/(re+pr))],["metric", "value"])
metrics.show()


# Looks like we've got a good *Precision*, but a low *Recall*, therefore our *F1* is not that good.

# ### Review the Area Under ROC
# Another way to assess the performance of a classification model is to measure the area under a ROC (Receiver Operating Characteristic) curve for the model. the spark.ml library includes a **BinaryClassificationEvaluator** class that we can use to compute this. The ROC curve shows the True Positive and False Positive rates plotted for varying thresholds.

# In[9]:


evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
aur = evaluator.evaluate(prediction)
print ("AUR = ", aur)


# So the AUR shows that our model is ok.
# Let's look deeper.

# ### View the Raw Prediction and Probability
# The prediction is based on a raw prediction score that describes a labelled point in a logistic function. This raw prediction is then converted to a predicted label of 0 or 1 based on a probability vector that indicates the confidence for each possible label value (in this case, 0 and 1). The value with the highest confidence is selected as the prediction.

# In[10]:


prediction.select("rawPrediction", "probability", "prediction", "trueLabel").show(100, truncate=False)


# Note that the results include rows where the probability for 0 (the first value in the probability vector) is only slightly higher than the probability for 1 (the second value in the probability vector). The default discrimination threshold (the boundary that decides whether a probability is predicted as a 1 or a 0) is set to 0.5; so the prediction with the highest probability is always used, no matter how close to the threshold.
# 
# And we can see from the results above that for those *truelabel* 1s that we predicted 0s, many of them the problibilty of 1 is just slightly less than the threshold 0.5.

# ## Tune Parameters 
# To find the best performing parameters, we can use the **CrossValidator** class to evaluate each combination of parameters defined in a **ParameterGrid** against multiple *folds* of the data split into training and validation datasets. Note that this can take a long time to run because every parameter combination is tried multiple times.

# ### Change the Discrimination Threshold
# The AUC score seems to indicate a reasonably good model, but the performance metrics seem to indicate that it predicts a high number of *False Negative* labels (i.e. it predicts 0 when the true label is 1), leading to a low *Recall*. We can improve this by lowering the threshold. Conversely, sometimes we may want to address a large number of *False Positive* by raising the threshold. 
# 
# In this case, I'll let the **CrossValidator** find the best threshold from 0.45, 0.4 and 0.35, regularization parameter from 0.3 and 0.1, and the maximum number of iterations allowed from 10 and 5.

# In[ ]:


paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.1]).addGrid(lr.maxIter, [10, 5]).addGrid(lr.threshold, 
                                                                                            [0.4, 0.3]).build()
cv = CrossValidator(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, 
                    numFolds=2)

model = cv.fit(train)


# In[ ]:


newPrediction = model.transform(test)
newPredicted = prediction.select("features", "prediction", "trueLabel")
newPredicted.show()


# Note that some of the **rawPrediction** and **probability** values that were previously predicted as 0 are now predicted as 1

# In[ ]:


# Recalculate confusion matrix
tp2 = float(newPrediction.filter("prediction == 1.0 AND truelabel == 1").count())
fp2 = float(newPrediction.filter("prediction == 1.0 AND truelabel == 0").count())
tn2 = float(newPrediction.filter("prediction == 0.0 AND truelabel == 0").count())
fn2 = float(newPrediction.filter("prediction == 0.0 AND truelabel == 1").count())
pr2 = tp2 / (tp2 + fp2)
re2 = tp2 / (tp2 + fn2)
metrics2 = spark.createDataFrame([
 ("TP", tp2),
 ("FP", fp2),
 ("TN", tn2),
 ("FN", fn2),
 ("Precision", pr2),
 ("Recall", re2),
 ("F1", 2*pr2*re2/(re2+pr2))],["metric", "value"])
metrics2.show()


# In[ ]:


# Recalculate the Area Under ROC
evaluator2 = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
aur2 = evaluator.evaluate(prediction)
print( "AUR2 = ", aur2)


# Looks pretty good! The new model improves the *Recall* from 0.11 to 0.37, the *F1* score from 0.20 to 0.54, without compromising other metrics.
# 
# ## Next Step
# 
# There is still much room to improve the model. For example, I can try more options of lower threshold, or use different classfication models, or prepare data better like adding new features. I'll write another one for this.
