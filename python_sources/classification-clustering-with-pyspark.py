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


get_ipython().system('pip install pyspark')


# In[ ]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *


# In[ ]:


spark = SparkSession.Builder().getOrCreate()


# In[ ]:


train = spark.read.csv('/kaggle/input/ccdata/CC GENERAL.csv',header = True,inferSchema=True)


# In[ ]:


train.limit(5).toPandas()


# # Pre-processing

# In[ ]:


train = train.na.drop(how='any')
train.limit(5).toPandas()


# In[ ]:


train = train.withColumn("label", train.PURCHASES_FREQUENCY>=0.5)
train = train.withColumn("label", train["label"].cast("string"))

from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="label", outputCol="target")
training = indexer.fit(train).transform(train)

training.limit(5).toPandas()


# # Lest's create our classification model

# In[ ]:


columns = [col for col in training.columns if col not in ['target','CUST_ID','label','PURCHASES_FREQUENCY']]


# In[ ]:


from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler().setInputCols(columns).setOutputCol("features")
train_calss = assembler.transform(training)


# In[ ]:


train_calss.select("features","target").show(5)


# # Random forest classifier

# In[ ]:


from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features',labelCol = "target")


# In[ ]:


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# In[ ]:


paramGrid = ParamGridBuilder()   .addGrid(rf.numTrees, [100, 200, 300])   .addGrid(rf.maxDepth, [1, 2, 3, 4, 5, 6, 7, 8])   .addGrid(rf.maxBins, [25, 28, 31])   .addGrid(rf.impurity, ["entropy", "gini"])   .build()


# In[ ]:


evaluator = BinaryClassificationEvaluator(labelCol = "target", rawPredictionCol = "prediction") 

crossval = CrossValidator(estimator = rf,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluator,
                          numFolds = 5)


# In[ ]:


train_rf, test_rf = train_calss.randomSplit([0.8, 0.2])


# In[ ]:


cvModel = crossval.fit(train_rf)


# In[ ]:


predictions = cvModel.transform(test_rf)


# In[ ]:


predictions.select("features","prediction","target").limit(5).toPandas()


# In[ ]:


evaluator = BinaryClassificationEvaluator(labelCol = "target", rawPredictionCol = "prediction") 
evaluator.evaluate(predictions)


# # Logistic regression

# In[ ]:


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol="features",labelCol="target",maxIter=10, regParam=0.3, elasticNetParam=0.8)


# In[ ]:


train_lr=train_calss.select("features","target")


# In[ ]:


training, testing = train_lr.randomSplit([0.8, 0.2])


# In[ ]:


model = lr.fit(training)


# In[ ]:


predictions = model.transform(testing)
predictions.select("prediction", "target", "features").show(5)


# In[ ]:


evaluator = BinaryClassificationEvaluator(labelCol = "target", rawPredictionCol = "prediction")
evaluator.evaluate(predictions)


# # Clustering

# # K-Means

# In[ ]:


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

columns = [col for col in training.columns if col not in ['target','CUST_ID','label']]
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler().setInputCols(columns).setOutputCol("features_clustering")

train_clustering = assembler.transform(training)


# In[ ]:


import numpy as np
cost = np.zeros(20)
for k in range(2,20):
    kmeans = KMeans()            .setK(k)            .setSeed(1)             .setFeaturesCol("features_clustering")            .setPredictionCol("cluster")

    model_k = kmeans.fit(train_clustering)
    cost[k] = model_k.computeCost(train_clustering)


# # how many K do I need

# In[ ]:


import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sbs
from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()


# In[ ]:


kmeans = KMeans().setK(2).setSeed(1).setFeaturesCol("features_clustering")
model = kmeans.fit(train_clustering)


# In[ ]:


# Make predictions
predictions = model.transform(train_clustering)


# In[ ]:


# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))


# In[ ]:


# Show up the centers.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[ ]:


predictions.select("features","prediction").limit(5).toPandas()


# # BisectingKMeans

# In[ ]:


from pyspark.ml.clustering import BisectingKMeans
bkm = BisectingKMeans().setK(2).setSeed(1)
model2= bkm.fit(train_clustering)


# In[ ]:


predictions2 = model2.transform(train_clustering)


# In[ ]:


evaluator2= ClusteringEvaluator()
silhouette2 = evaluator.evaluate(predictions2)
print("Silhouette with squared euclidean distance = " + str(silhouette2))


# In[ ]:




