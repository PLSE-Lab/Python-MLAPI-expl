#!/usr/bin/env python
# coding: utf-8

# # Source : https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa

# In[ ]:





# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("vivek_spark_app").getOrCreate()
sdf = spark.read.load("../input/bank.csv",format="csv",inferSchema= True,header=True,sep=";")
# sdf.take(2)
sdf = sdf.withColumnRenamed("y","deposit")
sdf.printSchema()


# In[ ]:


pdf = sdf.toPandas()
pdf


# In[ ]:


# pd.DataFrame(sdf.take(5), columns=sdf.columns).traspose()


# In[ ]:


# Numeric features
num_features = [t[0] for t in sdf.dtypes if t[1] == 'int']
num_features


# ### https://spark.apache.org/docs/latest/ml-features.html#stringindexer
# ### https://spark.apache.org/docs/latest/ml-features.html#onehotencoderestimator
# ### https://spark.apache.org/docs/latest/ml-features.html#vectorassembler

# In[ ]:


# Feature engineering

from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
    
label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
print(f"assemblerInputs : {assemblerInputs}")
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
print(f"assembler : {assembler}")
stages += [assembler]
print(f"stages : {stages}")


# In[ ]:


# from pyspark.ml import Pipeline
# cols = sdf.columns
# pipeline = Pipeline(stages = stages)
# pipelineModel = pipeline.fit(sdf)
# sdf = pipelineModel.transform(sdf)
# selectedCols = ['label', 'features'] + cols
# sdf = sdf.select(selectedCols)
sdf.printSchema()


# In[ ]:


print(sdf.select('features').take(2))
sdf.select('label').take(2)


# In[ ]:


train, test = sdf.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))
print(type(train))


# # Decision Tree Classifier

# In[ ]:





# In[ ]:


from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth=3) #maxDepth to avoide overfitting

dtModel = dt.fit(train) # train the model
predictions_dt = dtModel.transform(test) # test the model / make prediction 

# pd.DataFrame( predictions_dt.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').take(5)


# In[ ]:


pd.DataFrame( predictions_dt.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').take(5),
             columns=['age', 'job', 'label', 'rawPrediction', 'prediction', 'probability']).transpose()
# predictions_dt.columns


# In[ ]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
eval = BinaryClassificationEvaluator()
eval.evaluate(predictions_dt,{eval.metricName: "areaUnderROC"})
#ROC ?


# # Random Forest Classifier

# In[ ]:


from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label' )
rfModel = rf.fit(train)
predictions_rf = rfModel.transform(test)
predictions_rf.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').toPandas()


# In[ ]:


eval = BinaryClassificationEvaluator()
eval.evaluate(predictions_rf,{eval.metricName: 'areaUnderROC'})


# # Gradient-Boosted Tree Classifier
# 

# In[ ]:


from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier()
gbtModel = gbt.fit(train)
gbtPrediction =  gbtModel.transform(test)
gbtPrediction.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').toPandas()


# In[ ]:


eval.evaluate(gbtPrediction,{eval.metricName: "areaUnderROC"})

