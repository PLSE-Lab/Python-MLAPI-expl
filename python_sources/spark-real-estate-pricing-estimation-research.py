#!/usr/bin/env python
# coding: utf-8

# ## A Bit Different View on Real Estate

# For some reasons I'm going to use apache spark ml lib.
# So, before we begin let's import all nesessary *stuff*, that we need for our research.

# In[ ]:


#Initial
import os
import numpy as np
import pandas as pd
import math

#Spark
import pyspark as spark
from pyspark import SparkConf, SparkContext


sc = SparkContext.getOrCreate()


from pyspark.sql import SparkSession, SQLContext

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor, GeneralizedLinearRegression, AFTSurvivalRegression
from pyspark.ml.feature import VectorIndexer

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import DoubleType
 
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder  
    
#Plots    
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:





# ### Data Exploration 

# At this section, I'll use tipical python datascience tools because this is more reliable and suitable tools for EDA part. (I mean using pandas in place of spark's dataframes and etc.)

# Let's take a look on what do we have here

# In[ ]:


dataPath = "../input"
', '.join(os.listdir(dataPath))


# ### Dataset Exploration:

# In[ ]:


pdtrainset = pd.read_csv("../input/train.csv")


# In[ ]:


pdtrainset.head(10)


# In[ ]:


pdtrainset.describe()


# In[ ]:


pdtrainset["SalePrice"] = np.log(pdtrainset["SalePrice"])
pdtrainset.head(7)


# In[ ]:


pdtrainset.info()


# In[ ]:


correl = pdtrainset[1:].corr()
correl.drop(["SalePrice", "Id"], axis=1, inplace=True)


# Here we interrested in features which have more significant effect on sale price.

# In[ ]:


correl.iloc[-1].apply(lambda x: abs(x)).plot(kind='bar', figsize=(10, 6), title="Correlation with SalePrice", grid=True)


# Let's find a list of most valuable features, then sort it by descending.

# In[ ]:


salecorrel = pd.DataFrame(correl.transpose()["SalePrice"])
top_features = salecorrel[salecorrel["SalePrice"] >= 0.45].sort_values(by=["SalePrice"], 
                                                                      ascending=False)
topcolumns = top_features.transpose().columns.values
topcorrelVal = top_features.values

["{0}: {1:.4f}".format(col, topcorrelVal[i][0]) for i,col in enumerate(topcolumns)]


# Now compare valuable feature's correlation eachother.
# 
# (Here I should note, that In my city, there are many paradoxes when, for excample, age of a building has a strong positive correlation with the quality of supporting structures...)
#  
# I believe that is not everywhere, so I will solely rely  on digits.)

# In[ ]:


# [sns.lmplot(x="SalePrice",y=x,data=pdtrainset,
#            scatter_kws={'alpha':0.07}, aspect=2, height=4) for x in topcolumns]

topItemsCorrelation = pdtrainset[list(topcolumns)].corr().abs()

topItemsCorrelation


# In[ ]:


sns.heatmap(topItemsCorrelation, annot=True, fmt=".2f")
plt.show()


# Let's list a pars that have correlation more then 0.8

# In[ ]:


utcItems = pd.DataFrame(topItemsCorrelation.unstack(), columns=["c"])
utcItems[(utcItems["c"] > 0.8) & (utcItems["c"] < 1)]


# **GrLivArea** has higher correlation with SalePrice then TotRmsAbvGrd patam
# as **GarageCars** and **TotalBsmtSF**, so I'll exclude from dataset "TotRmsAbvGrd", "GarageArea" and "1stFlrSF" from our features research.

# In[ ]:


topcolumns = list(set(topcolumns) - set(["TotRmsAbvGrd", "GarageArea", "1stFlrSF", "GarageYrBlt"]))
topcolumns


# Totally we have seven features, two of them is year of **YearBuilt** and **YearRemodAdd** (Remodel date). 
# I guess, it will good idea to use them as categorical variable splited by decades.

# In[ ]:


getDecades = lambda col: col.apply(lambda x: math.ceil(float(x) / 10)*10)

pdtrainset["YearBuilt"] = getDecades(pdtrainset["YearBuilt"])
pdtrainset["YearRemodAdd"] = getDecades(pdtrainset["YearRemodAdd"])
pdtrainset = pdtrainset[topcolumns+["SalePrice"]]
pdtrainset.head()


# In[ ]:


valuableColumns = list(pdtrainset.columns.values)


# In[ ]:


[sns.lmplot(x=col,y="SalePrice",data=pdtrainset, 
            scatter_kws={'alpha':0.07}, aspect=3, height=5) for col in topcolumns]


# Before train our models, let split features on to two groups: continious and categorical

# I won't deskribe all valuable features here, but 

# ### Spark ML

# In[ ]:


sqlContext = SQLContext(sc)

sc.setLogLevel("ERROR")


# Then prepare spark's dataframes 

# In[ ]:


df = sqlContext.createDataFrame(pdtrainset)


# In[ ]:


df.describe()


# In[ ]:


# #sptrain = sqlContext.read.csv("../input/train.csv", header=True)

sptrain = df.withColumn("label", df.SalePrice.cast("double")).cache()

sptest = sqlContext.read.csv("../input/test.csv", header=True)


# In[ ]:


for col in valuableColumns[:-1]:
    # Of cause we can't change immutable values, but we can owerwrite them
    sptrain = sptrain.withColumn(col+"_d", sptrain[col].cast("double"))
    sptest = sptest.withColumn(col+"_d", sptest[col].cast("double"))
    
sptrain = sptrain.fillna(-1., subset=valuableColumns)
sptest = sptest.fillna(-1., subset=valuableColumns)
# sptrain = sptrain.fillna("no", subset=catColumn)
# sptest = sptest.fillna("no", subset=catColumn)    
sptrain


# In[ ]:


#let add few train model 
gbt = GBTRegressor(maxIter=100, 
                   maxDepth=5)\
                  .setLabelCol("label")\
                  .setFeaturesCol("features")

# Than let split our features on to group categorical (string) and continious (numerical)
# indexers = [StringIndexer(inputCol=column, outputCol=column+"_index")\
#             .setHandleInvalid("keep")\
#             .fit(sptrain) for column in catColumn]

#indexers = indexers + ([OneHotEncoder(inputCol= col+"_index", outputCol= col+"_oht") for col in catColumn])
#TODO: change all columns to a valuable only
indexers = []
indexers.append(VectorAssembler(
    inputCols=["{0}_d".format(col) for col in valuableColumns[:-1]], 
        outputCol="features"))
indexers.append(gbt)


# In[ ]:


(trainingData, subtestData) = sptrain.randomSplit([0.7, 0.3])

modelGBT = Pipeline(stages=tuple(indexers)).fit(trainingData)


# In[ ]:


predictions = modelGBT.transform(subtestData)

predictions.select(["prediction", "label", "features"]).show(7)


# In[ ]:


evaluator = RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = {0}".format(str(rmse)))


# In[ ]:


testResult = modelGBT.transform(sptest)
submit = testResult.select(["Id","prediction"])          .withColumn("SalePrice", testResult["prediction"])          .drop("prediction")


# In[ ]:


#submit.write.format("com.databricks.spark.csv").option("header","true").save("../input/output.csv")

js  = submit.dropna()


# In[ ]:





# In[ ]:


#submit.coalesce(1).write.csv('../input/submission_.csv', encoding="utf-8", emptyValue=-1)
# submit.write.format("com.databricks.spark.csv").option("header", "true").save("../input/submit__.csv")
submit = submit.fillna(-1.0)
#submit.show()
submit.dropna().coalesce(1).write.csv(os.path.join("submission.csv",dataPath), encoding="utf-8", emptyValue="na")


# In[ ]:




