#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
#from pyspark.sql.types import spark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql import DataFrameReader
#from pyspark.sql.types import sc
#from pyspark.sql.types import sqlResultsPD
#from pyspark.sql.types import predictionsPD
#import pyspark.sql.DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import RandomForestRegressor
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import datetime


#logFile = "E:\\anacanda/Lib/site-packages/pyspark/bin/README.md"  # Should be some file on your system
sc = SparkSession.builder.appName("SimpleApp").getOrCreate()
#logData = spark.read.text(logFile).cache()
# 1. Location of training data: contains Dec 2015 trip and fare data from NYC
#trip_file_loc = '../input/nyc-taxi/trip_data_sample.csv'
#fare_file_loc = '../input/nyc-taxi/trip_fare_sample.csv'

# 2. Location of the joined taxi+fare training file
#taxi_valid_file_loc = 'D:\\Ebi secure/project spark/taxi/valid_file_loc'

# 3. Set model storage directory path. This is where models will be saved.
#modelDir = 'D:\\Ebi secure/project spark/taxi'; 

# 4. Set data storage path. This is where data is sotred on the blob attached to the cluster.
#dataDir = 'D:\\Ebi secure/project spark/taxi'; # The last backslash is needed;

sqlContext = SQLContext(sc)
## READ IN TRIP DATA FRAME FROM CSV
trip_file_loc='../input/nyctaxinew/trip_data_new.csv'
trip = spark.read.csv(path=trip_file_loc, header=True, inferSchema=True)

## READ IN FARE DATA FRAME FROM CSV
fare_file_loc='../input/nyctaxi/trip_fare_sample_new.csv'
fare = spark.read.csv(path=fare_file_loc, header=True, inferSchema=True)
trip.printSchema()
fare.printSchema()
## REGISTER DATA-FRAMEs AS A TEMP-TABLEs IN SQL-CONTEXT
trip.createOrReplaceTempView("trip")
fare.createOrReplaceTempView("fare")

## USING SQL: MERGE TRIP AND FARE DATA-SETS TO CREATE A JOINED DATA-FRAME
## ELIMINATE SOME COLUMNS, AND FILTER ROWS WTIH VALUES OF SOME COLUMNS
sqlStatement = """SELECT t.medallion, t.hack_license,
  f.total_amount, f.tolls_amount,
  hour(f.pickup_datetime) as pickup_hour, f.vendor_id, f.fare_amount, 
  f.surcharge, f.tip_amount, f.payment_type, t.rate_code, 
  t.passenger_count, t.trip_distance, t.trip_time_in_secs 
  FROM trip t, fare f  
  WHERE t.medallion = f.medallion AND t.hack_license = f.hack_license 
  AND t.pickup_datetime = f.pickup_datetime 
  AND t.passenger_count > 0 and t.passenger_count < 8 
  AND f.tip_amount >= 0 AND f.tip_amount <= 25 
  AND f.fare_amount >= 1 AND f.fare_amount <= 250 
  AND f.tip_amount < f.fare_amount AND t.trip_distance > 0 
  AND t.trip_distance <= 100 AND t.trip_time_in_secs >= 30 
  AND t.trip_time_in_secs <= 7200 AND t.rate_code <= 5
  AND f.payment_type in ('CSH','CRD')"""
trip_fareDF = spark.sql(sqlStatement)

# REGISTER JOINED TRIP-FARE DF IN SQL-CONTEXT
trip_fareDF.createOrReplaceTempView("trip_fare")

## SHOW WHICH TABLES ARE REGISTERED IN SQL-CONTEXT
spark.sql("show tables").show()

# SAMPLE 10% OF DATA, SPLIT INTO TRAIINING AND VALIDATION AND SAVE IN BLOB
trip_fare_featSampled = trip_fareDF.sample(False, 0.1, seed=1234)
trainfilename = dataDir + "TrainData";
trip_fare_featSampled.repartition(10).write.mode("overwrite").parquet(trainfilename)

## READ IN DATA FRAME FROM CSV
taxi_train_df = spark.read.parquet(trainfilename)

## CREATE A CLEANED DATA-FRAME BY DROPPING SOME UN-NECESSARY COLUMNS & FILTERING FOR UNDESIRED VALUES OR OUTLIERS
taxi_df_train_cleaned = taxi_train_df.drop('medallion').drop('hack_license').drop('total_amount').drop('tolls_amount')    .filter("passenger_count > 0 and passenger_count < 8 AND tip_amount >= 0 AND tip_amount < 15 AND             fare_amount >= 1 AND fare_amount < 150 AND trip_distance > 0 AND trip_distance < 100 AND             trip_time_in_secs > 30 AND trip_time_in_secs < 7200" )

## PERSIST AND MATERIALIZE DF IN MEMORY
taxi_df_train_cleaned.persist()
taxi_df_train_cleaned.count()

## REGISTER DATA-FRAME AS A TEMP-TABLE IN SQL-CONTEXT
taxi_df_train_cleaned.createOrReplaceTempView("taxi_train")

taxi_df_train_cleaned.printSchema()

#%%sql -q -o sqlResultsPD
#SELECT fare_amount, passenger_count, tip_amount FROM taxi_train WHERE passenger_count > 0 AND passenger_count < 7 AND fare_amount > 0 AND fare_amount < 100 AND tip_amount > 0 AND tip_amount < 15

#%%local
#%matplotlib inline

## %%local creates a pandas data-frame on the head node memory, from spark data-frame,
## which can then be used for plotting. Here, sampling data is a good idea, depending on the memory of the head node

# TIP BY PAYMENT TYPE AND PASSENGER COUNT
ax1 = sqlResultsPD[['tip_amount']].plot(kind='hist', bins=25, facecolor='lightblue')
ax1.set_title('Tip amount distribution')
ax1.set_xlabel('Tip Amount ($)'); ax1.set_ylabel('Counts');
plt.figure(figsize=(4,4)); plt.suptitle(''); plt.show()

# TIP BY PASSENGER COUNT
ax2 = sqlResultsPD.boxplot(column=['tip_amount'], by=['passenger_count'])
ax2.set_title('Tip amount by Passenger count')
ax2.set_xlabel('Passenger count'); ax2.set_ylabel('Tip Amount ($)');
plt.figure(figsize=(4,4)); plt.suptitle(''); plt.show()

# TIP AMOUNT BY FARE AMOUNT, POINTS ARE SCALED BY PASSENGER COUNT
ax = sqlResultsPD.plot(kind='scatter', x= 'fare_amount', y = 'tip_amount', c='blue', alpha = 0.10, s=2.5*(sqlResultsPD.passenger_count))
ax.set_title('Tip amount by Fare amount')
ax.set_xlabel('Fare Amount ($)'); ax.set_ylabel('Tip Amount ($)');
plt.axis([-2, 80, -2, 20])
plt.figure(figsize=(4,4)); plt.suptitle(''); plt.show()

### CREATE FOUR BUCKETS FOR TRAFFIC TIMES
sqlStatement = """SELECT payment_type, pickup_hour, fare_amount, tip_amount, 
    vendor_id, rate_code, passenger_count, trip_distance, trip_time_in_secs, 
  CASE
    WHEN (pickup_hour <= 6 OR pickup_hour >= 20) THEN 'Night'
    WHEN (pickup_hour >= 7 AND pickup_hour <= 10) THEN 'AMRush' 
    WHEN (pickup_hour >= 11 AND pickup_hour <= 15) THEN 'Afternoon'
    WHEN (pickup_hour >= 16 AND pickup_hour <= 19) THEN 'PMRush'
    END as TrafficTimeBins,
  CASE
    WHEN (tip_amount > 0) THEN 1 
    WHEN (tip_amount <= 0) THEN 0 
    END as tipped
  FROM taxi_train"""

taxi_df_train_with_newFeatures = spark.sql(sqlStatement)




# DEFINE THE TRANSFORMATIONS THAT NEEDS TO BE APPLIED TO SOME OF THE FEATURES
sI1 = StringIndexer(inputCol="vendor_id", outputCol="vendorIndex");
sI2 = StringIndexer(inputCol="rate_code", outputCol="rateIndex");
sI3 = StringIndexer(inputCol="payment_type", outputCol="paymentIndex");
sI4 = StringIndexer(inputCol="TrafficTimeBins", outputCol="TrafficTimeBinsIndex");

# APPLY TRANSFORMATIONS
encodedFinal = Pipeline(stages=[sI1, sI2, sI3, sI4]).fit(taxi_df_train_with_newFeatures).transform(taxi_df_train_with_newFeatures);

trainingFraction = 0.75; testingFraction = (1-trainingFraction);
seed = 1234;

# SPLIT SAMPLED DATA-FRAME INTO TRAIN/TEST, WITH A RANDOM COLUMN ADDED FOR DOING CV (SHOWN LATER)
trainData, testData = encodedFinal.randomSplit([trainingFraction, testingFraction], seed=seed);

# CACHE DATA FRAMES IN MEMORY
trainData.persist(); trainData.count()
testData.persist(); testData.count()



## DEFINE REGRESSION FURMULA
regFormula = RFormula(formula="tip_amount ~ paymentIndex + vendorIndex + rateIndex + TrafficTimeBinsIndex + pickup_hour + passenger_count + trip_time_in_secs + trip_distance + fare_amount")

## DEFINE INDEXER FOR CATEGORIAL VARIABLES
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=32)

## DEFINE ELASTIC NET REGRESSOR
eNet = LinearRegression(featuresCol="indexedFeatures", maxIter=25, regParam=0.01, elasticNetParam=0.5)

## Fit model, with formula and other transformations
model = Pipeline(stages=[regFormula, featureIndexer, eNet]).fit(trainData)

## PREDICT ON TEST DATA AND EVALUATE
predictions = model.transform(testData)
predictionAndLabels = predictions.select("label","prediction").rdd
testMetrics = RegressionMetrics(predictionAndLabels)
print("RMSE = %s" % testMetrics.rootMeanSquaredError)
print("R-sqr = %s" % testMetrics.r2)

## PLOC ACTUALS VS. PREDICTIONS
predictions.select("label","prediction").createOrReplaceTempView("tmp_results");

from pyspark.ml.regression import GBTRegressor

## DEFINE REGRESSION FURMULA
regFormula = RFormula(formula="tip_amount ~ paymentIndex + vendorIndex + rateIndex + TrafficTimeBinsIndex + pickup_hour + passenger_count + trip_time_in_secs + trip_distance + fare_amount")

## DEFINE INDEXER FOR CATEGORIAL VARIABLES
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=32)

## DEFINE GRADIENT BOOSTING TREE REGRESSOR
gBT = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

## Fit model, with formula and other transformations
model = Pipeline(stages=[regFormula, featureIndexer, gBT]).fit(trainData)

## PREDICT ON TEST DATA AND EVALUATE
predictions = model.transform(testData)
predictionAndLabels = predictions.select("label","prediction").rdd
testMetrics = RegressionMetrics(predictionAndLabels)
print("RMSE = %s" % testMetrics.rootMeanSquaredError)
print("R-sqr = %s" % testMetrics.r2)

## PLOC ACTUALS VS. PREDICTIONS
predictions.select("label","prediction").createOrReplaceTempView("tmp_results");


## DEFINE REGRESSION FURMULA
regFormula = RFormula(formula="tip_amount ~ paymentIndex + vendorIndex + rateIndex + TrafficTimeBinsIndex + pickup_hour + passenger_count + trip_time_in_secs + trip_distance + fare_amount")

## DEFINE INDEXER FOR CATEGORIAL VARIABLES
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=32)

## DEFINE RANDOM FOREST ESTIMATOR
randForest = RandomForestRegressor(featuresCol = 'indexedFeatures', labelCol = 'label', numTrees=20,
                                   featureSubsetStrategy="auto",impurity='variance', maxDepth=6, maxBins=100)

## Fit model, with formula and other transformations
model = Pipeline(stages=[regFormula, featureIndexer, randForest]).fit(trainData)

## SAVE MODEL
datestamp = datetime.datetime.now().strftime('%m-%d-%Y-%s');
fileName = "RandomForestRegressionModel_" + datestamp;
randForestDirfilename = modelDir + fileName;
model.save(randForestDirfilename)

## PREDICT ON TEST DATA AND EVALUATE
predictions = model.transform(testData)
predictionAndLabels = predictions.select("label","prediction").rdd
testMetrics = RegressionMetrics(predictionAndLabels)
print("RMSE = %s" % testMetrics.rootMeanSquaredError)
print("R-sqr = %s" % testMetrics.r2)

## PLOC ACTUALS VS. PREDICTIONS
predictions.select("label","prediction").createOrReplaceTempView("tmp_results");

#%%sql -q -o predictionsPD
#SELECT * from tmp_results

#%%local


ax = predictionsPD.plot(kind='scatter', figsize = (5,5), x='label', y='prediction', color='blue', alpha = 0.25, label='Actual vs. predicted');
fit = np.polyfit(predictionsPD['label'], predictionsPD['prediction'], deg=1)
ax.set_title('Actual vs. Predicted Tip Amounts ($)')
ax.set_xlabel("Actual"); ax.set_ylabel("Predicted");
ax.plot(predictionsPD['label'], fit[0] * predictionsPD['label'] + fit[1], color='magenta')
plt.axis([-1, 15, -1, 15])
plt.show(ax)

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

## DEFINE RANDOM FOREST MODELS
randForest = RandomForestRegressor(featuresCol = 'indexedFeatures', labelCol = 'label',
                                   featureSubsetStrategy="auto",impurity='variance', maxBins=100)

## DEFINE MODELING PIPELINE, INCLUDING FORMULA, FEATURE TRANSFORMATIONS, AND ESTIMATOR
pipeline = Pipeline(stages=[regFormula, featureIndexer, randForest])

## DEFINE PARAMETER GRID FOR RANDOM FOREST
paramGrid = ParamGridBuilder()     .addGrid(randForest.numTrees, [10, 25, 50])     .addGrid(randForest.maxDepth, [3, 5, 7])     .build()

## DEFINE CROSS VALIDATION
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(metricName="rmse"),
                          numFolds=3)

## TRAIN MODEL USING CV
cvModel = crossval.fit(trainData)

## PREDICT AND EVALUATE TEST DATA SET
predictions = cvModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print("R-squared on test data = %g" % r2)

## SAVE THE BEST MODEL
datestamp = datetime.datetime.now().strftime('%m-%d-%Y-%s');
fileName = "CV_RandomForestRegressionModel_" + datestamp;
CVDirfilename = modelDir + fileName;
cvModel.bestModel.save(CVDirfilename);

from pyspark.ml import PipelineModel

savedModel = PipelineModel.load(randForestDirfilename)

predictions = savedModel.transform(testData)
predictionAndLabels = predictions.select("label","prediction").rdd
testMetrics = RegressionMetrics(predictionAndLabels)
print("RMSE = %s" % testMetrics.rootMeanSquaredError)
print("R-sqr = %s" % testMetrics.r2)

## READ IN DATA FRAME FROM CSV
taxi_valid_df = spark.read.csv(path=taxi_valid_file_loc, header=True, inferSchema=True)
taxi_valid_df.printSchema()

## READ IN DATA FRAME FROM CSV
taxi_valid_df = spark.read.csv(path=taxi_valid_file_loc, header=True, inferSchema=True)

## CREATE A CLEANED DATA-FRAME BY DROPPING SOME UN-NECESSARY COLUMNS & FILTERING FOR UNDESIRED VALUES OR OUTLIERS
taxi_df_valid_cleaned = taxi_valid_df.drop('medallion').drop('hack_license').drop('store_and_fwd_flag').drop('pickup_datetime')    .drop('dropoff_datetime').drop('pickup_longitude').drop('pickup_latitude').drop('dropoff_latitude')    .drop('dropoff_longitude').drop('tip_class').drop('total_amount').drop('tolls_amount').drop('mta_tax')    .drop('direct_distance').drop('surcharge')    .filter("passenger_count > 0 and passenger_count < 8 AND payment_type in ('CSH', 'CRD')     AND tip_amount >= 0 AND tip_amount < 30 AND fare_amount >= 1 AND fare_amount < 150 AND trip_distance > 0     AND trip_distance < 100 AND trip_time_in_secs > 30 AND trip_time_in_secs < 7200" )

## REGISTER DATA-FRAME AS A TEMP-TABLE IN SQL-CONTEXT
taxi_df_valid_cleaned.createOrReplaceTempView("taxi_valid")

### CREATE FOUR BUCKETS FOR TRAFFIC TIMES
sqlStatement = """ SELECT *, CASE
     WHEN (pickup_hour <= 6 OR pickup_hour >= 20) THEN "Night" 
     WHEN (pickup_hour >= 7 AND pickup_hour <= 10) THEN "AMRush" 
     WHEN (pickup_hour >= 11 AND pickup_hour <= 15) THEN "Afternoon"
     WHEN (pickup_hour >= 16 AND pickup_hour <= 19) THEN "PMRush"
    END as TrafficTimeBins
    FROM taxi_valid
"""
taxi_df_valid_with_newFeatures = spark.sql(sqlStatement)

## APPLY THE SAME TRANSFORATION ON THIS DATA AS ORIGINAL TRAINING DATA
encodedFinalValid = Pipeline(stages=[sI1, sI2, sI3, sI4]).fit(taxi_df_train_with_newFeatures).transform(taxi_df_valid_with_newFeatures)

## LOAD SAVED MODEL, SCORE VALIDATION DATA, AND EVALUATE
savedModel = PipelineModel.load(CVDirfilename)
predictions = savedModel.transform(encodedFinalValid)
r2 = evaluator.evaluate(predictions)
print("R-squared on validation data = %g" % r2)

datestamp = datetime.datetime.now().strftime('%m-%d-%Y-%s');
fileName = "Predictions_CV_" + datestamp;
predictionfile = dataDir + fileName;
predictions.select("label","prediction").write.mode("overwrite").csv(predictionfile)
spark.stop()


# In[ ]:


import pandas as pd
trip_data_sample = pd.read_csv("../input/nyc-taxi/trip_data_sample.csv")
trip_fare_sample = pd.read_csv("../input/nyc-yaxi/trip_fare_sample.csv")


# In[ ]:


import pandas as pd
trip_data_sample = pd.read_csv("../input/sample-dataset-nyc-taxi/trip_data_sample.csv")
trip_fare_sample = pd.read_csv("../input/sample-dataset-nyc-taxi/trip_fare_sample.csv")


# In[ ]:


import pandas as pd
trip_data_sample_new = pd.read_csv("../input/nyc-taxi-new/trip_data_sample_new.csv")


# In[ ]:


import pandas as pd
trip_fare_sample_new = pd.read_csv("../input/trip_fare_sample_new.csv")


# In[ ]:


import pandas as pd
trip_data_sample_new = pd.read_csv("../input/trip_data_sample_new.csv")
trip_fare_sample_new = pd.read_csv("../input/trip_fare_sample_new.csv")


# In[ ]:


import pandas as pd
trip_data_new = pd.read_csv("../input/trip_data_new.csv")


# In[ ]:


import pandas as pd
trip_data_new = pd.read_csv("../input/trip_data_new.csv")
trip_fare_sample_new = pd.read_csv("../input/trip_fare_sample_new.csv")


# In[ ]:


import pandas as pd
trip_joined_fare = pd.read_csv("../input/trip_joined_fare.csv")

