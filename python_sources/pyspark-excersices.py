#!/usr/bin/env python
# coding: utf-8

# # Personal PySpark Excersices
# I am learnning Spark and PySpark by myself, feel free to suggest ideas for improvements

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


# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


from pyspark.sql import SparkSession

# Create my_spark
my_spark = SparkSession.builder.getOrCreate()

# Print my_spark
print(my_spark)


# In[ ]:


print(my_spark.catalog.listTables())


# In[ ]:


# Create pd_temp
#pd_temp = pd.DataFrame(np.random.random(10))

# Read a dataframe
file_path = "../input/flights.csv"

# Create spark_temp from pd_temp
flights = my_spark.read.csv(file_path, header=True)

# Show the data
flights.show()

# Examine the tables in the catalog
print(my_spark.catalog.listTables())

# Add flights to the catalog
flights.createOrReplaceTempView("flights")

# Examine the tables in the catalog again
print(my_spark.catalog.listTables())


# In[ ]:


# Add duration_hrs
flights = flights.withColumn("duration_hrs", flights.air_time/60)


# In[ ]:


flights.toPandas().shape[0]


# #### duration_hrs histogram

# In[ ]:


flights.limit(flights.toPandas().shape[0]).toPandas()["duration_hrs"].hist()


# https://github.com/Bergvca/pyspark_dist_explore/

# In[ ]:


get_ipython().system('pip install pyspark_dist_explore')
# https://github.com/Bergvca/pyspark_dist_explore/


# In[ ]:


import numpy as np
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Read a dataframe
file_path = "../input/planes.csv"

# Create spark_temp from pd_temp
planes = my_spark.read.csv(file_path, header=True)

# Rename year column
planes = planes.withColumnRenamed("year", "plane_year")

# Join the DataFrames
model_data = flights.join(planes, on="tailnum", how="leftouter")


# In[ ]:


# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
model_data = model_data.withColumn("month", model_data.month.cast("integer"))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast("integer"))


# In[ ]:


# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)


# In[ ]:


# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.cast("integer"))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")


# In[ ]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

# Create a StringIndexer
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol="carrier_index", outputCol="carrier_fact")


# In[ ]:


# Create a StringIndexer
dest_indexer = StringIndexer(inputCol="dest", outputCol="dest_index")

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_fact")


# In[ ]:


# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")


# In[ ]:


# Import Pipeline
from pyspark.ml  import  Pipeline

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])


# In[ ]:


# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)


# In[ ]:


piped_data.toPandas().head(3)


# In[ ]:


# Split the data into training and test sets
training, test = piped_data.randomSplit([.7, .3])


# In[ ]:


# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression Estimator
lr = LogisticRegression()


# In[ ]:


# Import the evaluation submodule
import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")


# In[ ]:


# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()


# In[ ]:


# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )


# In[ ]:


# Call lr.fit()
best_lr = lr.fit(training)

# Print best_lr
print(best_lr)


# In[ ]:


# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))


# In[ ]:




