#!/usr/bin/env python
# coding: utf-8

# # Hello spark.ml
# This is a notebook that shows an example of using pyspark for data analysis and modeling. This notebook will use regression methods to predict home prices. It is not practical to use pyspark for this dataset because it is so small, but hopefully it should serve as a guide on how to complete a simple machine learning project with spark.
# ![Spark Logo](https://spark.apache.org/images/spark-logo-trademark.png)

# ## Setup
# The following code will set up the environment with the required software
# * Java 8 (already installed)
# * Spark (already installed)
# * pyspark (to be installed, [make sure internet is enabled](https://www.kaggle.com/questions-and-answers/36982))

# In[ ]:


## Make sure internet is enabled!
get_ipython().system('pip install pyspark==2.4')


# Importing libraries

# In[ ]:


# Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# PySpark imports
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


# #### Create spark session using local resources

# In[ ]:


# 'local[*]' means use all available cores in the local machine
spark = SparkSession     .builder     .appName("Hello spark.ml")     .master('local[*]')     .getOrCreate()


# In[ ]:


spark


# ## Read in Data
# Data can be read in with an inferred schema. For big data it would be recommended to define the schema explicitly because inferring the schema requires two passes reading the data.
# 
# The dataframe is read into only one partition by default because it is so small. The data is repartitioned into four partitions for demonstation purposes.

# In[ ]:


train_fname = r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv'
test_fname = r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv'

df = spark.read.csv(train_fname, header=True, inferSchema=True, nullValue='NA').repartition(4).persist()
df_test = spark.read.csv(test_fname, header=True, inferSchema=True, nullValue='NA').repartition(4).persist()


# In[ ]:


df.count(), df_test.count()


# ## Analyze the Target Variable
# 
# The target variable is `SalePrice`

# Compute basic summary statistics with the `summary` function

# In[ ]:



df.select('SalePrice').summary("count", "mean", "stddev", 
                               "min", "5%", "25%", "50%", "75%", "95%", "max").show()


# #### Plot a histogram of the variable.
# This is an example of converting a subset of the pyspark dataframe to a pandas dataframe to do a statistical function. Although it is possible to compute histograms in a pyspark dataframe, it is not as simplistic as using pandas.

# In[ ]:


df.select('SalePrice').sample(False, 0.5).toPandas().hist()


# ## Data Prep

# In[ ]:


# Convert MSSubClass to a string because it is a categorical field
df = df.withColumn('MSSubClass', F.col('MSSubClass').cast('string'))
df_test = df_test.withColumn('MSSubClass', F.col('MSSubClass').cast('string'))


# ## Analyze Input Variables

# #### Single Variable Analysis with Target

# In[ ]:


nominal_fields = ['MSSubClass',
 'MSZoning',
 'Street',
 'Alley',
 'LandContour',
 'Utilities',
 'LotConfig',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrType',
 'Foundation',
 'Heating',
 'CentralAir',
 'Electrical',
 'Functional',
 'GarageType',
 'PavedDrive',
 'MiscFeature',
 'SaleType',
 'SaleCondition']

ordinal_fields = [
 'LotShape',
 'LandSlope',
 'OverallQual',
 'OverallCond',
 'ExterQual',
 'ExterCond',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'HeatingQC',
 'KitchenQual',
 'FireplaceQu',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PoolQC',
 'Fence'   
]

categorical_fields = [
 'MSSubClass'
]

count_fields = [
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageCars'
]

timeseries_fields = [
 'MoSold',
 'YrSold'
]

continuous_fields = [
 'LotFrontage',
 'LotArea',
 'YearBuilt',
 'YearRemodAdd',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'GarageYrBlt',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'MiscVal',
 ]

target_fields = [
 'SalePrice'
]

id_fields = [
    'Id'
]


# In[ ]:


def describe_categorical(df, column, target='SalePrice', numRows=20):
    df.groupby(column).agg(F.count('ID').alias('count'), 
                           F.round(F.mean(target)).alias('mean'), 
                           F.round(F.stddev(target)).alias('stddev'),
                           F.min(target).alias('min'), 
                           F.max(target).alias('max')
                          ).orderBy('count', ascending=False).show(numRows)
    


# In[ ]:


for column, typ in df.dtypes:
    print(column)
    describe_categorical(df, column)


# In[ ]:


def scatter_plot(df, x, y='SalePrice', sampling_rate=1.0):
    df.select(x, y).sample(False, sampling_rate).toPandas().plot.scatter(x=x, y=y, title="{} vs {}".format(x,y))


# In[ ]:


for field in continuous_fields:
    scatter_plot(df, field)


# In[ ]:


def box_plot(df, x, y='SalePrice', sampling_rate=1.0):
    pdf = df.select(x, y).sample(sampling_rate).toPandas()
    sns.catplot(x=x, y=y, kind="box", data=pdf)
    plt.gca().set_title("{} vs {}".format(x,y))


# In[ ]:


for field in ordinal_fields:
    box_plot(df, field)


# ## Data Cleaning

# In[ ]:


# Handle Outliers


# ## Feature Engineering

# In[ ]:


#df = df.withColumn('IsNew', F.col('SaleType') == 'New')


# #### Handle missing values

# In[ ]:


# Drop fields with more than 10% missing values

