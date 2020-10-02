#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# installing pyspark
get_ipython().system('pip install pyspark')


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Loading spark context

# In[ ]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
# Creating spark session containing spark context
#sc = SparkContext(appName = "Santander_Customer_Satisfaction")
spark = SparkSession.Builder().getOrCreate()


# # Reading data from files

# In[ ]:


# Loading data


train = pd.read_csv('../input/santander-customer-satisfaction/train.csv')

test = pd.read_csv('../input/santander-customer-satisfaction/test.csv')


# In[ ]:


# Loading data to spark session
train_spark = spark.read.format("csv").option("header", "true").load('../input/santander-customer-satisfaction/train.csv')
test_spark = spark.read.format("csv").option("header", "true").load('../input/santander-customer-satisfaction/test.csv')


# # Data exploration

# #### Spark mixed with some python

# In[ ]:


# displaying first 5 rows
train_spark.toPandas().head(5)


# In[ ]:


# Looking at the distribution of the target column 
total_datapoints = len(train["TARGET"])
print(train["TARGET"].value_counts()/total_datapoints)
plt.figure(figsize=(6,4))
sns.barplot(y=train["TARGET"].value_counts()/total_datapoints, x=["0","1"])
sns.despine()
plt.suptitle("Distribution of TARGET column in percentage",fontsize=18)
plt.title("blue=0 (satisfied), orange=1 (unsatisfied)")

# 0 = Satisfied, 1= unsatisfied
# Alot more satisfied customers, about 96%


# In[ ]:


# Getting stats on each column
train_spark.describe().toPandas()


# Min value: -999999 for var3 as value seems a bit wierd!

# In[ ]:


train.var3.value_counts()


# In[ ]:


# Checking distribution of rows with feature "var3" = -999999
plt.figure(figsize=(6,4))
sns.barplot(y=train.loc[train.var3 == -999999].TARGET.value_counts()/len(train.loc[train.var3 == -999999]),x=["0","1"])
sns.despine()
plt.suptitle("Distribution for var3=-999999")
plt.title("count of TARGET=0 (blue) and TARGET=1 (orange)")


# The var3=-999999 seems to have a different distribution than the overall dataset
# when looking at the TARGET variable -> may contain information. So we keep those rows.
# OBS, subject of improvement, should be investigated further.
# 

# In[ ]:


print("Checking for nan-values:")
print(train.isnull().values.any())


# In[ ]:


# Assuming ID is not correlated with customer satisfaction
#train = train.drop(["ID"], axis=1)


# In[ ]:


# Assuming ID is not correlated with customer satisfaction so i drop it
train_spark_drop_id = train_spark.drop('ID')
#train_spark_drop_id.toPandas()


# The dataset is heavily skew towards satisfied customers.

# # Undersampling of data due to imbalanced target distribution

# #### Spark

# In[ ]:


# Creating one data frame for each class
train_spark_target_0 = train_spark_drop_id.filter("TARGET=0")
train_spark_target_1 = train_spark_drop_id.filter("TARGET=1")

# Counting the number of samples for each of them
num_target_0 = train_spark_target_0.count()
num_target_1 = train_spark_target_1.count()


# Downsampling the dataset of TARGET=0 to same about amount of rows as TARGET=1
# OBS. This function does not sample exact amount, subject for improvement
train_spark_target_0_under = train_spark_target_0.sample(True, num_target_1/num_target_0)

# Concatenating the undersampled with TARGET=0 and the ordinary TARGET=1
train_under_spark = train_spark_target_0_under.union(train_spark_target_1)

print("Precentage of each class after under sampling")
print(train_under_spark.toPandas()["TARGET"].value_counts()/train_under_spark.count())


# #### Python

# In[ ]:


#count_class_0, count_class_1 = train.TARGET.value_counts()
#train_target_0 = train[train['TARGET'] == 0]

#train_target_1 = train[train['TARGET'] == 1]

#train_target_0_under = train_target_0.sample(count_class_1)
#train_under =  pd.concat([train_target_0_under, train_target_1], axis=0, ignore_index=True)

#total_datapoints_under = len(train_under["TARGET"])
#print("Precentage of each class after under sampling")
#print(train_under["TARGET"].value_counts()/total_datapoints_under)
#train_under


# # Data cleaning
# ## Removing constant columns

# #### Spark

# In[ ]:


# Calculating the amount of unique values for each column
## This could maybe be solved in a better way than to cast as panda. Casting seems quite heavy
constant_in_train_spark= train_under_spark.toPandas().apply(lambda x: x.nunique(), axis=0)

# Extracting the list of the columns with only one unique value
c_train_ind_spark = list(constant_in_train_spark[constant_in_train_spark == 1].index.values)

# Dropping all the columns 
## below '*'' is used to sen the list as arguments to drop since it could not take a list as input
train_drop_1_spark = train_under_spark.drop(*c_train_ind_spark)
print('Number of cols dropped: ', len(c_train_ind_spark))
print(c_train_ind_spark)


# #### Python

# In[ ]:


# checking if some column is constant
# axis=0 : applies to all columns
#constant_in_train = train_under.apply(lambda x: x.nunique(), axis=0)
#c_train_ind = constant_in_train[constant_in_train == 1].index.values
#print('Number of cols dropped: ', len(c_train_ind)))
#print(list(c_train_ind))

#train_drop_1 = train_under

# Dropping constant columns
#train_drop_1.drop(list(c_train_ind), axis=1)


# ## Checking for highly correlated features

# #### Spark

# In[ ]:


# Spark
#Calculating the correlation matrix . From https://stackoverflow.com/questions/51831874/how-to-get-correlation-matrix-values-pyspark/51834729
from pyspark.mllib.stat import Statistics

# copy df except TARGET
df_corr = train_drop_1_spark.drop("TARGET")
# copying columns
col_names = df_corr.columns
# Creatting an rdd of all features
features = df_corr.rdd.map(lambda row: row[0:])
#Calculating correlation
corr_mat=Statistics.corr(features, method="pearson")
# Creating a data frame from result
corr_matrix_spark = pd.DataFrame(corr_mat)
# Setting column names of datafram
corr_matrix_spark.index, corr_matrix_spark.columns = col_names, col_names

corr_matrix_spark


# #### Python

# In[ ]:


# Calculating correlation matrix for all features
#corr_matrix = train_drop_1.corr()
#corr_matrix


# ## Dropping all features with correlation higher than 0.9

# #### Spark

# In[ ]:


cols_to_remove_spark = []

# Looping through 
for col in range(len(corr_matrix_spark.columns)):
    for row in range(col):
        if (corr_matrix_spark.iloc[row,col] >0.5             or corr_matrix_spark.iloc[row,col] < -0.5)             and (corr_matrix_spark.columns[row] not in cols_to_remove_spark):
                
            cols_to_remove_spark.append(corr_matrix_spark.columns[col])

train_drop_2_spark = train_drop_1_spark.drop(*cols_to_remove_spark)

print("Columns removed:")
print(len(cols_to_remove_spark))


# #### Python

# In[ ]:


#cols_to_remove = []
#for col in range(len(corr_matrix.columns)):
#    for row in range(col):
#        if (corr_matrix.iloc[row,col] >0.9 or corr_matrix.iloc[row,col] < -0.9) and (corr_matrix.columns[row] not in cols_to_remove):
#            cols_to_remove.append(corr_matrix.columns[col])

#train_drop_2 = train_drop_1.drop(cols_to_remove, axis=1)

#print("Columns removed:")
#print(len(cols_to_remove))


# ## Removing same columns for test data

# #### Spark

# In[ ]:


train_no_target_cols_spark = train_drop_2_spark.columns[0:-1]
test_ids = test_spark.toPandas()["ID"].values
test_remove_spark = test_spark.select(*train_no_target_cols_spark)
#test_remove_spark.toPandas()


# In[ ]:





# #### Python

# In[ ]:


#Removing same columns from test

#train_no_target_cols = train_drop_2.columns[0:-1]

#test_remove = test[train_no_target_cols]
#test_remove


# # Transforming from spark DF to RDD

# In[ ]:


# Creating RDD from panda
#from pyspark.mllib.regression import LabeledPoint

#training set
#s_df_train = spark.createDataFrame(train_drop_2.sample(300))
#RDD_train = s_df_train.rdd.map(lambda x: LabeledPoint(x["TARGET"], x[:-1]))


#test set
#s_df_test = spark.createDataFrame(test_remove)
#RDD_test = s_df_test.rdd.map(lambda x: x[:])


# In[ ]:


from pyspark.mllib.regression import LabeledPoint

RDD_train_spark = train_drop_2_spark.rdd.map(lambda x: LabeledPoint(x["TARGET"], x[:-1]))
RDD_test_spark = test_remove_spark.rdd.map(lambda x: x[:])


# # ML-part Random forest

# In[ ]:


from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

# Creating random forest model
model = RandomForest.trainClassifier(RDD_train_spark, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=200, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=15, maxBins=32, seed=12345)
#print(model.toDebugString())


# In[ ]:


# Predicting test values
predictions = model.predict(RDD_test_spark).collect()
print(predictions[0:100])


# In[ ]:


# Creating a datafram to submit results on test set
submission_df = pd.DataFrame(columns=["ID", "TARGET"])
submission_df["TARGET"] = predictions
submission_df["ID"] = test_ids
submission_df


# In[ ]:


# WRiting results to csv-files
submission_df.to_csv('santandersubmission_corma_test.cvs', index=False)


# # Final Score on test data : 0.75 

# In[ ]:





# In[ ]:




