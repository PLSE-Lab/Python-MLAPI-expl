#!/usr/bin/env python
# coding: utf-8

# ## Apache Spark : The Unified Analytics Engine
# 
# The largest open source project in data processing framework that can do ETL, analytics, machine learning and graph processing on large volumes of data at rest (batch processing) or in motion (streaming processing) with rich high-level APIs for the programming languages like Scala, Python, Java and R.
# 
# Spark has seen immense growth over the past several years. Hundreds of contributors working collectively have made Spark an amazing piece of technology powering the de facto standard for big data processing and data sciences across all industries.
# 
# Internet powerhouses such as Netflix, Yahoo, and eBay have deployed Spark at massive scale, collectively processing multiple petabytes of data on clusters of over 8,000 nodes.

# ## Why Spark ?
# 
# Typically when you think of a computer you think about one machine sitting on your desk at home or at work. This machine works perfectly well for applying machine learning on small dataset . However, when you have huge dataset(in tera bytes or giga bytes), there are some things that your computer is not powerful enough to perform. One particularly challenging area is data processing. Single machines do not have enough power and resources to perform computations on huge amounts of information (or you may have to wait for the computation to finish).
# 
# A cluster, or group of machines, pools the resources of many machines together allowing us to use all the cumulative resources as if they were one. Now a group of machines alone is not powerful, you need a framework to coordinate work across them. Spark is a tool for just that, managing and coordinating the execution of tasks on data across a cluster of computers.

# ## Spark Architecture
# 
# Apache Spark allows you to treat many machines as one machine and this is done via a master-worker type architecture where there is a driver or master node in the cluster, accompanied by worker nodes. The master sends work to the workers and either instructs them to pull to data from memory or from disk (or from another data source).
# 
# **Read more about Architecture**
# 
# https://spark.apache.org/docs/latest/cluster-overview.html

# ## Bank Marketing - Data Exploration, transformation & Modeling
# I am using freely available databricks stand alone community edition server (https://community.cloud.databricks.com) as Spark library currently not available in Kaggle directly.
# 
# The below are the summary of this exercise:
# 
# 1. Read this dataset in pyspark, say, as df 
# 2. Carry out the folowing operations on df :
# 
# 	a) Cache this dataset
# 	b) Show first 5 rows
# 	c) Count number of rows
# 	d) Print its data structure
# 
# 3. From this dataset, select just the following columns and overwrite  the previous dataframe, df :
# 
# ['age', 'job', 'marital', 'education', 'default',  'balance', 
# 'housing', 'loan', 'contact', 'duration',  'campaign',
#  'pdays', 'previous', 'poutcome', 'deposit']
# 
# 4. The list of categorical columns is as follows:
# 
# catCols = ['job', 'marital', 'education', 'default',
#                      'housing', 'loan', 'contact', 'poutcome']
# 
# Create a pipeline to transform each one of the categorical
# columns to as many Onehotencoded columns	by first using
#  StringIndexer and then OneHotEncoder. 
# 
# 5. Use VectorAssembler to aseemble all the OneHotEncoded columns and the following numerical columns in one column. Call this new assembled column as: 'rawFeatures' :
# 
# numericCols = ['age', 'balance', 'duration',  'campaign', 'pdays', 'previous']
# 
# 6. Print the 'rawFeatures' column
# 
# Here is Spark code for LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, Gradient-boosted tree classifier, NaiveBayes & Support Vector Machine on Bank marketing dataset
# 
# *Databricks Notebook :*
# 
# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/7066241366603179/2873197913727922/4729409196487951/latest.html
# 

# **How to increase accuracy of a model ?**
# 
# 
# * Add new features or drop existing features and train model.
# * Tune ML algorithm (https://spark.apache.org/docs/latest/ml-tuning.html)

# ## Reference
# 
# https://docs.databricks.com/spark/latest/gentle-introduction/gentle-intro.html
# 
# https://docs.databricks.com/spark/latest/gentle-introduction/gentle-intro.html#gentle-introduction-to-apache-spark
# 
# https://docs.databricks.com/spark/latest/gentle-introduction/for-data-scientists.html

# 
