#!/usr/bin/env python
# coding: utf-8

# # Big Data
# Big data is just data but big? Well big data can be described using "The Four V's" (other resources have 5 V's or 7 V's). The four V's are as follows: 
# 1. Volume
# 2. Velocity
# 3. Variety
# 4. Veracity 
# 
# Big data is described as large in volume (amount of data e.g. zetabytes), high velocity (streaming data e.g. sensor data or terabytes of trade information), coming as a variety (different forms of data, e.g. videos and tweets) and veracity which is the uncertainty of data (e.g. poor data quality).
# 
# Due to the processing overhead of big data, we need special tools that are optimized for calculations on this size. 

# # Compute Clusters
# 
# Previous section mention the overhead of processing big data. One computer won't do the job of processing large amounts of data classified as 'big data' but you can have multiple computers work together. This is what a **compute cluster** is, a group of computers that work together to do some work. 
# 
# Ok, so how do we manage to have a group of computers to work together to accomplish a task? This managment of work on clusters is actually hard dealing with concurrency, interprocess communication, scheduling, etc. with the addition of dealing distributed systems problems like computer failures or network latency. 

# # Hadoop
# Thankfully we have **Apache Hadoop** is a collection of tools that will assist us for managing clusters. 
# 
# - Yarn: manages compute jobs in the cluster
# - HDFS: (Hadoop Distributed File System), stores data on the cluster's nodes (computers)
# - Spark: a framework to do computation on the data 

# # Get started with Spark
# 1. Download [Spark (2.4.3)](https://spark.apache.org/) (or latest pre-built)
# 2. Set an environment variable (e.g. terminal on OS X). I use Python 3 so I did:
# > export PYSPARK_PYTHON=python3
# 3. Also set the path:
# > export PATH=${PATH}:/home/you/spark-2.4.4-bin-hadoop2.7/bin
# 4. If you run into a 'Py4JJavaError', you may need to install Java or OpenJDK version 8
# 
# These are things I did to set up Spark on my Mac but just Google if these instructions don't work or leave a comment, I can try to help out. Also these instructions are running for spark locally by entering the following in the terminal:
# > spark-submit spark-program.py

# # A Spark Program

# I just clicked 'Add Data' at the top right and picked 'Los Angeles Traffic Collision Dataset' so feel free to switch to another dataset when experiementing. Double check the file type though and switch the spark read method accordingly.

# In[ ]:


import os
input_dir = '../input'
os.listdir(input_dir)
file = 'traffic-collision-data-from-2010-to-present.csv'
path = os.path.join(input_dir,file)
print(path)


# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


import sys
from pyspark.sql import SparkSession, functions, types
 
spark = SparkSession.builder.appName('example 1').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

data = spark.read.csv(path, header=True,
                      inferSchema=True)
data.show()


# Yeah.. it doesn't look pretty. Let's see what we can do. First, we explore some methods with a Spark dataframe.

# In[ ]:


# let's see the schema
data.printSchema()


# In[ ]:


# select some columns
data.select(data['Crime Code'], data['Victim Age']).show()


# In[ ]:


# filter the data
data.filter(data['Victim Age'] < 40).select('Victim Age', 'Victim Sex').show()


# In[ ]:


# write to a json file
json_file = data.filter(data['Victim Age'] < 40).select('Victim Age', 'Victim Sex')
json_file.write.json('json_output', mode='overwrite')


# If you were expecting one json file well no, instead you get a **directory** of multiple json files. The concatenation of those files is the actual output. This is because of the way Spark computes. More on it later.

# In[ ]:


get_ipython().system('ls json_output')


# In[ ]:


# a few more things

# perform a calculation on a column and rename it
data.select((data['Council Districts']/2).alias('CD_dividedBy2')).show()

# rename columns 
data.withColumnRenamed('Victim Sex', 'Gender').select('Gender').show()

# drop columns and a cleaner vertical format for the top 10 
d = data.drop('Neighborhood Councils')
d.show(n=10, truncate=False, vertical=True)


# # Partitioning
# We previously saw the output of json file is resulted with a directory of multiple json files. This is because we said that big data is too big to be processed on one single computer which is why the Apache Hadoop toolset is there to be able to work with data and compute on multiple computers but can come together as one result as if the data was processed on one machine. This is why all Spark dataframes are partitioned this way no matter how small the data is. 
# 
# Ususally you would give an input directory of files as our "data" where each thread/process/core/executor reads an indvidual input file. When creating the output, each write is done in parallel and when each of the output files are combined they form the single output result. That is where HDFS plays a part as the shared filesystem for all of this parallelism to work. 
# 
# YARN is responsbile for managing the computation on each individual computer when actally working with a cluster of nodes. YARN manages the CPU and memory resources. Rather than moving the data to different nodes, YARN can move the compute work to where the data is.
# 
# On the local machine, we just use the local filesytem

# # Conclusion
# This was a very breif overview of Big Data and Spark. I am just studying for my final so I thought might as well write about it and share the knowledge with others as a way of studying. If this was useful let me know and I will continue with more details on more PySpark stuff like how it calculates, grouping data and joining data. Also I am not an expert on this stuff so if I am giving some wrong information let me know.  

# 

# In[ ]:




