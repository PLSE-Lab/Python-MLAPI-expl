#!/usr/bin/env python
# coding: utf-8

# # The prologue
# 
# Let me start with a situation from my life. When I reached an intermediate level of my datascience journey, I faced with this issue of huge datasets. The datasets are becoming bigger and with my laptop having average specs (I5--7th gen with 12 Gb of Ram), it feels like training of the models takes forever. Especially the memory demanding algorithms such as cross validation is taking heavy toll on my lap and me (as i have to run the algorithm over night and this makes me wakeup several times in the middle of night to check for execution completion).  
# 
# So to tackle this situation I found several methods 
# 
# *) AWS --- Costly but effective --- allows you to assign huge resources for your algorithms
# 
# *) paid Kernels ---- Same as Aws 
# 
# *) ........................
# 
# *) Last but not the least, we have spark --- free open source 
# 
# So this notebook is my Entire Solution for handling huge datasets in data science, in other words a solution for better sleep at night. 
# 

# ## Preknowledge requirement
# 
# 
# * Spark --- Begginer (This notebook is for absolute beginners of spark. For those who use spark already can still take some knowledge as we all think differently)
# 
# * Datascience --- Beginner, intermediate , advanced ( only thing is you should be new/intermediate to spark )
# 
# * Python --- Should love it.

# # Table of Contents
# 
# #### 1). Introduction.
# 
# #### 2). Importing and checking Datasets.
# 
#      2.1) Creating a DataFrame.
#      
#      2.2) Showing the data.
#      
#      2.3) Grabbing the data.
#      
#      2.4) Creating new columns.
# 
# #### 3). Basics operations on dataframes.
# 
#      3.1) Filtering data using SQL Queries.
#   
#      3.2) Filtering data using Dataframe methods.
# 
# #### 4). Handling missing data using spark.
# 
#      4.1) Keeping the missing data.
#   
#      4.2) Dropping the missing data.
#   
#      4.3) Filling the missing data.
# 
# #### 5). How spark works internally.
# 
# #### 6). Conclusion and further materials.
# 
# #### 7). Resources
# 
# 
# Feel free to skip to sections as per your requirements.

# # 1)  Introduction to pyspark
# 
# This is a begginer friendly notebook for people who want to dive into the world of big data and datascience. As the datasets becomes more and more bigger, the need for bigdata technologies such as hadoop, spark etc.. increases.In my research I found spark to be the best one among those technologies. This notebook is written for absolute begginers in spark. This notebook will be accompanied by several notebooks explaining more advanced topics in pyspark.
#      
# 
# Apache Spark is a distributed framework that handle's Big Data analysis. Apache Spark is written in Scala and can be integrated with Python, Scala, R, SQL, Java  languages. Spark is basically a computational engine, that works with huge sets of data by processing them in parallel and batch systems.PySpark is the Python API of Apache Spark.I like to keep the introduction short and want to show how to work using pyspark. For curious minds i'll be explaining how spark works internally in section 7.
# 
# 
# The main advantage of spark is that, it can do various machine learning tasks with ease.The Ml lib library which we see in the upcoming notebooks will help you to deal with datascience problems using huge datasets. 
# 

# # 2) Importing and Checking datasets.
# 
# Post Spark version2.0, Spark introduced dataframes.These are like advanced version of tables with rows and columns,for easy handling of large datasets.
# 
# These dataframes are the same ones which you might have used in python or R with additional properties. Spark DataFrame expand on a lot of these concepts, allowing you to transfer that knowledge easily by understanding the simple syntax of Spark DataFrames.
# 
# ### 2.1) Creating a DataFrame
# 
# First we need to start a SparkSession:
# 
# The below 2 commands are used to import findspark module which actually tells the program where spark folder (the one we downloads during installation) is residing in your computer.
# 

# In[ ]:


#import findspark


# In[ ]:


#findspark.init('/home/davinci/spark-2.4.5-bin-hadoop2.7')


# In[ ]:


get_ipython().system('pip install pyspark  #for installing spark in kaggle kernel')
from pyspark.sql import SparkSession


# Then start the SparkSession

# In[ ]:


# May take a little while on a local computer
spark = SparkSession.builder.appName("Basics").getOrCreate()


# We will first need to get the data from a file (or connect to a large distributed file like HDFS, we'll talk about this later once we move to larger datasets on AWS EC2). Here i'm importing data from a json file

# In[ ]:


#This is a dataset available online 
# Might be a little slow locally
df = spark.read.json('../input/peoplejson1/people.json')


# ### 2.2) Showing the data

# In[ ]:


# Note how data is missing!
df.show()


# Spark will automatically set null value for missing data.

# In[ ]:


df.printSchema()


# when we ran the above command spark understood the schema of the dataframe. This is an handy tool while using some datasets.

# There are data types that makes infering of  schema easier (like csv ). 
# 
# Still there is an option to define our own schema and can read the datasets in that schema.
# 
# Spark has all the tools you need for this, it just requires a very specific structure:

# In[ ]:


from pyspark.sql.types import StructField,StringType,IntegerType,StructType


# We have to create the list of Structure fields
#     * :param name: string, name of the field.
#     * :param dataType: :class:`DataType` of the field.
#     * :param nullable: boolean, whether the field can be null (None) or not.

# In[ ]:


data_schema = [StructField("age", IntegerType(), True),StructField("name", StringType(), True)]
final_struc = StructType(fields=data_schema)
df = spark.read.json('../input/peoplejson1/people.json', schema=final_struc)
df.printSchema()


# In the above piece of code we defined a a list of data schema using the parmaeters. Then by using the structtype we made a schema out of it. Then in 3rd sentense, we used that schema to read our json file. 

# To show the columns

# In[ ]:


df.columns


# To show the rows

# In[ ]:


df.head(2) #by default shows 1 row


# ### 2.3) Grabbing the data

# In[ ]:


print(df['age'])


# As you can see this returns a column object. But it is far less usefull. we can use .select() function to import it as a dataframe.

# In[ ]:


print(df.select('age'))


# In[ ]:


df.select('age').show()


# There are 2 types of functions .show() and .collect()
# 
# *) .show()  : print the data and shows to us
# 
# *) .collect() : can be used to assign data to a variable.

# To select more than one column, we can give the column names as a list

# In[ ]:


df.select(['age','name']).show()


# ### 2.4) Creating new columns
# 
# To create new columns we can use .withcolumn()

# In[ ]:


# Adding a new column which is copied from an old column.
df.withColumn('newage',df['age']).show()


# In[ ]:


# Renaming a column
df.withColumnRenamed('age','supernewage').show()


# More complicated operations to create new columns

# In[ ]:


df.withColumn('doubleage',df['age']*4).show()


# # 3) Basic operations on Dataframes

# ### 3.1) Filtering Data using SQL
# 
# A big ability of spark is quick filtering and searching capacity. Spark also allows us to use sql queries for filtering and other operations. So i'll be showing both sql and ordinary DF methods.

# In[ ]:


# Let Spark know about the header and infer the Schema types!
df = spark.read.csv('../input/applstock/appl_stock.csv',inferSchema=True,header=True)


# In[ ]:


df.show()


# In[ ]:


# Using SQL
df.filter("Close<500").show()


# In[ ]:


# Using SQL with .select()
df.filter("Close<500").select(['Open','Close']).show()


# ### 3.2) Filtering data using normal df methods.

# In[ ]:


# Using normal df methods.
df.filter(df["Close"] < 200).show()


# In[ ]:


# Make sure to add in the parenthesis separating the statements!
df.filter( (df["Close"] < 200) & (df['Open'] > 200) ).show()


# In[ ]:


# Make sure to add in the parenthesis separating the statements!
df.filter( (df["Close"] < 200) | (df['Open'] > 200) ).show()


# In[ ]:


# Make sure to add in the parenthesis separating the statements!
df.filter( (df["Close"] < 200) & ~(df['Open'] < 200) ).show()


# In[ ]:


df.filter(df["Low"] == 197.16).show()


# # 4) Handling Missing Data
# 
# There will be missing data or null data in your datasets. SO you can handle these missing data in three ways
# 
#     *) Keep the missing data points.
#     *) Drop them missing data points.
#     *) Fill them in with some other value.
# 
# 

# ### 4.1) Keeping the missing data
# 
# Here we dont have to do anything. Spark already understands the missing value and gives it null.

# In[ ]:


df = spark.read.csv("../input/containsnull1/ContainsNull.csv",header=True,inferSchema=True)


# In[ ]:


df.show()


# Notice how the data remains as a null.
# 

# ### 4.2) Drop the missing data
# 
# We can yse .na functions for missing data.
# 
# There are 3 parameters.
# 
#     df.na.drop(how='any', thresh=None, subset=None)
#     
#     * param how: 'any' or 'all'.
#     
#         If 'any', drop a row if it contains any nulls.
#         If 'all', drop a row only if all its values are null.
#     
#     * param thresh: int, default None
#     
#         If specified, drop rows that have less than `thresh` non-null values.
#         This overwrites the `how` parameter.
#         
#     * param subset: 
#         optional list of column names to consider.

# In[ ]:


# Drop any row that contains missing data
df.na.drop().show()


# In[ ]:


# Has to have at least 2 NON-null values
df.na.drop(thresh=2).show()


# In[ ]:


df.na.drop(subset=["Sales"]).show()


# In[ ]:


df.na.drop(how='any').show()


# In[ ]:


df.na.drop(how='all').show()


# ### 4.3) Fill the missing values
# 
# We can also fill the missing values with new values. If you have multiple nulls across multiple data types, Spark is actually smart enough to match up the data types. For example:

# In[ ]:


df.na.fill('NEW VALUE').show()


# In[ ]:


df.na.fill(0).show()


# Usually you should specify what columns you want to fill with the subset parameter

# In[ ]:


df.na.fill('No Name',subset=['Name']).show()


# A very common practice is to fill values with the mean value for the column, for example:

# In[ ]:


from pyspark.sql.functions import mean
mean_val = df.select(mean(df['Sales'])).show()


# # 5) How spark works internally.
# 
# 
# 
# You might have thought like why this session is kept here instead of placing it in the beginning. The sole purpose of it is that, this section explains how spark works internally and to understand we should know some terms such as sparkcontest and all which you will be quite familiar by now.
# 
# 
# 
# The whole code you have written for execution is called as driver program. Spark runs this programs on various clusters known as nodes. This Parallelization is the one helping you to achieve greater processing speeds. Spark uses executors for this. If you are using shells and writting the programs in an interactive way, these shells are considered as driver programs by itself. The next thing is sc -- the spark contest (we have used it in our code). This is actually the linkage between the driver program and the multiple executors. So whenever we reads a file the spark contest internally links to all the nodes. 
# 
# 
# 
# This is a basic introduction of how spark works internally.
# 

# # 6) Conclusion
# 
# As you have seen spark is a good way to work with huge datasets.Spark helps to solve the problem of larger processing times by using parallelization approach. 
# 
# The spark topic is huge.I wont be able to show everthing to you in one notebook. My sole purpose of creating this notebook was to introduce you to Pyspark and the basic operations on datasets using pyspark. I hope that is fulfilled. 
# 
# In the upcoming notebooks I'll be diving into the spark ML lib library used for machine learning and will introduce you to various machine learning algorithms using pyspark.
# 
# 1). Linear Regression.
# 
# 
# 2). Logistic Regression.
# 
# 
# 3). Decision Trees & Random forests.
# 
# 
# 4). K-means Clustering.
# 
# 
# 5). Natural Language Processing.
# 
# 
# 
# and more...
# 
# 

# # 7) Resources
# 
# First of all a big thanks to Jose Portilla and his course "Spark and Python for big data using pyspark" in udemy, for lifting of my yspark journe. Great course with lots of valuable materials and examples. 
# 
# https://spark.apache.org/ : The spark website -- Literally have everything we need 
# 
# 
# A lot of youtube videos and websites.
# 
# 
# 
# 
# Hope to meet you soon in next notebook.

# In[ ]:




