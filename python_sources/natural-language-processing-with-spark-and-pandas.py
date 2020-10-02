#!/usr/bin/env python
# coding: utf-8

# **Natural Languague Processing NLP using Spark and Pandas**
# 
# I created this notebook to do a short demostration about this library called NLP. 
# I found this exercise really fun and beginner friendly.
# 
# We are going to analize some dataset from Reddit and figure out what are the most common words. 
# Just to clarify, this dataset is really small and it works just for practice but you can apply the same methods to some others datasets too. 
# 
# To use this notebook you need to install 
# * pyspark
# * spark-nlp
# * pandas
# 
# You can do it just running the following code in Jupyter Notebook:
# 

# In[ ]:


get_ipython().system('pip install pyspark')
get_ipython().system('pip install spark-nlp==2.0.1')
get_ipython().system('pip install pandas')


# Import `pandas` Library and set the column width to 800. 

# In[4]:


import pandas as pd
pd.set_option('max_colwidth', 800)


# Let's create a `SparkSession`. We're going declare a Spark package to use the NLP library and count the most common words from our dataset. 

# In[5]:


from pyspark.sql import SparkSession

spark = SparkSession         .builder         .config("spark.jars.packages", "JohnSnowLabs:spark-nlp:1.8.2")         .getOrCreate()


# Declare a path variable and read the csv files with the `SparkSession` created before. 
# 
# Set a *header* option as true and *csv* format 

# In[6]:


path = '../input/*.csv'
df = spark.read.format('csv').option('header', 'true').load(path)
df.limit(5).toPandas()


# Our objective with this project is count the most common words, so we don't want null comments.
# 
# Let's filter all null rows from the comment column.

# In[7]:


df = df.filter('comment is not null')


# I'm going to create a new DataFrame using * explode * and * split * functions of `pyspark`.
# 
# The purpose of this is create a new column called word, this new column will contain all the words of our comments splited with spaces.

# In[8]:


from pyspark.sql.functions import split, explode, desc

dfWords = df.select(explode(split('comment', '\\s+')).alias('word'))                     .groupBy('word').count().orderBy(desc('word'))

dfWords.printSchema()


# In[9]:


dfWords.orderBy(desc('count')).limit(5).toPandas()


# Our new DataFrame doesn't looks so good, as you can see, we have blank rows, pronouns, etc.
# 
# Our goal is count the relevant words from posts. That's why we are going to use `NLP` library. 
# Natural Languague Processing library will clasify every word from the dataset as Noun, Pronoun, Verbs, etc.

# In[10]:


from com.johnsnowlabs.nlp.pretrained.pipeline.en import BasicPipeline as bp

dfAnnotated = bp.annotate(df, 'comment')
dfAnnotated.printSchema()


# * `text` original text from comment column.
# * `pos.metadata` will contain a key,value for every words.
# * `pos.result` column is an array with a bunch of tags for every word in the DataSet.
# 
# Here is the list of NLP tags https://cs.nyu.edu/grishman/jet/guide/PennPOS.html
# 

# In[11]:


dfPos = dfAnnotated.select("text", "pos.metadata", "pos.result")
dfPos.limit(5).toPandas()


# Let's create a new DataFrame with the `pos` struct

# In[12]:


dfSplitPos = dfAnnotated.select(explode("pos").alias("pos"))
dfSplitPos.limit(5).toPandas()


# I want to count every word with the tag NNP or NNPs which means:
# * NNP	Proper noun, singular 
# * NNPS	Proper noun, plural
# 

# In[13]:


NNPFilter = "pos.result = 'NNP' or pos.result = 'NNPs'"
dfNNPFilter = dfSplitPos.filter(NNPFilter)
dfNNPFilter.limit(10).toPandas()


# I'm going to use selectExpr function to create a new DataFrame with a *word* and *tag* columns

# In[14]:


dfWordTag = dfNNPFilter.selectExpr("pos.metadata['word'] as word", "pos.result as tag")
dfWordTag.limit(10).toPandas()


# Finally, we have our DataSet as we want and we can start counting the most common words. 

# In[15]:


dfCountWords = dfWordTag.groupBy('word').count().orderBy(desc('count'))
dfCountWords.limit(20).toPandas()


# Our DataSet is doens't say so much, but the idea is apply this method to more huge datasets, this can be works as practice to apply in other studies and projects. 
# 
# Please feel free to let me know your thoughts about this and what I can do better for a next exercise. 
# 
# You can reach me on Medium or Github
# 
# * https://github.com/kennycontreras
# * https://medium.com/@kennycontreras

# In[ ]:




