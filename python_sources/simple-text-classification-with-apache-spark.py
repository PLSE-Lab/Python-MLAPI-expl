#!/usr/bin/env python
# coding: utf-8

# ### Simple Text Processing and Classification with Apache Spark
# ---
# The aim of this notebook is to practise basic text processing using the Apache Spark with the use of the toxic comment text classification dataset. The machine learning and text processing used here are at a poor standard. The goal was mainly to convert the column `comment_text` into a column of sparse vectors for use in a classification algorithm in the spark `ml` library.  

# The `pyspark.ml` library is used for machine learning with Spark DataFrames. For machine learning with Spark RDDs use the `pyspark.mllib` library. 

# In[ ]:


import pandas as pd

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression


# In[ ]:


# Build a spark context
hc = (SparkSession.builder
                  .appName('Toxic Comment Classification')
                  .enableHiveSupport()
                  .config("spark.executor.memory", "4G")
                  .config("spark.driver.memory","18G")
                  .config("spark.executor.cores","7")
                  .config("spark.python.worker.memory","4G")
                  .config("spark.driver.maxResultSize","0")
                  .config("spark.sql.crossJoin.enabled", "true")
                  .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                  .config("spark.default.parallelism","2")
                  .getOrCreate())


# In[ ]:


hc.sparkContext.setLogLevel('INFO')


# In[ ]:


hc.version


# Unfortunately, as much as I love the addition of the csv reader in Spark version 2+ and the databricks spark-csv package, I was unable to use the packages to parse a multiline multi-character quoted record in a csv. As a result, I loaded the data into a DataFrame using Pandas, and then I converted the Pandas DataFrame to a Spark DataFrame.

# In[ ]:


def to_spark_df(fin):
    """
    Parse a filepath to a spark dataframe using the pandas api.
    
    Parameters
    ----------
    fin : str
        The path to the file on the local filesystem that contains the csv data.
        
    Returns
    -------
    df : pyspark.sql.dataframe.DataFrame
        A spark DataFrame containing the parsed csv data.
    """
    df = pd.read_csv(fin)
    df.fillna("", inplace=True)
    df = hc.createDataFrame(df)
    return(df)

# Load the train-test sets
train = to_spark_df("../input/train.csv")
test = to_spark_df("../input/test.csv")


# In[ ]:


out_cols = [i for i in train.columns if i not in ["id", "comment_text"]]


# In[ ]:


# Sadly the output is not as  pretty as the pandas.head() function
train.show(5)


# In[ ]:


# View some toxic comments
train.filter(F.col('toxic') == 1).show(5)


# In[ ]:


# Basic sentence tokenizer
tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
wordsData = tokenizer.transform(train)


# In[ ]:


# Count the words in a document
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
tf = hashingTF.transform(wordsData)


# In[ ]:


tf.select('rawFeatures').take(2)


# In[ ]:


# Build the idf model and transform the original token frequencies into their tf-idf counterparts
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(tf) 
tfidf = idfModel.transform(tf)


# In[ ]:


tfidf.select("features").first()


# Do a test first to practise with the LogisticRegression class. I like to create instances of objects first tocheck their methods and docstrings and figure out how to access data.
# 
# Build a logistic regression model for the binary toxic column.
# Use the features column (the tfidf values) as the input vectors, `X`, and the toxic column as output vector, `y`.

# In[ ]:


REG = 0.1


# In[ ]:


lr = LogisticRegression(featuresCol="features", labelCol='toxic', regParam=REG)


# In[ ]:


tfidf.show(5)


# In[ ]:


lrModel = lr.fit(tfidf.limit(5000))


# In[ ]:


res_train = lrModel.transform(tfidf)


# In[ ]:


res_train.select("id", "toxic", "probability", "prediction").show(20)


# In[ ]:


res_train.show(5)


# #### Select the probability column
# ---
# Create a user-defined function (udf) to select the second element in each row of the column vector

# In[ ]:


extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())


# In[ ]:


(res_train.withColumn("proba", extract_prob("probability"))
 .select("proba", "prediction")
 .show())


# ### Create the results DataFrame
# ---
# Convert the test text

# In[ ]:


test_tokens = tokenizer.transform(test)
test_tf = hashingTF.transform(test_tokens)
test_tfidf = idfModel.transform(test_tf)


# Initialize the new DataFrame with the id column

# In[ ]:


test_res = test.select('id')
test_res.head()


# Make predictions for each class

# In[ ]:


test_probs = []
for col in out_cols:
    print(col)
    lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=REG)
    print("...fitting")
    lrModel = lr.fit(tfidf)
    print("...predicting")
    res = lrModel.transform(test_tfidf)
    print("...appending result")
    test_res = test_res.join(res.select('id', 'probability'), on="id")
    print("...extracting probability")
    test_res = test_res.withColumn(col, extract_prob('probability')).drop("probability")
    test_res.show(5)


# In[ ]:


test_res.show(5)


# In[ ]:


test_res.coalesce(1).write.csv('./results/spark_lr.csv', mode='overwrite', header=True)


# The output is actually a directory and not a csv file. Within the directory there is one or more csv files, which together make up the entire csv results. I used the cat function to concatenate these csv files together.

# In[ ]:


get_ipython().system('cat results/spark_lr.csv/part*.csv > spark_lr.csv')


# In[ ]:


ls


# This submission scores 0.8797 on the public leaderboard.
