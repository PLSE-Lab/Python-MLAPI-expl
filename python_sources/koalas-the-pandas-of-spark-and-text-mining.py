#!/usr/bin/env python
# coding: utf-8

# # Intro to Koalas for Apache Spark 3.0.0
# <center>
# <table>
#     <tr>
# <td><img src="https://raw.githubusercontent.com/databricks/koalas/master/icons/koalas-logo.png" alt="Koalas" width="100"/></td>
# <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Heart_coraz%C3%B3n.svg/1200px-Heart_coraz%C3%B3n.svg.png" alt="Love" width="100"/></td>
# <td><img src="https://miro.medium.com/max/3494/1*Qw6DOZnKIGXzkxa7iZk2uQ.png" alt="Love" height="300" width="200"/></td>
#     </tr>
# </table>    
# </center>
# 
# Apache Spark has not been used as much as other dataframe libraries such as pandas mostly because of it's difficult and different API for the spark dataframe, but it is an powerful instrument for Big Data that each data scientist must know. Koalas is a library created by databricks to improve and make data scientist more productive by mapping Pandas's API on top of Spark, Pandas is, as of today, the most used DataFrame implementation in python.
# 
# In this notebook I will try to introduce Koalas, Pandas-like API for Apache Spark, and I will be using spark 3 (recentely released on 18 June), hitting two objectives with one stone.
# Of course there are some differences and I will try to cover them as well, Koalas is meant to be an introduction for new users, once you are in Spark world you might want to explore their APIs too.
# 
# <center><h2 style="color:red"> UPVOTE if you like this kernel! :)</h2></center>

# # Installing and Importing the libraries
# First thing first we need to upgrade the pyspark versione to 3.0.0 and install the koalas library, which is pretty easy, and we don't need to manage the java version problem because spark 3 now is compatible also with Java 11 (the one used by kaggle)!!

# In[ ]:


get_ipython().system('java -version')


# In[ ]:


get_ipython().system('pip install --upgrade --quiet pyspark==3.0.0')

get_ipython().system('pip install --quiet koalas')


# Import the libraries here and in the next cell I configure the Spark Session, as you can see I commented out the conf part, it's becuase I use the default configurations, but wanted to show that you can configure them, all you need to add is `.config(conf=conf)` in the SparkSession object creation. You can change the driver IP from `local[*]` to it's IP, btw `*` indicated to use all available processors it can be changed to a number if you don't want it, for example `local[4]` will use only 4 processors.

# In[ ]:


import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', 20)

import databricks.koalas as ks

import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as sqlF
from pyspark import SparkContext, SparkConf


# In[ ]:


#conf = SparkConf().setAll([('spark.executor.memory', '5g'), 
#                           ('spark.driver.memory','5g'),
#                           ('spark.driver.maxResultSize','0')])


spark = SparkSession             .builder.master('local[*]')            .appName("TutorialApp")            .getOrCreate()

sqlContext = SQLContext(sparkContext=spark.sparkContext, 
                        sparkSession=spark)


# Reading is pretty easy with all three APIs, but you can already start seeing that koalas one is practically identical to the pandas API, while the spark one seems to be a little with more verbose.

# In[ ]:


train_pandas = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
train_koalas = ks.read_csv("../input/tweet-sentiment-extraction/train.csv", escape="_")

train_spark  = spark.read.csv("../input/tweet-sentiment-extraction/train.csv",
                             inferSchema="true", header="true", escape="_")

# another ways of attaching koalas api to spark dataframe
#train_koalas = train_spark.to_koalas()


# In[ ]:


#This will be useful to show similarity between pandas and koalas dataframes
class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)


# # Basic Operations
# 
# ## Show/head
# 
# Here we see how to print the first 5 rows of the dataframe, you can see that in spark we use `.show(n_rows)` with with koalas and pandas we use the usual `.head(n_rows)`, while the output format is different between spark and the other two, the actual rows can change between koalas and padas when the dataframe is splitted into chunks (it might show any top 5 rows).

# In[ ]:


print("train_spark.show(5)")
train_spark.show(5)
display("train_koalas.head(5)","train_pandas.head(5)")


# ## dtypes and columns
# The dtypes for spark is different from the other two, spark doesn't have an object dtype for columns while koalas is very similar to pandas in this case.
# You can also see the result of `.columns` is different between spark and the other two.

# In[ ]:


print("Spark API:")
print(train_spark.dtypes)
print()
print("Koalas API:")
print(train_koalas.dtypes)
print()
print("Pandas API:")
print(train_pandas.dtypes)


# In[ ]:


print("Spark API:")
print(train_spark.columns)
print()
print("Koalas API:")
print(train_koalas.columns)
print()
print("Pandas API:")
print(train_pandas.columns)


# ## sort_index and sort_values
# Let's see how sorting works, both index and value wise.

# In[ ]:


print("SPARK doesn't use any index so there is no sort by index!")
display("train_koalas.sort_index(ascending=False).head(5)", "train_pandas.sort_index(ascending=False).head(5)")


# In[ ]:


print("train_spark.sort('text', ascending=False).show(5)")
train_spark.sort("text", ascending=False).show(5)
display("train_koalas.sort_values(by='text',ascending=False).head(5)", 
        "train_pandas.sort_values(by='text',ascending=False).head(5)")


# ## GroupBy
# Groupby has the same syntax in almost all three cases, 

# In[ ]:


print("train_spark.groupBy('sentiment').count().orderBy(sqlF.col('count').desc()).show()")
train_spark.groupBy("sentiment").count()    .orderBy(sqlF.col("count").desc())    .show()
display("train_koalas.groupby('sentiment')[['textID']].count().sort_values('textID', ascending=False)",
        "train_pandas.groupby('sentiment')[['textID']].count().sort_values('textID', ascending=False)")


# ## NaN values
# Here is one big difference, pandas removes the NaN values during the groupBy while spark doesn't. Personally I think it's better to keep the NaNs and remove them on demand. Let's see if there are any NaN and remove them.

# In[ ]:


#to drop
spark_dropna  = train_spark.na.drop()
koalas_dropna = train_koalas.dropna()
pandas_dropna = train_pandas.dropna()


# In[ ]:


#to fill
spark_fillna  = train_spark.na.fill('missing text')
koalas_fillna = train_koalas.fillna('missing text')
pandas_fillna = train_pandas.fillna('missing text')


# In[ ]:


print("INITIALLY")
train_spark.filter(train_spark.text.isNull()).show()
print("AFTER DROP NA")
spark_dropna.filter(spark_dropna.text.isNull()).show()
print("AFTER FILL NA")
spark_fillna.filter(spark_fillna.text == "missing text").show()


# # SQL on Spark
# One of the greates advantage of spark is that if you know SQL very well you can "convert" the dataframe to a tempView in SQL and use SQL queries to perform data transformation etc. we repeat some of the comand above using SQL: 

# In[ ]:


# First Create a tempView
train_spark.createOrReplaceTempView("train")


# In[ ]:


spark.sql("SELECT * FROM train").show(5)


# In[ ]:


spark.sql("""
            SELECT sentiment, count(*) AS total 
            FROM train 
            GROUP BY sentiment 
            ORDER BY total DESC
          """).show(5)


# In[ ]:


spark.sql("""
            SELECT * 
            FROM train 
            WHERE text IS NULL
          """).show(5)


# # Plots
# Let's see how to plot using koalas, it's pretty easy for mostly all kind of plot, you transform the data for the necessary plot and with the `.plot()` using the `kind=type_of_plot` plot what you need.

# In[ ]:


data = train_koalas.groupby('sentiment')['textID'].count()

data.plot(kind="bar", figsize=(5,4))
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()


# If you want to use any other library, you can still to it, assuming the data to be a pandas dataframe or series, just use `.to_numpy()` method to convert the result in numpy array which can be plotted. Overall pretty easy in my opinion :).
# 

# In[ ]:


sns.barplot(x=data.index.to_numpy(), y=data.to_numpy())
plt.ylabel("Frequency")
plt.xlabel("Sentiment")
plt.xticks(rotation=45)
plt.show()


# ## Some WordCloud
# Every One wants to do them!

# In[ ]:


import wordcloud
import re
words = koalas_dropna.to_spark().rdd.flatMap(lambda x: re.split("\s+",x[2]))                  .map(lambda word: (word, 1))                  .reduceByKey(lambda a, b: a + b)

schema = StructType([StructField("words", StringType(), True),
                 StructField("count", IntegerType(), True)])

words_df = sqlContext.createDataFrame(words, schema=schema)


# In[ ]:


print("Total number of words:")
words_df.groupBy().sum("count").show()


# In[ ]:


words_df.groupBy('words')        .agg(sqlF.mean("count")/195177)        .orderBy(sqlF.desc("(avg(count) / 195177)"))        .show(50)

word_cloud = words_df.orderBy(sqlF.desc("count"))                     .limit(200)                     .toPandas()                     .set_index('words')                     .T                     .to_dict('records')


# In[ ]:


wc = wordcloud.WordCloud(background_color="white", max_words=200)
wc.generate_from_frequencies(dict(*word_cloud))

plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation='bilinear')
plt.show()


# # Some Text Mining :)
# It's pretty simple we will create a very simple model using TF-IDF with unigrams (given the data quantity, if we ahd more data usually a bigram or trigram model works better.) of the data and perform a holdout validation performance check using a LinearSVC with OneVsRest technique for multiclass classification. We will see also how the Pipelines in spark works, we will convert the koalas dataframe to spark dataframe for the training using `df.to_spark()` where df is the koalas dataframe. You can also use other libraries at this point, to convert from koalas or spark to pandas just use `df.toPandas()`, there are way to speed up the latter process, may be will show it some other time.

# In[ ]:


from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, NGram
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC, OneVsRest


# In[ ]:


tokenizer = RegexTokenizer(inputCol="selected_text", outputCol="words", pattern="\\W")
ngram = NGram(inputCol="words", outputCol="n-gram").setN(1) #Unigram
tf = CountVectorizer(inputCol="n-gram", outputCol="tf")
idf = IDF(inputCol="tf", outputCol="features", minDocFreq=3)
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexer = StringIndexer(inputCol="sentiment", outputCol="sentiment_index")

pipeline = Pipeline(stages=[tokenizer, ngram, tf, idf, indexer])


# In[ ]:


tf_idf = pipeline.fit(spark_dropna)
training_data = tf_idf.transform(koalas_dropna.to_spark()).select("features","sentiment_index")


# In[ ]:


train, valid = training_data.randomSplit([0.7, 0.3], seed=41)
svc = LinearSVC()

classifierMod = OneVsRest(classifier=svc, featuresCol="features",
                         labelCol="sentiment_index")

model = classifierMod.fit(train)


# In[ ]:


valid_prediction = model.transform(valid)
train_prediction = model.transform(train)


evaluator = MulticlassClassificationEvaluator(labelCol="sentiment_index", 
                                              predictionCol="prediction")
print("Train Accuracy achieved:",round(evaluator.evaluate(train_prediction.select("sentiment_index","prediction"), {evaluator.metricName: "accuracy"}),3))
print("Valid Accuracy achieved:",round(evaluator.evaluate(valid_prediction.select("sentiment_index","prediction"), {evaluator.metricName: "accuracy"}),3))


# We achieved a 81% accuracy on validation set in this very simple model, without any hyper parameter optimization or feature selection, the model is overfitting on the trainset there is difference of almost 11% in terms of accuracy between the training and validation, but still not a bad model!

# In[ ]:


#close the spark session when done
spark.stop()


# # Conclusions
# 
# Here we saw how can spark be used in the Text Mining Sector with a very simple model, there are some external library like spark-nlp (not yet updated for spark 3, but works very well on  spark 2.4.x) to do some advanced stuff as well such as bert embedding, WordNet embeddings etc. This is the first version of the spark tutorial, will update this notebook in future to include some more cool stuff such as plots with Koalas API, more complex examples some more ML with may be some Parameter optimization and Cross Validation.
