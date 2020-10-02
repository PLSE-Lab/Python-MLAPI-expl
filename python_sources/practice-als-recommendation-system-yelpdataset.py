#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input/yelp-dataset"))
from pyspark import SparkContext, SparkConf, StorageLevel, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


# In[ ]:


# initialize spark session
spark = SparkSession.builder         .appName("Big Data Project")         .master("local[4]")         .config("spark.logConf", True)         .getOrCreate()
sc = spark.sparkContext
sc
sqlContext = SQLContext(sc)


# # Data Loading

# In[ ]:



# review_df = sqlContext.read.json("yelp_dataset/yelp_review.json")
review_df = spark.read.csv('../input/remove-text-column-of-review/output.csv', header=True)
business_df = sqlContext.read.json("../input/yelp-dataset/yelp_academic_dataset_business.json")


# # Data - Reduce Sparsity

# In[ ]:


# Filter Restaurant businesses
business_df = business_df.filter("categories like '%Restaurant%'").select(col("business_id"))

# Filter out shops/users that have very few (<10) reviews to forcefully restrict sparsity.
business_review_count = review_df.groupBy('business_id').                agg(F.count(review_df.stars).alias("biz_count"))

biz_with_10_ratings_or_more = business_review_count.filter("biz_count>=10")

user_review_count = review_df.groupBy('user_id').                agg(F.count(review_df.stars).alias("user_count"))

user_with_10_ratings_or_more = user_review_count.filter("user_count>=10")

# merge with review_df
review_df = business_df.join(review_df, "business_id", "inner")
review_df = biz_with_10_ratings_or_more.join(review_df, "business_id", "inner")
review_df = user_with_10_ratings_or_more.join(review_df, "user_id", "inner")


# In[ ]:


# assert schema for feeding into ALS
review_df = review_df.select(col("user_id"), col("business_id"), col("stars"))
print(review_df.schema)


# # ID Transformation for Feeding into ALS

# In[ ]:


user_num_df = review_df.select('user_id').distinct()
biz_num_df = review_df.select('business_id').distinct()


# In[ ]:


# converting business id and user id with to numerical index, as required by ALS
from pyspark.sql.window import Window as W
# user index
user_num_df = user_num_df.withColumn("idx", monotonically_increasing_id())
w = W.orderBy("idx")
user_num_df = user_num_df.withColumn("user_index", F.row_number().over(w))
user_num_df = user_num_df.drop("idx")
# business index
biz_num_df = biz_num_df.withColumn("idx1", monotonically_increasing_id())
w = W.orderBy("idx1")
biz_num_df = biz_num_df.withColumn("business_index", F.row_number().over(w))
biz_num_df = biz_num_df.drop("idx1")


# In[ ]:


review_df = review_df.withColumn("stars", review_df["stars"].cast(IntegerType()))
review_df = review_df.join(biz_num_df, "business_id")
review_df = review_df.join(user_num_df, "user_id")
print(review_df.schema)


# # Baseline Models

# In[ ]:


# split dataset into train, validation and test
(training, validation, test) = review_df.randomSplit([0.6,0.2,0.2], seed=42)
# cache these datasets for performance
training.cache()
validation.cache()
test.cache()

review_df.unpersist()
business_df.unpersist()
biz_num_df.unpersist()
user_num_df.unpersist()
biz_with_10_ratings_or_more.unpersist()
user_with_10_ratings_or_more.unpersist()

review_df = None
business_df = None
biz_num_df = None
user_num_df = None
biz_with_10_ratings_or_more = None
user_with_10_ratings_or_more = None


# In[ ]:


# Baseline model 1 - average rating overall
training_avg_rating = training.agg(avg(training.stars)).collect()[0][0]
# add a column with the average rating
test_for_avg_df = test.withColumn('prediction', F.lit(training_avg_rating))
evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars", predictionCol="prediction")
# get RMSE
test_avg_RMSE_1 = evaluator.evaluate(test_for_avg_df)
print("The baseline 1 RMSE is {0}".format(test_avg_RMSE_1))
#training_avg_rating.unpersist()
test_for_avg_df.unpersist()
training_avg_rating = None
test_for_avg_df = None
evaluator = None


# In[ ]:


# Baseline model 2 - average rating per business
training_avg_rating = training.groupBy('business_id').agg(avg(training.stars).alias("prediction"))
# add a column with the average rating
test_for_avg_df = test.join(training_avg_rating, "business_id")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars", predictionCol="prediction")
# get RMSE
test_avg_RMSE_2 = evaluator.evaluate(test_for_avg_df)
print("The baseline 2 RMSE is {0}".format(test_avg_RMSE_2))
training_avg_rating.unpersist()
test_for_avg_df.unpersist()
training_avg_rating = None
test_for_avg_df = None
evaluator = None


# In[ ]:


# Baseline model 3 - average rating per user
training_avg_rating = training.groupBy('user_id').agg(avg(training.stars).alias("prediction"))
# add a column with the average rating
test_for_avg_df = test.join(training_avg_rating, "user_id")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars", predictionCol="prediction")
# get RMSE
test_avg_RMSE_3 = evaluator.evaluate(test_for_avg_df)
print("The baseline 3 RMSE is {0}".format(test_avg_RMSE_3))
training_avg_rating.unpersist()
test_for_avg_df.unpersist()
training_avg_rating = None
test_for_avg_df = None
evaluator = None


# # Training

# In[ ]:


def train_ALS(rank, maxit, reg, train_df, test_df, seed = 42):
    als = ALS(maxIter = maxit,
              regParam = reg,
              rank = rank,
              userCol="user_index", 
              itemCol="business_index", 
              ratingCol="stars",
              coldStartStrategy="drop")

    # set the parameters for the method
    als.setSeed(seed)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars", predictionCol="prediction")
    # Create the model with these parameters.
    model = als.fit(train_df)
    # Run the model to create a prediction. Predict against the validation_df.
    predict_df = model.transform(test_df)
    error = evaluator.evaluate(predict_df)
    als = None
    model = None
    predict_df.unpersist()
    predict_df = None
    evaluator = None
    return(error)

numIterations = [5, 10, 15, 20]
ranks = [2, 4, 6, 8]
regs = list(np.arange(0.05, 0.4, 0.05))

RMSE_results = []
best_RMSE = 20
for rank in ranks:
    for maxit in numIterations:
        for reg in regs:
            error = train_ALS(rank = rank, maxit = maxit, reg = reg,train_df = training, test_df = validation)
            RMSE_results.append(error)
#             print(error)
#             print(maxit)
#             print(reg)
#             print(rank)
            if best_RMSE > error:
                best_RMSE = error
                best_maxit = maxit
                best_reg = reg
                best_rank = rank
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(range(112),RMSE_results, 'bs', range(112),np.full((112), test_avg_RMSE_1), 'r--',range(112),np.full((112), test_avg_RMSE_2), 'g--',range(112),np.full((112), test_avg_RMSE_3), 'y--')
RMSE_results.to_csv('RMSE_results.csv')


# In[ ]:


# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter = best_maxit,
          regParam = best_reg,
          rank = best_rank,
          userCol="user_index", itemCol="business_index", ratingCol="stars",
          coldStartStrategy="drop", seed = 42)
model = als.fit(training)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars", predictionCol="prediction")
# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))


# # Prediction

# In[ ]:


# Generate top 10 business recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each business
bizRecs = model.recommendForAllItems(10)
# Generate top 10 business recommendations for a specified set of users
users = review_df.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of businesses
bizs = review_df.select(als.getItemCol()).distinct().limit(3)
bizSubSetRecs = model.recommendForItemSubset(bizs, 10)
# $example off$
userRecs.show()
bizRecs.show()
userSubsetRecs.show()
bizSubSetRecs.show()

