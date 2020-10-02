#!/usr/bin/env python
# coding: utf-8

# # Sparkify Project Workspace

# This is a [Udacity nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) project (Data Science Capstone).
# This project uses users' event data from Sparkify, which is an imaninary digital music service similar to Spotify or Pandora, to build a model to predict users' churn.
# 
# The original dataset is 12GB but a small subset of the dataset (128MB) will be used in this notebook.

# In[ ]:


# import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes("pastel")
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')

from pyspark.sql import SparkSession, Window

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import sum as Fsum
from pyspark.sql.functions import min as Fmin
from pyspark.sql.functions import max as Fmax
from pyspark.sql.functions import avg, col, concat, count, desc, asc, explode, lit, split, stddev, udf, isnan, when, rank, from_unixtime

from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# In[ ]:


# Create spark session
spark = SparkSession     .builder     .appName("Sparkify")     .getOrCreate()


# In[ ]:


# Read in full sparkify dataset
event_data = "../input/mini_sparkify_event_data.json"
df = spark.read.json(event_data)


# # Load and Clean Dataset
# 
# In this notebook, the file name, `mini_sparkify_event_data.json`, will be loaded and cleaned such as handling of invalid or missing values.

# The first five rows of the dataset.

# In[ ]:


df.head(5)


# Schema information
# 
# * artist: Artist name (ex. Daft Punk)
# * auth: User authentication status (ex. Logged)
# * firstName: User first name (ex. Colin)
# * gender: Gender (ex. F or M)
# * itemInSession: Item count in a session (ex. 52)
# * lastName: User last name (ex. Freeman)
# * length: Length of song (ex. 223.60771)
# * level: User plan (ex. paid)
# * location: User's location (ex. Bakersfield)
# * method: HTTP method (ex. PUT)
# * page: Page name (ex. NextSong)
# * registration: Registration timestamp (unix timestamp) (ex. 1538173362000)
# * sessionId: Session ID (ex. 29)
# * song: Song (ex. Harder Better Faster Stronger)
# * status: HTTP status (ex. 200)
# * ts: Event timestamp(unix timestamp) (ex. 1538352676000)
# * userAgent: User's browswer agent (ex. Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0)
# * userId: User ID (ex. 30)

# In[ ]:


df.printSchema()


# ## Statistics

# Statistics of the whole dataset

# In[ ]:


df.describe().show()


# Statistics of the `artist` column

# In[ ]:


df.describe('artist').show()


# Statistics of the `sessionId` column

# In[ ]:


df.describe('sessionId').show()


# Statistics of the `userId` column

# In[ ]:


df.describe('userId').show()


# Total rows: 286,500

# In[ ]:


df.count()


# All the `page` events in the dataset:
# 
# - About
# - Add Friend
# - Add to Playlist
# - Cancel
# - Cancellation Confirmation: **This even wil be used as a flag of churn**
# - Downgrade
# - Error
# - Help 
# - Home
# - Login 
# - Logout
# - NextSong 
# - Register 
# - Roll Advert
# - Save Settings 
# - Settings
# - Submit Downgrade
# - Submit Registration
# - Submit Upgrade
# - Thumbs Down
# - Thumbs Up
# - Upgrade

# `page` kind

# In[ ]:


df.select("page").dropDuplicates().sort("page").show()


# ## missing values

# In[ ]:


def count_missing(df, col):
    """
    A helper function which count how many missing values in a colum of the dataset.
    
    This function is useful because the data can be either three cases below:
    
    1. NaN
    2. Null
    3. "" (empty string)
    """
    return df.filter((isnan(df[col])) | (df[col].isNull()) | (df[col] == "")).count()


# Check how many missing values in each column

# In[ ]:


print("[missing values]\n")
for col in df.columns:
    missing_count = count_missing(df, col)
    if missing_count > 0:
        print("{}: {}".format(col, missing_count))


# `userId` and `sessionId`
# 
# If the below Ids are null or empty, delete those rows:
# 
# * userId
# * sessionId

# In[ ]:


df_without_missing_id = df.dropna(how = "any", subset = ["userId", "sessionId"])
df_without_missing_id = df_without_missing_id.filter(df["userId"] != "") # `userId` should not be empty string


# In[ ]:


print("df:                    {}".format(df.count()))
print("df_without_missing_id: {}".format(df_without_missing_id.count())) # no missing values

if df.count() == df_without_missing_id.count():
    print("No missing values with userId and sessionId")
else:
    print("{} rows have been removed.".format(df.count() - df_without_missing_id.count()))


# # Exploratory Data Analysis

# Detect number columns and category columns.
# 
# * num_cols: Number columns (Long or Double)
# * cat_cols: Category columns (String)

# In[ ]:


num_cols = []
cat_cols = []

for s in df.schema:
    data_type = str(s.dataType)
    if data_type == "StringType":
        cat_cols.append(s.name)
    
    if data_type == "LongType" or data_type == "DoubleType":
        num_cols.append(s.name)


# In[ ]:


num_cols


# In[ ]:


cat_cols


# ## Number columns

# In[ ]:


df_without_missing_id.describe(num_cols).show()


# There are three HTTP status codes:
# 
# * 307: Temporary Redirect
# * 404: Not Found
# * 200: OK

# In[ ]:


df_without_missing_id.select("status").dropDuplicates().show()


# ### Category columns

# auth

# In[ ]:


df_without_missing_id.select("auth").dropDuplicates().show()


# gender

# In[ ]:


df_without_missing_id.select("gender").dropDuplicates().show()


# level

# In[ ]:


df_without_missing_id.select("level").dropDuplicates().show()


# location (only showing top 10)

# In[ ]:


df_without_missing_id.select("location").dropDuplicates().show(10)


# method

# In[ ]:


df_without_missing_id.select("method").dropDuplicates().show()


# page

# In[ ]:


df_without_missing_id.select("page").dropDuplicates().show()


# userAgent (only showing top 10)

# In[ ]:


df_without_missing_id.select("userAgent").dropDuplicates().show(10)


# ### Define Churn
# 
# Churn will be defined as when `Cancellation Confirmation` events happen, and users with the events are churned users in this analysis.

# churn: `Cancellation Confirmation`

# In[ ]:


df_without_missing_id.filter("page = 'Cancellation Confirmation'").show(10)


# In[ ]:


flag_churned_event = udf(lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
df_churned = df_without_missing_id.withColumn("churned", flag_churned_event("page"))


# churned rate (from total event logs)

# In[ ]:


churned_rate = df_churned.groupby("userId").agg({"churned": "sum"}).select(avg("sum(churned)")).collect()[0]["avg(sum(churned))"]
print("churned: {:.2f}%".format(churned_rate * 100))


# In[ ]:


df_churned.select(["userId", "gender", "level", "page", "status", "ts", "churned"]).show(30)


# In[ ]:


windowval = Window.partitionBy("userId").orderBy(asc("ts")).rangeBetween(Window.unboundedPreceding, 0)
df_phase = df_churned.withColumn("phase", Fsum('churned').over(windowval))
df_churn = df_phase.withColumn("churn", Fmax('churned').over(Window.partitionBy("userId")))


# In[ ]:


df_churn.select(["userId", "gender", "level", "page", "status", "ts", "churned", "phase", "churn"]).show(20)


# In[ ]:


df_churn.filter(df_churn["churn"] == 1).select(["userId", "gender", "level", "page", "status", "ts", "churned", "phase", "churn"]).show(20)


# 52 userIds were churned

# In[ ]:


churned_user_count = df_churn.filter(df_churn["churn"] == 1).select("userId").dropDuplicates().count()
print("churned user count: {} (total: {})".format(churned_user_count, df_churn.count()))
print("churned user rate: {:.2f}%".format(churned_user_count / df_churn.count() * 100))


# ### Explore Data
# 
# In this section, data exploration will be done comparing churned users with not churned users, inspecting if there are any big differences between the two groups.

# The below columns will be examined:
# 
# * artist
#   * [x] the number of artist
# * [x] gender: 0 or 1
# * length
#   * [x] the total length
# * [x] level: 0 or 1
# * page
#   * [x] the number of `Thumbs Up`
#   * [x] the number of `Thumbs Down`
# * song
#   * [x] the number of song

# Define a common function to convert churn value (0 or 1) to `Not Churn` or `Churn`
# 
# Both matplotlib and seaborn plot libraries require pandas dataframe, not pyspark dataframe, so I need to convert the pyspark dataframe to pandas one. I do this conversion every time for a small subset of the dataset because if I do this conversion for all the dataset, it takes time and causes an error.

# In[ ]:


func_churn_label = udf(lambda x: 'Churn' if x == 1 else 'Not Churn')


# In[ ]:


df_churn_user = df_churn.groupby("userId").max("churn").withColumnRenamed("max(churn)", "churn").select(["userId", "churn"])


# gender

# In[ ]:


pd_gender = df_churn.select(["userId", "gender", "churn"]).withColumn("churn", func_churn_label("churn")).toPandas()
pd_gender.head()


# In[ ]:


sns.countplot(x="gender", hue="churn", data=pd_gender);


# level

# In[ ]:


pd_level = df_churn.select(["userId", "level", "churn"]).withColumn("churn", func_churn_label("churn")).toPandas()
pd_level.head()


# In[ ]:


sns.countplot(x="level", hue="churn", data=pd_level);


# artist

# In[ ]:


pd_artist = df_churn_user.join(df_churn.groupby("userId")                                     .agg({"artist": "count"})                                     .withColumnRenamed("count(artist)", "artist_count"), ["userId"])                          .withColumn("churn", func_churn_label("churn")).toPandas()
pd_artist.head()


# In[ ]:


sns.boxplot(x="churn", y="artist_count", data=pd_artist);


# song

# In[ ]:


pd_song = df_churn_user.join(df_churn.groupby("userId")                                      .agg({"song": "count"})                                      .withColumnRenamed("count(song)", "song_count"), ["userId"])                        .withColumn("churn", func_churn_label("churn")).toPandas()
pd_song.head()


# In[ ]:


sns.boxplot(x="churn", y="song_count", data=pd_song);


# length

# In[ ]:


pd_length = df_churn_user.join(df_churn.groupby("userId")                                        .agg({"length": "sum"})                                        .withColumnRenamed("sum(length)", "total_length"), ["userId"])                           .withColumn("churn", func_churn_label("churn")).toPandas()
pd_length.head()


# In[ ]:


sns.boxplot(x="churn", y="total_length", data=pd_length);


# page: total visits

# In[ ]:


pd_visit = df_churn_user.join(df_churn.groupby("userId")                                       .count()                                       .withColumnRenamed("count", "visit_count"), ["userId"])                          .withColumn("churn", func_churn_label("churn")).toPandas()
pd_visit.head()


# In[ ]:


sns.boxplot(x="churn", y="visit_count", data=pd_visit);


# page: Thumbs Up / Thumbs Down

# up

# In[ ]:


pd_up = df_churn_user.join(df_churn.filter((df_churn["page"] == 'Thumbs Up'))                                    .groupby("userId")                                    .count()                                    .withColumnRenamed("count", "up_count"), ["userId"])                      .withColumn("churn", func_churn_label("churn")).toPandas()
pd_up.head()


# In[ ]:


sns.boxplot(x="churn", y="up_count", data=pd_up);


# down

# In[ ]:


pd_down = df_churn_user.join(df_churn.filter((df_churn["page"] == 'Thumbs Down'))                                    .groupby("userId")                                    .count()                                    .withColumnRenamed("count", "down_count"), ["userId"])                      .withColumn("churn", func_churn_label("churn")).toPandas()
pd_down.head()


# In[ ]:


sns.boxplot(x="churn", y="down_count", data=pd_down);


# # Feature Engineering

# ### Feature Engineering Ideas
# 
# * artist
#   * [x] the number of artist
# * [x] gender: 0 or 1
# * length
#   * [x] the total length
# * [x] level: 0 or 1
# * page
#   * [x] the number of `Thumbs Up`
#   * [x] the number of `Thumbs Down`
# * song
#   * [x] the number of song

# In[ ]:


df_churn.show(1)


# Original dataframe to be merged later

# In[ ]:


df_original = df_churn.groupby('userId').max("churn").withColumnRenamed("max(churn)", "target")


# In[ ]:


df_original.show(10)


# artist count per userId

# In[ ]:


user_artist = df_churn.groupby("userId").agg({"artist": "count"}).withColumnRenamed("count(artist)", "artist_count")
user_artist.show(5)


# gender

# In[ ]:


flag_gender = udf(lambda x: 1 if x == "F" else 0, IntegerType())
df_churn_with_gender = df_churn.withColumn("gender", flag_gender("gender"))
df_churn_with_gender.show(1)


# In[ ]:


user_gender = df_churn_with_gender.groupby('userId').agg({"gender": "max"}).withColumnRenamed("max(gender)", "gender")
user_gender.show(5)


# length

# In[ ]:


user_length = df_churn.groupby('userId').agg({"length": "sum"}).withColumnRenamed("sum(length)", "length")
user_length.show(5)


# Page
# 
# * Thumbs Up
# * Thumbs Down

# In[ ]:


user_thumbs_up = df_churn.filter(df_churn["page"] == 'Thumbs Up').groupby('userId').count().withColumnRenamed("count", "thumb_up")
user_thumbs_up.show(5)


# In[ ]:


user_thumbs_down = df_churn.filter(df_churn["page"] == 'Thumbs Down').groupby('userId').count().withColumnRenamed("count", "thumb_down")
user_thumbs_down.show(5)


# level

# In[ ]:


flag_level = udf(lambda x: 1 if x == "paid" else 0, IntegerType())
df_churn_with_level = df_churn.withColumn("level", flag_level("level"))
df_churn_with_level.show(1)


# In[ ]:


user_level = df_churn_with_level.groupby('userId').agg({"level": "max"}).withColumnRenamed("max(level)", "level")
user_level.show(5)


# song count per userId

# In[ ]:


user_song = df_churn.groupby("userId").agg({"song": "count"}).withColumnRenamed("count(song)", "song_count")
user_song.show(5)


# Join all the features

# In[ ]:


merged_df = df_original.join(user_artist, ['userId'])     .join(user_gender, ['userId'])     .join(user_length, ['userId'])     .join(user_level, ['userId'])     .join(user_thumbs_up, ['userId'])     .join(user_thumbs_down, ['userId'])     .join(user_song, ['userId'])


# In[ ]:


merged_df.show(5)


# # Modeling
# 
# 
# In this modeling section, the below tasks will be executed to build models. Three machine learning models will be examined and I will decide one of them based on the evaluation results for further hypyer parameter tuning.
# 
# * scaling
# * train/test split
# * build models
#   * Logistic Regression
#   * Random Forest classifier
#   * GBT classifier
# * evaluate the models based on f1 score since churned users in the dataset are fairly small so the distribution of target variables are extremely biased.

# Drop `userId` column (which is not necessary for modeling)

# In[ ]:


merged_df_without_user = merged_df.drop("userId")
feature_columns = [col for col in merged_df_without_user.columns if col!='target']
feature_columns


# Train/Test split

# In[ ]:


train, test = merged_df_without_user.randomSplit([0.7, 0.3], seed=0)


# Build model

# In[ ]:


def build_model(classifier, param):
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
    pipeline = Pipeline(stages=[assembler, scaler, classifier])

    model = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param,
        evaluator=MulticlassClassificationEvaluator(labelCol='target', metricName='f1'),
        numFolds=5,
    )
    return model


# First, I will try Logistic Regression as a baseline model. Logistic Regression model is much simpler model than other two models. The time needed for training is relatiely shorter than others so it would be a good idea to use this model as a baseline.

# In[ ]:


lr = LogisticRegression(featuresCol="scaled_features", labelCol="target")
param = ParamGridBuilder().build()
model = build_model(lr, param)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit_model = model.fit(train)')


# In[ ]:


pred = fit_model.transform(test)


# All the predicted values are 0 (Not churned)

# In[ ]:


pred.select("prediction").dropDuplicates().collect()


# Even if the predictions are all 0, f1 score is around 0.73 as a result of the imbalanced dataset.
# 
# I've decided to use this score as a baseline result and I will try to create a better model.

# In[ ]:


evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="target")
f1_score = evaluator.evaluate(pred, {evaluator.metricName: "f1"})
print("f1: {}".format(f1_score))


# ### Try different models

# Random Forest

# In[ ]:


rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="target")
rf_param = ParamGridBuilder().build()
rf_model = build_model(rf, rf_param)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf_fit_model = rf_model.fit(train)')


# In[ ]:


rf_pred = rf_fit_model.transform(test)


# In[ ]:


rf_pred.select("prediction").dropDuplicates().collect()


# In[ ]:


rf_f1_score = evaluator.evaluate(rf_pred, {evaluator.metricName: "f1"})
print("f1: {}".format(rf_f1_score))


# GBT

# In[ ]:


gbt =GBTClassifier(featuresCol="scaled_features", labelCol="target")
gbt_param = ParamGridBuilder().build()
gbt_model = build_model(gbt, gbt_param)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'gbt_fit_model = gbt_model.fit(train)')


# In[ ]:


gbt_pred = gbt_fit_model.transform(test)


# In[ ]:


gbt_pred.select("prediction").dropDuplicates().collect()


# In[ ]:


gbt_f1_score = evaluator.evaluate(gbt_pred, {evaluator.metricName: "f1"})
print("f1: {}".format(gbt_f1_score))


# With default parameters (without hyperparameter tuning), Random Forest gives me a better result than that of GBT classifier.
# Let's dig into more on Random Forest.

# ### Feature Importances

# Random Forest

# In[ ]:


rf_feature_importance_df = pd.DataFrame()
rf_feature_importance_df['feature'] = feature_columns
rf_feature_importance_df['importance'] = rf_fit_model.bestModel.stages[2].featureImportances.values.tolist()
rf_feature_importance_df = rf_feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
rf_feature_importance_df


# According to the feature importances provided by the Random Forest model, `Thumbs Up` and `Thumbs Down` seem to be important while the level of the users do not really matter.

# In[ ]:


plt.figure(figsize=(7,7))
sns.barplot(x='importance', y='feature', data=rf_feature_importance_df, color="b")
plt.title('Feature Importance')
plt.ylabel('');


# ### Hyperparameter Tuning

# In this section, hyperparameter tuning will be executed for Random Forest model.

# In[ ]:


classifier = RandomForestClassifier(featuresCol="scaled_features", labelCol="target")

param_grid = ParamGridBuilder()     .addGrid(classifier.maxDepth,[5, 10])     .addGrid(classifier.numTrees, [20, 50])     .addGrid(classifier.minInstancesPerNode, [1, 10])     .addGrid(classifier.subsamplingRate, [0.7, 1.0])     .build()

model_tuned = build_model(classifier, param_grid)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit_model_tuned = model_tuned.fit(train)')


# In[ ]:


best_model = fit_model_tuned.bestModel
best_model.stages[2].save("random_forest_tuned")


# In[ ]:


best_model_pred = best_model.transform(test)


# In[ ]:


best_model_pred.show(5)


# In[ ]:


best_f1_score = evaluator.evaluate(best_model_pred, {evaluator.metricName: "f1"})
print("f1: {}".format(best_f1_score))


# The final result is better than the Random Forest model with default parameters.
# Given that the full dataset (12GB) would provide different results compared with the small subset (128MB), further hyperparameter tuning will be required to gain stable results.
