#!/usr/bin/env python
# coding: utf-8

# <table><tr><td><img src="https://spark.apache.org/images/spark-logo-trademark.png"></td><td><img src="https://bento.cdn.pbs.org/hostedbento-prod/blog/20170114_200556_794501_pk-channel-16x9.jpeg"></td></tr></table>

# # Introduction

# For those like myself looking to familiarise themselves with big data tools, here is my data exploration of the PBS Kids dataset using PySpark.   Using PySpark for exploration on a training dataset of only 4 Gb is a case of overkill:  big data tools are typlically used where the dataset won't fit in RAM,otherwise the parallelisation overhead will generally make things slower than single machine toolsets like scitkitlearn.  And due to Spark's 'lazy evaluation" method it's certainly the wrong tool for the job for data exploration where lots of different metrics are required of a dataset.   But it was an opportunity to consolidate skills from the edX subject I've just completed "Big Data Analytics using Spark"  https://courses.edx.org/courses/course-v1:UCSanDiegoX+DSE230x+3T2019/course/.   I don't expect to achieve any great leaderboard result or push the boundaries of machine learning alorithims, rather to practise my PySpark skills.

# **You should run this notebook on a local machine** - Kaggle only provides 4.9G of scratch memory, too small for Pyspark as it tends to chew through memory, espcially when doing 'windows' functions.  I've had to use normal Pandas dataframes for the visualisation part of the notebook so it will run on Kaggle.
# 
# Worth noting  that the PBS competion doesn't allow kernels with internet access, and hence pyspark.  But you could run Pyspark offline and pickle the resulting models for use in the competition kernels.  
# 
# This notebook covers initial exploration, and I'll post others on feature engineering and on modelling.  I've based the analysis on the popular kernel "Data Science Bowl 2019 data exploration"  https://www.kaggle.com/erikbruin/data-science-bowl-2019-data-exploration by Erkin Bruin https://www.kaggle.com/erikbruin.  I've run the results on mt laptop in both Erkin's sensible single machine approach and in Pyspark, both to ensure accurancy and compare run-times.  All commentary is Eric's verbatim, unless noted with "[MH]".   

# ## Load modules and set up Spark context

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


import os

import pandas as pd
import sklearn as sk
import math
import psutil
from time import time
import calendar
import json

import seaborn as sns
import matplotlib.style as style
style.use('fivethirtyeight')


from pyspark.sql import SparkSession 
from pyspark.sql.functions import col,unix_timestamp,to_date,min,max,isnull,count,when
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType,TimestampType
import pyarrow.parquet as pq
get_ipython().run_line_magic('pylab', 'inline')



# In[ ]:


#Initialise the Spark context
os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

NumCores=4 #Kaggle offers 4 CPU cores/threads.  Change for local machine


Spark = SparkSession.builder.master(f'local[{int(NumCores)}]').appName("PBS_Kids_Spark").config("spark.executor.memory", "4g") .config("spark.driver.memory", "14g") .config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","10g").config("spark.driver.maxResultSize",0).config("spark.sql.execution.arrow.enabled",True).getOrCreate()


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Load data to DataFrames\n\nTrainDf=Spark.read.csv(\'../input/data-science-bowl-2019/train.csv\',header=True,quote=\'"\',escape=\'"\') #quote and escape options required to parse double quotes\nTrainlabelsDf=Spark.read.csv(\'../input/data-science-bowl-2019/train_labels.csv\',header=True,quote=\'"\',escape=\'"\')\nTestDf=Spark.read.csv(\'../input/data-science-bowl-2019/test.csv\',quote=\'"\',header=True,escape=\'"\')\n\n#Load smaller files as panda Dfs\nSpecsDf=pd.read_csv(\'../input/data-science-bowl-2019/specs.csv\')\nsample_submissionDf=pd.read_csv(\'../input/data-science-bowl-2019/sample_submission.csv\')')


# [MH] Comparison with pandas method (as per "Data Science Bowl 2019 Data Exploration" kernel):  Spark:  2.7s; pandas 46.1s.  Not really apples-to-apples comparison due to lazy evaluation in Spark

# # Understand the training data

# In[ ]:


get_ipython().run_cell_magic('time', '', "#What is the shape of the data?\nprint(f'rows :{TrainDf.count()}, columns: {len(TrainDf.columns)}')\n#I considered using countApprox to speed up, but the required conversion to rdd slowed things down")


# [MH] Comparison with pandas method (as per "Data Science Bowl 2019 Data Exploration" kernel):  Spark:  8.6 s; pandas 21.9 microseconds.  

# So we have 11 million rows and just 11 columns. However, Kaggle provided the following note: Note that the training set contains many installation_ids which never took assessments, whereas every installation_id in the test set made an attempt on at least one assessment.
# 
# As there is no point in keeping training data that cannot be used for training anyway, I am getting rid of the installation_ids that never took an assessment

# In[ ]:


get_ipython().run_cell_magic('time', '', 'TrainDf.createOrReplaceTempView("Train")\nkeepidDf=Spark.sql(f\'SELECT installation_id from Train WHERE type="Assessment"\').dropDuplicates()\nkeepidDf.createOrReplaceTempView("keepid")\nColumns=\',\'.join([\'Train.\'+a for a in TrainDf.columns])\nTrainDf=Spark.sql(f\'SELECT {Columns} from Train INNER JOIN keepid ON Train.installation_id=keepid.installation_id\')\\\n.repartition(NumCores) \n#repartition to ensure data is evenly spread to workers after the filter')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#convert timeestamp field to datetime.  \nTrainDf=TrainDf.withColumn(\'timestamp\',unix_timestamp(col(\'timestamp\'), "yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'").cast("timestamp"))')


# [MH] Check for null values -there are 1,740 in timestamp.  A tiny fraction for 11m rows - let's drop them

# In[ ]:


get_ipython().run_cell_magic('time', '', 'NullDf=TrainDf.agg(*[count(when(isnull(c),c)).alias(c) for c in TrainDf.columns])\nNullDf.show()')


# In[ ]:


TrainDf=TrainDf.na.drop()


# In[ ]:


get_ipython().run_cell_magic('time', '', "print(f'rows :{TrainDf.count()}, columns: {len(TrainDf.columns)}')")


# As you can see we have dropped about 3 million rows

# [MH] Comparison with pandas method (as per "Data Science Bowl 2019 Data Exploration" kernel):  Spark:  17.4s; pandas 5.1s.  

# In[ ]:


print(f'rows :{keepidDf.count()}, columns: {len(keepidDf.columns)}')


# The number of unique installations in our "smaller" train set is now 4242.

# I will first visualize some of the existing columns.  

# In[ ]:


get_ipython().run_cell_magic('time', '', '\'\'\'\'We want to put the data in a pandas dataframe in order to do graphs etc.  \nThe most memory and time efficient method to convert from Spark to Pandas is via a parquet file save\nand read via PyArrow.  But Kaggle machines doen\'t have sufficient memory for this, so I\'ve commented that code out and used a normal Pandas dataframe load of the .csv source\n\'\'\'\n\n# TrainDf.write.mode("overwrite").save(\'trainDf.parquet\')  #Uncomment if using local machine\n# TrainPdDf=pq.read_table(\'trainDf.parquet\').to_pandas()\n\nTrainPdDf=pd.read_csv(\'../input/data-science-bowl-2019/train.csv\', parse_dates= [\'timestamp\']) #comment out if using local machine')


# [MH] Time to populate a Pandas dataframe:  Comparison with pandas method (as per "Data Science Bowl 2019 Data Exploration" kernel):  Spark:  52s; pandas 46.1s.  

# In[ ]:


plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(211)
ax1 = sns.countplot(y="type", data=TrainPdDf, color="blue", order = TrainPdDf.type.value_counts().index)
plt.title("number of events by type")

ax2 = fig.add_subplot(212)
ax2 = sns.countplot(y="world", data=TrainPdDf, color="blue", order = TrainPdDf.world.value_counts().index)
plt.title("number of events by world")

plt.tight_layout(pad=0)
plt.show()


# In[ ]:


plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(12,10))
se = TrainPdDf.title.value_counts().sort_values(ascending=True)
se.plot.barh()
plt.title("Event counts by title")
plt.xticks(rotation=0)
plt.show()


# [MH]  Check if any installation IDs are overrepresented - likely webbots

# In[ ]:


plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(12,10))
se = TrainPdDf.installation_id.value_counts().sort_values(ascending=False).head(200)
se.plot.bar()
plt.title("Event counts by installation id (top 200)")
plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'Counts=TrainDf.groupBy(\'installation_id\').agg(count(\'installation_id\').alias(\'NumEvents\'))\nCounts.select("NumEvents").describe().show()\nprint(f\'50% and 90% quartile :{Counts.approxQuantile(["NumEvents"],[0.5,0.9],0.05)}\')  #Use approxQuantile rather than Quantile for speed')


# [MH] Given the median number of events per id is ~1.3k and 90% are less than ~7k events, it seems safe to assume that >15k events are  webbots - or else children with way too much screentime ;-)  Let's remove

# In[ ]:


get_ipython().run_cell_magic('time', '', 'Counts.createOrReplaceTempView("Counts")\nwebbotsDf=Spark.sql(f\'SELECT * from Counts WHERE NumEvents>15000\').select(\'installation_id\')\nprint(f\'Number of suspected webbots: {webbotsDf.count()}\')\nNotWebbotsDf=Spark.sql(f\'SELECT * from Counts WHERE NumEvents<=15000\').select(\'installation_id\')\nNotWebbotsDf.registerTempTable("NotWebbots")  \nTrainDf.createOrReplaceTempView("Train")  \n\nColumns=\',\'.join([\'Train.\'+a for a in TrainDf.columns])\nTrainDf=Spark.sql(f\'SELECT {Columns} from Train \\\nINNER JOIN NotWebbots ON Train.installation_id=NotWebbots.installation_id\')\\\n.repartition(NumCores) \n#repartition to ensure data is evenly spread to workers after the filter')


# I will now add some new columns based on the timestamp, and visualize these.

# In[ ]:


def get_time(df):
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df
train = get_time(TrainPdDf)


# In[ ]:


fig = plt.figure(figsize=(12,10))
se = train.groupby('date')['date'].count()
se.plot()
plt.title("Event counts by date")
plt.xticks(rotation=90)
plt.show()


# When looking at the day of the week, we see no major difference. Of course, we are talking about kids who don't have to go to work ;-)

# In[ ]:


fig = plt.figure(figsize=(12,10))
se = train.groupby('dayofweek')['dayofweek'].count()
se.index = list(calendar.day_abbr)
se.plot.bar()
plt.title("Event counts by day of week")
plt.xticks(rotation=0)
plt.show()


# When looking at the numbers by hour of the day, I find the distribution a little bit strange. Kids seem up late at night and don't do much early in the morning. Has this something to do with time zones perhaps?  

# In[ ]:


fig = plt.figure(figsize=(12,10))
se = train.groupby('hour')['hour'].count()
se.plot.bar()
plt.title("Event counts by hour of day")
plt.xticks(rotation=0)
plt.show()


# # Understanding the test set

# In[ ]:


get_ipython().run_cell_magic('time', '', "#What is the shape of the data?\nprint(f'rows :{TestDf.count()}, columns: {len(TestDf.columns)}')")


# In[ ]:


TestDf.select('installation_id').dropDuplicates().count()


# So we have 1.1 million rows on a thousand unique installation_ids in the test set. Below, you can see that we have this same amount of rows in the sample submission. This means that there are no installation_ids without assessment in the test set indeed.

# In[ ]:


sample_submissionDf.shape[0]


# Another thing that I would like to check is if there is any overlap with regards to installation_id's in the train and test set. As you can see, there are no installation_id's that appear in both train and test.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'TestDf.createOrReplaceTempView("Test")\nSpark.sql(f\'SELECT Train.title from Train INNER JOIN Test ON Train.installation_id=Test.installation_id\').count()')


# [MH] Comparison with pandas method (as per "Data Science Bowl 2019 Data Exploration" kernel):  Spark: 10.4s; pandas 310ms.  

# What about the date ranges?

# In[ ]:


get_ipython().run_cell_magic('time', '', '#convert timeestamp field to datetime.  \nTestDf=TestDf.withColumn(\'timestamp\',unix_timestamp(col(\'timestamp\'), "yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'").\\\n                         cast(TimestampType()))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'TestDates=TestDf.select(to_date(TestDf[\'timestamp\']).alias(\'date\'))\nTrainDates=TrainDf.select(to_date(TrainDf[\'timestamp\']).alias(\'date\'))\nTest_min_date, Test_max_date = TestDates.select(min("date"), max("date")).first()\nTrain_min_date, Train_max_date = TrainDates.select(min("date"), max("date")).first()\nprint(f\'The date range in train is: {Train_min_date} to {Train_max_date}\')\nprint(f\'The date range in test is: {Test_min_date} to {Test_max_date}\')')


# [MH] Comparison with pandas method (as per "Data Science Bowl 2019 Data Exploration" kernel): Spark: 24.3s; pandas 5.9s

# The date range is more or less the same, so we are talking about a dataset that seems (randomly) split on installation_id. Well actually "sort of" as Kaggle seems to have done this on installation_id's with assessments first, and added the "left-overs" with no assessments taken to the train set.

# ## Understanding and visualizing the train labels

# The outcomes in this competition are grouped into 4 groups (labeled accuracy_group in the data):
# 
# 3: the assessment was solved on the first attempt
# 
# 2: the assessment was solved on the second attempt
# 
# 1: the assessment was solved after 3 or more attempts
# 
# 0: the assessment was never solved
# 
# I started by visualizing some of these columns

# In[ ]:


get_ipython().run_cell_magic('time', '', '\'\'\'\'We want to put the data in a pandas dataframe in order to do graphs etc.  \nThe most memory and time efficient method to convert from Spark to Pandas is via a parquet file save\n\'\'\'\n\n# TrainlabelsDf.write.mode("overwrite").save(\'TrainlabelsDf.parquet\')  #uncomment if using local machine\n# TrainlabelsPdDf=pq.read_table(\'TrainlabelsDf.parquet\').to_pandas()\n\nTrainlabelsPdDf=pd.read_csv(\'../input/data-science-bowl-2019/train_labels.csv\') #comment out if using local machine')


# In[ ]:


plt.rcParams.update({'font.size': 22})

plt.figure(figsize=(12,6))
sns.countplot(y="title", data=TrainlabelsPdDf, color="blue", order = TrainlabelsPdDf.title.value_counts().index)
plt.title("Counts of titles")
plt.show()


# Below, you can see that a lot of Chest Sorter assessments were never solved. Bird Measurer also seems hard with a relatively small amount solved on the first attempt.

# In[ ]:


plt.rcParams.update({'font.size': 16})

se = TrainlabelsPdDf.groupby(['title', 'accuracy_group'])['accuracy_group'].count().unstack('title')
se.plot.bar(stacked=True, rot=0, figsize=(12,10))
plt.title("Counts of accuracy group")
plt.show()


# As the match between the train dataframe and the train_labels dataframe is not straightforward, it tried to figure out how these dataframes are to be matched by focussing on just one particular installation_id.

# In[ ]:


TrainlabelsPdDf[TrainlabelsPdDf.installation_id == "0006a69f"]


# From Kaggle: The file train_labels.csv has been provided to show how these groups would be computed on the assessments in the training set. Assessment attempts are captured in event_code 4100 for all assessments except for Bird Measurer, which uses event_code 4110. If the attempt was correct, it contains "correct":true.
# 
# However, in the first version I already noticed that I had one attempt too many for this installation_id when mapping the rows with the train_labels for. It turns out that there are in fact also assessment attemps for Bird Measurer with event_code 4100, which should not count (see below). In this case that also makes sense as this installation_id already had a pass on the first attempt

# In[ ]:


get_ipython().run_cell_magic('time', '', 'Spark.sql(f\'SELECT {Columns} from Train WHERE event_code = 4100 AND installation_id = "0006a69f"\\\nAND title == "Bird Measurer (Assessment)"\').toPandas()')


# [MH] Comparison with pandas method (as per "Data Science Bowl 2019 Data Exploration" kernel): Spark: 11.6s; pandas 780 ms

# When we exclude the Bird Measurer/4100 rows we get the correct match with the numbers in train_labels for this installation_id (4 correct, 12 incorrect)

# Now the question arises: Could there be installation_id's who did assessments (we have already taken out the ones who never took one), but without results in the train_labels? As you can see below, yes there are 628 of those.
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'TrainlabelsDf.createOrReplaceTempView("Trainlabel")\nUniqueTrainlabelsDf=Spark.sql(f\'SELECT installation_id from Trainlabel\').dropDuplicates()\nUniqueTrainlabelsDf.createOrReplaceTempView("UnqTrainlabel")\nkeepidDf.createOrReplaceTempView("keepid")  #we created a list of unique train ids earlier\nSpark.sql(f\'SELECT keepid.installation_id as one, UnqTrainlabel.installation_id as two FROM keepid \\\nLEFT JOIN UnqTrainlabel ON keepid.installation_id=UnqTrainlabel.installation_id \\\nWHERE UnqTrainlabel.installation_id IS NULL\').count()')


# [MH] Comparison with pandas method (as per "Data Science Bowl 2019 Data Exploration" kernel): Spark: 22.4s; pandas 288 ms

# As we can not train on those installation_id's anyway, I am taking them out of the train set. This reduces our train set further from 8.3 million rows to 7.7 million.

# In[ ]:


get_ipython().run_cell_magic('time', '', "Columns=','.join(['Train.'+a for a in TrainDf.columns])\nTrainDf=Spark.sql(f'SELECT {Columns} from Train \\\nINNER JOIN UnqTrainlabel ON Train.installation_id=UnqTrainlabel.installation_id')\\\n.repartition(NumCores) \n#repartition to ensure data is evenly spread to workers after the filter")


# [MH] Comparison with pandas method (as per "Data Science Bowl 2019 Data Exploration" kernel): Spark: 40.6s; pandas 100s

# Check if game_session alone is the unique identifier in train_labels 

# In[ ]:


Count1=TrainlabelsDf.count()
Count2=TrainlabelsDf.select('game_session').dropDuplicates().count()
print(f'Number of rows in train_labels: {Count1}')
print(f'Number of unique game_sessions in train_labels: {Count2}')


# [MH] And so the exploration is done.  Save the parquet files for use in the next phase:  feature engineering

# In[ ]:


#Uncomment these if running on local machine.  Don't need to save in Kaggle, will load data into subsequent notebooks
# TrainDf.write.mode("overwrite").save('TrainDf.parquet')
# TestDf.write.mode("overwrite").save('TestDf.parquet')
# TrainlabelsDf.write.mode("overwrite").save('TrainlabelsDf.parquet')


# In[ ]:




