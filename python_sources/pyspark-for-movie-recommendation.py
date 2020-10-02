#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# In this notebook I am going to show how Spark MLlib can be used for movie recommendation right from Kaggle Kernels.
# 
# Dataset is taken from [this](https://www.kaggle.com/c/megogochallenge) challenge. 
# 
# Task was to recommend 10 movies for each user from test set. 
# Metric is [MAP@10](http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html). 

# ### Why spark? 
# 
# At the moment this competition starts I have no any practical expirience in building recommendation systems. As well as in spark, by the way. 
# 
# Of course I read some great articles on the subject, including [this one](https://medium.com/@james_aka_yale/the-4-recommendation-engines-that-can-predict-your-movie-tastes-bbec857b8223) and [this kernel](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system), but when real task has come I have no idea what to do.
# It was good for me that organizers decided to share some [baselines](https://github.com/SantyagoSeaman/megogo_challenge_solutions). 
# 
# Best of it by score was a [Spark](https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html) one, so after trying to play with first two baselines I decided to reproduce it.
# I have few not really great hours looking how my local jupyter notebook freezes after I launch it. And that's how idea to run it in Kaggle comes to me. 
# 
# I tried to google some kernels, but there was no single one. That's why I write this one.

# ## Installing pyspark

# There is 2 ways to do it here. 
# * Run magic command, as in cell below
# * Or go to Packages and enter pyspark (it will take some time, but this will form your own version of docker and you'll no need to waste time while running next versions of kernel)
# 

# In[1]:


get_ipython().system('pip install pyspark')


# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

get_ipython().run_line_magic('env', 'JOBLIB_TEMP_FOLDER=/tmp')
#https://www.kaggle.com/getting-started/45288 - this helps some with 'no space left on device'


# Now importing all needed spark modules. Pay attention how sc and spark variables were initialized. 

# In[5]:


import pyspark.sql.functions as sql_func
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext('local') #https://stackoverflow.com/questions/30763951/spark-context-sc-not-defined
spark = SparkSession(sc)


# ** let's read data **

# In[6]:


data_schema = StructType([
    StructField('session_start_datetime',TimestampType(), False),
    StructField('user_id',IntegerType(), False),
    StructField('user_ip',IntegerType(), False),
    StructField('primary_video_id',IntegerType(), False),
    StructField('video_id',IntegerType(), False),
    StructField('vod_type',StringType(), False),
    StructField('session_duration',IntegerType(), False),
    StructField('device_type',StringType(), False),
    StructField('device_os',StringType(), False),
    StructField('player_position_min',LongType(), False),
    StructField('player_position_max',LongType(), False),
    StructField('time_cumsum_max',LongType(), False),
    StructField('video_duration',IntegerType(), False),
    StructField('watching_percentage',FloatType(), False)
])
final_stat = spark.read.csv(
    '../input/train_data_full.csv', header=True, schema=data_schema
).cache()


# In[7]:


ratings = (final_stat
    .select(
        'user_id',
        'primary_video_id',
        'watching_percentage',
    )
).cache()


# In[8]:


get_ipython().run_cell_magic('time', '', 'ratings.count()')


# In[9]:


import gc #This is to free up the memory
gc.collect()
gc.collect()


# ** training model **

# In[10]:


get_ipython().run_cell_magic('time', '', 'als = ALS(rank=100, #rank s the number of latent factors in the model (defaults to 10). Higher value - better accuracy (at this competition), longer training\n          maxIter=2, #maxIter is the maximum number of iterations to run (defaults to 10). Higher value - more memory used\n          implicitPrefs=True, #implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data (defaults to false)\n          regParam=1, #regParam specifies the regularization parameter in ALS (defaults to 1.0)\n          alpha=50, #alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations (defaults to 1.0)\n          userCol="user_id", itemCol="primary_video_id", ratingCol="watching_percentage",\n          numUserBlocks=32, numItemBlocks=32,\n          coldStartStrategy="drop")\n\nmodel = als.fit(ratings)')


# ** made predicts **

# In[ ]:


get_ipython().run_cell_magic('time', '', 'userRecsDf = model.recommendForAllUsers(10).cache()\nuserRecsDf.count()')


# In[ ]:


userRecs = userRecsDf.toPandas()
userRecs.shape


# In[ ]:


userRecs[:2]


# In[ ]:


predicted_dict = userRecs.set_index('user_id').to_dict('index')
predicted_dict = {user_id:[r[0] for r in recs['recommendations']] for user_id, recs in predicted_dict.items()}
len(predicted_dict)


# ** reading test data ** 

# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission_full.csv')


# In[ ]:


sample_submission['als_predicted_primary_video_id'] = sample_submission.user_id.apply(
    lambda user_id: ' '.join([str(v) for v in predicted_dict[user_id]]) if user_id in predicted_dict else None)


# In[ ]:


sample_submission[:5]


# In[ ]:


sample_submission['primary_video_id'] = sample_submission.als_predicted_primary_video_id.combine_first(
    sample_submission.primary_video_id)
del sample_submission['als_predicted_primary_video_id']


# In[ ]:


sample_submission.to_csv('sample_submission_full_als.csv',
                         header=True, index=False)


# This will form results very close to shared baseline. With maxIter = 10 it will be almost same. 

# ** now, let's make some very basic approach to proceed with cold users **

# In[ ]:


train_data = pd.read_csv('../input/train_data_full.csv')
train_needed_users =  train_data[train_data.user_id.isin(sample_submission.user_id)]
users_with_history = list(set(train_needed_users.user_id))
cold_users = list(set(sample_submission.user_id) - set(users_with_history))
print('number of users presented in history: ', len(users_with_history), ' % of users with hist data: ',len(users_with_history)/len(sample_submission.user_id))
print('number of cold start users ', len(cold_users), ' % of users without hist data: ', len(cold_users)/len(sample_submission.user_id))


# In[ ]:


top_10_videos = train_data[train_data['watching_percentage']>=0.5].loc[train_data.session_start_datetime >= '2018-09-20 00:00:00', # Supposing that 10 days closest to testing period is most representative
                               'primary_video_id'].value_counts()[:10].index.tolist()


# In[ ]:


sample2 = sample_submission.copy()
sample2['forcold'] = ' '.join([str(v) for v in top_10_videos])


# In[ ]:


sample2.loc[sample2.user_id.isin(cold_users),'primary_video_id'] = sample2['forcold'][1]


# In[ ]:


del sample2['forcold']


# In[ ]:


sample2.head()


# In[ ]:


sample2.to_csv('sparkasl_withcold.csv', #forming another file, with basic processing of cold users 
                         header=True, index=False)


# ## Conclusion
# 
# It's very nice that Kaggle Kernels allow to play out with Spark libs, which is giving quite good result in a competition. 
# Also many thanks to the organizers for sharing baselines. 
# 
# Hope this kernel was useful as very short introduction to Spark MLib for recommendation task.
# 
# Please share your approach for this type of task in discussion. 
