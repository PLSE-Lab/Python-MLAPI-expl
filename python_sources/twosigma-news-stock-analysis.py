#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import pandas as pd
import numpy as np
from kaggle.competitions import twosigmanews
#from sklearn.naive_bayes import GaussianNB


# In[ ]:


env = twosigmanews.make_env()


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


market_train_df.dropna(inplace = True)


# In[ ]:


market_train_df.reset_index(inplace = True)


# In[ ]:


df_market_1 = market_train_df[['time','assetName','open','close','volume']]
df_news_1 = news_train_df[['time','assetName','sentimentNegative','sentimentPositive']]


# In[ ]:


df_market_1['time_new'] = df_market_1['time'].dt.floor('d')
df_news_1['time_new'] = df_news_1['time'].dt.floor('d')


# In[ ]:


df_news_1.drop(['time'], inplace = True, axis = 1)
df_market_1.drop(['time'], inplace = True, axis =1 )


# In[ ]:


df_market_apple = df_market_1[df_market_1['assetName'] == 'Apple Inc']
df_news_apple = df_news_1[df_news_1['assetName'] == 'Apple Inc']


# In[ ]:


df_market_apple.head()


# In[ ]:


df_news_apple.head()


# In[ ]:


days = env.get_prediction_days()


# In[ ]:


#(market_obs_df, news_obs_df, predictions_template_df) = next(days)


# In[ ]:


next(days)


# In[ ]:


def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0


# In[ ]:


make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)


# In[ ]:


(market_obs_df, news_obs_df, predictions_template_df) = next(days)


# In[ ]:


make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)


# In[ ]:


for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_random_predictions(predictions_template_df)
    env.predict(predictions_template_df)
print('Done!')


# In[ ]:


env.write_submission_file()


# In[ ]:


import os
print([filename for filename in os.listdir('.') if '.csv' in filename])


# In[ ]:


import pandas as pd


# In[ ]:


submissions = pd.read_csv('./submissions.csv')
submissions.head()


# In[ ]:




