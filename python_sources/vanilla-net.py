#!/usr/bin/env python
# coding: utf-8

# # Vanilla Net

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain

get_ipython().run_line_magic('matplotlib', 'inline')

REDUCED = False # Reduce the data size for development and testing


# In[ ]:


def clean_train_data(news_df, market_df):
    '''Clean and preprocess the news and market data for training.
    
    Parameters
    ----------
    news_df : dataframe
        See https://www.kaggle.com/c/two-sigma-financial-news/data for full description of the dataframe.
    market_df : dataframe
        See https://www.kaggle.com/c/two-sigma-financial-news/data for full description of the dataframe.
    
    Returns
    -------
    dataframe 
        Cleaned data ready to be fed to the model.
    
    '''
    # assetCode, time, volume, open, returnsOpenPrevMktres1, 
    # returnsOpenPrevMkres10, returnsOpenNextMktres10
    # sentimentNegative, sentimentNeutral, sentimentPositive
    cols = ['assetCode', 'time', 'volume', 'open', 'returnsOpenPrevMktres1', 
            'returnsOpenPrevMkres10', 'returnsOpenNextMktres10']
    cleaned_df = market_df.loc[cols]
    
    return None


# In[ ]:


#TODO: Add cleaned data specifications
#TODO: Define Returns
def train_model(train_df):
    '''Train the model using the given trianing data.
    
    Parameters
    ----------
    train_data : dataframe
        Cleaned data. (Specifications)
        
    Returns
    -------

    '''
    
    return None


# ## Get competition environment

# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# ## Get training data

# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()

if REDUCED:
    market_train_df = market_train_df.tail(100_000)
    news_train_df = news_train_df.tail(300_000)


# ## Preprocess and clean the data

# In[ ]:


# Select columns and drop NA
cols = ['assetCode', 'time', 'volume', 'open', 'returnsOpenPrevMktres1', 
        'returnsOpenPrevMktres10', 'returnsOpenNextMktres10']
market_train_df = market_train_df.loc[:,cols]
market_train_df.dropna(inplace=True)


# In[ ]:


# Select columns and drop NA
cols = ['time','assetCodes', 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive']
news_train_df = news_train_df.loc[:,cols]
news_train_df.dropna(inplace=True)


# In[ ]:


# Normalize time
market_train_df.loc[:, 'time'] = market_train_df.time.dt.normalize()
news_train_df.loc[:, 'time'] = news_train_df.time.dt.normalize()

# assetCodes from String to List
news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'")


# In[ ]:


# Explode news on assetCodes
assetCodes_expanded = list(chain(*news_train_df['assetCodes']))
assetCodes_index = news_train_df.index.repeat(news_train_df['assetCodes'].apply(len))

assert len(assetCodes_expanded) == len(assetCodes_index)


# In[ ]:


assetCodes_df =  pd.DataFrame({'index': assetCodes_index, 'assetCode': assetCodes_expanded})
news_train_df_exploded = news_train_df.merge(assetCodes_df, 'right', right_on='index', left_index=True, validate='1:m')
news_train_df_exploded.drop(['assetCodes', 'index'], 1, inplace=True)


# In[ ]:


# Compute means for same date and assetCode
news_agg_dict = {
    'sentimentNegative':'mean'
    ,'sentimentNeutral':'mean'
    ,'sentimentPositive':'mean'
}
news_train_df_agg = news_train_df_exploded.groupby(['time', 'assetCode'], as_index=False).agg(news_agg_dict)


# In[ ]:


# Merge on market data
X = market_train_df.merge(news_train_df_agg, 'left', ['time', 'assetCode'])


# ## Train the model

# In[ ]:


train_model(train_df)


# ## Make predictions on test data

# In[ ]:


days = env.get_prediction_days()


# In[ ]:


import numpy as np
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0


# In[ ]:


for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_random_predictions(predictions_template_df)
    env.predict(predictions_template_df)
print('Done!')


# In[ ]:


env.write_submission_file()

