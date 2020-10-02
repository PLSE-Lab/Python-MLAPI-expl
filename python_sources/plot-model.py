#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import date, datetime, timedelta
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews


# In[ ]:


env = twosigmanews.make_env()


# # Merge & clean data

# In[ ]:


start_date = date(2015,1,1)

# (market_train, _) = env.get_training_data()
(market_train_orig, news_train_orig) = env.get_training_data()
market_train_df = market_train_orig.copy()
news_train_df = news_train_orig.copy()
print('Market train shape: ',market_train_df.shape)
print('News train shape: ', news_train_df.shape)

# Sort data
market_train_df = market_train_df.sort_values('time')
market_train_df['date'] = market_train_df['time'].dt.date

# Fill nan
market_train_fill = market_train_df
column_market = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
column_raw = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10']
for i in range(len(column_raw)):
    market_train_fill[column_market[i]] = market_train_fill[column_market[i]].fillna(market_train_fill[column_raw[i]])
market_train_orig = market_train_orig.sort_values('time')
news_train_orig = news_train_orig.sort_values('time')
market_train_df = market_train_orig.copy()
news_train_df = news_train_orig.copy()
del market_train_orig
del news_train_orig
market_train_df = market_train_df.loc[market_train_df['time'].dt.date>=start_date]
news_train_df = news_train_df.loc[news_train_df['time'].dt.date>=start_date]
market_train_df['close_open_ratio'] = np.abs(market_train_df['close']/market_train_df['open'])
threshold = 0.5
print('In %i lines price increases by 50%% or more in a day' %(market_train_df['close_open_ratio']>=1.5).sum())
print('In %i lines price decreases by 50%% or more in a day' %(market_train_df['close_open_ratio']<=0.5).sum())
market_train_df = market_train_df.loc[market_train_df['close_open_ratio'] < 1.5]
market_train_df = market_train_df.loc[market_train_df['close_open_ratio'] > 0.5]
market_train_df = market_train_df.drop(columns=['close_open_ratio'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
column_market = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
column_raw = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10']
#the top hundred words.
vectorizer = CountVectorizer(max_features=1000, stop_words={"english"})
#we do this with TF-IDF.
X = vectorizer.fit_transform(news_train_df['headline'].values)
tf_transformer = TfidfTransformer(use_idf=False).fit(X)
X_train_tf = tf_transformer.transform(X)
X_train_vals = X_train_tf.mean(axis=1)


del vectorizer
del X
del X_train_tf

#mean tf-idf score for news article.
d = pd.DataFrame(data=X_train_vals)
news_train_df['tf_score'] = d

market_train_df = market_train_df.loc[market_train_df['time'].dt.date>=start_date]
news_train_df = news_train_df.loc[news_train_df['time'].dt.date>=start_date]

#add indicator features
market_train_df['rolling_average_close_mean'] = market_train_df.groupby('assetCode')['close'].transform('mean')
market_train_df['rolling_average_vol_mean'] = market_train_df.groupby('assetCode')['volume'].transform('mean')
market_train_df['rolling_average_close_std'] = market_train_df.groupby('assetCode')['close'].transform('std')
market_train_df['rolling_average_vol_std'] = market_train_df.groupby('assetCode')['volume'].transform('std')
#some more refined instruments
market_train_df['moving_average_7_day'] = market_train_df.groupby('assetCode')['close'].transform(lambda x: x.rolling(window=7).mean())
ewma = pd.Series.ewm
market_train_df['ewma'] =  market_train_df.groupby('assetCode')['close'].transform(lambda x : ewma(x, span=30).mean())
market_train_df['moving_average_7_day'] = market_train_df['moving_average_7_day'].fillna(0)
market_train_df['ewma'] = market_train_df['ewma'].fillna(0)
for i in range(len(column_raw)):
    market_train_df[column_market[i]] = market_train_df[column_market[i]].fillna(market_train_df[column_raw[i]])
    print('Removing outliers ...')
column_return = column_market + column_raw + ['returnsOpenNextMktres10']
orig_len = market_train_df.shape[0]
for column in column_return:
    market_train_df = market_train_df.loc[market_train_df[column]>=-2]
    market_train_df = market_train_df.loc[market_train_df[column]<=2]
new_len = market_train_df.shape[0]
rmv_len = np.abs(orig_len-new_len)
print('There were %i lines removed' %rmv_len)
print('Removing strange data ...')
orig_len = market_train_df.shape[0]
market_train_df = market_train_df[~market_train_df['assetCode'].isin(['PGN.N','EBRYY.OB'])]
#market_train_df = market_train_df[~market_train_df['assetName'].isin(['Unknown'])]
new_len = market_train_df.shape[0]
rmv_len = np.abs(orig_len-new_len)
print('There were %i lines removed' %rmv_len)
# Function to remove outliers
def remove_outliers(data_frame, column_list, low=0.02, high=0.98):
    for column in column_list:
        this_column = data_frame[column]
        quant_df = this_column.quantile([low,high])
        low_limit = quant_df[low]
        high_limit = quant_df[high]
        data_frame[column] = data_frame[column].clip(lower=low_limit, upper=high_limit)
    return data_frame
columns_outlier = ['takeSequence', 'bodySize', 'sentenceCount', 'wordCount', 'sentimentWordCount', 'firstMentionSentence','noveltyCount12H',                  'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H', 'volumeCounts24H',                  'volumeCounts3D','volumeCounts5D','volumeCounts7D']
print('Clipping news outliers ...')
news_train_df = remove_outliers(news_train_df, columns_outlier)
asset_code_dict = {k: v for v, k in enumerate(market_train_df['assetCode'].unique())}
drop_columns = [col for col in news_train_df.columns if col not in ['sourceTimestamp', 'urgency', 'takeSequence', 'bodySize', 'companyCount', 
               'sentenceCount', 'firstMentionSentence', 'relevance','firstCreated', 'assetCodes']]
columns_news = ['firstCreated','relevance','sentimentClass','sentimentNegative','sentimentNeutral',
               'sentimentPositive','noveltyCount24H','noveltyCount7D','volumeCounts24H','volumeCounts7D','assetCodes','sourceTimestamp',
               'assetName','audiences', 'urgency', 'takeSequence', 'bodySize', 'companyCount', 
               'sentenceCount', 'firstMentionSentence','time', 'tf_score']
def data_prep(market_df,news_df):
    market_df['date'] = market_df.time.dt.date
    market_df['close_to_open'] = market_df['close'] / market_df['open']
    market_df.drop(['time'], axis=1, inplace=True)
    
    news_df = news_df[columns_news]
    news_df['sourceTimestamp']= news_df.sourceTimestamp.dt.hour
    news_df['firstCreated'] = news_df.firstCreated.dt.date
    news_df['assetCodesLen'] = news_df['assetCodes'].map(lambda x: len(eval(x)))
    news_df['assetCodes'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])
    news_df['asset_sentiment_count'] = news_df.groupby(['assetName', 'sentimentClass'])['time'].transform('count')
    news_df['len_audiences'] = news_train_df['audiences'].map(lambda x: len(eval(x)))
    kcol = ['firstCreated', 'assetCodes']
    news_df = news_df.groupby(kcol, as_index=False).mean()
    market_df = pd.merge(market_df, news_df, how='left', left_on=['date', 'assetCode'], 
                            right_on=['firstCreated', 'assetCodes'])
    del news_df
#     market_df['assetCodeT'] = market_df['assetCode'].map(asset_code_dict)
    market_df = market_df.drop(columns = ['firstCreated','assetCodes','assetName']).fillna(0) 
#     print(market_df.count)
    return market_df
print('Merging data ...')
market_train_df = data_prep(market_train_df, news_train_df)
market_train_df = market_train_df.loc[market_train_df['date']>=start_date]
market_train = market_train_df

del market_train_df


# In[ ]:


market_train.describe().round(3)


# # Prepare model

# In[ ]:


# cat_cols = ['assetCode']
# num_cols = ['volume',
#             'close',
#             'open',
#             'returnsClosePrevRaw1',
#             'returnsOpenPrevRaw1',
#             'returnsClosePrevMktres1',
#             'returnsOpenPrevMktres1',
#             'returnsClosePrevRaw10',
#             'returnsOpenPrevRaw10',
#             'returnsClosePrevMktres10',
#             'returnsOpenPrevMktres10']
cat_cols = ['assetCodeT']
num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 
               'returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'close_to_open', 'rolling_average_close_mean', 'rolling_average_vol_mean', 'rolling_average_close_std', 'ewma', 'rolling_average_close_std', 'sourceTimestamp', 'urgency', 'companyCount', 'takeSequence', 'bodySize', 'sentenceCount',
               'moving_average_7_day','relevance', 'sentimentClass', 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
               'noveltyCount24H','noveltyCount7D','volumeCounts24H','volumeCounts7D','assetCodesLen', 'asset_sentiment_count', 'len_audiences', 'tf_score']


# In[ ]:


from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(market_train.index.values,
                                              test_size=0.25,
                                              random_state=42)


# # Handling categorical variables

# In[ ]:


def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{} for cat in cat_cols]


# for i, cat in enumerate(cat_cols):
#     print('encoding %s ...' % cat, end=' ')
#     encoders[i] = {l: id for id, l in enumerate(market_train.loc[train_indices, cat].astype(str).unique())}
#     market_train[cat] = market_train[cat].astype(str).apply(lambda x: encode(encoders[i], x))
#     print('Done')

market_train['assetCodeT'] = market_train['assetCode'].astype(str).apply(lambda x: encode(encoders[0], x))
embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets


# # Define NN Architecture

# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout
from keras.losses import binary_crossentropy

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

#categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
categorical_logits = Flatten()(categorical_embeddings[0])
categorical_logits = Dense(32,activation='relu')(categorical_logits)
categorical_logits = Dropout(0.5)(categorical_logits)
categorical_logits = BatchNormalization()(categorical_logits)
categorical_logits = Dense(32,activation='relu')(categorical_logits)

numerical_inputs = Input(shape=(len(num_cols),), name='num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)
numerical_logits = Dense(128,activation='relu')(numerical_logits)

numerical_logits = Dropout(0.5)(numerical_logits)
numerical_logits = BatchNormalization()(numerical_logits)
numerical_logits = Dense(128,activation='relu')(numerical_logits)
numerical_logits = Dense(64,activation='relu')(numerical_logits)

logits = Concatenate()([numerical_logits,categorical_logits])
logits = Dense(64,activation='relu')(logits)
out = Dense(1, activation='sigmoid')(logits)

model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
model.compile(optimizer='adam',loss=binary_crossentropy)


# In[ ]:


from keras.utils import plot_model
plot_model(model, to_file='model.png')


# In[ ]:


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

