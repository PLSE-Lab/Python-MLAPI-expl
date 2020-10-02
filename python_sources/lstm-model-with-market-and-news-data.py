#!/usr/bin/env python
# coding: utf-8

# ## A simple LSTM model
# 
# I combined techinques and code from two notebooks that I found, and converted it to an LSTM. 
# 
# References to other notebooks used: 
# 
# https://www.kaggle.com/christofhenkel/market-data-nn-baseline
# 
# https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-and-news-data

# In[ ]:


from kaggle.competitions import twosigmanews
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# get the data from two sigma environment
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


# gets used later to aggregate news into marker data 
news_cols_agg = {
    'bodySize': ['min', 'max', 'mean', 'std'],
    'sentimentNegative': ['min', 'max', 'mean', 'std'],
    'sentimentNeutral': ['min', 'max', 'mean', 'std'],
    'sentimentPositive': ['min', 'max', 'mean', 'std'],
}
# specify categorical columns
categorical_cols = ['assetName', 'dayofweek', 'month', 'year']
# lengths of embeddings of categorical columns
embedding_lengths = [100, 2, 2, 3]
encodings = {}


# In[ ]:


def join_market_news(market_train_df, news_train_df):
    # Fix asset codes (str -> list)
    news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'")    
    
    # Expand assetCodes -- converts ['AAPL', 'GOOG'] --> 'APPL', 'GOOG'
    assetCodes_expanded = list(chain(*news_train_df['assetCodes']))
    assetCodes_index = news_train_df.index.repeat( news_train_df['assetCodes'].apply(len) )

    assert len(assetCodes_index) == len(assetCodes_expanded)
    df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})

    # Create expandaded news (will repeat every assetCodes' row)
    news_cols = ['time', 'assetCodes'] + sorted(news_cols_agg.keys())
    news_train_df_expanded = pd.merge(df_assetCodes, news_train_df[news_cols], left_on='level_0', right_index=True, suffixes=(['','_old']))

    # Free memory
    del news_train_df, df_assetCodes

    # Aggregate numerical news features
    news_train_df_aggregated = news_train_df_expanded.groupby(['time', 'assetCode']).agg(news_cols_agg)
    
    # Free memory
    del news_train_df_expanded

    # Convert to float32 to save memory
    news_train_df_aggregated = news_train_df_aggregated.apply(np.float32)

    # Flat columns
    news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]

    # Join with train
    market_train_df = market_train_df.join(news_train_df_aggregated, on=['time', 'assetCode'])

    # Free memory
    del news_train_df_aggregated
    
    return market_train_df


# In[ ]:


def get_xy(market_train_df, news_train_df):
    x = get_x(market_train_df, news_train_df)
    y = market_train_df['returnsOpenNextMktres10'].clip(-1, 1)
    return x, y

def label_encode(series, min_counts=2):
    vc = series.value_counts()
    #reserve 0 for unknown
    le = {c : i+1 for i, c in enumerate(vc.index[vc > min_counts])}
    le['UNKN'] = 0
    return le

def get_encodings(df, cat_cols):
    if len(encodings) == 0:
        for col in cat_cols:
            encodings[col] = label_encode(df[col])
    return encodings

def map_encodings(df, cat_cols, encs):
    for col in cat_cols:
        df[col] = df[col].map(encs[col]).fillna(0).astype(int)
        
def get_x(market_train_df, news_train_df, isTrain=True):
    # Split date into before and after 22h (the time used in train data)
    # E.g: 2007-03-07 23:26:39+00:00 -> 2007-03-08 00:00:00+00:00 (next day)
    #      2009-02-25 21:00:50+00:00 -> 2009-02-25 00:00:00+00:00 (current day)    
    news_train_df['time'] = (news_train_df['time'] - np.timedelta64(22,'h')).dt.ceil('1D')

    # Round time of market_train_df to 0h of curret day
    market_train_df['time'] = market_train_df['time'].dt.floor('1D')

    # Join market and news
    x = join_market_news(market_train_df, news_train_df)
    
    x['dayofweek'], x['day'], x['month'], x['year'] = x.time.dt.dayofweek, x.time.dt.day, x.time.dt.month, x.time.dt.year

    encodings = get_encodings(x, categorical_cols)
    map_encodings(x, categorical_cols, encodings) 
    if isTrain:
        cols_to_drop = ['returnsOpenNextMktres10', 'universe', 'time']
    else: 
        cols_to_drop = ['time']
    
    x.drop(columns=cols_to_drop, inplace=True)
        
    return x


# In[ ]:


# This will take some time...
X, y = get_xy(market_train_df, news_train_df)


# In[ ]:


#Save universe data for latter use
universe = market_train_df['universe']
time = market_train_df['time']

# Free memory
del market_train_df, news_train_df


# In[ ]:


#get all the numeric columns
num_cols = [x for x in X.columns if x not in categorical_cols]

#remove assetCode from num_cols
num_cols = [x for x in num_cols if x not in ['assetCode']]


# In[ ]:


#scale numeric cols
def scale_numeric(df):
    df[num_cols] = df[num_cols].fillna(0)

    scaler = StandardScaler()
    
    #need to do this due to memory contraints
    for i in range(0, len(num_cols), 4):
        cols = num_cols[i:i + 3]
        df[cols] = scaler.fit_transform(df[cols].astype(float))
        
scale_numeric(X)


# In[ ]:


#split dataset into 80% for training and 20% for validation 
n_train = int(X.shape[0] * 0.8)

X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
X_valid, y_valid = X.iloc[n_train:], y.iloc[n_train:]


# In[ ]:


# For valid data, keep only those with universe > 0. This will help calculate the metric
u_valid = (universe.iloc[n_train:] > 0)
t_valid = time.iloc[n_train:]

X_valid = X_valid[u_valid]
y_valid = y_valid[u_valid]
t_valid = t_valid[u_valid]

d_valid = t_valid.dt.date

del u_valid


# In[ ]:


#seperate the columns into categorical and numerical
def get_cat_num_split(df):
    X = {} 
    X['num'] = df.loc[:, num_cols].values
    X['num'] = np.reshape(X['num'], (X['num'].shape[0], 1, X['num'].shape[1]))
    for cat in categorical_cols:
        X[cat] = df.loc[:, cat].values
    return X


# In[ ]:


#seperate the columns into categorical and numerical
X_train_split = get_cat_num_split(X_train)
X_valid_split = get_cat_num_split(X_valid)

#set y to a binary representation of returns, true if it's 0-1 and false if i'ts -1-0
y_train_bin = (y_train >= 0).values
y_valid_bin = (y_valid >= 0).values


# In[ ]:


encoding_len = {k: len(encodings[k]) + 1 for k in encodings}


# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, LSTM, Dropout, Reshape
from keras.losses import binary_crossentropy, mse
from keras.regularizers import l2
import keras.backend as K

DROPOUT_RATE = 0.2

cat_inputs = [Input(shape=[1], name=cat) for cat in categorical_cols]
embeddings = [Embedding(encoding_len[cat], embedding_lengths[i])(cat_inputs[i]) for i, cat in enumerate(categorical_cols)]
categorical_logits = Concatenate()([(cat_emb) for cat_emb in embeddings])
categorical_logits = LSTM(128, activation='relu', input_shape=(1, len(categorical_cols)), return_sequences=True,
                         kernel_regularizer=l2(1e-5), kernel_initializer='random_uniform')(categorical_logits)

numerical_inputs = Input(shape=(1, len(num_cols)), name='num')
numerical_logits = LSTM(256, activation='relu', input_shape=(1, len(num_cols)), return_sequences=True,
                        kernel_regularizer=l2(1e-5), kernel_initializer='random_uniform')(numerical_inputs)
numerical_logits = Dropout(DROPOUT_RATE)(numerical_logits)

logits = Concatenate()([numerical_logits,categorical_logits])
logits = LSTM(256, activation='relu', kernel_initializer='random_uniform')(logits)
out = Dense(1, activation='sigmoid', name='confidence_level')(logits)

model = Model(inputs = cat_inputs + [numerical_inputs], outputs=out)
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[ ]:


model.summary()


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint

check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=1,verbose=True)
model.fit(X_train_split, y_train_bin,
          validation_data=(X_valid_split, y_valid_bin),
          epochs=1,
          verbose=True,
          callbacks=[early_stop,check_point]) 


# In[ ]:


from sklearn.metrics import accuracy_score

# distribution of confidence that will be used as submission
model.load_weights('model.hdf5')
# scaled condifence value  from 0 - 1 to -1 - 1
confidence_valid = model.predict(X_valid_split)[:,0]*2 -1

plt.hist(confidence_valid, bins='auto')
plt.title("predicted confidence")
plt.show()


# In[ ]:


#r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_valid * y_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)


# # Train full model
# Now we train a full model with `num_boost_round` found in validation.

# In[ ]:


def make_predictions(predictions_template_df, market_obs_df, news_obs_df):
    inp = get_x(market_obs_df, news_obs_df, False)
    scale_numeric(inp)
    inp_split = get_cat_num_split(inp)
    scaled_pred = model.predict(inp_split) * 2 - 1    
    predictions_template_df.confidenceValue = np.clip(scaled_pred, -1, 1)


# In[ ]:


days = env.get_prediction_days()

x1,y1,z1 = None, None, None

for (market_obs_df, news_obs_df, predictions_template_df) in days:
    x1,y1,z1 = predictions_template_df, market_obs_df, news_obs_df
    make_predictions(x1,y1,z1)
    env.predict(predictions_template_df)
print('Done!')


# In[ ]:


env.write_submission_file()

