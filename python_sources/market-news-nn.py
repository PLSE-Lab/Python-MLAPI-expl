#!/usr/bin/env python
# coding: utf-8

# This is heavily based on following two kernels - 
# https://www.kaggle.com/bguberfain/a-simple-model-using-the-market-and-news-data
# https://www.kaggle.com/dmitrypukhov/eda-and-nn-for-market-and-news

# In[ ]:


import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from time import time
import seaborn as sns
import os
import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer,StandardScaler, MinMaxScaler,OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import tensorflow

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


toy=False


# In[ ]:


# This competition settings
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


market_train_df.shape


# In[ ]:


news_train_df.shape


# In[ ]:


# We will reduce the number of samples for memory reasons
if toy:
    market_train_df = market_train_df.tail(100_000)
    news_train_df = news_train_df.tail(300_000)
else:
    market_train_df = market_train_df.tail(750_000)
    news_train_df = news_train_df.tail(1500_000)


# In[ ]:


news_cols_agg = {
    'urgency': ['min', 'count'],
    'takeSequence': ['max'],
    'bodySize': ['min', 'max', 'mean', 'std'],
    'wordCount': ['min', 'max', 'mean', 'std'],
    'sentenceCount': ['min', 'max', 'mean', 'std'],
    'companyCount': ['min', 'max', 'mean', 'std'],
    'marketCommentary': ['min', 'max', 'mean', 'std'],
    'relevance': ['min', 'max', 'mean', 'std'],
    'sentimentNegative': ['min', 'max', 'mean', 'std'],
    'sentimentNeutral': ['min', 'max', 'mean', 'std'],
    'sentimentPositive': ['min', 'max', 'mean', 'std'],
    'sentimentWordCount': ['min', 'max', 'mean', 'std'],
    'noveltyCount12H': ['min', 'max', 'mean', 'std'],
    'noveltyCount24H': ['min', 'max', 'mean', 'std'],
    'noveltyCount3D': ['min', 'max', 'mean', 'std'],
    'noveltyCount5D': ['min', 'max', 'mean', 'std'],
    'noveltyCount7D': ['min', 'max', 'mean', 'std'],
    'volumeCounts12H': ['min', 'max', 'mean', 'std'],
    'volumeCounts24H': ['min', 'max', 'mean', 'std'],
    'volumeCounts3D': ['min', 'max', 'mean', 'std'],
    'volumeCounts5D': ['min', 'max', 'mean', 'std'],
    'volumeCounts7D': ['min', 'max', 'mean', 'std']
}


# In[ ]:


def join_market_news(market_train_df, news_train_df):
    # Fix asset codes (str -> list)
    news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'")    
    
    # Expand assetCodes
    assetCodes_expanded = list(chain(*news_train_df['assetCodes']))
    assetCodes_index = news_train_df.index.repeat( news_train_df['assetCodes'].apply(len) )

    assert len(assetCodes_index) == len(assetCodes_expanded)
    df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})

    # Create expandaded news (will repeat every assetCodes' row)
    news_cols = ['time', 'assetCodes'] + sorted(news_cols_agg.keys())
    news_train_df_expanded = pd.merge(df_assetCodes, news_train_df[news_cols], left_on='level_0', right_index=True, suffixes=(['','_old']))

    # Free memory
    del news_train_df, df_assetCodes
    gc.collect()

    # Aggregate numerical news features
    news_train_df_aggregated = news_train_df_expanded.groupby(['time', 'assetCode']).agg(news_cols_agg)
    
    # Free memory
    del news_train_df_expanded
    gc.collect()

    # Convert to float32 to save memory
    news_train_df_aggregated = news_train_df_aggregated.apply(np.float32)

    # Flat columns
    news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]

    # Join with train
    market_train_df = market_train_df.join(news_train_df_aggregated, on=['time', 'assetCode'])

    # Free memory
    del news_train_df_aggregated
    gc.collect()
    
    return market_train_df


# In[ ]:


market_df = join_market_news(market_train_df, news_train_df)


# In[ ]:


market_df.tail()


# In[ ]:


market_df.shape


# In[ ]:


market_df.columns


# In[ ]:


# Features
cat_cols = ['assetCode']
time_cols=['year', 'week', 'day', 'dayofweek']
mkt_numeric_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                'returnsOpenPrevMktres10']

news_numeric_cols = [
#        'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
#        'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
#        'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
#        'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
#        'returnsOpenNextMktres10', 'universe', 'urgency_min', 'urgency_count',
#        'takeSequence_max', 'bodySize_min', 'bodySize_max', 'bodySize_mean',
#        'bodySize_std', 'wordCount_min', 'wordCount_max', 'wordCount_mean',
#        'wordCount_std', 'sentenceCount_min', 'sentenceCount_max',
#        'sentenceCount_mean', 'sentenceCount_std', 'companyCount_min',
#        'companyCount_max', 'companyCount_mean', 'companyCount_std',
#        'marketCommentary_min', 'marketCommentary_max', 'marketCommentary_mean',
#        'marketCommentary_std', 
    'relevance_min', 'relevance_max',
    'relevance_mean', 'relevance_std', 'sentimentNegative_min',
    'sentimentNegative_max', 'sentimentNegative_mean',
    'sentimentNegative_std', 'sentimentNeutral_min', 'sentimentNeutral_max',
    'sentimentNeutral_mean', 'sentimentNeutral_std',
    'sentimentPositive_min', 'sentimentPositive_max',
    'sentimentPositive_mean', 'sentimentPositive_std',
    'sentimentWordCount_min', 'sentimentWordCount_max',
    'sentimentWordCount_mean', 'sentimentWordCount_std',
    'noveltyCount12H_min', 'noveltyCount12H_max', 'noveltyCount12H_mean',
    'noveltyCount12H_std', 'noveltyCount24H_min', 'noveltyCount24H_max',
    'noveltyCount24H_mean', 'noveltyCount24H_std', 'noveltyCount3D_min',
    'noveltyCount3D_max', 'noveltyCount3D_mean', 'noveltyCount3D_std',
    'noveltyCount5D_min', 'noveltyCount5D_max', 'noveltyCount5D_mean',
    'noveltyCount5D_std', 'noveltyCount7D_min', 'noveltyCount7D_max',
    'noveltyCount7D_mean', 'noveltyCount7D_std']
#         'volumeCounts12H_min',
#        'volumeCounts12H_max', 'volumeCounts12H_mean', 'volumeCounts12H_std',
#        'volumeCounts24H_min', 'volumeCounts24H_max', 'volumeCounts24H_mean',
#        'volumeCounts24H_std', 'volumeCounts3D_min', 'volumeCounts3D_max',
#        'volumeCounts3D_mean', 'volumeCounts3D_std', 'volumeCounts5D_min',
#        'volumeCounts5D_max', 'volumeCounts5D_mean', 'volumeCounts5D_std',
#        'volumeCounts7D_min', 'volumeCounts7D_max', 'volumeCounts7D_mean',
#        'volumeCounts7D_std']

numeric_cols = mkt_numeric_cols + news_numeric_cols

feature_cols = cat_cols + time_cols + numeric_cols

# Labels
label_cols = ['returnsOpenNextMktres10']


# In[ ]:


print (np.unique(feature_cols).shape)
print (len(feature_cols))
print (numeric_cols)


# ## Split to train, validation and test

# In[ ]:


# Split to train, validation and test.
# ToDo: remove shuffle, use generator.
#market_train_df, market_test_df = train_test_split(market[market.time > '2009'].sample(100000, random_state=42), shuffle=True, random_state=24)
market_train_df, market_test_df = train_test_split(market_df, shuffle=True, random_state=24)
market_train_df, market_val_df = train_test_split(market_train_df, shuffle=True, random_state=24)

# Look at min/max and quantiles
market_train_df.describe()


# ## Preprocess the data
# Scale numeric columns, encode categorical. Split to X, y for training.

# In[ ]:


class Prepro:
    """
    Bring all preprocessing here: scale, encoding
    Should be fit on train data and called on train, validation or test data
    """
    
    def __init__(self, feature_cols, cat_cols, time_cols, numeric_cols, label_cols):
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.time_cols = time_cols
        self.numeric_cols = numeric_cols
        self.label_cols = label_cols
        self.cats={}
    
    def transformXy(self, df):
        """
        Preprocess and return X,y
        """
        df = df.copy()
        # Scale, encode etc. features
        X = self.transform(df)
        # Scale labels
        df[self.label_cols] = self.y_scaler.transform(df[self.label_cols])
        y = df[self.label_cols]
        return(X,y)
    
    def transform(self, df):
        """
        Preprocess and return X
        """
        # Add day, week, year
        df = self.prepare_time_cols(df)
        # Fill nans
        df[self.numeric_cols] = df[self.numeric_cols].fillna(0)
        # Preprocess categorical features
        for col in cat_cols:
            df[col] = df[col].apply(lambda cat_name: self.prepare_cat_cols(cat_name, col))
        # Scale numeric features and labels
        df[self.numeric_cols+self.time_cols] = self.numeric_scaler.transform(df[self.numeric_cols+self.time_cols])
        # Return X
        return df[self.feature_cols]
    
    def fit(self, df):
        """
        Fit preprocessing scalers, encoders on given train df
        To be called on train df only
        """
        # Extract day, week, year from time
        df = df.copy()
        df = self.prepare_time_cols(df)
        # Handle strange cases, impute etc.
        df = self.prepare_train_df(df)
        # Use QuantileTransformer to handle outliers
        # Fit for labels
        self.y_scaler = QuantileTransformer()
        self.y_scaler.fit(df[self.label_cols])
        # Fit for numeric and time
        self.numeric_scaler = QuantileTransformer()
        self.numeric_scaler.fit(df[self.numeric_cols + self.time_cols])
        # Fit for categories
        # Organize dictionary, each category column has list with values
        self.cats=dict()
        for col in cat_cols:
            self.cats[col] = list(df[col].unique())

    def prepare_train_df(self, train_df):
        """
        Clean na, remove strange cases.
        For train dataset only. 
        """
        # Handle nans
        train_df = train_df.copy()
        # Need better imputer
        # for col in numeric_cols:
        #     market_train_df[col] = market_train_df[col].fillna(market_train_df[col].mean())
        train_df.tail()
        train_df[self.numeric_cols] = train_df[self.numeric_cols].fillna(0)

        # # Remove strange cases with close/open ratio > 2
        max_ratio  = 2
        train_df = train_df[np.abs(train_df['close'] / train_df['open']) <= max_ratio]
        return(train_df)

    def prepare_time_cols(self, df):
        """ 
        Extract time parts, they are important for time series 
        """
        df = df.copy()
        df['year'] = df['time'].dt.year
        # Maybe remove month because week of year can handle the same info
        df['day'] = df['time'].dt.day
        # Week of year
        df['week'] = df['time'].dt.week
        df['dayofweek'] = df['time'].dt.dayofweek
        return(df)
        
    def prepare_cat_cols(self, cat_name, col):
        """
        Encode categorical features to numbers
        """
        try:
            # Transform to index of name in stored names list
            index_value = self.cats[col].index(cat_name)
        except ValueError:
            # If new value, add it to the list and return new index
            self.cats[col].append(cat_name)
            index_value = len(self.cats[col])
        index_value = 1.0/(index_value+1.0)
        return(index_value)


# In[ ]:


# Preprocess and split to X_train, X_val, X_test, y_train ...
prepro = Prepro(feature_cols, cat_cols, time_cols, numeric_cols, label_cols)
prepro.fit(market_train_df)

# Clean train df,handle strange cases
market_train_df = prepro.prepare_train_df(market_train_df)

X_train, y_train = prepro.transformXy(market_train_df)
X_val, y_val = prepro.transformXy(market_val_df)
X_test, y_test = prepro.transformXy(market_test_df)

# Display for visual check. 
pd.concat([X_train,y_train], axis=1).describe()
#X_train.head()


# # 5. Base model
# NN with Dense layers.
# 

# ## Create the model

# In[ ]:


# Initialize the constructor
model = Sequential()

# Add an input layer 
input_size = X_train.shape[1]

# Add layers - no worries which are.
# ToDo: find a good architecture of NN
model.add(Dense(256, input_shape=(input_size,), kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
model.summary()


# ## Train market model

# In[ ]:


weights_file='best_weights.h5'

# We'll stop training if no improvement after some epochs
earlystopper = EarlyStopping(patience=5, verbose=1)

# Low, avg and high scor training will be saved here
# Save the best model during the traning
checkpointer = ModelCheckpoint(weights_file
    ,verbose=1
    ,save_best_only=True
    ,save_weights_only=True)

reduce_lr = ReduceLROnPlateau(factor=0.2,
                              patience=5, min_lr=0.001)

# Train
training = model.fit(X_train,y_train
                                ,batch_size=512
                                ,epochs=100
                                ,validation_data=[X_val, y_val]
                                #,steps_per_epoch=100
                                 #, validation_steps=100
                                ,callbacks=[earlystopper, checkpointer, reduce_lr])
# Load best weights saved
model.load_weights(weights_file)


# ## Evaluate market model
# 
# ### Loss function by epoch 

# In[ ]:


# f, axs = plt.subplots(1,2, figsize=(12,4))
# axs[0].plot(training.history['loss'])
# axs[0].set_xlabel("Epoch")
# axs[0].set_ylabel("Loss")
# axs[0].set_title("Loss")
# axs[1].plot(training.history['val_loss'])
# axs[1].set_xlabel("Epoch")
# axs[1].set_ylabel("val_loss")
# axs[1].set_title("Validation loss")
# plt.tight_layout()
# plt.show()
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title("Loss and validation loss")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.show()


# ### Predict on test data

# In[ ]:


pred_size=100
X_test2 = X_test.values[:pred_size]
y_pred2 = model.predict(X_test2) [:,0]*2-1
y_test2 = y_test.values[:pred_size][:,0]*2-1

ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax1.plot(y_test2, linestyle='none', marker='.', color='darkblue')
ax1.plot(y_pred2, linestyle='none', marker='.', color='darkred')
ax1.legend(["Ground truth","Predicted"])
ax1.set_title("Both")
ax1.set_xlabel("Epoch")
ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1,rowspan=1)
ax2.plot(y_test2, linestyle='none', marker='.', color='darkblue')
ax2.set_title("Ground truth")
ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1,rowspan=1)
ax3.plot(y_pred2, linestyle='none', marker='.', color='darkred')
ax3.set_title("Predicted")
plt.tight_layout()
plt.show()


# ### Predict on random asset

# In[ ]:


def predict_random_asset():
    """
    Get random asset from test set, predict on it, plot ground truth and predicted value
    """
    # Get any asset
    ass = market_test_df.assetName.sample(1, random_state=24).iloc[0]
    test_ass_df = market_test_df[market_test_df['assetName'] == ass]
    # Preprocess
    X,y = prepro.transformXy(test_ass_df)
    y.index = test_ass_df.time
    # Predict
    pred = pd.DataFrame(model.predict(X)*2 -1)
    pred.index = test_ass_df.time
    # Plot
    plt.plot(y*2-1, linestyle='none', marker='.', color='darkblue')
    plt.plot(pred, linestyle='none', marker='.', color='orange')
    plt.title(ass)
    plt.legend(["Ground truth", "predicted"])
    plt.show()
    
predict_random_asset()


# In[ ]:


# Accuracy metric
confidence = model.predict(X_test)*2 -1
print("Accuracy: %f" % accuracy_score(y_test > 0, confidence > 0))

# Show distribution of confidence that will be used as submission
plt.hist(confidence, bins='auto')
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.title("predicted confidence")
plt.show()


# # Submission

# In[ ]:


def make_predictions(market_obs_df, news_obs_df, predictions_template_df):
    """
    Predict confidence for one day and update predictions_template_df['confidenceValue']
    @param market_obs_df: market_obs_df returned from env
    @param predictions_template_df: predictions_template_df returned from env.
    @return: None. prediction_template_df updated instead. 
    """
    # Preprocess the data
    market_obs_df = join_market_news(market_obs_df, news_obs_df)
    X = prepro.transform(market_obs_df)
    # Predict
    y_pred = model.predict(X)
    confidence_df=pd.DataFrame(y_pred*2-1, columns=['confidence'])

    # Merge predicted confidence to predictions template
    pred_df = pd.concat([predictions_template_df, confidence_df], axis=1).fillna(0)
    predictions_template_df.confidenceValue = pred_df.confidence


# In[ ]:


##########################
# Submission code

# Save data here for later debugging on it
days_saved_data = []

# Store execution info for plotting later
predicted_days=[]
predicted_times=[]
last_predictions_template_df = None

# Predict day by day
days = env.get_prediction_days()

for (market_obs_df, news_obs_df, predictions_template_df) in days:
    # Store the data for later debugging on it
    days_saved_data.append((market_obs_df, news_obs_df, predictions_template_df))
    # For later plotting
    predicted_days.append(market_obs_df.iloc[0].time.strftime('%Y-%m-%d'))
    time_start = time()

    # Call prediction func
    make_predictions(market_obs_df, news_obs_df, predictions_template_df)
    #!!!
    env.predict(predictions_template_df)
    
    # For later plotting
    last_predictions_template_df = predictions_template_df
    predicted_times.append(time()-time_start)
    #print("Prediction completed for ", predicted_days[-1])


# In[ ]:


# Plot execution time 
sns.barplot(np.array(predicted_days), np.array(predicted_times))
plt.title("Execution time per day")
plt.xlabel("Day")
plt.ylabel("Execution time, seconds")
plt.show()

# Plot predicted confidence for last day
last_predictions_template_df.plot(linestyle='none', marker='.')
plt.title("Predicted confidence for last observed day: %s" % predicted_days[-1])
plt.xlabel("Observation No.")
plt.ylabel("Confidence")
plt.show()


# In[ ]:


# We've got a submission file!
# !!! Write submission after all days are predicted
env.write_submission_file()
print([filename for filename in os.listdir('.') if '.csv' in filename])


# In[ ]:




