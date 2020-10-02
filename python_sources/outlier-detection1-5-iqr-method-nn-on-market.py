#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews


#***********************************import keras
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.losses import binary_crossentropy
import keras.callbacks as callbacks
from keras.callbacks import Callback
from keras.applications.xception import Xception
from keras.layers import multiply

import keras
from keras import optimizers
from keras.legacy import interfaces
from keras.utils.generic_utils import get_custom_objects

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.regularizers import l2
from keras.layers.core import Dense, Lambda
from keras.layers.merge import concatenate, add
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras.optimizers import SGD


# In[ ]:


env = twosigmanews.make_env()
(market_train, _) = env.get_training_data()


# In[ ]:


cat_cols = ['assetCode']
num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10']


# In[ ]:


from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(market_train.index.values,test_size=0.25, random_state=23)


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


for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(market_train.loc[:, cat].astype(str).unique())}
    market_train[cat] = market_train[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets


# # Handling numerical variables

# In[ ]:


#market_train['timess'] =market_train.time.dt.strftime("%Y%m%d").astype(int)
#market_train = market_train.loc[market_train['timess'] > 20110000]


# In[ ]:


market_train.head()
market_train.shape[0]


# In[ ]:


types = market_train['assetCode'].unique()
d = {type: market_train[market_train['assetCode'] == type] for type in types} #creating dataframes for each unique assetCode


# In[ ]:


market_train.tail()


# In[ ]:


a=0
for type in types:
    a=a+d[type].shape[0]
a    


# In[ ]:


market_train.shape[0]


# In[ ]:


d[0]


# In[ ]:


#Setting quantiles and outlier borders

low = .25
high = .75

bounds = {}
for type in types:
    filt_df = d[type].loc[:, d[type].columns != 'assetCode']# Remove 'Type' Column
    filt_df = filt_df.loc[:, filt_df.columns != 'universe']
    quant_df = filt_df.quantile([low, high])
    IQR = quant_df.iloc[1,:]-  quant_df.iloc[0,:]
    quant_df.iloc[0,:] = quant_df.iloc[0,:] - 1.5*IQR
    quant_df.iloc[1,:] = quant_df.iloc[1,:] + 1.5*IQR
    bounds[type] = quant_df
    bounds[type] = bounds[type].reset_index()
    bounds[type] = bounds[type].drop("index", axis=1)
bounds[1]


# In[ ]:


bounds[1].loc[1,"volume"]


# In[ ]:


market_train.head()


# In[ ]:


market_train1=pd.DataFrame()
for type in types:
    for column in num_cols:
        d[type] = d[type].loc[d[type][column]>=bounds[type].loc[0,column]]
        d[type] = d[type].loc[d[type][column]<=bounds[type].loc[1,column]]
    market_train1=pd.concat([market_train1,d[type]], ignore_index=True)


# In[ ]:


orig_len=market_train.shape[0]
new_len = market_train1.shape[0]
rmv_len1 = np.abs(orig_len-new_len)
print('There were %i lines removed' %rmv_len1)


# In[ ]:





# In[ ]:


market_train1['close_open_ratio'] = np.abs(market_train1['close']/market_train1['open'])
market_train1 = market_train1.loc[market_train1['close_open_ratio'] < 1.4]
market_train1 = market_train1.loc[market_train1['close_open_ratio'] > 0.4]
#market_train1 = market_train1.drop(columns=['close_open_ratio'])

newer_len=market_train1.shape[0]
rmv_len2 = np.abs(new_len-newer_len)
print('There were %i lines removed additionally' %rmv_len2)


# In[ ]:


market_train.head()


# In[ ]:


market_train1=market_train1.dropna()
rmv_len3 = np.abs(newer_len-market_train1.shape[0])
print('There were %i lines removed additionally' %rmv_len3)
deleted_rows=rmv_len1+rmv_len2+rmv_len3
print('There were %i lines in total removed' %deleted_rows)


# In[ ]:



market_train1['average'] = (market_train1['close'] + market_train1['open'])/2
market_train1['pricevolume'] = market_train1['volume'] * market_train1['close']


# In[ ]:


market_train1.shape[0]


# In[ ]:


num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10','close_open_ratio','average','pricevolume']


# In[ ]:


market_train=market_train1.copy()


# In[ ]:


train_indices, val_indices = train_test_split(market_train.index.values,test_size=0.25, random_state=23)


# In[ ]:


from sklearn.preprocessing import StandardScaler
 
market_train[num_cols] = market_train[num_cols].fillna(0)
print('scaling numerical columns')

scaler = StandardScaler()

#col_mean = market_train[col].mean()
#market_train[col].fillna(col_mean, inplace=True)
scaler = StandardScaler()
market_train[num_cols] = scaler.fit_transform(market_train[num_cols])


# In[ ]:


market_train.head()


# In[ ]:


#pd.merge(market_train, newsgp, how='left', on=['time', 'assetCode']


# # Define NN Architecture

# Todo: add explanaition of architecture

# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

#categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
categorical_logits = Flatten()(categorical_embeddings[0])
#categorical_logits = Dense(32,activation='relu')(categorical_logits)
#categorical_logits =Dropout(0.5)(categorical_logits)
#categorical_logits =BatchNormalization()(categorical_logits)
categorical_logits = Dense(32,activation='relu')(categorical_logits)

a=len(num_cols)
numerical_inputs = Input(shape=(a,), name='num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)

numerical_logits = Dense(128,activation='relu')(numerical_logits)
#numerical_logits=Dropout(0.3)(numerical_logits)
#numerical_logits = BatchNormalization()(numerical_logits)
#numerical_logits = Dense(128,activation='relu')(numerical_logits)
numerical_logits = Dense(64,activation='relu')(numerical_logits)

logits = Concatenate()([numerical_logits,categorical_logits])
logits = Dense(64,activation='relu')(logits)
out = Dense(1, activation='sigmoid')(logits)

model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
model.compile(optimizer='adam',loss=binary_crossentropy)


# In[ ]:


# Lets print our model
model.summary()


# In[ ]:


def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices,'returnsOpenNextMktres10']>=0).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(market_train, train_indices)
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train, val_indices)


# In[ ]:


market_train.head()


# In[ ]:


class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            callbacks.ModelCheckpoint("model.hdf5",monitor='val_my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1),
            swa,
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)


# # Train NN model

# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

"""
epochs = 10
snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1,init_lr=1e-3)
batch_size = 32
swa = SWA('model_swa.hdf5',6)
history = model.fit(X_train,y_train.astype(int),
                    validation_data=(X_valid,y_valid.astype(int)),
                    epochs=epochs,
                    #batch_size=batch_size,
                    callbacks=snapshot.get_callbacks(),shuffle=True,verbose=2)
                    
early_stop = EarlyStopping( mode = 'max',patience=15, verbose=1)
check_point = ModelCheckpoint('model.hdf5', mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau( mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
#check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
#early_stop = EarlyStopping(patience=5,verbose=True)
model.fit(X_train,y_train.astype(int),
                    validation_data=(X_valid,y_valid.astype(int)), 
                    epochs=15,
                    callbacks=[check_point,reduce_lr,early_stop], 
                    verbose=2)

model.fit(X_train,y_train.astype(int),
          validation_data=(X_valid,y_valid.astype(int)),
          epochs=10,
          verbose=True,
          callbacks=[early_stop,check_point]) 
"""


check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5,verbose=True)
model.fit(X_train,y_train.astype(int),
          validation_data=(X_valid,y_valid.astype(int)),
          epochs=5,
          verbose=True,
          callbacks=[early_stop,check_point]) 


# In[ ]:


"""
try:
    print('using swa weight model')
    model.load_weights('model_swa.hdf5')
except:
    model.load_weights('model.hdf5')
"""


# # Evaluation of Validation Set

# In[ ]:


# distribution of confidence that will be used as submission
model.load_weights('model.hdf5')
confidence_valid = model.predict(X_valid)[:,0]*2 -1
print(accuracy_score(confidence_valid>0,y_valid))
plt.hist(confidence_valid, bins='auto')
plt.title("predicted confidence")
plt.show()


# In[ ]:


# calculation of actual metric that is used to calculate final score
r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)


# # Prediction

# In[ ]:


days = env.get_prediction_days()


# In[ ]:


n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    
    t = time.time()

    market_obs_df['assetCode_encoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    market_obs_df['close_open_ratio'] = np.abs(market_obs_df['close']/market_obs_df['open'])
    market_obs_df['average'] = (market_obs_df['close'] + market_obs_df['open'])/2
    market_obs_df['pricevolume'] = market_obs_df['volume'] * market_obs_df['close']
    market_obs_df[num_cols] = market_obs_df[num_cols].fillna(0)
    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])
    X_num_test = market_obs_df[num_cols].values
    X_test = {'num':X_num_test}
    X_test['assetCode'] = market_obs_df['assetCode_encoded'].values
    
    prep_time += time.time() - t
    
    t = time.time()
    market_prediction = model.predict(X_test)[:,0]*2 -1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() -t
    
    t = time.time()
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')


# In[ ]:


# distribution of confidence as a sanity check: they should be distributed as above
plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()

