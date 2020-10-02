#!/usr/bin/env python
# coding: utf-8

# Inspired by **@raddar** with his great kernel ([https://www.kaggle.com/raddar/target-true-meaning-revealed/)] reveals that the 'target' is actually a ratio between historical purchase behaviour and further purchase behavour.
# 
# Therefore, as the transction data literally includes the time series info of each customer purchasing behaviour (from -13 to 2 month in 'month_lag' columns ), I try to do some feature engineering to use time series model (LSTM) and convolutional networks (SqueezeNet) for predicting the target.
# 
# The logic is to conduct feature engineering to describe the customer purchasing behaviour by month_lag.
# eg. month -13,  [purchase mean, std, mode], [city_code], [ratio of feature 1]  etc
# 
# And because the preprocessing takes times to finish, I make my dataset public for better use in this kernel.
#  
#  This is my first Kaggle Kernel and hope you enjoy it! Feel free to leave any comment to make it better.
#  Thanks.
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
import datetime
import gc
from keras.models import Model
from keras.layers import LSTM,Dropout,Input,Dense,BatchNormalization,Reshape,concatenate
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,AveragePooling2D,ConvLSTM2D, Concatenate,MaxPool2D,GRU
import gc
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop
import keras.backend as K
from sklearn.model_selection import StratifiedKFold


# With some exploration, we can find there's lots of 'purchase_amount' is too large to be a reasonable data point in transaction but can also find out those transactions did not valid as their 'authorized_flag' is false.
# 
# Therefore, the following preprocessing considered only the entry with 'authorized_flag' == Y and also introduce an ratio of vaid transaction per month.
# 
# 
# Detail columns for each customer feature for each month:
# 
# 
# 
# cols = ['_authorized_ratio',  
# '_city_id_mean', 
# '_city_id_count', 
# '_category_1_mean',
#  '_installments_mean',
#  '_installments_std',
#  '_installments_min',
#  '_installments_max',
#  '_installments_median', 
#  '_merchant_category_id_nunique',
#  '_merchant_id_nunique',         
#  '_purchase_amount_mean',
#  '_purchase_amount_std',          
#  '_purchase_amount_min',
#  '_purchase_amount_max',
#  '_purchase_amount_median',
#  '_state_id_nunique',
#  '_state_id_mode',
#  '_subsector_id_nunique',
#  '_subsector_id_mode',
#  '__category_20_mean',
#  '__category_21_mean',
#  '__category_22_mean',
#  '__category_23_mean',
#  '__category_24_mean',
#  '__category_25_mean',
#  '__category_3A_mean',
#  '__category_3B_mean',
#  '__category_3C_mean']
# 

# In[ ]:



'''df_timefeature = pd.read_csv('historical_transactions.csv')
#df_timefeature = pd.read_csv('new_merchant_transactions.csv')
df_timefeature.authorized_flag = df_timefeature.authorized_flag.map({'Y':1,'N':0})
df_timefeature.category_1 = df_timefeature.category_1.map({'Y':1,'N':0})
df_timefeature.category_2 = df_timefeature.category_2.fillna('0').astype('int')
df_timefeature.category_3 = df_timefeature.category_3.fillna('D')
df_timefeature.merchant_id = df_timefeature.merchant_id.fillna('NO')
df_timefeature = df_timefeature[['authorized_flag', 'card_id', 'city_id', 
                                  'category_1', 'installments','category_3', 'merchant_category_id', 'merchant_id', 'month_lag',
                                  'purchase_amount', 'purchase_date', 'category_2', 'state_id',
                                  'subsector_id']] 

for ft in ['category_2','category_3']:
    ft_ = pd.get_dummies(df_timefeature[ft])
    ft_.columns = [ft.join('_%s'%(cols)) for cols in ft_.columns]
    df_timefeature = df_timefeature.join(ft_, how = 'left')
    
df_timefeature = df_timefeature[['authorized_flag', 'card_id', 'city_id', 'category_1', 'installments',
       'merchant_category_id', 'merchant_id', 'month_lag',
       'purchase_amount', 'state_id','subsector_id', 
       '_category_20', '_category_21', '_category_22',
       '_category_23', '_category_24', '_category_25', '_category_3A',
       '_category_3B', '_category_3C']]
aggs = {'city_id':['mean','count']
 , 'category_1':['mean']
 , 'installments':['mean','std','min','max','median']
 , 'merchant_category_id':['nunique']
 , 'merchant_id':['nunique']
 ,'purchase_amount':['mean','std','min','max','median']
 , 'state_id':['nunique', lambda x: x.value_counts().index[0]]
 ,'subsector_id':['nunique', lambda x: x.value_counts().index[0]]
 ,'_category_20':['mean']
 ,'_category_21':['mean']
 ,'_category_22':['mean']
 ,'_category_23':['mean']
 ,'_category_24':['mean']
 ,'_category_25':['mean']
 ,'_category_3A':['mean']
 ,'_category_3B':['mean']
 ,'_category_3C':['mean']}


df_timefeature = df_timefeature.groupby(['card_id','month_lag']).agg(aggs)
'''


# In[ ]:


df_all= pd.read_pickle('../input/elo-timeseries-fe/df_all_Time.pkl')
df_all = df_all.set_index('card_id')
df_all = df_all.reset_index()

drop_columns = ['Train','card_id','target','outlier']
cate_columns = ['_feature_12','_feature_13','_feature_14','_feature_15','_feature_22','_feature_23','_feature_30',
               'active_year','active_month']
df_all.head()


# As the outliers matter in this competition , I construct Train and Valid set both includes outliers using random sampling with the same ratio of outliers.

# In[ ]:


train = df_all[df_all.Train == True]
y = train.target.values
#y = 10**(train['target'].values*np.log10(2))
#y_std = y.std()
#y_mean = y.mean()
#y = (y-y_mean)/y_std


val_idx = train[train.outlier == 1].sample(450).index.tolist()
val_idx = train[train.outlier == 0].sample(int(train[train.outlier == 0].shape[0]/5)).index.tolist()+val_idx

trn_idx = train[~train.index.isin(val_idx)].index.tolist()

x_train = train[train.index.isin(trn_idx)].drop(drop_columns,axis = 1)
x_train = dict(cate = x_train[cate_columns].values,
              mem = x_train.drop(cate_columns,axis = 1).values)
#x_train = x_train.drop(cate_columns,axis = 1).values
#y_train =  train[train.index.isin(trn_idx)].target.values
y_train =  y[trn_idx]



x_val = train[train.index.isin(val_idx)].drop(drop_columns,axis = 1)
x_val = dict(cate = x_val[cate_columns].values,
              mem = x_val.drop(cate_columns,axis = 1).values)
#x_val = x_val.drop(cate_columns,axis = 1).values
#y_val=  train[train.index.isin(val_idx)].target.values
y_val =  y[val_idx]

print('done!')
#print('Trn shape: ', x_train.shape)
#print('Val shape: ', x_val.shape)


# In[ ]:


plt.plot(np.sort(y),label= 'y')
plt.plot(np.sort(np.array(y_train).reshape(-1,)), label = 'y_train')
plt.plot(np.sort(np.array(y_val).reshape(-1,)), label = 'y_val')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(np.sort(y),label= 'y')
sns.distplot(np.sort(np.array(y_train).reshape(-1,)), label = 'y_train')
sns.distplot(np.sort(np.array(y_val).reshape(-1,)), label = 'y_val')
plt.legend()
plt.show()


# In[ ]:





# I did some experiences on the following four models and in this kernel I used the model with one LSTM layer for time series data  and dense layers for categorical data.

# In[ ]:


np.random.seed(0)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean((K.square(y_pred - y_true))))
    
def get_model(cate_shape, lstm_shape):
    Category = Input(cate_shape, name = 'cate')
    Lstm_Input = Input(lstm_shape, name = 'mem')
    Norm_1 = BatchNormalization()(Lstm_Input)
    C_D1 = Dense(64, activation= 'relu')(Category)
    C_D1 = Dense(32, activation= 'relu')(C_D1)
    C_D1 = Dense(16, activation= 'relu')(C_D1)
    
    #R_1 = Reshape((-1,464))(Lstm_Input)
    R_1 = Reshape((-1,29))(Lstm_Input)
    L_1 = LSTM(256)(R_1)
    
    main = concatenate([C_D1,L_1])
    main = Dense(128, activation= 'relu')(main)
    main = Dense(64, activation= 'relu')(main)
    main = Dense(32, activation= 'relu')(main)
    out = Dense(1, activation= 'linear')(main)
    
    model = Model([Category,Lstm_Input],out)
    model.summary()
    
    return model



print('Finish Construct Models')


# In[ ]:


def get_model_lstmonly(lstm_shape):
    #Category = Input(cate_shape, name = 'cate')
    Lstm_Input = Input(lstm_shape, name = 'mem')
    Norm_1 = BatchNormalization()(Lstm_Input)
    R_1 = Reshape((-1,29))(Norm_1)
    L_1 = LSTM(256)(R_1)
    #L_1 = LSTM(64)(L_1)
    
    #main = concatenate([C_D1,L_1])
    main = Dense(128, activation= 'relu')(L_1)
    main = Dense(64, activation= 'relu')(main)
    main = Dense(32, activation= 'relu')(main)
    #main = Dense(16, activation= 'relu')(main)
    out = Dense(1, activation= 'linear')(main)
    
    model = Model(Lstm_Input,out)
    model.summary()
    
    return model


def SqueezeNet(lstm_shape):
    def fire(np_filters, name="fire"):
        def layer(x):
            sq_filters, ex1_filters, ex2_filters = np_filters
            squeeze = Conv2D(sq_filters, (1, 1), activation='relu', padding='same', name=name + "/squeeze1x1")(x)
            expand1 = Conv2D(ex1_filters, (1, 1), activation='relu', padding='same', name=name + "/expand1x1")(squeeze)
            expand2 = Conv2D(ex2_filters, (3, 3), activation='relu', padding='same', name=name + "/expand3x3")(squeeze)
            out = Concatenate(axis=-1, name=name+'/concat')([expand1, expand2])
            return out
        return layer
    Lstm_Input = Input(lstm_shape, name = 'mem')
    R_1 = Reshape((-1,29,1))(Lstm_Input)
    Norm_1 = BatchNormalization()(R_1)
    Conv1 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv1')(Norm_1)
    Max_pool1 =MaxPool2D((2, 2), strides=(2, 2), name='max_pool1')(Conv1) #(10,10)
    fire1 = fire((32,64,64),name = 'fire1')(Max_pool1)
    fire2 = fire((32,64,64),name = 'fire2')(fire1)
    Max_pool2 = MaxPool2D((2, 2), strides=(2, 2), name='max_pool2')(fire2)#(5,5)
    Drop_1 = Dropout(0.25, name = 'dropout_1')(Max_pool2)
    fire3 = fire((32,128,128),name = 'fire3')(Drop_1)
    fire4 = fire((32,128,128),name = 'fire4')(fire3)
    Max_pool3 = MaxPool2D((2, 2), strides=(2, 2), name='max_pool3')(fire4)#(2,2)
    Drop_2 = Dropout(0.25, name = 'dropout_2')(Max_pool3)
    fire5 = fire((16,32,32),name = 'fire5')(Drop_2)
    fire6 = fire((16,64,64),name = 'fire6')(fire5)
    fire7 = fire((16,128,128),name = 'fire7')(fire6)
    Gl_avg_pooling = GlobalAveragePooling2D(name = 'gl_avg')(fire7)
    D_1= Dense(32,activation='linear', name='predictions')(Gl_avg_pooling)
    D_1= Dense(16,activation='linear', name='predictions')(D_1)
    Output_layer = Dense(1,activation='linear', name='predictions')(D_1)
    model = Model(Lstm_Input,Output_layer, name = 'SqueezeNet')
    model.summary()
        

    return model

def get_model_cnnlstm(lstm_shape):
     
    #Category = Input(cate_shape, name = 'cate')
    Lstm_Input = Input(lstm_shape, name = 'mem')
    Norm_1 = BatchNormalization()(Lstm_Input)
    R_1 = Reshape((-1,29,1))(Norm_1)
    
    C_1 = Conv2D(32,(3,3),strides=(1,1),data_format="channels_last",padding='same',activation='relu')(R_1)
    C_1 = Conv2D(32,(3,3),strides=(1,1),data_format="channels_last",padding='same',activation='relu')(C_1)
    C_1 = Conv2D(32,(3,3),strides=(1,1),data_format="channels_last",padding='same',activation='relu')(C_1)
    Pool = MaxPool2D(pool_size=(2,2),strides=(2,2))(C_1)
    R_2 = Reshape(((8, 14*32)))(Pool)
    L_1 = LSTM(256)(R_2)
    #L_1 = LSTM(64)(L_1)
    
    #main = concatenate([C_D1,L_1])
    main = Dense(32, activation= 'relu')(L_1)
    main = Dense(16, activation= 'relu')(main)
    out = Dense(1, activation= 'linear')(main)
    
    model = Model(Lstm_Input,out)
    model.summary()
    
    return model


def get_model_Squeezelstm(lstm_shape):
    def fire(np_filters, name="fire"):
        def layer(x):
            sq_filters, ex1_filters, ex2_filters = np_filters
            squeeze = Conv2D(sq_filters, (1, 1), activation='relu', padding='same', name=name + "/squeeze1x1")(x)
            expand1 = Conv2D(ex1_filters, (1, 1), activation='relu', padding='same', name=name + "/expand1x1")(squeeze)
            expand2 = Conv2D(ex2_filters, (3, 3), activation='relu', padding='same', name=name + "/expand3x3")(squeeze)
            out = Concatenate(axis=-1, name=name+'/concat')([expand1, expand2])
            return out
        return layer
    #Category = Input(cate_shape, name = 'cate')
    Lstm_Input = Input(lstm_shape, name = 'mem')
    #Norm_1 = BatchNormalization()(Lstm_Input)
    R_1 = Reshape((-1,29,1))(Lstm_Input)
    Conv1 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1')(R_1)
    Max_pool1 =MaxPool2D((2, 2), strides=(2, 2), name='max_pool1')(Conv1) #(10,10)
    fire1 = fire((16,32,32),name = 'fire1')(Max_pool1)
    fire2 = fire((16,64,64),name = 'fire2')(fire1)
    fire3 = fire((16,64,64),name = 'fire3')(fire2)
    Max_pool2 = MaxPool2D((2, 2), strides=(2, 2), name='max_pool2')(fire3)#(5,5)
    R_2 = Reshape(((4, 7*128)))(Max_pool2)
    L_1 = LSTM(128)(R_2)
    main = Dense(32, activation= 'relu')(L_1)
    main = Dense(16, activation= 'relu')(main)
    main = Dense(8, activation= 'relu')(main)
    out = Dense(1, activation= 'linear')(main)
    
    model = Model(Lstm_Input,out)
    model.summary()
    
    return model


# In[ ]:


model = get_model((9,), (464,))
#model = get_model_lstmonly((464,))
#model = SqueezeNet((464,))
#model =  get_model_cnnlstm((464,))
#model = get_model_Squeezelstm((464,))

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.compile(loss='mse', optimizer=Adam(),  metrics=[root_mean_squared_error])
history = model.fit(x_train, y_train, epochs=20,verbose=1, 
                    batch_size=32,validation_data =(x_val,y_val), 
                    callbacks=[early_stopping])


# In[ ]:


#%%

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.show()


# In[ ]:



submission = pd.read_csv('../input/elo-merchant-category-recommendation/sample_submission.csv')
x_ts = df_all[df_all.Train == False]
x_ts = x_ts.drop(drop_columns,axis = 1)
x_ts = dict(cate = x_ts[cate_columns].values,
              mem = x_ts.drop(cate_columns,axis = 1).values)

y_pre = model.predict(x_ts)
#y_pre = (y_pre*y_std)+y_mean
submission['target'] = y_pre
submission.to_csv('LSTM.csv',index = False)
 


# In[ ]:


plt.plot(np.sort(y),label= 'y')
plt.plot(np.sort(np.array(y_pre).reshape(-1,)), label = 'y_pre')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(np.sort(y),label= 'y')
sns.distplot(np.sort(np.array(y_pre).reshape(-1,)), label = 'y_pre')
plt.legend()
plt.show()


# The result might not be very outstanding as other boosting method and great kernels because of the sample of outliers are too less for a DL model. Some upsampling method might be required for further improment. 
# 
# However, it's a good practice to use different method other than boosting for this competiton.
# I'll keep improve the the kernel in the following days.
# 
# Thanks for reading my kernel! Hope you enjoy it.
# 
