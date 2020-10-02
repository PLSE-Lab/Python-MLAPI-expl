#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install git+git://github.com/allen-chiang/Kaggle-M5.git@master')
import os
import numpy as np 
import pandas as pd
import tensorflow as tf
import pywt
from time_series_transform.base import *
from time_series_transform.tensorflow_adopter import *
from time_series_transform.sequence_transfomer import *
from sklearn.preprocessing import LabelEncoder,StandardScaler,OrdinalEncoder
from time_series_transform.time_series_transformer import Pandas_Time_Series_Dataset
for dirname, _, filenames in os.walk('/kaggle/input'):
   for filename in filenames:
       print(os.path.join(dirname, filename))


# # Data Preparation

# Feature Selection
# * Sequences of sells number grouping by window size
# * The current weekday
# * The current sells prices
# * Store id, state id, item id, department id, category id
# * The year, month
# * Past and future event (To-do)
# * Snap of the state

# In[ ]:


calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
sell_prices['id'] = sell_prices.store_id + sell_prices.item_id
sell_prices.index = sell_prices.id
sell_prices = sell_prices.drop(['store_id','item_id'],axis =1)
calendar = calendar[['d','wm_yr_wk']]
calendar['d'] = calendar.d.apply(lambda x: x.replace('d_','')).astype(int)
sell_prices = sell_prices.merge(calendar,how = 'left',on = 'wm_yr_wk')
sell_prices = sell_prices.pivot(index='id', 
                 columns='d', 
                 values='sell_price'
                )
sell_prices.columns = map(lambda x: 'p_'+str(x),sell_prices.columns)


# In[ ]:


def calendar_join(calendar, df, df_date_header,calendar_header, calendarCol,date_ahead=0):
    day = df.columns[df.columns.str.contains(df_date_header)].tolist()
    tmp = calendar[calendar.d.isin(day)][calendarCol]
    tmp = pd.DataFrame(tmp).transpose()
    tmp.columns = map(lambda x:f'{calendar_header}{x+1+date_ahead}',tmp.columns)
    for i in tmp:
        df[i] = tmp[i].values[0]
    return df


# In[ ]:


calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
calendar['date_ndash'] = calendar.date.apply(lambda x: x.replace('-',''))
calendar.tail(10)


# In[ ]:


df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
df.index = df.id
df = df.drop('id',axis=1)
df = calendar_join(calendar,df,'d_','w_','wday')
df = calendar_join(calendar,df,'d_','snapca_','snap_CA',10)
df = calendar_join(calendar,df,'d_','snaptx_','snap_TX',10)
df = calendar_join(calendar,df,'d_','snapwi_','snap_WI',10)


# In[ ]:


sell_prices['merge_id'] = sell_prices.index
df['merge_id'] = df.store_id+df.item_id
df = df.merge(sell_prices,how = 'left' ,on="merge_id")
del sell_prices
df = calendar_join(calendar,df,'d_','month_','month')


# In[ ]:


Rolling_Window = 30
DateList = df.columns[df.columns.str.contains('d_')]
tmp = df[DateList].rolling(Rolling_Window,axis =1)
kurt = tmp.kurt().fillna(0)
mean = tmp.mean().fillna(0)
kurt.columns = map(lambda x: x.replace('d_','kurt_'),kurt.columns)
mean.columns = map(lambda x: x.replace('d_','mean_'),mean.columns)
for i in kurt:
    df[i] = kurt[i]
for i in mean:
    df[i] = mean[i]
del tmp
del kurt
del mean


# In[ ]:


df.info()


# # Train and Test Split
# * Training date --> day 1601~1912
# * Test date --> Day 1913-Window Size ~ 1914

# In[ ]:


def _tag_list(tag,tagList):
    return list(map(lambda x: f"{tag}_{x}",tagList))

def tag_list(train_date,test_date,tag):
    return _tag_list(tag,train_date),_tag_list(tag,test_date)


# In[ ]:


CATEGORICAL_DIM = [
    'item_id','dept_id','cat_id','store_id','state_id'
]


WINDOW_SIZE = 15
TEST_RANGE = WINDOW_SIZE + 20
TRAIN = list(range(1601,1912))
TEST = list(range(1913-TEST_RANGE,1914))

Train_Date,Test_Date = tag_list(TRAIN,TEST,'d')
Train_WeekDay, Test_WeekDay = tag_list(TRAIN,TEST,'w')
Train_PriceDay,Test_PriceDay = tag_list(TRAIN,TEST,'p')
Train_SnapCA,Test_SnapCA = tag_list(TRAIN,TEST,'snapca')
Train_SnapTX,Test_SnapTX = tag_list(TRAIN,TEST,'snaptx')
Train_SnapWI,Test_SnapWI = tag_list(TRAIN,TEST,'snapwi')
Train_Month,Test_Month = tag_list(TRAIN,TEST,'month')
Train_Kurt,Test_Kurt = tag_list(TRAIN,TEST,'kurt')
Train_Mean,Test_Mean = tag_list(TRAIN,TEST,'mean')


Train_Seq_Len = len(TRAIN)
Test_Seq_Len= len(TEST)


# In[ ]:


labelEncoderDict = {}
embeddingNum = {}
for i in CATEGORICAL_DIM:
    le = LabelEncoder()
    le.fit(df[i])
    labelEncoderDict[i] = le
    df[i] = le.transform(df[i])
    embeddingNum[i] = len(le.classes_)


# In[ ]:


testDate = Test_Date+Test_WeekDay+Test_PriceDay+Test_SnapCA+Test_SnapTX+Test_SnapWI+Test_Month + Test_Kurt+Test_Mean
trainDate = Train_Date+Train_WeekDay+Train_PriceDay+Train_SnapCA+Train_SnapTX+Train_SnapWI+Train_Month+ Train_Kurt+Train_Mean

test = df[CATEGORICAL_DIM+testDate]
train = df[CATEGORICAL_DIM+trainDate]


# In[ ]:


ptt = Pandas_Time_Series_Dataset(train)
for i in CATEGORICAL_DIM:
    ptt.set_config(name = i,
                   colNames = [i],
                   tensorType = 'category',
                   windowSize=WINDOW_SIZE,
                   seqSize=Train_Seq_Len,
                   outType=np.float32,
                   sequence_stack = None,
                   isResponseVar = False
                  )
ptt.set_config(name = 'sells',
               colNames = Train_Date,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt.set_config(name = 'sells_kurt',
               colNames = Train_Kurt,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = 'sells',
               isResponseVar = False
              )
ptt.set_config(name = 'sells_mean',
               colNames = Train_Mean,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = 'sells',
               isResponseVar = False
              )
ptt.set_config(name = 'label',
               colNames = Train_Date,
               tensorType = 'label',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = True
              )
ptt.set_config(name = 'weekDay',
               colNames = Train_WeekDay,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt.set_config(name = 'price',
               colNames = Train_PriceDay,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt.set_config(name = 'month',
               colNames = Train_Month,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt.set_config(name = 'snapca',
               colNames = Train_SnapCA,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt.set_config(name = 'snapwi',
               colNames = Train_SnapWI,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt.set_config(name = 'snaptx',
               colNames = Train_SnapTX,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )


# In[ ]:


def data_transform(X):
    # we only use the current week day and current price as feature
    # ignore the rest of data within the window since the change is very limited.
    X['weekDay'] = X['weekDay'][:,-1,0] 
    X['price'] = X['price'][:,-1,0]
    X['month'] = X['month'][:,-1,0]
    return (X,X['label'])


# In[ ]:


gen = ptt.make_data_generator()
tfg = TFRecord_Generator("train.tfRecord")
tfg.write_tfRecord(gen)
dataset = tfg.make_tfDataset()
train_dataset = dataset.map(data_transform)


# In[ ]:


ptt_val = Pandas_Time_Series_Dataset(test)
for i in CATEGORICAL_DIM:
    ptt_val.set_config(name = i,
                   colNames = [i],
                   tensorType = 'category',
                   windowSize=WINDOW_SIZE,
                   seqSize=Test_Seq_Len,
                   outType=np.float32,
                   sequence_stack = None,
                   isResponseVar = False
                  )
ptt_val.set_config(name = 'sells',
               colNames = Test_Date,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Test_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt_val.set_config(name = 'sells_kurt',
               colNames = Test_Kurt,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Test_Seq_Len,
               outType=np.float32,
               sequence_stack = 'sells',
               isResponseVar = False
              )
ptt_val.set_config(name = 'sells_mean',
               colNames = Test_Mean,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Test_Seq_Len,
               outType=np.float32,
               sequence_stack = 'sells',
               isResponseVar = False
              )
ptt_val.set_config(name = 'label',
               colNames = Test_Date,
               tensorType = 'label',
               windowSize=WINDOW_SIZE,
               seqSize=Test_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = True
              )
ptt_val.set_config(name = 'weekDay',
               colNames = Test_WeekDay,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Test_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt_val.set_config(name = 'price',
               colNames = Test_PriceDay,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt_val.set_config(name = 'month',
               colNames = Test_Month,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt_val.set_config(name = 'snapca',
               colNames = Test_SnapCA,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt_val.set_config(name = 'snapwi',
               colNames = Test_SnapWI,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )
ptt_val.set_config(name = 'snaptx',
               colNames = Test_SnapTX,
               tensorType = 'sequence',
               windowSize=WINDOW_SIZE,
               seqSize=Train_Seq_Len,
               outType=np.float32,
               sequence_stack = None,
               isResponseVar = False
              )


# In[ ]:


vgen = ptt_val.make_data_generator()
vtfg = TFRecord_Generator("val.tfRecord")
vtfg.write_tfRecord(vgen)
dataset = vtfg.make_tfDataset()
val_dataset = dataset.map(data_transform)


# # LSTM and Embedding Model

# In[ ]:


with tf.device("/gpu:0"):
    sells_input = tf.keras.layers.Input(shape=(WINDOW_SIZE,3),name = 'sells')
    state_input = tf.keras.layers.Input(shape=1,name = 'state_id')
    store_input = tf.keras.layers.Input(shape=1,name = 'store_id')
    item_input =  tf.keras.layers.Input(shape=1,name = 'item_id')
    cat_input = tf.keras.layers.Input(shape=1,name = 'cat_id')
    dept_input = tf.keras.layers.Input(shape=1,name = 'dept_id')
    month_input = tf.keras.layers.Input(shape=1,name= 'month')

    weekDay_input = tf.keras.layers.Input(shape=1,name = 'weekDay')
    price_input = tf.keras.layers.Input(shape=1,name = 'price')
    snapca_input = tf.keras.layers.Input(shape=(WINDOW_SIZE,1),name = 'snapca')
    snapwi_input = tf.keras.layers.Input(shape=(WINDOW_SIZE,1),name = 'snapwi')
    snaptx_input = tf.keras.layers.Input(shape=(WINDOW_SIZE,1),name = 'snaptx')

    month = tf.keras.layers.Embedding(13,1)(month_input)
    state  = tf.keras.layers.Embedding(embeddingNum['state_id'],1)(state_input)
    store  = tf.keras.layers.Embedding(embeddingNum['store_id'],1)(store_input)
    item  = tf.keras.layers.Embedding(embeddingNum['item_id'],1)(item_input)
    dept  = tf.keras.layers.Embedding(embeddingNum['dept_id'],1)(dept_input)
    cat  = tf.keras.layers.Embedding(embeddingNum['cat_id'],1)(cat_input)
    weekDay = tf.keras.layers.Embedding(8,1)(weekDay_input)


    snapca = tf.keras.layers.Flatten()(snapca_input)
    snapca = tf.keras.layers.Embedding(2,1,input_length= WINDOW_SIZE)(snapca)


    snaptx = tf.keras.layers.Flatten()(snaptx_input)
    snaptx = tf.keras.layers.Embedding(2,1,input_length= WINDOW_SIZE)(snaptx)


    snapwi = tf.keras.layers.Flatten()(snapwi_input)
    snapwi = tf.keras.layers.Embedding(2,1,input_length= WINDOW_SIZE)(snapwi)

    lstm = tf.keras.layers.Concatenate()([snapwi,snaptx,snapca,sells_input])
    lstm = tf.keras.layers.GRU(2)(lstm)



    state  = tf.keras.layers.Flatten()(state)
    store  = tf.keras.layers.Flatten()(store)
    item  = tf.keras.layers.Flatten()(item)
    dept  = tf.keras.layers.Flatten()(dept)
    cat  = tf.keras.layers.Flatten()(cat)
    weekDay = tf.keras.layers.Flatten()(weekDay)
    month = tf.keras.layers.Flatten()(month)


    loc = tf.keras.layers.Concatenate()([state,store])
    product = tf.keras.layers.Concatenate()([item,dept,cat,weekDay,month])




    cate = tf.keras.layers.Concatenate()([product,loc,lstm])
    cate = tf.keras.layers.Dense(10,'elu')(cate)
    dense  = tf.keras.layers.Dense(1)(cate)


    lstm_embed = tf.keras.models.Model({
        'sells':sells_input,
        'state_id':state_input,
        'store_id':store_input,
        'item_id':item_input,
        'dept_id':dept_input,
        'cat_id':cat_input,
        'weekDay':weekDay_input,
        'price':price_input,
        'snapca':snapca_input,
        'snapwi':snapwi_input,
        'snaptx':snaptx_input,
        'month':month_input
    },dense)


    rmsp = tf.keras.optimizers.RMSprop(lr = 0.2,
                                       decay = 1e-5,
                                   )

    lstm_embed.compile(optimizer=rmsp, 
                       loss='mse',
                       metrics = ['mae','mse']
                      )


# In[ ]:


lstm_embed.summary()


# In[ ]:


tf.keras.utils.plot_model(lstm_embed,
                          show_shapes=True
                         )


# In[ ]:


del train


# In[ ]:


EVALUATION_INTERVAL = 50
Validation_STEP =  50
EPOCHS = 100

batchsize = (len(Train_Date)-WINDOW_SIZE)*int(len(df)/EVALUATION_INTERVAL)+1
train_dataset = train_dataset.unbatch().batch(batchsize).prefetch(tf.data.experimental.AUTOTUNE).repeat()

batchsize = (len(Test_Date)-WINDOW_SIZE)*int(len(df)/Validation_STEP)+1
val_dataset= val_dataset.unbatch().batch(batchsize).prefetch(tf.data.experimental.AUTOTUNE).repeat()


# In[ ]:


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    min_delta=0.001, 
    patience=10, 
    verbose=1, 
    mode='auto',
    baseline=None, 
    restore_best_weights=True
)

lstm_embed.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch= EVALUATION_INTERVAL,
    validation_data = val_dataset,
    validation_steps = Validation_STEP,
    callbacks = [early_stopping]
    )


# In[ ]:


dataset = vtfg.make_tfDataset()
test_data = dataset.map(data_transform).prefetch(tf.data.experimental.AUTOTUNE)
prd = lstm_embed.predict(test_data,verbose = 1,steps = 50)
# prdDf = pd.DataFrame(prd.reshape((len(df),-1)))
prdDf = pd.DataFrame(prd.reshape((50,-1)))


# In[ ]:


testDf = pd.DataFrame(test[Test_Date[-21:]].to_records()).drop('index',axis =1)


# In[ ]:


from matplotlib import pyplot as plt
for ix in range(0,5):
    dataFm = pd.DataFrame()
    dataFm['prd'] = prdDf.iloc[ix].values
    dataFm['real'] = testDf.iloc[ix].values
    dataFm.plot(marker='x')


# In[ ]:




