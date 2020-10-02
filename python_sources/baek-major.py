#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import plaidml.keras
# plaidml.keras.install_backend()


# In[2]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[3]:


#cd /content/gdrive/My Drive/Colab Notebooks/Beck/iic_new


# In[4]:


get_ipython().system('pip install pandas')


# In[ ]:





# In[5]:


from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


# In[6]:


2+3


# In[7]:


import time
from keras.models import Sequential
from array import *
import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math, time
import tensorflow as tf
import glob   
# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
#from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error
import statistics
from itertools import combinations
import scipy
from pandas_datareader import data as pdr
#import fix_yahoo_finance
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
from keras.layers import Flatten
from keras.models import load_model
from keras import optimizers
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.layers.normalization import BatchNormalization


# In[8]:


2+3


# In[9]:


# def get(tickers, startdate, enddate):
#   def data(ticker):
#     return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
#   datas = map (data, tickers)
#   return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

# tickers = ['AAPL', 'MSFT', 'AMZN', 'CSCO', 'ADBE', 'AMGN', 'NVDA', 'SBUX', 'ASML', 'GILD', 'QCOM', 'INTU', 'ISRG', 'CELG', 'ILMN', 'VRTX', 'CTSH', 'REGN', 'AMAT', 'EQIX',
#            'ROST', 'XLNX', 'FISV', 'EBAY', 'ORLY']
# all_data = get(tickers, datetime.datetime(2000, 1, 4), datetime.datetime(2017, 7, 27))

# all_data.describe()
get_ipython().system('pip install keras')


# In[ ]:





# In[ ]:





# In[29]:


prices_df_sp = pd.read_csv('../input/prices_df_sp.csv')

prices_df_sp['Date'] = pd.to_datetime(prices_df_sp['Date'])
prices_df_sp = prices_df_sp.set_index('Date')


# In[30]:


prices_df_sp = prices_df_sp.replace([np.inf, -np.inf], np.nan)
prices_df_sp = prices_df_sp.fillna(0)
prices_df_sp.head()


# In[13]:


# index_sp = pdr.get_data_yahoo('^GSPC', start=datetime.datetime(2000, 1, 4), end=datetime.datetime(2017, 7, 27))
# index_sp = index_sp.drop(['High','Adj Close','Low', 'Open', 'Volume'],1)
# index_sp.head()


# In[28]:


index_sp = pd.read_csv('../input/index_sp.csv')
index_sp['Date'] = pd.to_datetime(index_sp['Date'])
index_sp = index_sp.set_index('Date')
index_sp.head()


# In[31]:


com_all_names = prices_df_sp.columns
com_all_names


# In[32]:


com_p = []
for i in range(prices_df_sp.shape[1]):
  c_p = scipy.stats.pearsonr(prices_df_sp.iloc[:,i], index_sp['Close'])
  if c_p[1] < 0.5 :
    com_p.append(com_all_names[i])

com_p


# In[33]:


com_c = []
for i in com_p:
  c_p = scipy.stats.pearsonr(prices_df_sp[i], index_sp['Close'])
  if c_p[0] >= 0.8 :
    com_c.append(i)

com_c


# In[34]:


#sorting 
for i in range(len(com_c)): 
      
    # Find the minimum element in remaining  
    # unsorted array 
    min_idx = i 
    for j in range(i+1, len(com_c)): 
        #c_p = scipy.stats.pearsonr(prices_df_sp[com_c[min_idx]], index_sp['Close'])
        if scipy.stats.pearsonr(prices_df_sp[com_c[min_idx]], index_sp['Close'])[0] > scipy.stats.pearsonr(prices_df_sp[com_c[j]], index_sp['Close'])[0]:
            min_idx = j 
              
    # Swap the found minimum element with  
    # the first element         
    com_c[i], com_c[min_idx] = com_c[min_idx], com_c[i] 


# In[35]:


com_c


# In[36]:


ind = list(range(0,10))


# In[37]:


comb = list(combinations(ind, 5))


# In[38]:


comb


# In[39]:


look_back = 20
com_names = com_c[:10]


# In[40]:


def normalize_data(df):
  n_com = df.shape[1]
  max_c = df.max(axis = 0)
  
  min_c = df.min(axis = 0)
  for com in range(n_com):
    #print(mean_c[com])
    #print(stand_c[com])
    temp = df.iloc[:,com].values
    df.iloc[:,com] = (temp-min_c[com])/(max_c[com] - min_c[com])
  return df
prices_df_sp = normalize_data(prices_df_sp)
index_sp_pred = index_sp.copy()
index_sp_pred  = normalize_data(index_sp_pred)


# In[41]:


def load_data_prev(stock, seq_len):
    amount_of_features = len(stock.columns) # 5
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days
    
    result = np.array(result)
    #row = round(0.9 * result.shape[0]) # 90% split
    row = 2009
    train = result[:int(row), :] # 90% date, all features 
    
    x_train = train[:, :-1] 
    #y_train = train[:, -1][:,-1]
    
    x_test = result[int(row):, :-1] 
    #y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, x_test]


# In[42]:


def load_data_pred(stock, seq_len):
    amount_of_features = len(stock.columns) # 5
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days
    
    result = np.array(result)
    #row = round(0.9 * result.shape[0]) # 90% split
    row = 2009
    train = result[:int(row), :] # 90% date, all features 
    
    x_train = train[:, :-1] 
    y_train = train[:, -1][:,-1]
    
    x_test = result[int(row):, :-1] 
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, x_test]


# In[43]:


def load_data_target(stock, seq_len):
    amount_of_features = len(stock.columns) # 5
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days
    
    result = np.array(result)
    #row = round(0.9 * result.shape[0]) # 90% split
    row = 2009
    train = result[:int(row), :] # 90% date, all features 
    
    #x_train = train[:, :-1] 
    y_train = train[:, -1][:,-1]
    
    #x_test = result[int(row):, :-1] 
    y_test = result[int(row):, -1][:,-1]

    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [ y_train, y_test]


# In[44]:


#Xi_train, yi_train, Xi_test, yi_test = load_data_pred(index_sp_pred, look_back)


# In[72]:


def compile_fit(model,df,X_train, Xi_train,yi_train,i):
  
  #X_train = numpy.reshape(X_train, (trainX.shape[0], 1, trainX.shape[1]))
  NAME = "TEN_N" + str(i)
  #X_train = numpy.reshape(X_train, (trainX.shape[0], 1, trainX.shape[1]))
  tensorboard = TensorBoard(log_dir="logs/{}".format(NAME),write_graph=True)
  
  history = model.fit([X_train, Xi_train], yi_train, 
                epochs=200,batch_size = 60,
                validation_split=0.02,
                verbose=0,callbacks=[tensorboard])
  save_weight = 'model_weight_combination_new_' + str(i) + '.h5' 
  model.save(save_weight)
  # serialize model to JSON
  model_json = model.to_json()
  save_weight = 'model_weight_combination_new_' + str(i) + '.json'
  with open(save_weight, "w") as json_file:
    json_file.write(model_json)
  score = model.evaluate([X_train,Xi_train], yi_train, verbose=1)
  print('Train loss:', score)
  return model
  


# In[73]:


def build_model():
  # define input
  visible1 = Input(shape=(look_back,5))
  # feature extraction
  extract1 = LSTM(output_dim=5,activation='relu',return_sequences=False,kernel_regularizer=regularizers.l2(0.001))(visible1)
  # first interpretation model
  interp1 = Dense(output_dim = 3, activation='relu',kernel_regularizer=regularizers.l2(0.001))(extract1)
  # second interpretation model
  visible2 = Input(shape=(look_back,1))
  extract2 = LSTM(output_dim = 4,activation='relu',return_sequences=False,kernel_regularizer=regularizers.l2(0.001))(visible2)
  # first interpretation model
  interp2 = Dense(output_dim = 2, activation='relu',kernel_regularizer=regularizers.l2(0.001))(extract2)
  merge = concatenate([interp1, interp2])
  #flat = Flatten()(merge)
  # output
  pre_output1 = Dense(output_dim = 2, activation='relu',kernel_regularizer=regularizers.l2(0.001))(merge)
  pre_output2 = BatchNormalization()(pre_output1)
  output = Dense(output_dim = 1, activation='linear',kernel_regularizer=regularizers.l2(0.001))(pre_output2)
  model = Model(inputs=[visible1,visible2], outputs=output)
  # summarize layers
  #print(model.summary())
  adam = optimizers.Adam(lr=0.00005,decay=0.00005/200)
  model.compile(loss='mean_squared_error', optimizer = adam )
  return model


# In[74]:


def eval():
  
  Xi_train, Xi_test = load_data_pred(index_sp_pred, look_back)
  yi_train, yi_test = load_data_target(index_sp,look_back)
  model = build_model()
  for i in range(3):
    df = pd.DataFrame(index = index_sp.index)
    for j in range(5):
      start = time.time()
      #print("i = ",i," j = ",j)
      df[com_names[comb[i][j]]] = prices_df_sp[com_names[comb[i][j]]]
      #print('df = ',df)
    X_train, X_test = load_data_prev(df, look_back)
    print("i = ",i)
    print(df.columns)
    model = compile_fit(model,df,X_train, Xi_train, yi_train,i)
    end = time.time()
    print("time ",end - start," for combination ",i+1)


# In[75]:



print("it will start")
starte = time.time()
"the code you want to test stays here"
eval()
ende = time.time()
print(ende - starte)
print("FINISHED")


# In[76]:


pwd


# In[77]:


ls


# In[78]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[80]:



from zipfile import ZipFile 
import os 
 
def get_all_file_paths(directory): 
 
   # initializing empty file paths list 
   file_paths = [] 
 
   # crawling through directory and subdirectories 
   for root, directories, files in os.walk(directory): 
       for filename in files: 
           # join the two strings in order to form the full filepath. 
           filepath = os.path.join(root, filename) 
           file_paths.append(filepath) 
 
   # returning all file paths 
   return file_paths         
 
def main(): 
   # path to folder which needs to be zipped 
   directory = './'
 
   # calling function to get all file paths in the directory 
   file_paths = get_all_file_paths(directory) 
 
   # printing the list of all files to be zipped 
   print('Following files will be zipped:') 
   for file_name in file_paths: 
       print(file_name) 
 
   # writing files to a zipfile 
   with ZipFile('my_python_files.zip','w') as zip: 
       # writing each file one by one 
       for file in file_paths: 
           zip.write(file) 
 
   print('All files zipped successfully!')         
 
 
if __name__ == "__main__": 
   main() 


# In[81]:


2+2


# In[ ]:


get_ipython().system('pip install findspark')


# In[ ]:


import findspark
findspark.init()


# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:





# In[ ]:


from pyspark import SparkContext
from pyspark import SparkConf


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D
from keras.layers import MaxPooling2D, Dropout,Flatten
from keras import backend as K
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Expect to see a numpy n-dimentional array of (60000, 28, 28)
type(X_train), X_train.shape, type(X_train)


# In[ ]:


#Flatten each of our 28 X 28 images to a vector of 1, 784
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
#Check shape
X_train.shape, X_test.shape


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
def scaleData(data):       
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data) 
X_train = scaleData(X_train)
X_test = scaleData(X_test)


# In[ ]:


input_shape = (1,28,28) if K.image_data_format() == 'channels_first' else (28,28, 1)
keras_model = Sequential()
keras_model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape, padding='same'))
keras_model.add(MaxPooling2D(pool_size=(2, 2)))
keras_model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
keras_model.add(MaxPooling2D(pool_size=(2, 2)))
keras_model.add(Flatten())
keras_model.add(Dense(512, activation='relu'))
keras_model.add(Dropout(0.5))
keras_model.add(Dense(10, activation='softmax'))
keras_model.summary()


# In[ ]:


#from  import Keras2DML
from systemml import MLContext
import math
epochs = 5
batch_size = 100
samples = 60000
max_iter = int(epochs*math.ceil(samples/batch_size))
sysml_model = MLContext.Keras2DML(spark, keras_model, input_shape=(1,28,28), weights='weights_dir', batch_size=batch_size, max_iter=max_iter, test_interval=0, display=10)

sysml_model.fit(X_train, y_train)


# In[ ]:


################################### Keras2DML: Parallely training neural network with SystemML####################################### 
import tensorflow as tf
import keras
from keras.models import Sequential
from systemml.mllearn import Keras2DML

from keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling2D, Dropout,Flatten
from keras import backend as K
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt


from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Expect to see a numpy n-dimentional array of (60000, 28, 28)

type(X_train), X_train.shape, type(X_train)


#This time however, we flatten each of our 28 X 28 images to a vector of 1, 784

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# expect to see a numpy n-dimentional array of : (60000, 784) for Traning Data shape and (10000, 784) for Test Data shape
type(X_train), X_train.shape, X_test.shape


#We also use sklearn's MinMaxScaler for normalizing

from sklearn.preprocessing import MinMaxScaler
def scaleData(data):
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)

X_train = scaleData(X_train)
X_test = scaleData(X_test)


# We define the same Keras model as earlier

input_shape = (1,28,28) if K.image_data_format() == 'channels_first' else (28,28, 1)
keras_model = Sequential()
keras_model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape, padding='same'))
keras_model.add(MaxPooling2D(pool_size=(2, 2)))
keras_model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
keras_model.add(MaxPooling2D(pool_size=(2, 2)))
keras_model.add(Flatten())
keras_model.add(Dense(512, activation='relu'))
keras_model.add(Dropout(0.5))
keras_model.add(Dense(10, activation='softmax'))
keras_model.summary()


# Import the Keras to DML wrapper and define some basic variables

epochs = 5
batch_size = 100
samples = 60000
max_iter = int(epochs*math.ceil(samples/batch_size))

# Now create a SystemML model by calling the Keras2DML method and feeding it your spark session, Keras model, its input shape, and the  # predefined variables. We also ask to be displayed the traning results every 10 iterations.

sysml_model = Keras2DML(spark, keras_model, input_shape=(1,28,28), weights='weights_dir', batch_size=batch_size, max_iter=max_iter, test_interval=0, display=10)

# Initiate traning. More spark workers and better machine configuration means faster training!

sysml_model.fit(X_train, y_train)

# Test your model's performance on the secluded test set, and re-iterate if required 
sysml_model.score(X_test, y_test)


# In[ ]:


dir(systemml)


# In[ ]:


from systemml import MLContext


# In[ ]:


from sklearn import datasets, neighbors
from systemml.mllearn import LogisticRegression


# In[ ]:


import findspark
findspark.init()

import pyspark
import random

sc = pyspark.SparkContext(appName="Pi")
num_samples = 100000000

def inside(p):     
  x, y = random.random(), random.random()
  return x*x + y*y < 1

count = sc.parallelize(range(0, num_samples)).filter(inside).count()

pi = 4 * count / num_samples
print(pi)

sc.stop()


# In[ ]:


session_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)

