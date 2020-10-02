#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
os.chdir("/kaggle/input")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np 
import pandas as pd
import pickle
import os
import datetime
from tqdm import tqdm
from statistics import mean 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking, Bidirectional, TimeDistributed, Input,concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder 
from keras.utils import to_categorical
from sklearn.metrics import f1_score

# le = preprocessing.LabelEncoder()
# le.classes_ = np.load('../input/audiolstm/le_BSH.npy')

# onehotencoder = OneHotEncoder() 

duration_list=[]
batch_sizes=[]
epochs_list=[]
optimizer_list=[]
training_loss_list=[]
test_loss_list=[]
pearson_list=[]
spearman_list=[]


# In[ ]:



with open('../input/audiolstm/audio_featDict.pkl', 'rb') as f:
    audio_featDict=pickle.load(f)
    
with open('../input/audiolstm/audio_featDictMark2.pkl', 'rb') as f:
    audio_featDictMark2=pickle.load(f)
    
    
traindf= pd.read_csv("../input/audiolstm/train_split_BSH.csv")
testdf=pd.read_csv("../input/audiolstm/test_split_BSH.csv")
valdf=pd.read_csv("../input/audiolstm/val_split_BSH.csv")

    
 
with open('../input/audiolstm/OG_multimodal_mittens_train.pkl', 'rb') as f:
    text_train=pickle.load(f)
    
with open('../input/audiolstm/OG_multimodal_mittens_test.pkl', 'rb') as f:
    text_test=pickle.load(f)
    
with open('../input/audiolstm/OG_multimodal_mittens_val.pkl', 'rb') as f:
    text_val=pickle.load(f)

    

error=[]
error_text=[]


def get_label(inp):
    if inp=="Buy":
        return [1,0,0]
    elif inp=="Hold":
        return [0,1,0]
    elif inp=="Sell":
        return [0,0,1]
    

def change_df(df):
    df['BSH_day3'] = df.apply(lambda row : get_label(row.BSH_day3), axis = 1)
    df['BSH_day7'] = df.apply(lambda row : get_label(row.BSH_day7), axis = 1)
    df['BSH_day15'] = df.apply(lambda row : get_label(row.BSH_day15), axis = 1)
    df['BSH_day30'] = df.apply(lambda row : get_label(row.BSH_day30), axis = 1) 
    return df
    

train_df = change_df(traindf)
val_df = change_df(valdf)
test_df = change_df(testdf)
    
    

def ModifyData(df,text_dict):
    X=[]
    X_text=[]
    y_3days=[]
    y_7days=[]
    y_15days=[]
    y_30days=[]

    for index,row in df.iterrows():
        
        try:
            X_text.append(text_dict[row['text_file_name'][:-9]])
        except:
            error_text.append(row['text_file_name'][:-9])

        lstm_matrix_temp = np.zeros((520, 26), dtype=np.float64)
        i=0
        
        try:
            speaker_list=list(audio_featDict[row['text_file_name'][:-9]])
            speaker_list=sorted(speaker_list, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
            for sent in speaker_list:
                lstm_matrix_temp[i, :]=audio_featDict[row['text_file_name'][:-9]][sent]+audio_featDictMark2[row['text_file_name'][:-9]][sent]
                i+=1
            X.append(lstm_matrix_temp)

        except:
            Padded=np.zeros((520, 26), dtype=np.float64)
            X.append(Padded)
            error.append(row['text_file_name'][:-9])
            
        
        y_3days.append((row['BSH_day3']))
        y_7days.append((row['BSH_day7']))
        y_15days.append((row['BSH_day15']))
        y_30days.append((row['BSH_day30']))
        
    X=np.array(X)
    X_text=np.array(X_text)
    X=np.nan_to_num(X)
    
    y_3days=np.array(y_3days)
    y_7days=np.array(y_7days)
    y_15days=np.array(y_15days)
    y_30days=np.array(y_30days)
    
        
    return X,X_text,y_3days,y_7days,y_15days,y_30days



X_train_audio,X_train_text,y_train3days, y_train7days, y_train15days, y_train30days=ModifyData(traindf,text_train)

X_test_audio,X_test_text, y_test3days, y_test7days, y_test15days, y_test30days=ModifyData(testdf,text_test)

X_val_audio,X_val_text,y_val3days, y_val7days, y_val15days, y_val30days=ModifyData(valdf,text_val)



input_audio_shape = (X_train_audio.shape[1], X_train_audio.shape[2])
input_text_shape = (X_train_text.shape[1],X_train_text.shape[2])


# In[ ]:


get_ipython().system('pip install tensorflow_addons')


# In[ ]:


import json
import os,sys
os.chdir("/kaggle/input/ensemble/attenion_align/")
import pathlib
import numpy as np
from delta.models.audio_only_clf_model import AlignClassModel
import numpy as np 
import pandas as pd
import pickle
import delta.compat as tf
import datetime
from tqdm import tqdm
from statistics import mean 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# import tf.keras.optimizers.Adam as Adam
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import sys


# In[ ]:





# In[ ]:


def get_labels(inp):
    labels = np.argmax(inp,axis=1)
    return labels

batch_size = 32
epochs = 1
learning_rate = 0.001
optimizer_name = 'adam'

def train(duration,labels_train,labels_val,labels_test,units,dropout,optimizer,flag):
#     model_save_path = os.path.join(plt_save_dir,'mdl_wts_'+str(duration)+'.hdf5')
#     mcp_save = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')

    input_text = tf.keras.layers.Input(shape = [516,300] )
    input_speech = tf.keras.layers.Input(shape = [520,26] )
        
    output,embedding =   AlignClassModel(dropout=dropout,units = units)([input_text,input_speech])


    model = tf.keras.Model(inputs =[input_text,input_speech] ,outputs = [output])
    
    if optimizer=='adam':
        optim = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimizer =='adadelta':
        optim = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
        
#     adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile( optimizer=optim,loss='categorical_crossentropy')
    

    if flag=='best_val':
        history = model.fit([X_train_text,X_train_audio],labels_train,batch_size=batch_size,validation_data=([X_val_text,X_val_audio],labels_val),verbose=1,epochs=epochs,callbacks=[mcp_save])
    
    if flag=='last_epoch':
        history = model.fit([X_train_text,X_train_audio],labels_train,batch_size=batch_size,validation_data=([X_val_text,X_val_audio],labels_val),verbose=1,epochs=epochs)


    if flag=='best_val':
        model.load_weights(model_save_path)

    test_pred = model.predict([X_test_text,X_test_audio])
    train_pred = model.predict([X_train_text,X_train_audio])
    
    test_pred_labels = get_labels(test_pred)
    train_pred_labels = get_labels(train_pred)

    y_test_labels = get_labels(labels_test)
    y_train_labels = get_labels(labels_train)
    
    train_f1 = f1_score(y_train_labels,train_pred_labels,average = 'weighted')
    test_f1 = f1_score(y_test_labels,test_pred_labels,average = 'weighted')
    
    print("Train F1 for {duration} days : {train_loss}".format(duration = duration,train_loss = train_f1))
    print("Test F1 for {duration} days : {test_loss}".format(duration = duration,test_loss = test_f1))
    
    
    save_path ="duration_"+str(duration) +"epochs="+str(epochs)+"_learning-rate"+str(learning_rate)
    
    save_pkl="y_pred_"+save_path+".pkl"
    
    with open(save_pkl,'wb') as f:
        pickle.dump(test_pred,f)
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    
    
    
    plt.savefig(save_path+".png")
    plt.close()
    
    model.save(save_path+"_model.h5")
    
    model.save_weights("model_wts"+save_path+"_model.h5")
    return


# In[ ]:


val_3  = train(3,y_train3days,y_val3days,y_test3days,units=50,dropout=0.4,optimizer='adam',flag='last_epoch')


# In[ ]:


val_7  = train(7,y_train7days,y_val7days,y_test7days,units=50,dropout=0.4,optimizer='adam',flag='last_epoch')


# In[ ]:


val_15 = train(15,y_train15days,y_val15days,y_test15days,units=50,dropout=0.45,optimizer='adam',flag='last_epoch')


# In[ ]:


val_30 = train(30,y_train30days,y_val30days,y_test30days,units=50,dropout=0.4,optimizer='adam',flag='last_epoch')



# In[ ]:


print("Train_F1 Test_F1")

print("3 days :",val_3)
print("7 days :",val_7)
print("15 days :",val_15)
print("30 days :",val_30)

