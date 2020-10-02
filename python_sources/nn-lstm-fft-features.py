#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import zipfile
import os
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy import fftpack
import xgboost
import warnings
from sklearn.model_selection import cross_val_predict, cross_validate
import seaborn as sns; sns.set()
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from keras.layers import LSTM,Input,Dense,Flatten,SpatialDropout1D,Dropout,CuDNNLSTM,Reshape,Concatenate
from keras.layers import Lambda,concatenate,BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K 
from sklearn.preprocessing import LabelEncoder


# In[ ]:


directory = "../input/"
X_test_path = os.path.join(directory,"X_test.csv")
X_train_path = os.path.join(directory,"X_train.csv")
X_test_data = pd.read_csv(X_test_path)
X_train_data = pd.read_csv(X_train_path)
y_train_data = pd.read_csv(os.path.join(directory,"y_train.csv"))
sample_submission = pd.read_csv(os.path.join(directory,"sample_submission.csv"))


# **Converting class labels to binary matrix representation**

# In[ ]:


le = LabelEncoder()
le.fit(list(y_train_data["surface"]))
y_train_dataset_for_nn = to_categorical(le.transform(list(y_train_data["surface"])))


# **Creating X-datasets for LSTM - shape = (num_samples,128,10)**

# In[ ]:


def dataset_for_nn(X_dataset):
    num_samples = X_dataset.shape[0]//128
    X_dataset_for_nn = np.zeros((num_samples,128,10))
    for i in range(num_samples):
        subset = np.array(X_dataset.iloc[i*128:(i+1)*128,3:])
        X_dataset_for_nn[i,:,:] = subset
    return X_dataset_for_nn


# In[ ]:


X_train_for_nn = dataset_for_nn(X_train_data)
X_test_for_nn = dataset_for_nn(X_test_data)


# **Function for extracting Fourier transform with averaging**

# In[ ]:


def freqs(dataset,width):
    X = np.abs(fftpack.fft(dataset))
    squeezed_dataset = []
    for i in range(64//width):
        squeezed_dataset.append(np.mean(X[i*width:(i+1)*width]))
    return squeezed_dataset


# **Creating features**

# In[ ]:


def X_features(X_dataset,width=3):
    num_samples = len(list(set(X_dataset["series_id"])))
    num_cols = 64//width
    features = np.zeros((num_samples,40+10*num_cols))
    for i in range(num_samples):
        X_train_subset = np.array(X_dataset.iloc[i*128:(i+1)*128,3:])
        features[i,:10] = np.mean(X_train_subset,axis=0)
        features[i,10:20] = np.std(X_train_subset,axis=0)
        features[i,20:30] = np.max(X_train_subset,axis=0)-np.min(X_train_subset,axis=0)
        features[i,30:40] = X_train_subset[-1,:]-X_train_subset[0,:]
        for j in range(X_train_subset.shape[1]):
            features[i,40+j*num_cols:40+(j+1)*num_cols] = freqs(X_train_subset[:,j],width)
    return features


# In[ ]:


X_train_features = X_features(X_train_data)
X_test_features = X_features(X_test_data)


# **Neural network architecture**

# In[ ]:


def LSTM_NN(drop):
    inp = Input(shape=(128,10))
    x = SpatialDropout1D(0.1)(inp)
    inp_2 = Input(shape=(250,))
    x_2 = Dense(250, input_shape=(250,), activation="sigmoid")(inp_2)
    x_2 = Dropout(drop)(x_2)
    x_2 = Dense(120, activation="sigmoid")(x_2)
    x_2 = Dropout(drop)(x_2)
    x_2 = Dense(60, activation="sigmoid")(x_2)
    x_2 = Dropout(drop)(x_2)
    x_2 = BatchNormalization()(x_2)
    x = CuDNNLSTM(units=200, return_sequences=True, return_state=False, go_backwards=False)(x)
    x = Dropout(drop)(x)
    x = CuDNNLSTM(units=100, return_sequences=False, return_state=False, go_backwards=False)(x)
    x = concatenate([x,x_2])
    x = Dropout(drop)(x)
    outp = Dense(9, activation="sigmoid")(x)
    model = Model(inputs=[inp,inp_2], outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer=Adam())
    return model


# **Training model and prediction**

# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_splits=5\nkfold = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)\ny_test = np.zeros((X_test_for_nn.shape[0],9*n_splits))\ntrain_preds = np.zeros((X_train_for_nn.shape[0],9))\nX = X_train_for_nn\nX_test = X_test_for_nn\nY = np.array(list(y_train_data["surface"]))\nfor i, (train_index, valid_index) in enumerate(kfold.split(X, Y)):\n    X_train, X_val =  X[list(train_index),:,:],X[list(valid_index),:,:]\n    X_train_feat, X_val_feat = X_train_features[list(train_index),:],X_train_features[list(valid_index),:]\n    Y_train, Y_val = Y[list(train_index)], Y[list(valid_index)]\n    Y_train = to_categorical(le.transform(Y_train))\n    Y_val = to_categorical(le.transform(Y_val))\n    model = LSTM_NN(0.15)\n    model.fit([X_train,X_train_feat], Y_train, epochs=120, validation_data=([X_val,X_val_feat], Y_val), verbose=2) \n    y_pred = model.predict([X_val,X_val_feat], verbose=2)\n    y_test[:,i*9:(i+1)*9] = model.predict([X_test,X_test_features])\n    train_preds[list(valid_index),:] = np.squeeze(y_pred)')


# **CV score**

# In[ ]:


res_train = np.argmax(train_preds,axis=1)
ans_train = np.argmax(to_categorical(le.transform(Y)),axis=1)
print ("CV score",round(accuracy_score(res_train,ans_train),4))


# **Exctracting test prediction (most frequent)**

# In[ ]:


def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0]

res_test_inter = np.zeros((y_test.shape[0],n_splits))
res_test=[]
for i in range(n_splits):
    inter_arr = y_test[:,i*9:(i+1)*9]
    res_test_inter[:,i] = np.argmax(inter_arr,axis=1)
for j in range(y_test.shape[0]):
    res_test.append(int(most_frequent(res_test_inter[j,:])))


# In[ ]:


test_for_sub=le.inverse_transform(res_test)
print (test_for_sub[:5])
test_size = len(list(set(X_test_data["series_id"])))
Y_test_pred_array = np.zeros((test_size,2))
Y_test_for_submission = pd.DataFrame(Y_test_pred_array,columns = ["series_id","surface"])
Y_test_for_submission.iloc[:,0] = list(range(test_size))
Y_test_for_submission.iloc[:,1] = test_for_sub
Y_test_for_submission.to_csv("submission.csv",index=None)


# In[ ]:


from IPython.display import FileLink
FileLink('submission.csv')

