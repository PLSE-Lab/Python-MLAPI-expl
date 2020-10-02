#!/usr/bin/env python
# coding: utf-8

# Notice that it is just a simple example in the deep-learning areas. The purpose of this kernel is to implement deep-learning algorithms for financial asset forecasting (e.g. stock index price movement). Any discussion is encouraged.
# 
# This kernal will be updating continuously...

# In[ ]:


#load come basic package
import pandas as pd
import numpy as np

#preprocessing
from sklearn.preprocessing import MinMaxScaler

#deep learning package
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt


# **preparation for the normal 2-dimension data**

# In[ ]:


seed=123
data=pd.read_csv('../input/SP500.csv')  

#delete the date column
data=data.drop(['Date'],axis=1)

pred_result=pd.DataFrame()

#here we use 80% data for training and validation, and 20% data for testing
#training and validation set
X_train=data.drop(['LABEL'],axis=1).loc[:1958,:].values
y_train=np.asarray(data.loc[:1958,'LABEL'].values).astype('float64')

#test set
X_test=data.drop(['LABEL'],axis=1).loc[1959:,:].values
y_test=np.asarray(data.loc[1959:,'LABEL']).astype('float64')

#here we normalized the features
#range from 0 to 1
scaler=MinMaxScaler().fit(X_train)
X_train_norm=scaler.transform(X_train)
X_test_norm=scaler.transform(X_test)


# **DNN**

# In[ ]:


Model_Deep_DNN=Sequential()
Model_Deep_DNN.add(layers.Dense(32,input_shape=(X_train.shape[1],)))
Model_Deep_DNN.add(layers.Dropout(0.1))
Model_Deep_DNN.add(layers.Dense(32))
Model_Deep_DNN.add(layers.Dropout(0.1))
Model_Deep_DNN.add(layers.Dense(32))
Model_Deep_DNN.add(layers.Dropout(0.1))
Model_Deep_DNN.add(layers.Dense(1,activation='sigmoid'))

Model_Deep_DNN.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
history=Model_Deep_DNN.fit(X_train,y_train,batch_size=128,epochs=100,validation_split=0.2)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'ro',label='Validation acc')
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')

plt.legend() 
plt.figure()


# 

# Transform the 2-dimension data to the 3-dimension data (here the timestep =1)

# In[ ]:


timestep=1
#reshape the data for 3d input
X_deep_matrix=np.append(X_train,X_test,axis=0)
#transform the matrix data to tensor data
X_deep_tensor=np.empty((X_deep_matrix.shape[0]-timestep+1,timestep,X_deep_matrix.shape[1]))

for i in range(timestep-1,X_deep_matrix.shape[0]):
    X_deep_tensor[i-timestep+1]=X_deep_matrix[i-timestep+1:i+1,:]

del X_deep_matrix
#generate training and test sets for deep learning algorithm    
X_train_deep=X_deep_tensor[:X_deep_tensor.shape[0]-y_test.shape[0]]
X_test_deep=X_deep_tensor[X_deep_tensor.shape[0]-y_test.shape[0]:]

del X_deep_tensor

y_train_deep=y_train[timestep-1:len(y_train)]


# **RNN**

# In[ ]:


Model_Deep_RNN=Sequential()
Model_Deep_RNN.add(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True,
                                    input_shape=(X_train_deep.shape[1],X_train_deep.shape[2])))
Model_Deep_RNN.add(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=True))
Model_Deep_RNN.add(layers.SimpleRNN(32,dropout=0.1,recurrent_dropout=0.1,return_sequences=False))
Model_Deep_RNN.add(layers.Dense(1,activation='sigmoid'))

Model_Deep_RNN.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
history=Model_Deep_RNN.fit(X_train_deep,y_train_deep,batch_size=128,epochs=100,validation_split=0.2)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'ro',label='Validation acc')
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')

plt.legend() 
plt.figure()


# **bidirectional RNN (BRNN)**

# In[ ]:


Model_Deep_BRNN=Sequential()
Model_Deep_BRNN.add(layers.Bidirectional(layers.SimpleRNN(5,dropout=0.1,recurrent_dropout=0.1,return_sequences=True,
                                     input_shape=(X_train_deep.shape[1],X_train_deep.shape[2]))))
Model_Deep_BRNN.add(layers.Bidirectional(layers.SimpleRNN(5,dropout=0.1,recurrent_dropout=0.1,return_sequences=True)))
Model_Deep_BRNN.add(layers.Bidirectional(layers.SimpleRNN(5,dropout=0.1,recurrent_dropout=0.1,return_sequences=False)))
Model_Deep_BRNN.add(layers.Dense(1,activation='sigmoid'))

Model_Deep_BRNN.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
history=Model_Deep_BRNN.fit(X_train_deep,y_train_deep,batch_size=128,epochs=100,validation_split=0.2)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'ro',label='Validation acc')
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')

plt.legend() 
plt.figure()


# **RNN-LSTM**

# In[ ]:


Model_Deep_RNNLSTM=Sequential()
Model_Deep_RNNLSTM.add(layers.LSTM(5,dropout=0.5,recurrent_dropout=0.5,return_sequences=True,
                                    input_shape=(X_train_deep.shape[1],X_train_deep.shape[2])))
Model_Deep_RNNLSTM.add(layers.LSTM(5,dropout=0.5,recurrent_dropout=0.5,return_sequences=True))
Model_Deep_RNNLSTM.add(layers.LSTM(5,dropout=0.5,recurrent_dropout=0.5,return_sequences=False))
Model_Deep_RNNLSTM.add(layers.Dense(1,activation='sigmoid'))

Model_Deep_RNNLSTM.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
history=Model_Deep_RNNLSTM.fit(X_train_deep,y_train_deep,batch_size=128,epochs=20,validation_split=0.2)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'ro',label='Validation acc')
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')

plt.legend() 
plt.figure()


# **RNN-GRU**

# In[ ]:


#RNN-GRU
Model_Deep_RNNGRU=Sequential()
Model_Deep_RNNGRU.add(layers.GRU(3,dropout=0.5,recurrent_dropout=0.5,return_sequences=True,
                                    input_shape=(X_train_deep.shape[1],X_train_deep.shape[2])))
Model_Deep_RNNGRU.add(layers.GRU(3,dropout=0.5,recurrent_dropout=0.5,return_sequences=True))
Model_Deep_RNNGRU.add(layers.GRU(3,dropout=0.5,recurrent_dropout=0.5,return_sequences=False))
Model_Deep_RNNGRU.add(layers.Dense(1,activation='sigmoid'))

Model_Deep_RNNGRU.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['accuracy'])
history=Model_Deep_RNNGRU.fit(X_train_deep,y_train_deep,batch_size=128,epochs=20,validation_split=0.2)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'ro',label='Validation acc')
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')

plt.legend() 
plt.figure()


# In[ ]:


pred_Deep=pd.concat([pd.DataFrame(Model_Deep_RNN.predict(X_test_deep)),
                     pd.DataFrame(Model_Deep_BRNN.predict(X_test_deep)),
                     pd.DataFrame(Model_Deep_RNNLSTM.predict(X_test_deep)),
                     pd.DataFrame(Model_Deep_RNNGRU.predict(X_test_deep))],axis=1)

pred_Deep[pred_Deep>0.5]=1
pred_Deep[pred_Deep<=0.5]=0

pred_result=pd.concat([pd.DataFrame(y_test),pred_Deep],axis=1)

#rename
pred_result.columns=['LABEL','RNN_Pred','BRNN_Pred','RNNLSTM_Pred','RNNGRU_Pred']

pred_result


# Any issues about the reproduce
# 
# https://github.com/keras-team/keras/issues/2280
# 
# https://stackoverflow.com/questions/42412660/non-deterministic-gradient-computation
