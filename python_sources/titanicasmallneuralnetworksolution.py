# coding: utf-8
import pandas as pd
data = pd.read_csv('../input/train.csv')
x_train=data.iloc[:,2:].values.reshape((891,10))
y_train=data.iloc[:,1].values.reshape((891,1))
#Throw away names and index
x1 = x_train[:,0]/6-1#Pclass
x2 = x_train[:,3]#Age
x2[pd.isnan(x2)]=30
x2 = x2/100 - 1
x3 = x_train[:,2]#Sex
x3[x3=='male'] = -0.5
x3[x3=='female'] = 0.5
x4 = x_train[:,4]/8-1#SibSp
#x5 = x_train[:,6]#Ticket all this information seems comparable in other forms
x5 = x_train[:,5]/6-1#Parents/Children
#x6 = x_train[:,0]
x7 = x_train[:,7]/100-1 #Faire
x8 = x_train[:,8]
##This is silly messy 
#Basically group the passagers by cabinets
for j,k in enumerate(x8):
    try:
      if k[0]=='A':x8[j]=0.0  
      elif k[0]=='B':x8[j]=0.2
      elif k[0]=='C':x8[j]=0.4
      elif k[0]=='D':x8[j]=0.6
      elif k[0]=='E':x8[j]=0.8
      elif k[0]=='F':x8[j]=1
      elif k[0]=='G':x8[j]=1.2
      elif k[0]=='T':x8[j]=1.4
    except TypeError:
      x8[j]=-1.1
      pass

x8=x8-1
x9 = x_train[:,9]
x9[x9=='Q'] = -1
x9[x9=='C'] = 0
x9[x9=='S'] = 1.
x9[pd.isna(x9)] = -2.1

import numpy as np

from keras.layers import Dense, Input, add, Dropout, activations, Flatten
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, concatenate
from keras.optimizers import Nadam
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils
K.set_image_data_format('channels_first')

batch_size = 100
nb_epoch = 100
inputs = Input(shape = (1,28,28))

x_train = np.array([x1,x2,x3,x4,x5,x7,x8,x9]).T.astype(float)
y_train = np_utils.to_categorical(y_train, 2)
X_train, X_test, Y_train, Y_test = train_test_split(x_train,
                                                   y_train,
                                                   test_size = 0.025,
                                                   random_state = 30,
                                                   stratify = y_train)
inputs = Input(shape = (8,))

x = Dense(70, activation='tanh')(inputs)
for i in range(3):
	x = Dense(30, activation='tanh', activity_regularizer=regularizers.l1(0.0000), kernel_regularizer= regularizers.l2(0.000))(x)
x = Dense(2, activation='tanh', activity_regularizer=regularizers.l1(0.001), kernel_regularizer= regularizers.l2(0.001))(x)
reducelr= ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.000000001)

model = Model(inputs=inputs,outputs=x)
model.compile(optimizer="Nadam", loss='binary_crossentropy', metrics=['accuracy'])

model.fit([X_train], Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test), shuffle=True, callbacks=[reducelr])
model.save('Titanics.h5')

test = pd.read_csv('../input/test.csv')

x_train=test.iloc[:,1:].values.reshape((418,10))
#Throw away names and index
x1 = x_train[:,0]/6-1#Pclass
x2 = x_train[:,3]#Age
x2[pd.isnan(x2)]=30
x2 = x2/100 - 1
x3 = x_train[:,2]#Sex
x3[x3=='male'] = -0.5
x3[x3=='female'] = 0.5
x4 = x_train[:,4]/4-1#SibSp
#x5 = x_train[:,6]#Ticket all this information seems comparable in other forms
x5 = x_train[:,5]/4-1#Parents/Children
#x6 = x_train[:,0]
x7 = x_train[:,7]/100-1 #Faire
x8 = x_train[:,8]
##This is silly messy 
#Basically group the passagers by cabinets
for j,k in enumerate(x8):
    try:
      if k[0]=='A':x8[j]=0.0  
      elif k[0]=='B':x8[j]=0.2
      elif k[0]=='C':x8[j]=0.4
      elif k[0]=='D':x8[j]=0.6
      elif k[0]=='E':x8[j]=0.8
      elif k[0]=='F':x8[j]=1
      elif k[0]=='G':x8[j]=1.2
      elif k[0]=='T':x8[j]=1.4
    except TypeError:
      x8[j]=-2.1
      pass

x8=x8-1
x9 = x_train[:,9]
x9[x9=='Q'] = -1
x9[x9=='C'] = 0
x9[x9=='S'] = 1.
x9[pd.isna(x9)] = 0

x_train = np.array([x1,x2,x3,x4,x5,x7,x8,x9]).T.astype(float)
predictions=model.predict(x_train)
for i in range(418):print(str(i+892)+" : "+str(np.maxarg(predictions[i])))
np.save('Predictions.npy',predictions)

#Randomness can be removed
preds = pd.DataFrame(np.argmax(r+0.2*np.random.randn(418,2),axis=1),index=[i+892 for i in range(418)],columns=['Survived'])
preds.to_csv('predictiontitanic.csv')