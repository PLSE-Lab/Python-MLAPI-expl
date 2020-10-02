# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model,load_model
from keras.callbacks import LambdaCallback
from keras.layers import Lambda,Dense,TimeDistributed,AveragePooling1D,Flatten,Reshape,Input,Concatenate,RepeatVector,Add,Activation,BatchNormalization
import keras.backend as K
from keras.optimizers import Adam
import keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.





X=np.load("X.npy") #Numpy array shape [-1,n_atoms,3]. XYZ for every atom in molecule. Centered in the midpoint of the coupling pair.
X2=np.load("X2.npy") #Numpy array shape [-1,n_atoms, 7] One-hot encoding of atom-type(5 features). Plus two columns indicating whether the the current atom is in the coupling pair.
Y=np.load("Y.npy") #Numpy array shape [-1,1] Scalar coupling constant

##
# If you want to look at more than one copuling type at the time, you need to add another input for coupling type.
###



n_atoms = 29 #We use zero-padding. No molecule with more than 29 atoms. 



#Lets scale the atom coordinates, plus make two distance features:
def pinp(x):
    d=K.sqrt(K.sum(x**2,-1,keepdims=True))

    d0 = d/10.0
    d1 = K.sqrt(d)/2.0

    return K.concatenate([x/10.0,d0,d1])



inp1 = Input(X.shape[1::])
inp2 = Input(X2.shape[1::])

hi = Concatenate()([inp2,Lambda(pinp)(inp1) ])

#Run the atom datapoints through some dense layers individually.
h = TimeDistributed(Dense(256,activation='relu'))(hi)
h = TimeDistributed(Dense(256,activation='relu'))(h)

#Then Aggregate them using a aggregation function that is permutation invariant (Max, average, sum etc.)
h = Flatten()(AveragePooling1D(n_atoms)(h))


h = Dense(256,activation='relu')(h)
out = Dense(1)(h)

model = Model([inp1,inp2],out)
model.compile(loss='mae',optimizer=Adam(1e-3))

model.fit([X,X2],Y)

