#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will demonstrate how to implement a Taylor series as a neural network architecture and make some hand written digits preditions. Let us get started by loading necessary libraries

# In[ ]:


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from keras.layers import Input, Dense, add, Lambda, Activation
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical


# Then let us load the data and select the training and testing datasets

# In[ ]:


TRAINING = "../input/train.csv" 
dataset = pd.read_csv(TRAINING, sep=',',header=0)
x_train = dataset[:40000].drop('label', 1).values
y_train = to_categorical(dataset[:40000]["label"].values, num_classes=10)
x_test = dataset[40000:].drop('label', 1).values
y_test = to_categorical(dataset[40000:]["label"].values, num_classes=10)


# Next, let us define a power activation function to be used in the Taylor series architecture

# In[ ]:


def pown(x,n):
  return x**n
get_custom_objects().update({'pown': Activation(pown)})


# Now, let us implement the Taylor series architecture by first implementing a coordinate transformation layer and compiling the model

# In[ ]:


def Taylor(input_shape, output_shape, approx_order):
  inputs = Input(shape=(input_shape,)) 
  x = Dense(input_shape)(inputs)
  y = Dense(output_shape)(x)
  for i in range(2,approx_order+1):
    y = add([y,Dense(output_shape)(Activation(lambda x: pown(x, n=i))(x))])
  outputs = Activation('softmax')(y)
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
  return model


# Finally, let us impute, standardize, normalize the data and use the fifth order approximation, then fit the model and evaluate its accuracy.

# In[ ]:


pipe = Pipeline([
    ("imputer", SimpleImputer()),
    ("standardScaler",StandardScaler()), 
    ("normalizer",Normalizer()), 
    ('nn', KerasClassifier(
        build_fn=Taylor, 
        input_shape = x_train.shape[1], 
        output_shape= y_train.shape[1], 
        approx_order = 5, 
        epochs=10, 
        batch_size=256)
    )
])
pipe.fit(x_train,y_train)
accuracy_score = pipe.score(x_test,y_test)
print("\n Test Accuracy: {}".format(accuracy_score))


# It is amazing how this simple, but yet powerfull Taylor series model quickly achieves a training accuracy of 99% and testing accuracy of 97% in just ten epochs. 
