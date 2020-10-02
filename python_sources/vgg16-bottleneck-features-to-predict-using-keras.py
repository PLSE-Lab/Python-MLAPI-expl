#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# Load images data set NPY files

# In[ ]:


# open npy file
arr_img = np.load('../input/handlanguagedigit/X.npy') 


# In order to use VGG16 extract the bottleneck features image input must be 3 chanels, but our dataset are grayscale images.
# We need to copy and stack up from 1 chanel to 3 chanels.
# 
# We choose VGG16 pre-trained by ImageNet as a feature extractor.
# 
# Then, our input in VGG16  is 64 x 64 x 3 dimensions and output will be flattened to be 2048 dimensions.
# So, output array choose be 2062 x 2048 (sample x featrues)

# In[ ]:


samples = 2062
h_size = 64
w_size = 64
chanel = 3

bottleneck_features = np.empty([samples,2048])

model = VGG16(weights=None, include_top=False) #kernel cannot download weights actually we use weights='imagenet' instead

for i in range(samples):
    arr_img_3d = np.asarray(np.dstack((arr_img[i], arr_img[i], arr_img[i])), dtype=np.float64)
    img = np.expand_dims(arr_img_3d, axis=0)
    feature = model.predict(img)
    bottleneck_features[i] = feature.flatten()
    


# According to computation comsumption, we will save the features to disk and load when we are going to use it.

# In[ ]:


features = np.load('../input/handlanguagedigit/bottleneck_features (1).npy')


# In[ ]:


features.shape


# In[ ]:


label = np.load('../input/handlanguagedigit/Y.npy') 


# In[ ]:


label.shape


# Split dataset into train and test set with ratio 80:20.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)


# Create two layer ANN
# 
# 2048  number of input,
# 512 number of first layer (ReLu),
# 512 number of second layer (ReLu),
# 10 number of output (soft max)
# 
# to predict the results.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

input_num = features.shape[1]

# create model
model = Sequential()
model.add(Dense(512, input_dim=input_num, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
# Compile model
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


train_history = model.fit(X_train, y_train, epochs=50, batch_size=32,  verbose=2)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(train_history.history['acc'])


# Test the test set

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

prediction = model.predict(X_test)
test_result = np.argmax(y_test, axis=1)
prediction_result = np.argmax(prediction, axis=1)
print(classification_report(test_result, prediction_result))
print(confusion_matrix(test_result, prediction_result))


# Good result for first trying!

# Conclusion
# 1. pre-trained model VGG16 is useful to extract features from image data.
# 2. deep ANN can build on top the pre-trained model to predict images belonging to classes.
# 3. find tune only the ANN building on top in order to get better results
