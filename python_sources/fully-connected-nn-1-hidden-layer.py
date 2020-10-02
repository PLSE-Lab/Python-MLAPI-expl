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


train = pd.read_csv("../input/train.csv")
X_test= pd.read_csv("../input/test.csv")
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 
del train


# In[ ]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[ ]:


img_rows = img_cols = 28
X_train = X_train.values.reshape(X_train.shape[0],img_rows,img_cols)
X_test = X_test.values.reshape(X_test.shape[0],img_rows,img_cols)


# In[ ]:


# from tensorflow import keras
# from tensorflow.keras.utils import to_categorical
# Y_train = to_categorical(Y_train, num_classes = 10)


# In[ ]:


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout


# In[ ]:


# model = Sequential([
#   Flatten(input_shape=(28, 28)),
#   Dense(512, activation='relu'),
#   Dropout(0.2),
#   Dense(10, activation='softmax')
# ])


# In[ ]:


# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


# In[ ]:


def create_model(units):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model


# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from time import time

start=time()
model = KerasClassifier(build_fn=create_model)
# units = [128, 256, 512, 768, 1024]
units = [512]
# epochs = [20, 25, 30, 35, 40]
epochs=[40, 45, 50, 55, 60]
param_grid = {'units': units, 'epochs': epochs}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=2, cv=5)
grid_result = grid.fit(X_train, Y_train)


# In[ ]:


grid_result.cv_results_


# In[ ]:


gsResult = grid_result.cv_results_
params = gsResult["params"]
mean_scores = gsResult["mean_test_score"]
list(zip(params, mean_scores))


# In[ ]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("total time:",time()-start)


# In[ ]:


preds = grid_result.best_estimator_.predict(X_test)


# In[ ]:


preds = model.predict_classes(X_test)


# In[ ]:


np.savetxt('submission.csv', 
           np.c_[range(1,len(X_test)+1),preds], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

