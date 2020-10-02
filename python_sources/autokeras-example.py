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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


pip install git+git://github.com/keras-team/autokeras@master


# In[ ]:


pip install git+git://github.com/keras-team/keras-tuner@master


# In[ ]:


import numpy as np
import autokeras as ak
from keras.datasets import mnist

# Prepare the data.
(x_train, y_classification), (x_test, y_test) = mnist.load_data()
x_image = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

x_structured = np.random.rand(x_train.shape[0], 100)
y_regression = np.random.rand(x_train.shape[0], 1)

# Build model and train.
automodel = ak.AutoModel(
   inputs=[ak.ImageInput(),
           ak.StructuredDataInput()],
    max_trials=100,
   outputs=[ak.RegressionHead(metrics=['mae']),
            ak.ClassificationHead(loss='categorical_crossentropy',
                                  metrics=['accuracy'])])
automodel.fit([x_image, x_structured],
              [y_regression, y_classification],
              validation_split=0.2)

