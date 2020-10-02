#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import os
import xgboost as xgb
from sklearn import cross_validation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


categoryToIndex = {}
indexToCategory = {}

trainCategory = [];
trainImage    = [];
for fishType in os.listdir("../input/train"):
    if fishType == ".DS_Store":
        continue
        
    i = None
    if fishType in categoryToIndex:
        i = categoryToIndex[fishType]
    else:
        i = len(categoryToIndex)
        categoryToIndex[fishType] = i
        indexToCategory[i] = fishType
        
    for f in os.listdir("../input/train/" + fishType):
        if f == ".DS_Store":
            continue
        trainCategory.append(i)
        trainImage.append(f)


# In[ ]:


NUM_SAMPLES = 32
randomIndexes = np.random.choice(len(trainCategory), NUM_SAMPLES)

trainInput = []
trainOutput = []
maxWidth = 0
maxHeight = 0
for i in randomIndexes:
    filename = trainImage[i]
    category = trainCategory[i]
    img = Image.open("../input/train/" + indexToCategory[category] + "/" + filename)
    data = np.asarray(img, dtype="float32") / 255.0
    out = np.zeros(len(indexToCategory))
    out[category] = 1.0
    
    if data.shape[0] > maxWidth:
        maxWidth = data.shape[0]
        
    if data.shape[1] > maxHeight:
        maxHeight = data.shape[1]
    
    trainInput.append(data)
    trainOutput.append(int(category))
    
for i in range(NUM_SAMPLES):
    padImage = np.zeros((maxWidth, maxHeight, 3))
    img = trainInput[i]
    
    padImage[:img.shape[0],:img.shape[1],:3] = img
    trainInput[i] = padImage


# In[ ]:


trainInput  = np.array(trainInput)
trainOutput = np.array(trainOutput)


# In[ ]:


x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    trainInput.reshape((NUM_SAMPLES, maxWidth*maxHeight*3)), trainOutput, 
    random_state=42,
    test_size=0.20
)


# In[ ]:


clf = xgb.XGBClassifier(
    max_depth=5,
    n_estimators=500,
    learning_rate=0.1,
    nthread=-1,
    objective='multi:softmax',
    seed=42
)

clf.fit(
    x_train, y_train,
    early_stopping_rounds=30,
    eval_metric="mlogloss",
    eval_set=[(x_test, y_test)]
)


# In[ ]:


trainOutput

