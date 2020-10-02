#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from sklearn.model_selection import train_test_split

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


trainDf = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')
print('train data set has got {} rows and {} columns'.format(trainDf.shape[0],trainDf.shape[1]))


# In[ ]:


trainDf.info()


# 

# In[ ]:


for col in trainDf.columns:
    if (col != 'id' and col != 'target'):
        print(col + ": ", trainDf[col].unique())


# 

# In[ ]:


for col in trainDf.columns:
    if trainDf[col].dtype == 'object':
        trainDf[col] = trainDf[col].apply(lambda x: x.strip().replace(' ', ''))


# In[ ]:


trainDf['nom_3'].unique()


# In[ ]:


trainDf['ord_2'].unique()


# In[ ]:


for col in trainDf.columns:
    print(col + ": ", trainDf[col].isna().sum())


# 

# In[ ]:


# separate target from array

targets = trainDf['target'].values
trainDf = trainDf.drop(['nom_8', 'nom_9', 'target'], axis = 1)


# In[ ]:


samples = []

for col in trainDf.columns:
    if trainDf[col].dtype == 'object':
        for val in trainDf[col].unique():
            samples.append(val)


# In[ ]:


tokenizer = text.Tokenizer(num_words=len(samples))
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index

print("Found %s unique tokens." % len(word_index))


# In[ ]:


def myFunc(row):
    if row.lower() in word_index:
        return word_index[row.lower()]
    else:
        print("Value not found in word index: ", row)

for col in trainDf.columns:
    if trainDf[col].dtype == object:
        series = trainDf[col].apply(lambda x: myFunc(x))
        trainDf[col] = series


# Since we have replaced values, we should look for any nulls again

# In[ ]:


for col in trainDf.columns:
    print(col + ": ", trainDf[col].isna().sum())


# Oct 6, 2019: Looking back at our unique values for nom_3 and ord_2, there are some values that include spaces which may be messing up our encoding. 

# In[ ]:


trainArray = trainDf.values
trainTargets = targets

print(trainArray.shape)
print(trainTargets.shape)


# In[ ]:


seed = 7
np.random.seed(seed)

xTrain, xTest, yTrain, yTest = train_test_split(trainArray, trainTargets, test_size=0.33, random_state=seed)


# In[ ]:


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(xTrain.shape[1], )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


partial_x_train = xTrain[:120000]
partial_y_train = yTrain[:120000]


# In[ ]:


x_val = xTrain[120000:]
y_val = yTrain[120000:]


# In[ ]:


np.isnan(partial_x_train).any()


# In[ ]:


history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=1000,
                    validation_data=(x_val, y_val))#


# Plot metrics later

# In[ ]:


import matplotlib.pyplot as plt

history_dict = history.history


# In[ ]:


history_dict


# 

# In[ ]:


testDf = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

testDf = testDf.drop(['nom_8', 'nom_9'], axis = 1)

for col in testDf.columns:
    if testDf[col].dtype == 'object':
        testDf[col] = testDf[col].apply(lambda x: x.strip().replace(' ', ''))

testSamples = []

for col in testDf.columns:
    if testDf[col].dtype == 'object':
        for val in testDf[col].unique():
            testSamples.append(val)


# In[ ]:


def testEncode(row):
    if row.lower() in word_index:
        return word_index[row.lower()]
    else:
        print("Value not found in word index: ", row)

for col in testDf.columns:
    if testDf[col].dtype == object:
        print(col)
        series = testDf[col].apply(lambda x: testEncode(x))
        testDf[col] = series
        
testArray = testDf.values


# In[ ]:


testArray.shape


# In[ ]:


np.isnan(testArray).any()


# In[ ]:


target = model.predict(testArray, verbose=1)


# In[ ]:


target


# In[ ]:


testDf['target'] = target


# In[ ]:


submission = testDf[['id', 'target']]


# In[ ]:


submission


# In[ ]:


submission.to_csv('submission20191009.csv', index=False)

