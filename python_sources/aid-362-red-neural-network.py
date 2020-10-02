#!/usr/bin/env python
# coding: utf-8

# ## Bioassay Neural Network

#    This Bioassay dataset presents a classification problem in which we are trying to see if the compounds in the study are binding to their targets. If the compound binds then it is recored as active and if not it is recorded as inactive. The dataset given has many features that may or may not actually help predict if the compound if active or not so we will need to do some feature selecting before building the neural network. This notebook only addresses one of the datasets, the rest are on my github. 

# In[ ]:


# read csv file into a pandas dataframe
import numpy as np
import pandas as pd

train = pd.read_csv('../input/AID362red_train.csv')
test = pd.read_csv('../input/AID362red_test.csv')
test.head(5)


# In[ ]:


# Function for converting categorical label into a numerical one
def outcome_to_numeric(x):
    if x=='Inactive':
        return 0
    if x=='Active':
        return 1


# In[ ]:


# Apply function to label column
train['label'] = train['Outcome'].apply(outcome_to_numeric)
test['label'] = test['Outcome'].apply(outcome_to_numeric)
test.head()


# In[ ]:


# Drop categorical column
train=train.drop('Outcome', axis=1)
test=test.drop('Outcome', axis=1)


# In[ ]:


# Split datasets into feature and label dataframes
x_train = train.drop('label', axis=1)
y_train = train['label']

x_test = test.drop('label', axis=1)
y_test = test['label']


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

model = ExtraTreesClassifier()
model.fit(x_train, y_train)

feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[ ]:


x_train = x_train[['WBN_GC_L_0.50', 'WBN_GC_H_1.00', 'MW', 'WBN_EN_H_0.50', 'WBN_EN_H_0.75']]

x_test = x_test[['WBN_GC_L_0.50', 'WBN_GC_H_1.00', 'MW', 'WBN_EN_H_0.50', 'WBN_EN_H_0.75']]


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.noise import AlphaDropout
from keras import optimizers
from keras import layers


model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

opt = optimizers.Adadelta(lr=.01)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train, y_train,
          epochs=30,
          batch_size=128)


# In[ ]:


score = model.evaluate(x_test, y_test, batch_size=128)


# In[ ]:


score


# In[ ]:




