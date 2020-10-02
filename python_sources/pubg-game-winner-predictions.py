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


# Let's use a simple XGBoost model to do the predictions- remove the identifiers in the intial feature set. With the data formatted and in numbers, let's not spend much time in data pre-processing.
# 
# Then spend more time in Permutation Importance, partial dependency plot and SHAP analysis to identify the key features and make qualitative sense of what impacts a good game performance in PUBG.

# Reading the data files.

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
submission=pd.read_csv('../input/sample_submission.csv')


# Making feature and prediction set from training and test data. Have removed only the identifiers for now.

# In[ ]:


x_train=train.copy()
y_train=train['winPlacePerc']
x_train=x_train.drop(columns=['Id','groupId','matchId','numGroups','maxPlace','winPlacePerc','damageDealt','headshotKills','roadKills','vehicleDestroys'])
x_test=test.copy()
x_test=x_test.drop(columns=['Id','groupId','matchId','numGroups','maxPlace','damageDealt','headshotKills','roadKills','vehicleDestroys'])

x_train.sort_index(axis=1,inplace=True)
x_test.sort_index(axis=1,inplace=True)


# Using the simple XGBoost model to make predictions and then will use methods to understand the model and do feature engineering.

# In[ ]:


import xgboost as xgb

model=xgb.XGBRegressor()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


# Preparing the submission file.

# In the simple XGBoost model, the score comes to around 0.0833. Let's try to understand the features in detail and improve the model.

# In[ ]:


from matplotlib import pyplot as plt
from xgboost import plot_importance

plot_importance(model)
plt.show()


# Let's remove the 4 features with 0 feature importance and then see the improvement in accuracy score. Removing the 4 features with 0 importance doesn't alter the predictions at all. However, if I remove any more features with low importance, then the predictions get slightly worse. So keeping all the features with non-zero importance.
# Either we do hyperparameter tuning or better try Deep learning. Dataset with 4.5 million rows is huge, can try with a simple feed forward neural network to compare predictions with XGBoost.
# 
# Before getting into Deep Learning, let's use SHAP values to better understand the dataset in current model.

# In[ ]:


import shap

explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(x_test)

shap.summary_plot(shap_values,x_test)


# In[ ]:


shap.dependence_plot('walkDistance',shap_values,x_test,interaction_index='killPlace')


# Let's use Deep Learning model to predict.

# In[ ]:


from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout

model1=models.Sequential()
model1.add(layers.Dense(512,activation='relu',input_shape=(x_train.shape[1],)))
model1.add(Dropout(0.2))
model1.add(layers.Dense(512,activation='relu'))
model1.add(Dropout(0.2))
model1.add(layers.Dense(1))

model1.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

model1.fit(x_train,y_train,epochs=16,batch_size=512)

y_pred_DL=model1.predict(x_test)


# In[ ]:


submission['winPlacePerc']=y_pred_DL
submission.to_csv('sample_submission.csv',index=False)

