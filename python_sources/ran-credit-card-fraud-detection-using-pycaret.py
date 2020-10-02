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


df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)
df.describe()


# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


df['Class'].value_counts()


# From the above, it's very clear that the data is highly imbalanced towards to non-fraud class.

# In[ ]:


df_sample_0 = df.loc[df.loc[:, 'Class'] == 0].sample(frac=0.01, random_state=25)
df_sample_0.shape


# In[ ]:


df_1 = df.loc[df.loc[:, 'Class'] == 1]
df_sample = df_sample_0.append(df_1)
df_sample = df_sample.sample(frac=1)
df_sample['Class'].value_counts()


# In[ ]:


from pycaret.classification import *


# In[ ]:


clf1 = setup(data = df_sample, 
            target = 'Class',
            silent=True,
            sampling=False)


# In[ ]:


compare_models(fold=10)


# In[ ]:


model  = create_model('catboost')


# In[ ]:


tunedModel = tune_model('catboost', fold=5, optimize="F1")


# In[ ]:


plot_model(estimator = tunedModel, plot = 'class_report')


# In[ ]:


plot_model(estimator = tunedModel, plot = 'auc')


# In[ ]:


plot_model(estimator = tunedModel, plot = 'confusion_matrix')


# In[ ]:


plot_model(estimator = tunedModel, plot = 'feature')


# In[ ]:


plot_model(estimator = tunedModel, plot = 'parameter')


# In[ ]:


calibratedModel = calibrate_model(tunedModel)
# Taking 30min of time; can change params


# In[ ]:


#predictions = predict_model(tuned_model, data=test)
#predictions.head()


# In[ ]:


finalModel = finalize_model(tunedModel)


# In[ ]:


rfC  = create_model('rf');
etC  = create_model('et');
xgbC = create_model('xgboost');

#blending 3 models
blendedModels = blend_models(estimator_list=[rfC, etC, xgbC])


# In[ ]:




