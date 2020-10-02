#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import warnings
warnings.filterwarnings('always')


# In[ ]:


# Read the data
filepath = "../input/phishing-website-detector/phishing.csv"
data = pd.read_csv(filepath)
data.head()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.describe()


# In[ ]:


X = data.drop(columns='class',axis=1)
X


# In[ ]:


y=data["class"]
y=pd.DataFrame(y)
y.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# In[ ]:


corr = data.corr()
fig,ax= plt.subplots(figsize=(20,20))
sns.heatmap(corr,annot=True,linewidth=2.5,ax=ax)


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!git clone --recursive https://github.com/VowpalWabbit/vowpal_wabbit.git \n!cd vowpal_wabbit/; make \n!cd vowpal_wabbit/; make install')


# In[ ]:


get_ipython().system('vw --help')


# In[ ]:


phishingwebsite =('../input/phishing-website-detector')


# In[ ]:


print('' + phishingwebsite)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


def to_vw_format(document, label=None):
    return str(label or '') + ' |data ' + ' '.join(re.findall('\w{3,}', document.lower())) + '\n'

to_vw_format(data, 1 if data.columns() == 'rec.autos' else -1)


# In[ ]:


with open(os.path.join('X_train'), 'w') as vw_train_data:
    for data, data.columns() in zip(X_train, y_train):
        vw_train_data.write(to_vw_format(data,data.columns()))
with open(os.path.join('y_test'), 'w') as vw_test_data:
    for text in y_train:
        vw_test_data.write(to_vw_format(text))

