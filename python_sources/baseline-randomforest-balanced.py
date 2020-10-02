#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from imblearn.over_sampling import ADASYN
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Classes
# **1** = extreme poverty  
# **2** = moderate poverty  
# **3** = vulnerable households   
# **4** = non vulnerable households  

# In[ ]:


data = pd.read_csv('../input/train.csv')
data.fillna(value = 0, inplace=True)
data.tail(5)


# In[ ]:


## not considering idhogar for the moment
features = data.drop(columns = ['Id','idhogar','dependency','edjefe','edjefa'], axis = 1)


# ## Check if different household have different classifications

# In[ ]:


dif_list = []
all_families = list(set(data.idhogar))
for item in all_families:
    check_target = list(set(data[data.idhogar == item]['Target']))
    if len(check_target)>1:
        dif_list.append(item)
print(list(dif_list))


# ## Check class imbalance

# In[ ]:


total = len(features)
target_1 = len(features[features['Target']==1])
target_2 = len(features[features['Target']==2])
target_3 = len(features[features['Target']==3])
target_4 = len(features[features['Target']==4])

print("Total: {}".format(total))
print("Class 1: {0:.2f}%".format(100*target_1/total))
print("Class 2: {0:.2f}%".format(100*target_2/total))
print("Class 3: {0:.2f}%".format(100*target_3/total))
print("Class 4: {0:.2f}%".format(100*target_4/total))


# ## Perform Oversampling for classes

# In[ ]:


ada = ADASYN(random_state = 199)
X = features.drop(columns = ['Target'], axis = 1)
y = features['Target']
X_res, y_res = ada.fit_sample(X, y)


# ## Check classes after re-sampling

# In[ ]:


total = len(y_res)
target_1 = len(y_res[y_res==1])
target_2 = len(y_res[y_res==2])
target_3 = len(y_res[y_res==3])
target_4 = len(y_res[y_res==4])

print("Total: {}".format(total))
print("Class 1: {0:.2f}%".format(100*target_1/total))
print("Class 2: {0:.2f}%".format(100*target_2/total))
print("Class 3: {0:.2f}%".format(100*target_3/total))
print("Class 4: {0:.2f}%".format(100*target_4/total))


# In[ ]:


## balanced
x_train, x_test, y_train, y_test = train_test_split(X_res, 
                                                    y_res, 
                                                    test_size=0.20, 
                                                    random_state=42)
## Imbalanced
x_train_imba, x_test_imba, y_train_imba, y_test_imba = train_test_split(X, 
                                                            y, 
                                                            test_size=0.20, 
                                                            random_state=42)


# In[ ]:


num_estimators = 30


# In[ ]:


## Balanced
model_random = RandomForestClassifier(n_estimators = num_estimators, 
                                      random_state=0)
model_random.fit(X_res, y_res)
preds_random = model_random.predict(x_test)
accuracy_score(y_test, preds_random)


# In[ ]:


## Imbalanced
model_random_imba = RandomForestClassifier(n_estimators = num_estimators, 
                                      random_state=0)
model_random_imba.fit(x_train_imba, y_train_imba)
preds_random_imba = model_random_imba.predict(x_test_imba)
accuracy_score(y_test_imba, preds_random_imba)


# ## Checking Test predictions

# In[ ]:


test = pd.read_csv('../input/test.csv')
test.fillna(value = 0, inplace=True)
test.drop(columns = ['idhogar','dependency','edjefe','edjefa'], axis = 1, inplace = True)
test.tail()


# In[ ]:


get_ipython().system('ls ../input/')


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
print(sample_submission.columns)
sample_submission.tail(5)


# In[ ]:


submission = pd.DataFrame(columns = ['Id', 'Target'])
submission


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, preds_random))


# In[ ]:


for i in range(len(test)):
    ID = test['Id'].loc[i]
    in_test = np.array(test.drop(columns = ['Id'], axis = 1).loc[i]).reshape(1, -1)
    pred = model_random.predict(in_test)
    submission.loc[i] = ([ID,pred[0]])


# In[ ]:


submission = submission.reset_index(drop=True)
submission


# In[ ]:


submission.to_csv('sample_submission.csv', index=False )


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('cat sample_submission.csv')


# In[ ]:




