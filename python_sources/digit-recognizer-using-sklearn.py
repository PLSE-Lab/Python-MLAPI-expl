#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax)
    #  square=True,
    plt.ylabel('Label')
    plt.xlabel('Prediction')


# In[ ]:


train_data=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:





# In[ ]:


train_data.describe()


# In[ ]:


y=train_data["label"]
X=train_data.copy()
del X["label"]


# In[ ]:


print(X,y)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train_X,val_X,train_y,val_y = train_test_split(X,y)


# In[ ]:


from sklearn.preprocessing import normalize
train_X_norm=normalize(train_X)
val_X_norm=normalize(val_X)


# In[ ]:


from sklearn.svm import LinearSVC

model=LinearSVC()
model.fit(train_X_norm[:5000],train_y[:5000])
accuracy_score(val_y,model.predict(val_X_norm))


# In[ ]:


from sklearn.linear_model import SGDClassifier

model=SGDClassifier(loss="log", max_iter=50)
model.fit(train_X_norm[:5000],train_y[:5000])
accuracy_score(val_y,model.predict(val_X_norm))


# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


model=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(200, 100), random_state=1,learning_rate="invscaling",max_iter=500)
norm_X=normalize(X)
model.fit(norm_X[:5000],y[:5000])
preds=model.predict(val_X_norm)
print(accuracy_score(val_y,preds))


# In[ ]:


model=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(40,20), random_state=1,max_iter=400,learning_rate="invscaling")
model.fit(train_X_norm[:5000],train_y[:5000])
print(accuracy_score(val_y,model.predict(val_X_norm)))


# In[ ]:


model=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=( 400,200,150,100), random_state=1,max_iter=400,learning_rate="invscaling")
model.fit(train_X_norm[:5000],train_y[:5000])
preds=model.predict(val_X_norm)
print(accuracy_score(val_y,preds))


# In[ ]:


model=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=( 400,200,150,100,50), random_state=1,max_iter=400,learning_rate="invscaling")
norm_X=normalize(X)
model.fit(train_X_norm[:5000],train_y[:5000])
preds=model.predict(val_X_norm)
print(accuracy_score(val_y,preds))


# In[ ]:


plot_confusion_matrix(val_y,preds)


# In[ ]:


model=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=( 400,200,150,100,50), random_state=1,max_iter=400,learning_rate="invscaling")
norm_X=normalize(X)
model.fit(norm_X,y)
preds=model.predict(val_X_norm)
print(accuracy_score(val_y,preds))
sol=model.predict(normalize(test_data))
df=pd.DataFrame(sol)
df.index+=1
df.to_csv("/kaggle/working/sol_final.csv",index=True,header=["Label"],index_label=["ImageId"])


# In[ ]:




