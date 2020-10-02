#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Import all required libraries for machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")


# In[ ]:


train.head(2)
# test.head(2)


# In[ ]:


print(train.shape)
print(test.shape)


# # Create an Empty DataFrame and add the audio data in flat array format for each row

# In[ ]:


train_df = pd.DataFrame()


# In[ ]:


test_df = pd.DataFrame()


# In[ ]:


for frame in train.audio_embedding:
    tmp = pd.DataFrame(np.array(frame).reshape(1,-1))
    train_df = train_df.append(tmp)


# In[ ]:


for frame in test.audio_embedding:
    tmp = pd.DataFrame(np.array(frame).reshape(1,-1))
    test_df = test_df.append(tmp)


# # Fill NaN with mean value for Train dataset

# In[ ]:


for k in train_df:
    train_df[k].fillna(train_df[k].mean(),inplace=True)


# 
# # Fill NaN with mean value for Test dataset

# In[ ]:


for k in train_df:
    test_df[k].fillna(test_df[k].mean(),inplace=True)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(train_df,train.is_turkey,test_size=.2)


# # Evaluate Train model with Logistic Regression

# In[ ]:


lr = LogisticRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


y_pred = lr.predict(X_test)
y_pred


# # Accuracy of the model is 

# In[ ]:


accuracy_score(y_test,y_pred)*100


# In[ ]:


conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[ ]:


label = ["Turkey","Not_Turkey"]
sns.heatmap(conf_mat,annot=True,fmt="d",xticklabels=label,yticklabels=label)


# # ROC curve for trained mmodel

# In[ ]:


logit_roc_auc = roc_auc_score(y_test, lr.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
plt.figure(figsize=(8,4))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('LogisticRegression_ROC')
plt.show()


# # Predict the Turkey voice 0|1 wuth given data set

# In[ ]:


y_pred2 = lr.predict(test_df.iloc[:-1,])
y_pred2


# # Total Non-Turkey vice

# In[ ]:


y_pred2[y_pred2==0].size


# # Total Turkey voice

# In[ ]:


y_pred2[y_pred2==1].size


# # Writing prediction to the file

# In[ ]:


submit = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


submit.is_turkey[1:] = y_pred2


# In[ ]:


submit.to_csv('submission.csv', index=False)


# In[ ]:


pd.read_csv("submission.csv")


# In[ ]:




