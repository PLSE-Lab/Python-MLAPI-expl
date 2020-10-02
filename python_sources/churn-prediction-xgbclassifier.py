#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,precision_recall_curve,precision_recall_fscore_support


# In[ ]:


data = pd.read_csv('../input/churn-predictions-personal/Churn_Predictions.csv')


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


# remove unwanted cols
data.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)
data = data.sample(frac=1)


# In[ ]:


# As data is imbalance - Balancing it
Exit_case = np.sum(data.Exited)
non_exit_indices = []
non_exit_ini = 0

for i in range(len(data)):
    if data['Exited'][i]==0:
        non_exit_indices.append(i)
        non_exit_ini += 1
        if non_exit_ini==Exit_case:
            break


# In[ ]:


# Merge Exit and non Exit
exit_case_df = data[data['Exited']==1]
non_exit_df = data.iloc[non_exit_indices,:]
data = exit_case_df.append(non_exit_df)


# In[ ]:


data = pd.get_dummies(data,columns=['Geography','Gender'],drop_first=True)


# In[ ]:


data.reset_index(drop=True)
data = data.sample(frac=1)


# In[ ]:


X,y = data.iloc[:,[ 0,  1,  2,  3,  4,  5,  6,  7,  9, 10, 11]],data.Exited


# In[ ]:


# Train-Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


# Modelling
clf = XGBClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


recall_score(y_test,y_pred),precision_score(y_test,y_pred),accuracy_score(y_test,y_pred)


# For an ML Accuracy, Recall and Precision should be in same range, which we can observe here

# 
