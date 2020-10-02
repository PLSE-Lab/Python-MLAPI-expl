#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv("../input/train.csv")
train.info()


# In[ ]:


test = pd.read_csv("../input/test.csv")
test.info()


# In[ ]:


X = train.drop(['id','target'], axis = 1)
X.head(1)


# In[ ]:


y = train['target']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression(solver = 'liblinear')


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)
predictions[:5]


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score

print(accuracy_score(y_test,predictions))
print("\n")
print(confusion_matrix(y_test,predictions))
print("\n")
print(classification_report(y_test,predictions))


# In[ ]:


TestForPred = test.drop(['id'], axis = 1)


# In[ ]:


t_pred = logmodel.predict(TestForPred).astype(int)


# In[ ]:


id = test['id']


# In[ ]:


logSub = pd.DataFrame({'id': id, 'target':t_pred })
logSub.head()


# In[ ]:


logSub.to_csv("1_Logistics_Regression_Submission.csv", index = False)

