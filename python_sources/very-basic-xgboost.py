#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier


# ## Loading Data

# In[ ]:


test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


# In[ ]:


X = train.drop(["label"],axis=1)
y = train["label"]


# ## Scaling Data

# In[ ]:


scale = StandardScaler()
X = scale.fit_transform(X)
test = scale.transform(test)


# ## Creating Train and Test Split

# In[ ]:


X_train,X_eval,y_train,y_eval = train_test_split(X,y,test_size=0.2,stratify=y)


# ## XGBoost Model

# In[ ]:


xgb_model = XGBClassifier()


# In[ ]:


bag_model = BaggingClassifier()
xt_model = ExtraTreesClassifier()


# In[ ]:


voting = VotingClassifier(estimators=[('xgb',xgb_model),('bag',bag_model), ('xt',xt_model)])


# In[ ]:


voting.fit(X_train,y_train)


# In[ ]:


voting.score(X_eval,y_eval)


# > ## Prediction

# In[ ]:


predict = voting.predict(test)


# In[ ]:


my_submission = pd.DataFrame({'ImageId': list(range(1,len(predict)+1)), 'Label': predict})
my_submission.to_csv('submission_xgb2.csv', index=False)


# In[ ]:




