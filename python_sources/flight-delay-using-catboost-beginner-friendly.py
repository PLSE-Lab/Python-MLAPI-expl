#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score


# In[ ]:


train = pd.read_csv("/kaggle/input/flight-delays-fall-2018/flight_delays_train.csv.zip")
test = pd.read_csv("/kaggle/input/flight-delays-fall-2018/flight_delays_test.csv.zip")


# In[ ]:


train.head(10)


# In[ ]:


test.head(10)


# In[ ]:


# changing target to numerical: N to 0 & Y to 1
train.loc[(train.dep_delayed_15min == 'N'), 'dep_delayed_15min'] = 0
train.loc[(train.dep_delayed_15min == 'Y'), 'dep_delayed_15min'] = 1


# In[ ]:


# Clean month, day of month and day of week
train['Month'] = train['Month'].str[2:].astype('int')
train['DayofMonth'] = train['DayofMonth'].str[2:].astype('int')
train['DayOfWeek'] = train['DayOfWeek'].str[2:].astype('int')

# Check the results
train.head(15)


# In[ ]:


# Clean month, day of month and day of week
test['Month'] = test['Month'].str[2:].astype('int')
test['DayofMonth'] = test['DayofMonth'].str[2:].astype('int')
test['DayOfWeek'] = test['DayOfWeek'].str[2:].astype('int')

# Check the results
test.head(15)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lb= LabelEncoder()
train["UniqueCarrier_new"] = lb.fit_transform(train["UniqueCarrier"])
train[["UniqueCarrier_new", "UniqueCarrier"]].head(11)


# In[ ]:


train["Origin_new"] = lb.fit_transform(train["Origin"])
train["Dest_new"] = lb.fit_transform(train["Dest"])


# In[ ]:


train.head()


# In[ ]:


X= train[['Month','DayofMonth','DayOfWeek','DepTime','Distance','UniqueCarrier_new','Origin_new','Dest_new']]
y= train['dep_delayed_15min']


# In[ ]:


y=y.astype('int')


# In[ ]:


#cleaning the test set
test["Origin_new"] = lb.fit_transform(test["Origin"])
test["Dest_new"] = lb.fit_transform(test["Dest"])
test["UniqueCarrier_new"] = lb.fit_transform(test["UniqueCarrier"])

test = test.drop(['UniqueCarrier','Origin','Dest'],1)


# In[ ]:


test.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)


# In[ ]:


from catboost import CatBoostClassifier
classifier = CatBoostClassifier()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn import metrics

print(metrics.classification_report(y_test,y_pred))


# In[ ]:


accuracy = classifier.score(y_pred,y_test)
print(accuracy*100,'%')


# In[ ]:


test_pred = classifier.predict_proba(X_test)[:,1]


# In[ ]:


roc_auc_score(y_test,test_pred )


# In[ ]:


test_pred


# In[ ]:


predictions = classifier.predict_proba(test)[:, 1]
predictions


# In[ ]:


submission = pd.DataFrame({'id':range(100000),'dep_delayed_15min':predictions})
submission.head(1001)


# In[ ]:


filename = 'Flight_delay_predictions.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:




