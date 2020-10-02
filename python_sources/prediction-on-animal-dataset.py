#!/usr/bin/env python
# coding: utf-8

# # My work on animal dataset

# In[ ]:


#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


#importing data from my workshop
file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter04/Dataset/openml_phpZNNasq.csv'


# In[ ]:


#reading data
df = pd.read_csv(file_url)


# In[ ]:


#data preprocessing
df.drop(columns='animal', inplace=True)
y = df.pop('type')


# In[ ]:


#spiltting test , train data.
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.4, random_state=188)


# In[ ]:


#applying random forest classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=1)
rf_model.fit(X_train, y_train)


# In[ ]:


#prediction on model
train_preds = rf_model.predict(X_train)
test_preds = rf_model.predict(X_test)


# In[ ]:


#accuracy on model
train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)


# In[ ]:


#printing for accuracy
print(train_acc)
print(test_acc)


# # let me train and test the model again for higher accuracy.

# In[ ]:


rf_model2 = RandomForestClassifier(random_state=42, n_estimators=30)
rf_model2.fit(X_train, y_train)


# In[ ]:


train_preds2 = rf_model2.predict(X_train)
test_preds2 = rf_model2.predict(X_test)


# In[ ]:


train_acc2 = accuracy_score(y_train, train_preds2)
test_acc2 = accuracy_score(y_test, test_preds2)


# In[ ]:


print(train_acc2)
print(test_acc2)


# In[ ]:





# In[ ]:




