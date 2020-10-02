#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


id=test.iloc[:,0].values
test.drop('id',axis=1)


# In[ ]:


id


# In[ ]:


X = train.iloc[:, 1:11].values
y = train.iloc[:, 0].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 99, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


cm


# In[ ]:


#for calculating accuracy
(131+6090)/(131+6090+92+241)


# In[ ]:


test.drop(['id'], axis=1, inplace = True)


# In[ ]:


test.head()


# In[ ]:


test


# In[ ]:


test_pred = classifier.predict(test)


# The result of the test set given is saved in the variable test_pred
# 

# In[ ]:


test_pred


# In[ ]:


submission = pd.DataFrame({'Id':id,'Action':test_pred})


# In[ ]:


submission


# In[ ]:


final_submission=submission.iloc[0:58921,:].values


# In[ ]:


final_submission


# In[ ]:


final_submission =  pd.DataFrame({'Id':final_submission[:,0],'Action':final_submission[:,-1]})


# In[ ]:


final_submission


# In[ ]:


filename = 'Amazon Employee Access .csv'

final_submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

