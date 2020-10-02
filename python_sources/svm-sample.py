#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()


# In[ ]:


digits.target


# In[ ]:


dir(digits)


# In[ ]:


digits.target_names


# In[ ]:


df = pd.DataFrame(digits.data,digits.target)
df.head()


# In[ ]:


df['target'] = digits.target
df.head(20)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.2)


# In[ ]:


from sklearn.svm import SVC
rbf_model = SVC(kernel='rbf')


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


rbf_model.fit(X_train, y_train)


# In[ ]:


rbf_model.score(X_test,y_test)


# In[ ]:


'''Using Linear kernel'''


# In[ ]:


linear_model = SVC(kernel='linear')
linear_model.fit(X_train,y_train)


# In[ ]:


linear_model.score(X_test,y_test)


# In[ ]:




