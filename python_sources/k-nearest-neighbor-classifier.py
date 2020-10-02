#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv('train.csv')
del df['PlayerID']
del df['Name']
df['GPMIN'] = df['GP'] * df['MIN']
df['3PAP3P'] = df['3PA'] * df['3P%']
df['FTAFT%'] =df['FTA'] * df['FT%']


# In[ ]:


y=df['TARGET_5Yrs']
del df['TARGET_5Yrs']


# In[ ]:


y = y.as_matrix()


# In[ ]:


X = df.as_matrix()


# In[ ]:


import numpy as np
t = np.where(np.isnan(X))
for i in range(len(t[0])):
    X[t[0][i]][t[1][i]] = 0


# In[ ]:


np.where(np.isnan(X))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
clf = KNeighborsClassifier(n_neighbors=150)
clf.fit(X_train, y_train)
y_pre = clf.predict(X_test)
print(np.mean(y_pre==y_test))


# In[ ]:





# In[ ]:


df = pd.read_csv('test.csv')
del df['PlayerID']
del df['Name']
df['GPMIN'] = df['GP'] * df['MIN']
df['3PAP3P'] = df['3PA'] * df['3P%']
df['FTAFT%'] = df['FTA'] * df['FT%']


# In[ ]:


df


# In[ ]:


test=df.as_matrix()


# In[ ]:


test


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clv=KNeighborsClassifier(n_neighbors=100)
clv.fit(X, y)
y_pre = clv.predict(test)


# In[ ]:


y_pre


# In[ ]:


sub = pd.read_csv('sample_submission.csv')
sub


# In[ ]:


del sub['TARGET_5Yrs']
sub['TARGET_5Yrs'] =y_pre


# In[ ]:


sub = sub.set_index('PlayerID')
sub


# In[ ]:


sub.to_csv('ANS1.csv')


# In[ ]:


pd.read_csv('ANS1.csv')


# In[ ]:





# In[ ]:




