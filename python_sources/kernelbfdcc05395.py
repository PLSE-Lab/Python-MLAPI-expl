#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


model = pd.read_csv('../input/train_V2.csv')


# In[ ]:


model = model.dropna(axis='rows')


# In[ ]:


x = model.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]


# In[ ]:


y = model.iloc[:, 28]


# In[ ]:


model = RandomForestRegressor(n_estimators=20,random_state=0)


# In[ ]:


model.fit(x,y)


# In[ ]:


test = pd.read_csv('../input/test_V2.csv')


# In[ ]:


test = test.dropna(axis='rows')


# In[ ]:


x = test.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]


# In[ ]:


y = model.predict(x)


# In[ ]:


x = test.iloc[:, 0]


# In[ ]:


out = np.vstack((x, y))


# In[ ]:


out = np.transpose(out)


# In[ ]:


out = pd.DataFrame(out, columns=['Id', 'winPlacePerc'])


# In[ ]:


out.to_csv('submission1.csv', index=False)

