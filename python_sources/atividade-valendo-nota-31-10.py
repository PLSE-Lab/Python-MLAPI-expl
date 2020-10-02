#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import sklearn.datasets as dt        


# In[ ]:


dic = dt.load_digits()
dic.keys()


# In[ ]:


dic.data


# In[ ]:


dic.data.shape


# In[ ]:


dic.images.shape


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(dic.images[880])


# In[ ]:


x=dic.data
y=dic.target


# In[ ]:


dy = pd.DataFrame(y)
dy.nunique()
dy.columns


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7)
modelo = knn.fit(x_train, y_train)
y_pred = modelo.predict(x_test)
y_score = modelo.score(x_test, y_test)
y_score


# In[ ]:


comparacao = pd.DataFrame(y_test)
comparacao['pred'] = y_pred
comparacao.head(50)

