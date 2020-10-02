#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/heights-and-weights/data.csv")


# In[ ]:


data.head()


# In[ ]:


data.dropna()


# In[ ]:


data.mean()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = data.iloc[:,:-1].values


# In[ ]:


y = data.iloc[:,1].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


regression = LinearRegression()


# In[ ]:


regression.fit(X_train,y_train)


# In[ ]:


predict = regression.predict(X_test)


# In[ ]:


data_output = pd.DataFrame({"actual":y_test,"predicted":predict})


# In[ ]:


data_output


# In[ ]:


plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regression.predict(X_train),color="blue")
plt.title("height vs weight")
plt.xlabel("h")
plt.ylabel("w")
plt.show()


# In[ ]:


plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regression.predict(X_train),color="blue")
plt.title("height vs weight")
plt.xlabel("h")
plt.ylabel("w")
plt.show()


# In[ ]:




