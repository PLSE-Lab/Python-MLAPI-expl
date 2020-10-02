#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


from sklearn import linear_model


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


dataframe = pd.read_fwf("../input/data.txt")
print("sample data imported !!!")


# In[ ]:


x_values = dataframe[["Brain"]]


# In[ ]:


y_values = dataframe[["Body"]]


# In[ ]:


body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)


# In[ ]:


plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()


# In[ ]:




