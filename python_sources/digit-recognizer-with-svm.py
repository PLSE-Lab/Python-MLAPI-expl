#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


raw_data = pd.read_csv("../input/train.csv")
raw_data.head(5)


# In[ ]:


data = raw_data.drop("label",axis=1)
label = raw_data["label"]


# # Original Image

# In[ ]:


image_data = data.iloc[5]
plt.imshow(image_data.values.reshape(28,28), cmap="Greens")


# # Modified Image 1

# In[ ]:


temp = data.iloc[5]
temp = temp/255
plt.imshow(temp.values.reshape(28,28), cmap="Greens")


# # Modified Image 2

# In[ ]:


temp1 = data.iloc[5]
temp1[temp1>1]=1
plt.imshow(temp1.values.reshape(28,28), cmap="Greens")


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model = SVC(gamma='scale')


# In[ ]:


data[data>0]=1


# In[ ]:


model.fit(data, label)


# In[ ]:


test_data = pd.read_csv("../input/test.csv")


# In[ ]:


test_data[test_data>0]=1


# In[ ]:


prediction = model.predict(test_data)


# In[ ]:


results = pd.Series(prediction,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("results.csv",index=False)


# In[ ]:





# In[ ]:




