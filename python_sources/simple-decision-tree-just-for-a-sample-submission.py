#!/usr/bin/env python
# coding: utf-8

# ## Simple Decision Tree just for a Sample Submission

# ### Abstract

# The only purpose is to make a sample submission. I prefer not to make it with a dummy file, instead, I have modeled very simple decision tree by using only train.csv and test.csv.

# ### Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  


# ### Loading and Displaying Datasets

# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


print("length of train dataset:", len(df_train))
print("length of test dataset:", len(df_test))


# ### Creating Numpy Arrays from Datasets

# In[ ]:


y = df_train["target"].values
X = df_train[["card_id", "feature_1", "feature_2", "feature_3"]].values
Xsub = df_test[["card_id", "feature_1", "feature_2", "feature_3"]].values

print("input features for train set:\n",X)
print("--------------------")
print("input features for test set:\n",Xsub)
print("--------------------")
print("output:\n",y)


# ### Create Very Simple Decisin Tree

# In[ ]:


model = DecisionTreeRegressor(max_depth=5)


# ### Train the Model

# In[ ]:


model.fit(X[:,1:], y)


# ### Make Predictions

# In[ ]:


y_pred = model.predict(Xsub[:,1:])


# ### Create Result Numpy Array

# In[ ]:


y_pred = y_pred.reshape(len(y_pred),1)


# In[ ]:


resultarray = np.append(Xsub, y_pred, axis=1)


# In[ ]:


print(resultarray)


# ### Create a Dataframe for Submission

# In[ ]:


resultdf = pd.DataFrame(resultarray, columns=["card_id", "f1", "f2", "f3", "target"])


# In[ ]:


resultdf = resultdf.drop(['f1', 'f2', 'f3'], axis=1)


# In[ ]:


resultdf.head()


# ### Create Submission csv File

# In[ ]:


resultdf.to_csv("submission.csv", sep=',', index=False)

