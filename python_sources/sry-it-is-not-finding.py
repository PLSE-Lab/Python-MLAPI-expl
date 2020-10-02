#!/usr/bin/env python
# coding: utf-8

# I am sorry that this kernel is not right.<br> You can see onodera's comment. The features are normal distribution. So it is not a matter.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


feature = [c for c in train_df.columns if c not in ['ID_code', 'target']]


# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# In[ ]:


med = train_df[feature].median()
med = med.reset_index()
med.columns = ["feature", "value"]
med.sort_values(by="value", inplace=True)
f = list(med.feature)


# In[ ]:


for i in range(10):
    train_df.loc[i, f].plot()
    plt.show()


# > It looks same shape.
# time series or there are some secrets.
