#!/usr/bin/env python
# coding: utf-8

# ## Load and review the data set

# In[ ]:


import numpy as np
import pandas as pd
data = pd.read_csv("../input/add.csv",low_memory=False)


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


# Reading first 20 rows
data.head(20)


# ## There are 1559 columns in the data.Each row in the data represent one image which is tagged as ad or nonad in the last column.column 0 to 1557 represent the actual numerical attributes of the images

# In[ ]:


# Check whether a given value is a missing value, if yes change it to NaN
def toNum(cell):
    try:
        return np.float(cell)
    except:
        return np.nan
    
# Apply missing value check to a column / Pandas series
def seriestoNum(series):
    return series.apply(toNum)


# In[ ]:


# Missing  values have been replaced by Nan
train_data=data.iloc[0:,0:-1].apply(seriestoNum)
train_data.head(20)


# ## Remove the missing value

# In[ ]:


# Apply pandas dropna function for removing all the missing value rows
train_data=train_data.dropna()
train_data.head(20)


# In[ ]:


# Converting Training label to 0 or 1
def toLabel(str):
    if str=="ad.":
        return 1
    else:
        return 0
    
train_labels=data.iloc[train_data.index,-1].apply(toLabel)
train_labels


# ## Training the data

# In[ ]:


# using row 50 t0 2200 for training
from sklearn.svm import LinearSVC 
clf = LinearSVC()
clf.fit(train_data[50:2200],train_labels[50:2200])


# ## Testing

# In[ ]:


clf.predict(train_data.iloc[12].values.reshape(1,-1))


# In[ ]:


clf.predict(train_data.iloc[-1].values.reshape(1,-1))


# In[ ]:




