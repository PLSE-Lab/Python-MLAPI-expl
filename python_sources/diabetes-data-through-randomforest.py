#!/usr/bin/env python
# coding: utf-8

# **DATA LOADING**
# *  import pandas library to load data
# *  import numpy lib to make test and trian data .. you can use sklearn lib to divide your data

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


dataset = pd.read_csv('../input/diabetes/diabetes.csv')
#let see the data
dataset


# In[ ]:


len(dataset)


# In[ ]:


datasets=np.array(dataset)
datasets


# In[ ]:


dataset.shape


# In[ ]:


len(datasets)


# **SPLITTING THE DATA**

# In[ ]:


train_data = datasets[1:1512,:8]
train_data


# In[ ]:


train_labels = datasets[1:1512,-1]
train_labels


# In[ ]:


test_data = datasets[1512:,:8]
test_data


# In[ ]:


test_labels = datasets[1512:,-1]
test_labels


# In[ ]:


test_data.shape


# In[ ]:


test_labels.shape


# In[ ]:


train_data.shape


# In[ ]:


train_labels.shape


# In[ ]:


train_data.dtype


# **DATA VISUALIZATION**

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


corr = train_data[:8,:9]
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True)


# In[ ]:


# RELATION BETWEEN SKIN THICKNESS AND HAVING DIABETIES OR NOT
ax = sns.barplot(x="SkinThickness", y="Outcome",data=dataset[:40])


# In[ ]:


## RELATION BETWEEN AGE AND HAVING DIABETIES OR NOT
Bax = sns.barplot(x="Age", y="Outcome",data=dataset[:30])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model = RandomForestClassifier()


# In[ ]:


model.fit(train_data,train_labels)


# In[ ]:


model.score(train_data,train_labels)


# In[ ]:


predict= model.predict(test_data)


# In[ ]:


model.score(test_data,predict)


# In[ ]:





# In[ ]:




