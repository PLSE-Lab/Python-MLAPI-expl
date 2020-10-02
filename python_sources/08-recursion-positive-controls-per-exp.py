#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# ## How many controls in each experiments ?

# In[ ]:


train_controls = pd.read_csv('../input/train_controls.csv')
test_controls = pd.read_csv('../input/test_controls.csv')


# In[ ]:


d = {}
for i,exp in enumerate(train_controls.experiment.unique()):
    d[exp] = len(train_controls[train_controls.experiment == exp].sirna.unique())
d


# In[ ]:


d = {}
for i,exp in enumerate(test_controls.experiment.unique()):
    d[exp] = len(test_controls[test_controls.experiment == exp].sirna.unique())
d


# ## How many replicates of each control siRNA per experiments ?

# In[ ]:


siRNA = train_controls.sirna.unique()
siRNA


# The same controls siRNA are presents in `test_controls`.

# In[ ]:


test_siRNA = test_controls.sirna.unique()
sorted(test_siRNA) == sorted(siRNA)


# In[ ]:


train_exp = train_controls.experiment.unique()
d = {}
for exp in train_exp:
    siRNA_per_exp = []
    siRNA_per_exp.extend([len(train_controls[(train_controls.experiment == exp) & (train_controls.sirna == i)]) for i in list(siRNA)])
    d[exp] = siRNA_per_exp


# In[ ]:


df = pd.DataFrame(d, index=siRNA)


# In[ ]:


sns.heatmap(df)
plt.title('Number of siRNA replicates per experiment [train]')
plt.show()


# In[ ]:


test_exp = test_controls.experiment.unique()
d = {}
for exp in test_exp:
    siRNA_per_exp = []
    siRNA_per_exp.extend([len(test_controls[(test_controls.experiment == exp) & (test_controls.sirna == i)]) for i in list(siRNA)])
    d[exp] = siRNA_per_exp


# In[ ]:


df = pd.DataFrame(d, index=siRNA)


# In[ ]:


sns.heatmap(df)
plt.title('Number of siRNA replicates per experiment [test]')
plt.show()


# ## Conclusion
# We can observe that most of the positive controls are present 4 times. Some siRNA positive controls are only present 3 times in train experiments or 2 times in test experiments. These missing replicates became a negative control (untreated cell: siRNA 1138). 
# All experiments have 31 controls.

# In[ ]:




