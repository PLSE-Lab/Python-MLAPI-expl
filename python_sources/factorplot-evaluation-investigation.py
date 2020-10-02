#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


x2 = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
y2 = [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]


# In[ ]:


import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.DataFrame({'Pclass': x2, 'Survived': y2})


# In[ ]:


test.head()


# In[ ]:


import seaborn as sns

sns.factorplot('Pclass', 'Survived', data=test)


# In[ ]:


test3 = pd.DataFrame(test)


# In[ ]:


test3


# In[ ]:


test3.describe()


# In[ ]:


# the point is exactly the mean(expectation) of y. For example, when x = 1, 
# we have (1, 0), (1, 3), (1, 3), and (1, 3), so the mean is (0 + 3 + 3 + 3) / 4 = 2.25,
# while the range is [0, 3]. The bar is from 0.75 to 3.0, why?

sns.factorplot(x='Pclass', y='Survived', data=test3)


# In[ ]:


# now use factorplot to determine if this feature is important
import seaborn as sns

sns.factorplot(x='Pclass', y='Survived', data=train)


# In[ ]:


sns.factorplot('Pclass', 'Survived', col='Sex', data=train)


# In[ ]:


sns.factorplot('Sex', 'Survived', col='Pclass', data=train)

