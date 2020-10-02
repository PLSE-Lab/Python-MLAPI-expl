#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as ss


# In[20]:


df = pd.read_csv('../input/heart.csv')
df.head()


# In[21]:


df.info()


# It seems that all columns dont have nulls. Also, from the documentation I don't see any special values that can be nulls. Hence, we might not have to worry about filling missing values.

# In[22]:


df.describe()


# In[23]:


for col in df.columns:
    print(col)
    print(df[col].unique())


# So, columns sex, cp, fbs, restecg, exang, slope, ca, thal are all categoricals. Also, other columns like age, trestbps, chol, thalach can be turned into categoricals.

# In[51]:


categoricals = "sex cp fbs restecg exang slope ca thal target".split()
for col in categoricals:
    df[col] = df[col].astype('category')
df.dtypes


# Let's understand the target variable

# In[52]:


df['target'].value_counts()


# So, we have similar counts for each class such that it's approximately balanced classes

# In[53]:


corrmat = df.corr()
corrmat


# In[54]:


sns.heatmap(corrmat, annot=True)


# So, none of the continuous variables are strongly correlated with others, which might mean overlapping features & we might have to consider PCA.

# Now, we'll see how the categorical variables correlate to the target variable

# In[55]:


# Thanks to this article by Shaked Zychlinski!
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
import math
from collections import Counter
def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    :param x: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :param y: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :return: float
    """
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


# In[57]:


for col in categoricals:
    print(f"Correlation of {col} with target : {theils_u(df[col], df['target'])}")


# There isn't any 1 feature with a strong correlation with the target.
# We can do this visually too with plots where we can see which values correlate to which target values.

# In[59]:


for col in categoricals:
    sns.countplot(x=col, hue='target', data=df)
    plt.show()


# There are some correlations between the values, but nothing that suggests a simple model should have 100% accuracy.

# We can look at scatter plots & distributions for the continuous variables

# In[60]:


continuous = set(df.columns) - set(categoricals)
continuous


# In[68]:


sns.pairplot(data=df[list(continuous)])


# Sure, some distributions are skewed, but the scatter plots appear random, rather than correlated.

# In[ ]:




