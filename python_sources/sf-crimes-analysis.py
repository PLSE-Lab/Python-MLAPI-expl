#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# Let's start with some visualization to explore the data a bit.

# In[ ]:


# data visualization imports
import seaborn as sns
sns.set(style="white", color_codes=True)

import matplotlib.pyplot as plt
sf_features_train = pd.read_csv("../input/train.csv")
sf_features_train.head()


# OK, so there aren't a huge number of features in the data, so doing some direct investigation may provide interesting toe-holds.  
# Let's start with just looking at the Category (the target feature) to see what the distribution is like.

# In[ ]:


sf_features_train.groupby('Category').size().plot(kind='bar')


# Some easy observations right away: there are 6 categories that really stick out as common,
# though really just 4 after you take out "NON-CRIMINAL" and "OTHER".
# There are also 18 that strike me as pretty rare so some kind of algorithm that can use initial probabilities
# as an input might be interesting to explore.  I wonder if they cluster by anything.
# Let's look at DayOfWeek for the most common crime, LARCENY/THEFT.

# In[ ]:


theft_data = sf_features_train[sf_features_train['Category'] == 'LARCENY/THEFT']
sns.countplot(x='DayOfWeek', data=theft_data)


# Fairly even across the board, though Friday and Saturday are slightly elevated.
# I wonder what sorts of resolutions there are, and whether those are fairly
# even too.

# In[ ]:


sf_features_train.groupby('Resolution').size().plot(kind='bar')


# Clearly not, "NONE", "ARREST, BOOKED", and "ARREST, CITED" are all way more common than other
# handlings.  I do notice that several of these resolutions indicate that the offender
# is a juvenile, I wonder if that changes the type of cases at all?

# In[ ]:


juvenile_resolutions = [
    'CLEARED-CONTACT JUVENILE FOR MORE INFO',
    'JUVENILE ADMONISHED',
    'JUVENILE BOOKED',
    'JUVENILE CITED',
    'JUVENILE DIVERTED'
]
criterion = sf_features_train['Resolution'].map(lambda x: x in juvenile_resolutions)
juv_cases = sf_features_train[criterion]
juv_cases.groupby('Category').size().plot(kind='bar')


# This is relevent, juvenile cases aren't nearly as predominantly Larceny.
# Well, we can come back and do more exploration in a bit, but I think I want
# to start trying some fairly naive classifiers and see how well we can do right out of
# the box. One of the first problems I see is that most of this data (with the exception of time
# and location) is already categorical, and therefore it's hard for me
# to visualize what a linear decision boundary would look like.  I think I want to start with a
# decision tree.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)


# Great, now what features do I care about?

# In[ ]:




