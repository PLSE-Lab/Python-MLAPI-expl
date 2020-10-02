#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORTING THE DATA SET #
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data_set = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')


# In[ ]:


data_set.head()


# In[ ]:


data_set.describe()


# In[ ]:


data_set.hist(column = 'blueWins', bins = 10)
plt.show()


# In[ ]:


# SEPARATING THE VARIABLES IN PREDICTORS AND TARGET VARIABLES #

x = data_set['blueWins']
y = data_set.drop('blueWins', axis=1)


# In[ ]:


# CREATING THE TRAINING AN TEST DATA SET #

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[ ]:


# MODEL CREATION #

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(y_train, x_train)


# In[ ]:


# PRINTING THE RESULTS #

result = (model.score(y_test, x_test))*100
print(f'Performance:  {result}' '%')

