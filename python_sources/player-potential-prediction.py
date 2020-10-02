#!/usr/bin/env python
# coding: utf-8

# ### Try to predict a player's Potential given just a few of his skills.

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/fifa19/data.csv')


# ## Exploratory data analysis

# In[ ]:


df.columns


# In[ ]:


df.describe()['Potential']


# In[ ]:


df = df.filter(['Potential', 'Age', 'ShortPassing', 'LongPassing', 'Agility', 'BallControl', 'Vision', 'Aggression'])


# In[ ]:


from sklearn.utils import shuffle

df = shuffle(df)


# In[ ]:


corr = df.corr()
corr['Potential'].sort_values(ascending=False)


# In[ ]:


df.info()


# ## Data transformations

# In[ ]:


df.dropna(subset=df.columns, inplace=True)


# In[ ]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2)


# In[ ]:


train_set[:10]


# In[ ]:


train_set.info()


# In[ ]:


players = train_set.drop('Potential', axis=1)
players_labels = train_set['Potential'].copy()


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])


# In[ ]:


players_prepared = pipeline.fit_transform(players)
players_prepared[:10]


# ## Training

# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(players_prepared, players_labels)


# In[ ]:


some_data = players.iloc[:5]
some_labels = players_labels.iloc[:5]
some_data_prepared = pipeline.transform(some_data)
print('Predictions:', lin_reg.predict(some_data_prepared))
print('Labels:', list(some_labels))


# In[ ]:


from sklearn.metrics import mean_squared_error
players_predictions = lin_reg.predict(players_prepared)
lin_mse = mean_squared_error(players_labels, players_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(players_prepared, players_labels)

players_predictions = tree_reg.predict(players_prepared)
tree_mse = mean_squared_error(players_labels, players_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# ## Evaluation

# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(lin_reg, players_prepared, players_labels, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-scores)


# In[ ]:


def display_scores (scores):
    print("Scores:", scores) 
    print("Mean:", scores.mean()) 
    print("Standard deviation:", scores.std()) 


# In[ ]:


display_scores(lin_rmse_scores)


# In[ ]:


X_test = test_set.drop('Potential', axis=1)
y_test = test_set['Potential'].copy()

X_test_prepared = pipeline.transform(X_test)
final_predictions = lin_reg.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[ ]:




