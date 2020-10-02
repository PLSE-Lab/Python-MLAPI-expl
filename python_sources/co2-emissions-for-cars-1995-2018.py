#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import utils, pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 50)


# # All the data is splitted in 20 csv files and the headings have been erased.

# In[ ]:


dataframes = []
filename = '../input/Fuel Consumption Ratings'
indexes = ['MODEL YEAR', 'MAKE', 'MODEL', 'VEHICLE CLASS', 'ENGINE SIZE', 'CYLINDERS','TRANSMISSION','FUEL TYPE','FUEL CONSUMPTION CITY(L/100 km)','HWY(L/100 km)','COMB(L/100 km)','COMB(mpg)','CO2 EMISSIONS(g/km)']

for i in range(20):
    to_add = pd.read_csv(filename+str(i)+'.csv', names=indexes)
    dataframes.append(to_add)
df = pd.concat(dataframes)
df.T


# In[ ]:


print(df.shape)
#checking if there is NaN values
df.isna().sum()


# In[ ]:


#checking for innecesary data; i.e. just 1 value for feature.
df.nunique()


# In[ ]:


df.drop_duplicates(keep='first', inplace=True)


# # Dropping the MODEL YEAR atribute, since its value is already represented in other features and has no relevance anymore.

# In[ ]:


df.drop(['MODEL YEAR'], axis=1, inplace=True)
df.T


# # Separating categoric from numeric variables:

# In[ ]:


num_vars = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]
num_vars


# In[ ]:


cat_vars = [c for c in df if not c in num_vars]
cat_vars


# # Converting object variables into categoric ones and storing the relation in cat_dict

# In[ ]:


cat_dict = {}

for n, col in df.items():
    if n in cat_vars:
        df[n] = df[n].astype('category')
        cat_dict[n] = {i+1:e for i,e in enumerate(df[n].cat.categories)}
cat_dict


# # Replacing categoric variables by their codes:

# In[ ]:


for n,col in df.items():
    if n in cat_vars:
        df[n] = df[n].cat.codes + 1
df.T


# In[ ]:


df.dtypes


# # Now that we have every variable as numeric, we can start with the training

# In[ ]:


#splitting independent variables from the label
x = df.drop(['CO2 EMISSIONS(g/km)'], axis=1)
y = df['CO2 EMISSIONS(g/km)']

#splitting the data for testing and training:
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1507)


# # Training the data and seeing its test and Out-Of-Bag score

# In[ ]:


m = RandomForestRegressor(1000, n_jobs=-1, oob_score=True)
m.fit(x_train, y_train)


# In[ ]:


def score():
    print(f'Scores:')
    print(f'Train      = {m.score(x_train, y_train):.4}')
    print(f'Validation = {m.score(x_val, y_val):.4}')
    if hasattr(m, 'oob_score_'): print(f'OOB        = {m.oob_score_:.4}')


# In[ ]:


score()


# # Feature Importance

# In[ ]:


imp = pd.DataFrame({'cols':x_train.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)
imp.style.bar(color='lightblue')


# In[ ]:




