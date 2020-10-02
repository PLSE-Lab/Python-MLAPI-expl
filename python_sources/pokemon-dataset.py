#!/usr/bin/env python
# coding: utf-8

# # Here's what we are going to be doing this class
# - Import the dataset
# - Count null values
# - Look for any null y values and drop those rows
# - Drop all columns that don't matter of the bat, like pokedex_number and name
# - Fill null values
# - Make dummy variables

# In[ ]:


import pandas as pd
pokemon = pd.read_csv("../input/pokemon/pokemon.csv")

pokemon.head()


# In[ ]:


pokemon.isna().sum()


# In[ ]:


pokemon.info()


# In[ ]:


pokemon.describe()


# 
# 
# ## Count how many legendary Pokemon there are

# In[ ]:


count = 0
for data in pokemon.is_legendary:
    if data == 1:
        count += 1

print(count)


# 
# 
# ## Split X and y, and delete all non important columns

# In[ ]:


y = pokemon['is_legendary']
X = pokemon
X = X.drop(columns=['is_legendary'])
X = X.drop(columns=['type2'])
X = X.drop(columns=['name'])
X = X.drop(columns=['japanese_name'])
X = X.drop(columns=['abilities'])
X = X.drop(columns=['pokedex_number'])

# This line can be commented out to run all X variables
X = X[['attack', 'sp_attack', 'sp_defense', 'speed', 'weight_kg', 'percentage_male', 'height_m', 'defense', 'base_egg_steps', 'type1']]
X = pokemon[['attack', 'speed', 'type1','weight_kg', 'percentage_male', 'height_m']]
X.head(25)


# 
# 
#    ## Check to see what null values need filled in our X

# In[ ]:


X.isna().sum()


# 
# 
# ## Get averages for height and weight, then fill them in the missing columns

# In[ ]:


average_height = pokemon.height_m.mean()
average_weight = pokemon.weight_kg.mean()
num = 50

X.height_m.fillna(average_height, inplace=True)
X.weight_kg.fillna(average_weight, inplace=True)
X.percentage_male.fillna(num, inplace=True)


# 
# 
# ## Make sure they are filled:

# In[ ]:


X.isna().sum()


# ## Get dummy variables for any strings

# In[ ]:


X = pd.get_dummies(X, drop_first=True)
X.head()


# In[ ]:


X.corr()


# 
# 
# ## Normalize the data from 0-1

# In[ ]:


from sklearn import preprocessing

cols = X.columns
x = X.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled, index=X.index, columns=cols)


# In[ ]:


X.head()


# In[ ]:


y.head()


# ---
# # We are going to stop here for this class. Next class we will do the linear regression stuff below.

# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=0)

logreg.fit(X_train,y_train)


# In[ ]:


predictions = logreg.predict(X_test)


# In[ ]:



from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)


# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)


# In[ ]:


X.info()


# In[ ]:


from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

test = X['attack']

X2 = sm.add_constant(test)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:





# In[ ]:


from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

test = pokemon[['attack', 'speed']]

X2 = sm.add_constant(test)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:


from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

test = X[['speed']]
#test = X[['attack', 'speed', 'percentage_male', 'defense', 'sp_defense', 'base_egg_steps' ]]
test = pokemon[['speed', 'base_egg_steps']]

X2 = sm.add_constant(test)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(test, y, test_size=0.25, random_state=0)

logreg1 = LogisticRegression(random_state=0)

logreg1.fit(X_train,y_train)

predictions = logreg1.predict(X_test)

accuracy_score(y_test, predictions)


# In[ ]:




