#!/usr/bin/env python
# coding: utf-8

# # Understand Regression model step by step
# 
# ### Contents
# 1. Load require Libraries.
# 1. Load and understand datasets.
# 1. Data Visualization
#  1. Univarient Analysis
#  1. Multivarient Analysis
# 1. Missing Value Treatment
# 1. Feature Engineering
# 1. Scaling Data
# 1. One Hot Engineering
# 1. Model Creation
# 

# ## Load Libraries

# In[ ]:


#Import Basic Libraries
import numpy as np 
import pandas as pd 

#Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Scaling Data
from sklearn.preprocessing import MinMaxScaler

#Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold

#Model
from sklearn.linear_model import LinearRegression, ElasticNet,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

import os
print(os.listdir("../input"))


# ## Loan Dataset

# In[ ]:


pokemon = pd.read_csv('../input/pokemon.csv')
pokemon.head(5)


# In[ ]:


#Here First Column name is #. 
pokemon = pokemon.rename(columns = {'#':"ID"})
pokemon.head(5)


# In[ ]:


#Read Combat
combat = pd.read_csv('../input/combats.csv')
combat.head(5)


# In[ ]:


#Get Shape of two dataframes
print('Pokemon shape is ' + str(pokemon.shape))
print('Combat shape is ' + str(combat.shape))


# In[ ]:


#Get Info
print('Pokemon info :')
pokemon.info()
print('\nCombat info :')
combat.info()


# In[ ]:


#Get Missing value
pokemon.isnull().sum()


# In[ ]:


combat.isnull().sum()


# In[ ]:


#Total Number Match of each pockemon
FirstCombat = combat.First_pokemon.value_counts().reset_index(name = 'FirstCombat')
SecondCombat = combat.Second_pokemon.value_counts().reset_index(name = 'SecondCombat')
TotalCombat = pd.merge(FirstCombat, SecondCombat, how = 'left', on = 'index')
TotalCombat['TotalMatch'] = TotalCombat['FirstCombat']+TotalCombat['SecondCombat']

TotalCombat.sort_values('index').head()


# In[ ]:


#Match winning details
FirstWin = combat['First_pokemon'][combat['First_pokemon'] == combat['Winner']].value_counts().reset_index(name = 'FirstWin')
SecondWin = combat['Second_pokemon'][combat['Second_pokemon'] == combat['Winner']].value_counts().reset_index(name = 'SecondWin')
TotalWin = pd.merge(FirstWin, SecondWin, how  = 'left', on = 'index')
TotalWin['TotalWin'] = TotalWin['FirstWin']+ TotalWin['SecondWin']
TotalWin.head(5)


# In[ ]:


#Here we have 3 data frame. Let's combine all
result = pd.merge(pokemon, TotalCombat, how = 'left', left_on= 'ID', right_on = 'index')
result = pd.merge(result, TotalWin, how = 'left', on = 'index')
result = result.drop(['index'], axis = 1)
result.head(10)


# In[ ]:


#Winning Percentage
pd.set_option('precision', 0)
result['WinningPercentage'] = (result.TotalWin / result.TotalMatch) * 100
result.head(5)


# In[ ]:


#Some Pokemon don't have Type2 char. So we can replace it with null char
result['Type 2'].fillna('Not Applicable', inplace = True)
result.head(10)


# ## Data Visualization

# In[ ]:


categ = ['Type 1','Type 2','Generation','Legendary']
conti = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']


# ### Univarient Analysis

# In[ ]:


#Univarient Analysis
plt.figure(figsize= (7,40))
i = 0
for cat in categ:
    plt.subplot(8,2,i+1)
    sns.countplot(x = cat, data = result);
    plt.xticks(rotation = 90)
    i+=1
for cont in conti:
    plt.subplot(8,2,i+1)
    sns.distplot(result[cont])
    i+=1
plt.show()


# ### Bivarient Analysis

# In[ ]:


#Now Visulaize how char related with WinningPercentage 
plt.figure(figsize = (8,30))
i =0
for cat in categ:
    plt.subplot(8,2,i+1)    
    sns.barplot(x = cat, y = 'WinningPercentage', data = result);
    plt.tight_layout()
    plt.xticks(rotation = 90)
    i+=1

for cont in conti:
    plt.subplot(8,2,i+1)
    sns.scatterplot(x = 'WinningPercentage', y = cont, data = result)
    i+=1
plt.show()


# ## Missing Value Treatment

# In[ ]:


result.info()


# In[ ]:


#drop na values in our dataframe
result = result.dropna()
result.info()


# ## Feature Engineering

# In[ ]:


result.loc[result['Type 2'] != 'Not Applicable', 'Char'] = 'Both_Char'
result.loc[result['Type 2'] == 'Not Applicable', 'Char'] = 'Only_One_Char'
result.head(5)


# ## Scaling Data

# In[ ]:


pd.set_option('display.float_format', '{:.2f}'.format)
Scaleing_result = result

from sklearn.preprocessing import StandardScaler

col_name = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','FirstWin','SecondWin','TotalWin']
scale = StandardScaler()
Scaleing_result[col_name] = scale.fit_transform(Scaleing_result[col_name])
Scaleing_result.head(5)


# ## One Hot Encoding

# In[ ]:


#Let's drop ID, Name Column
Encoding_result = Scaleing_result.drop(['ID','Name','FirstCombat','SecondCombat','TotalMatch'],axis =1)
Encoding_result['Legendary'] = Encoding_result['Legendary'].astype(str)
Encoding_result = pd.get_dummies(Encoding_result, drop_first = True)
Encoding_result.head(5)


# In[ ]:


#Correlation Matrix
plt.figure(figsize = (5,5))
sns.heatmap(Encoding_result.corr(), cmap = 'Greens')
plt.show()


# In[ ]:


#Split Dependent and Target Variable
WinningPercentage = Encoding_result['WinningPercentage']
Encoding_result.drop(['WinningPercentage'], axis =1, inplace = True)


# In[ ]:


#Split Dataset
x_train, x_test, y_train, y_test = train_test_split(Encoding_result,WinningPercentage, test_size = 0.2, random_state = 10)


# ## Model Creation

# In[ ]:


#Let's Create Model
models = []
models.append(('LR',LinearRegression()))
models.append(('EN', ElasticNet()))
models.append(('Lasso', Lasso()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('GB', GradientBoostingRegressor()))
models.append(('Ada', AdaBoostRegressor()))


# In[ ]:


model_results = []
names = []
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 10)
    cv_result = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'neg_mean_squared_error')
    model_results.append(cv_result)
    names.append(name)
    msg = '%s %f (%f)' % (name, cv_result.mean(), cv_result.std())
    print(msg)


# In[ ]:


#Visualize our result
plt.figure(figsize = (5,5))
sns.boxplot(x = names, y = model_results)


# Above graph shows Gradient Boosting gives better result. Now we can predict Winning percentage of each pokemon.

# In[ ]:


GBM = GradientBoostingRegressor()
GBM.fit(x_train, y_train)
pred = GBM.predict(x_test)


# #### Let's Compare Result

# In[ ]:


plt.figure(figsize = (7,7))
sns.regplot(y_test, pred)
plt.show()


# In[ ]:


plt.figure(figsize = (18,3))
sns.lineplot(x=y_test.index.values, y=y_test, color = 'purple')
sns.lineplot(x=y_test.index.values, y=pred, color = 'orange')
plt.show()


# Our model shows predicted values is very close to actual target values in test set .
