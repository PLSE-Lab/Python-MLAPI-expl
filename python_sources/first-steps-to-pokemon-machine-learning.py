#!/usr/bin/env python
# coding: utf-8

# Heya, just a notebook for me to play around.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import warnings

# Any results you write to the current directory are saved as output.


# Import the required data, and take a look at all the columns.

# In[ ]:


data = pd.read_csv("../input/SeventhGenPokemon3.csv", encoding='latin-1')
print(data.info())


# Yay! No empty columns. But that is alot of columns... also, why doesn't type 2 contain empty data?

# In[ ]:


print(data.groupby('Type2').Type2.count())


# Oh! So they do have a none inside! How helpful! Now let's decide what to do with the data. Let's see what are the main factors in deciding the Base_Total! For this, I guess we can drop some columns. And lets remove the Alolan Column. I do not want to add it in...

# In[ ]:


data_want = data.copy()
data_want = data_want.drop(['Normal_Dmg', 'Fire_Dmg', 'Water_Dmg', 'Eletric_Dmg', 'Grass_Dmg', 'Ice_Dmg', 
                            'Fight_Dmg', 'Poison_Dmg', 'Ground_Dmg', 'Flying_Dmg', 'Psychic_Dmg', 'Bug_Dmg', 
                            'Rock_Dmg', 'Ghost_Dmg', 'Dragon_Dmg', 'Dark_Dmg', 'Steel_Dmg', 'Fairy_Dmg', 
                           'isAlolan', 'hasAlolan'], axis=1)
print(data_want.info())


# Let's get graphing! Let's begin with a heat map showing correlations!

# In[ ]:


import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# In[ ]:


corr = data_want.corr()
hm = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)


# Wow...thats tiny :/ From the heat map, the Base_Total seems to have a strong negative correlation to Female_Pct and Base_Happiness, and a strong positive correlation to Special Defense and Special Attack (Kinda duh right, I mean these add to the Base_Total?) Let's take a look at some of the more interesting stuff!

# In[ ]:


bh = sns.relplot(y='Base_Total', x='Base_Happiness', size='Female_Pct', hue='Egg_Steps', col='Generation',
                 col_wrap=True, data=data_want)


# That's interesting...The 0 Egg_Steps pokemon tend to have higher base totals for the first 6 generations. In fact, all pokemon with 0 Egg_Steps have a 0% Female_Pct. Perhaps they are the legendaries! Let's take them out then!

# In[ ]:


bh = sns.relplot(y='Base_Total', x='Base_Happiness', size='Female_Pct', hue='Egg_Steps', col='Generation',
                 row='Legendary', data=data_want)


# Yep, looks like it. It is interesting though, to note that among the legendaries, there exists some that could be female... Guess i never played enough D:
# 
# Anyways, that is all fine and dandy, but what about the Height and Weight?

# In[ ]:


hgt = sns.relplot(y='Base_Total', x='Height.m.', data=data_want, hue='Generation')


# In[ ]:


hgt = sns.relplot(y='Base_Total', x='Weight.kg.', data=data_want, hue='Generation')


# Looks quite promising too...
# 
# I guess we can further cut some stuff out again. Espcially region (which is almost the same as generation, but less numerical.

# In[ ]:


data_want = data_want.drop(['National', 'Mega_Evolutions', 'Region', 'Male_Pct'], axis=1)
print(data_want.info())


# And also, change the objects to a int value.

# In[ ]:


data_want['Exp_Speed'] = data_want['Exp_Speed'].map({'Erratic':1, 'Fast':2, 'Fluctuating':3, 'Medium':4, 'Medium Fast':5, 
                                       'Medium Slow':6, 'Slow':7})
data_want['Group1'] = data_want['Group1'].map({'Amorphous':1, 'Bug':2, 'Ditto':3, 'Dragon':4, 'Fairy':5, 'Field':6, 
                                    'Flying':7, 'Grass':8, 'Human-like':9, 'Mineral':10, 'Monster':11,
                                    'None':12, 'Water 1':13, 'Water 2':14, 'Water 3':15})
data_want['Group2'] = data_want['Group2'].map({'Amorphous':1, 'Bug':2, 'Ditto':3, 'Dragon':4, 'Fairy':5, 'Field':6, 
                                    'Flying':7, 'Grass':8, 'Human-like':9, 'Mineral':10, 'Monster':11,
                                    'None':12, 'Water 1':13, 'Water 2':14, 'Water 3':15})
data_want['Type1'] = data_want['Type1'].map({'bug':1, 'dark':2, 'dragon':3, 'electric':4, 'fairy':5, 'fighting':6,
                                            'fire':7, 'flying':8, 'ghost':9, 'grass':10, 'ground':11, 'ice':12,
                                            'normal':13, 'poison':14, 'psychic':15, 'rock':16, 'steel':17,
                                            'water':18})
data_want['Type2'] = data_want['Type2'].map({'bug':1, 'dark':2, 'dragon':3, 'electric':4, 'fairy':5, 'fighting':6,
                                            'fire':7, 'flying':8, 'ghost':9, 'grass':10, 'ground':11, 'ice':12,
                                            'normal':13, 'poison':14, 'psychic':15, 'rock':16, 'steel':17,
                                            'water':18})
data_want['Capt_Rate'] = data_want['Capt_Rate'].replace('30 (Meteorite)255 (Core)', '255')
data_want['Capt_Rate'] = data_want['Capt_Rate'].astype(int)
print(data_want.info())


# In[ ]:


corr = data_want.corr()
plt.figure(figsize=(25, 25))
hm = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)


# Looks like the Type1, Type2 and Generation have pretty weak correlations too...Guess we can remove them.

# In[ ]:


data_want=data_want.drop(['Type1', 'Type2', 'Generation'], axis=1)
data_want=data_want.drop('Name', axis=1)


# Now for some Machine Learning part!

# In[ ]:


from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection


# In[ ]:


Y = data_want.Base_Total
X = data_want.drop('Base_Total',axis=1)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


scaler = preprocessing.StandardScaler().fit(X_train)
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=400))
hyperparameters = {'randomforestregressor__min_samples_split': [2],
                   'randomforestregressor__min_samples_leaf': [1],
                   'randomforestregressor__max_features': ['sqrt'],
                   'randomforestregressor__max_depth': [None],
                  'randomforestregressor__bootstrap': [False]}

clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_validation)
print(r2_score(Y_validation, Y_pred))
print(mean_squared_error(Y_validation, Y_pred))


# Whew, I'm not too bad it seems :D
