#!/usr/bin/env python
# coding: utf-8

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

# Any results you write to the current directory are saved as output.


# In[ ]:


# Epicurious Recipe Dataset: https://www.kaggle.com/hugodarwood/epirecipes
data = pd.read_json("../input/full_format_recipes.json")
data.head()


# In[ ]:


# Libraries
import nltk
from nltk import pos_tag, word_tokenize
import re


# In[ ]:


# Pre-processing: Remove all NaN values within the dataset, and any recipes with improper syntax.
data.drop([6,123,1946,2423,3299,3838,4248,4650,5063,5259,6560,7162,9307,10562,11089,12648,15243,16874,19369,19822], inplace = True) 
data.fillna('', inplace = True)
#data['ingredients'] = data['ingredients'].apply(lambda y: np.nan if len(y)==0 else y)
#data.dropna(subset = ['ingredients'], inplace = True)
data.reset_index(drop = True, inplace = True)


# In[ ]:


# Reduce each ingredient description to a one-word descriptor. 
simplified_ingredients = [] # store the reduced ingredients for each recipe
ingredients_list = [] # list of all ingredients in the dataset
for i in range(len(data['ingredients'])): 
    simplified_ingredients.append([])
    for j in range(len(data['ingredients'][i])):
        tokenized_ingredients = nltk.word_tokenize(data['ingredients'][i][j]).
        # Remove any parenthesized items - main ingredient unlikely to be found there.
        if '(' in tokenized_ingredients: 
            tokenized_ingredients = tokenized_ingredients[0:tokenized_ingredients.index('(')] + tokenized_ingredients[tokenized_ingredients.index(')')+1:len(tokenized_ingredients)-1]       
        # Remove items after comma - main ingredients unlikely to be found there.
        if ',' in tokenized_ingredients:
            tokenized_ingredients = tokenized_ingredients[0:tokenized_ingredients.index(',')]
        tagged_ingredients = nltk.pos_tag(tokenized_ingredients)
        # Select last noun token. 
        tagged_ingredients = [k for k in tagged_ingredients if k[1] == 'NN' or k[1] == 'NNS']
        tagged_ingredients = tagged_ingredients[-1:]
        if len(tagged_ingredients) != 0:
            simplified_ingredients[i].append(tagged_ingredients[0][0]) # add only the token
            ingredients_list.append(tagged_ingredients[0][0])


# In[ ]:


# Find all unique ingredients.
ingredients_set = set(ingredients_list)
ingredients_list = list(ingredients_set)

# Use ingredients as our features. Store counts of each ingredient per recipe.
ingredients_features = np.zeros((len(simplified_ingredients), len(ingredients_list)))
for n in range(len(simplified_ingredients)): 
    for m in range(len(ingredients_list)):
        ingredients_features[n][m] = simplified_ingredients[n].count(ingredients_list[m])


# In[ ]:


# Label recipes as dessert or non-dessert recipes.
recipe_type = np.zeros((len(data['categories']), 1))
for a in range(len(data['categories'])):
    if 'Dessert' in data['categories'][a]:
        recipe_type[a] = 1
    else:
        recipe_type[a] = 0


# In[ ]:


# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

feature_selector = SelectKBest(chi2, k = 10) # reduce features to the 10 best features
selected_ingredients = feature_selector.fit_transform(ingredients_features, recipe_type)

mask = feature_selector.get_support() # mask of chosen features 
for b in range(len(mask)):
    if(mask[b] == True):
        print(ingredients_list[b])


# In[ ]:


# Split into training and testing sets 80/20
end_index_train = int((len(data) * .80) - 1)
beg_index_test = int(len(data) * .80)
x_train = selected_ingredients[0:end_index_train][:]
x_test = selected_ingredients[beg_index_test:][:]
y_train = recipe_type[0:end_index_train][:] 
y_test = recipe_type[beg_index_test:][:]


# In[ ]:


# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='liblinear').fit(x_train, y_train.ravel())
classifier.predict(x_test)
classifier.score(x_test, y_test) # accuracy of classifier

