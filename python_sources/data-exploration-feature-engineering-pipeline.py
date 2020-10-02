#!/usr/bin/env python
# coding: utf-8

# # Data exploration + pipeline for ingredients one hot encodings
# 
# # Import data
#  
# Curently I'm considering only *ingredients*, *nutrition* and free features like *n_ingredients* and *n_steps* for exploration and one hot encodings. 

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

r_recipes = pd.read_csv('../input/food-com-recipes-and-user-interactions/RAW_recipes.csv')
test = pd.read_csv('../input/food-com-recipes-and-user-interactions/interactions_test.csv')
train = pd.read_csv('../input/food-com-recipes-and-user-interactions/interactions_train.csv')
validation = pd.read_csv('../input/food-com-recipes-and-user-interactions/interactions_validation.csv')


# In the following I will get required data about recipes and interactions, merge train and validation data sets as we use later cross-validation, calculate the average rating per recipe in train/test/validate, and join recipes data with rating data. 

# In[ ]:


r_recipes = r_recipes[['id', 'ingredients', 'nutrition', 'n_steps', 'n_ingredients']]
r_recipes.columns = ['recipe_id', 'ingredients', 'nutrition', 'n_steps', 'n_ingredients']
r_recipes = r_recipes.set_index('recipe_id')

train = pd.concat([train[['user_id', 'recipe_id', 'rating']], validation[['user_id', 'recipe_id', 'rating']]], axis = 0)

train_rating = pd.DataFrame(train.groupby(['recipe_id']).mean()['rating'])
test_rating = pd.DataFrame(test.groupby(['recipe_id']).mean()['rating'])

recipes_rating_train = r_recipes.join(train_rating, how = 'inner')
recipes_rating_test = r_recipes.join(test_rating, how = 'inner')

recipes_rating_train['rating'] = recipes_rating_train['rating'].apply(lambda x: round(x))
recipes_rating_test['rating'] = recipes_rating_test['rating'].apply(lambda x: round(x))

train_test = pd.concat([recipes_rating_train[0:round(recipes_rating_train.shape[0]*0.3)], recipes_rating_test[0:round(recipes_rating_test.shape[0]*0.3)]])


# # Some data exploration
# 
# To continue with prediction of ratings I will check for missing values, dublicates, look for outliers and imbalances. 

# In[ ]:


def avoidRowsWithMissValues(df):
  if(df.isnull().values.any()): 
    columns = df.columns
    for column in columns: 
      df[df[column].isnull()] = ""
      df[df[column]=='NaN'] = ""
      df[pd.isna(df[column])] = ""
  return df

recipes_rating_train = avoidRowsWithMissValues(recipes_rating_train)
recipes_rating_test = avoidRowsWithMissValues(recipes_rating_test)


recipes_rating_train.drop_duplicates()
recipes_rating_test.drop_duplicates()


# #### Ratings distribution 
# 
# Most of recipes are rated with 5 in both set. So we need later to take care about imbalanced classes. Class 4 appears more often in test class. 
# 

# In[ ]:


sns.set(style = "whitegrid")
ax = sns.boxenplot(x = recipes_rating_train['rating'])
ax.set_xticks(np.arange(0,6))
ax.set_xlabel('Ratings in train set')
plt.show()


# In[ ]:


sns.set(style = "whitegrid")
ax = sns.boxenplot(x = recipes_rating_test['rating'])
ax.set_xticks(np.arange(0,6))
ax.set_xlabel('Ratings in test set')
plt.show()


# #### Number of ingredients
# Is similarly distributed in train and test sets and do not include critical outliers.
# 

# In[ ]:


sns.set(style = "whitegrid")
ax = sns.boxenplot(x = recipes_rating_train['n_ingredients'])
ax.set_xticks(np.arange(0,20))
ax.set_xlabel('Number of ingredients per recipe in train set')
plt.show()


# In[ ]:


sns.set(style = "whitegrid")
ax = sns.boxenplot(x = recipes_rating_test['n_ingredients'])
ax.set_xticks(np.arange(0,20))
ax.set_xlabel('Number of ingredients per recipe in test set')
plt.show()


# #### Number of steps per recipe 
# The distribution of steps in train and test sets are similar. Some of recipes in both sets have unsualy many steps (> 22). The number of such recipes are similarly distributed in test and train sets and the rating disctribution of outliers corresponds to general rating distribution.  

# In[ ]:


sns.set(style = "whitegrid")
ax = sns.boxenplot(x = recipes_rating_train['n_steps'])
ax.set_xticks(np.arange(0,40, 2))
ax.set_xlabel('Number of steps per recipe')
plt.show()


# In[ ]:


sns.set(style = "whitegrid")
ax = sns.boxenplot(x = recipes_rating_test['n_steps'])
ax.set_xticks(np.arange(0,40, 2))
ax.set_xlabel('Number of steps per recipe')
plt.show()


# In[ ]:


recipes_rating_train[recipes_rating_train['n_steps'] > 22]['rating'].value_counts()


# In[ ]:


recipes_rating_test[recipes_rating_test['n_steps'] > 22]['rating'].value_counts()


# # Feature engineering
# 
# Following I will choose a bunch of features to be ready to run classification algorithms. Firstly I check correlations and transform ingredients and nutrition values into a suitable format. With the help of CountVectorizer all ingredients are added to a vocabulary. So each recipe ingredients will be represented by onehot-encodings. Nutrition data will be as well separated into different columns. 
# 
# For deployment it is better when all the transformations with the input data are done in a pipeline. For this purpose i created Scikit Learn -designed transformers that will transform the current and future input data. But the input data is expected to be in particular format like includes *recipe_id* and other required columns.

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

def strToList(list_l, splitSymbol):
    list_l = list_l.split(splitSymbol)
    temp = list()
    for l in list_l: 
        l = l.replace("[",'').replace("]",'').replace("'", '').replace(" ", '')
        temp.append(l)
    return temp

class ingredientsToList(BaseEstimator, TransformerMixin): 
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self, X):
        return self
    def transform(self, X): 
      for column in self.columns:
        X[column] = X[column].apply(lambda x : strToList(x, ','))
      return X
    
class ingredientsToOneHot(BaseEstimator, TransformerMixin): 
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self, X):
        return self
    def transform(self, X): 
        cv = CountVectorizer(analyzer=lambda x: x)
        for column in self.columns:
            test = cv.fit_transform(X[column].to_list())
            test_columns = [x for x in cv.get_feature_names()]
            X = X.join(pd.DataFrame(test.toarray(), columns = test_columns, index = X.index))
            #X = X.join(pd.DataFrame(test.toarray(), index = X.index))
        return X

class nutritionDataIntoCol(BaseEstimator, TransformerMixin): 
    def fit(self, X):
        return self

    def transform(self, X): 
      nutrition_X = pd.DataFrame(X['nutrition'].to_list(), columns = ['calories', 'total fat', 'sugar_nutrition', 'sodium', 'protein', 'saturated fat', 'carbohydrates'], index = X.index)
      
      nutrition_X_col = nutrition_X.columns
      for col in nutrition_X_col: 
        nutrition_X[col] = nutrition_X[col].apply(lambda x: float(x))

      X = X.join(nutrition_X)
      return X

class getFeatureColumns(BaseEstimator, TransformerMixin): 
    def fit(self, X):
        return self
    def transform(self, X):
        col = list(X.columns)
        for c in ['ingredients', 'nutrition', 'n_steps', 'n_ingredients']:
            col.remove(c)
        return X[col]


# In[ ]:


from sklearn.pipeline import Pipeline

pip = Pipeline([
    ('ingredientsToList', ingredientsToList(columns = ['ingredients', 'nutrition'])), 
    ('ingredientToOneHotColumns', ingredientsToOneHot(columns = ['ingredients'])),
    ('nutritionData', nutritionDataIntoCol()), 
    ('getFeatureColumns', getFeatureColumns())
])
all_withFeatures = pip.transform(train_test)
all_withFeatures.head()


# ## Clean RAM
# Here I remove some of variables that are not required any more. It allows to conduct further memory-consuming experiments.

# In[ ]:


del r_recipes
del train
del train_rating
del train_test
del recipes_rating_train
del recipes_rating_test


# ## Avoid rare ingredients
# The idea here is to avoid ingredients that appear rarely in recipes. Since they can disturb models while looking for the fit. So I consider further ingredients that appear at least in 1% of recipes. 

# In[ ]:


col = list(all_withFeatures.columns)
for column in ['rating', 'calories', 'total fat', 'sugar_nutrition', 'sodium', 'protein',
       'saturated fat', 'carbohydrates']:
    col.remove(column)
sum_ingredients = pd.DataFrame(all_withFeatures[col].sum(axis = 0)/all_withFeatures.shape[0])

sns.set(style = "whitegrid")
ax = sns.boxenplot(x = sum_ingredients.values*100)
ax.set_xticks(np.arange(0,10, 2))
ax.set_xlabel('Number of time an ingredient was used')
plt.show()

sum_freq_ingred_index = list(sum_ingredients[sum_ingredients[0]>0.01].index)

