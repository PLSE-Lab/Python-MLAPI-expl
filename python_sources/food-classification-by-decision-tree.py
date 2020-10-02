#!/usr/bin/env python
# coding: utf-8

# ## Example of food classification based on diet nutrition using decision tree.
# 
# This is an example of decision tree classfication of food based on diet nutrition data.
# 
# #### Reference
# 
# - Sebastian Raschka and Vahid Mirjalili, Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow, 2nd Edition. (Capter 3, A Tour of Machine Learning Classifiers Using scikit-learn)
# 
# ## 1. Import libraries

# In[ ]:


get_ipython().system('pip install pydotplus')

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image, display_png
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz


# ## 2. Load data

# In[ ]:


df = pd.read_csv('../input/Emoji Diet Nutritional Data (g) - EmojiFoods (g).csv')
df


# Nice cute emojis ;-).
# 
# ## 3. Add labels
# 
# Since the data has no labels for classification, we add the following labels to each row.
# 
# - 0 (fruits): grapes, melon, watermelon, tangerine, lemon, banana, pineapple, red apple, green apple, pear, peach, cherries, strawberry, kiwifruit, avocado
# - 1 (vegitables): tomato, eggplant, potato, carrot, corn, hot pepper, cucumber, mushroom, peanuts, chestnut
# - 2 (grain crops): bread, croissant, french bread, pancakes, rice crackers, rice, spaghetti
# - 3 (animals/fishes): cheese, beef, chicken, bakon, fried shrimp
# - 4 (junk foods): hamburger, french fries, pizza, hotdog, taco, burrito, popcorn
# - 5 (desserts): ice cream, doughnut, cookie, cake, chocolate bar, candy, custard flan, honey
# - 6 (drinks): milk, black tea, sake, champagne, red wine, beer

# In[ ]:


labels =[0,0,0,0,0,0,0,0,0,0,
         0,0,0,0,1,0,1,1,1,1,
         1,1,1,1,1,2,2,2,2,3,
         3,3,3,4,4,4,4,4,4,4,
         2,2,2,3,5,5,5,5,5,5,
         5,5,6,6,6,6,6,6]
len(labels)

df['labels']= labels
df.info()


# ## 4. Make decision tree
# 
# We try to make a decision tree to classify the food into these seven classes by using the following nine features; Calories (kcal), Carbohydrates (g), Total Sugar (g), Protein (g), Total Fat (g), Saturated Fat (g), Monounsaturated Fat (g), Polyunsaturated Fat (g), Total Fiber (g)                 

# In[ ]:


tree = DecisionTreeClassifier(criterion='gini', random_state =1, min_samples_leaf=1)

X_train = df.iloc[:, 2:11]
y_train = df['labels']

tree.fit(X_train, y_train)
dot_data = export_graphviz(tree, filled = True, rounded = True, class_names = ['fruits','vegetables', 'grain crops', 'animals/fishes', 'junk foods', 'desserts', 'drinks'],
                          feature_names = df.columns[2:11].values, out_file = None)

graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')
display_png(Image('tree.png'))


# ## 5. Result summary
# 
# From the decision tree, we can see the following patterns.
# 
# - Most of drinks are low calory (<=1.13) and low sugar (<=0.013), except for sake which is made from rice and has nutrition similar to grain crops category.
# - Fruits and vegetables have similar nutrition. The main difference is the amount of sugar. Most of foods have total sugar > 0.055.
# - Most of desserts have high calories (> 1.13) and high sugar (> 0.162).
# - Junk foods and animals/fishes have high saturated fat (>0.021).
