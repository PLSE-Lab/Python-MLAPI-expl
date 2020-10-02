#!/usr/bin/env python
# coding: utf-8

# ## About this notebook
# This notebook kernel was created to help you understand more about machine learning. I intend to create tutorials with several machine learning algorithms from basic to advanced. I hope I can help you with this data science trail. For any information, you can contact me through the link below.
# 
# Contact me here: https://www.kaggle.com/vitorgamalemos/machine-learning-01-decision-tree
# 
# ## What is a Decision Tree?
# 
# **Another notebook on decision trees:** https://www.kaggle.com/vitorgamalemos/machine-learning-02-simple-decision-tree
# 
# <p style="text-align: justify;"> A decision tree is a decision support tool that uses a tree-like graph or model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements. In short, a decision tree is a function as a decision model. These decision trees are trained according to a training data set. The algorithms most used to the construction of these trees are IDE, C4.5, and ASSISTANT.</p>

# ## An example dataset

# In[ ]:


import pandas as pd 
  
data = [['Day1', 'Sun', 'Warm', 'High', 'Weak', 'No'], 
        ['Day2', 'Sun', 'Warm', 'High', 'Strong', 'No'],
        ['Day3', 'Cloud', 'Warm', 'High', 'Weak', 'Yes'],
        ['Day4', 'Rain', 'Soft', 'High', 'Weak', 'Yes'], 
        ['Day5', 'Rain', 'Fresh', 'Normal', 'Weak', 'Yes'],
        ['Day6', 'Rain', 'Fresh', 'Normal', 'Strong', 'No'],
        ['Day7', 'Cloud', 'Fresh', 'Normal', 'Weak', 'Yes'], 
        ['Day8', 'Sun', 'Soft', 'High', 'Weak', 'No'],
        ['Day9', 'Sun', 'Fresh', 'Normal', 'Weak', 'Yes'],
        ['Day10', 'Rain', 'Soft', 'Normal', 'Strong', 'Yes'],
        ['Day11', 'Sun', 'Soft', 'Normal', 'Strong', 'Yes'], 
        ['Day12', 'Cloud', 'Soft', 'High', 'Strong', 'Yes'],
        ['Day13', 'Cloud', 'Warm', 'Normal', 'Weak', 'Yes'],
        ['Day14', 'Rain', 'Soft', 'High', 'Strong', 'No']]


df = pd.DataFrame(data, columns = ['Day', 'Aspect', 'Temperature', 'Humidity','Wind','Output']) 
df.head(12)


# ## Categorization (Transforming to numeric data)

# In[ ]:


df_type_1 = df.select_dtypes(include=['int64']).copy()
df_type_2 = df.select_dtypes(include=['float64']).copy()
df_type_3 = pd.concat([df_type_2, df_type_1], axis=1, join_axes=[df_type_1.index])


# ## Categorization using Sklearn

# In[ ]:


from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder

categorization = preprocessing.LabelEncoder()
categorization.fit(df["Aspect"].astype(str))
list(categorization.classes_)

df_object = df.astype(str).apply(categorization.fit_transform)
df_formated = pd.concat([df_type_3, df_object], axis=1, join_axes=[df_type_3.index])
df_formated.head(12)


# In[ ]:


import matplotlib.pyplot as plt
df_formated.hist(alpha=0.5, figsize=(15, 15), color='blue')
plt.show()


# In[ ]:


import pandas
from pandas.plotting import scatter_matrix

scatter_matrix(df_formated, alpha=0.5, figsize=(20, 20))
plt.show()


# In[ ]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X = df_formated.ix[:,'Day':'Wind':] 
y = df_formated.ix[:, 'Output':]


# ## Training decision tree

# In[ ]:


tree_clf = DecisionTreeClassifier(max_depth=6)
tree_clf.fit(X, y)


# ## Visualizing our decision tree

# In[ ]:


from sklearn.tree import export_graphviz
from IPython.display import Image
from subprocess import call
from graphviz import Digraph

t = export_graphviz(tree_clf,out_file='tree_decision.dot',
                    feature_names=['Day','Aspect','Temperature','Humidity','Wind'],
                    class_names=['Yes', 'No'],
                    rounded=True,filled=True)

call(['dot', '-Tpng', 'tree_decision.dot', '-o', 'tree.png', '-Gdpi=800'])
Image(filename = 'tree.png')


# ## Entropy Calculation
# The entropy of a set can be defined as the purity of the set. Given a set $S$, with instances belonging to class $i$, with probability $pi$, the entropy is defined as:
# 
# To build a decision tree, we need to calculate the entropy using the frequency table of only one attribute and the entropy of the two attributes.
# 
# ### Entropy Calculation only One Attribute:
# $$
# Entropy(S) = \sum_{i=1}^{c}({{-P_i}} ({{log_2}} {{P_i}})
# $$
# 
# ### Entropy Calculation Two Attributes:
# $$
# Entropy(T, X) = \sum_{c {E} x}^{c}({{P(C)}} {{Entropy(C)}})
# $$
