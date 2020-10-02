#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set()


# In[ ]:


data = pd.read_csv('/kaggle/input/gender-classification/Transformed Data Set - Sheet1.csv')


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.head()


# ## This small dummy data I have used takes four independent factors to predict the Gender of the person.

# In[ ]:


_ = sns.countplot(x = data.Gender, y=None)


# # Label Encoding to plug into model

# In[ ]:


from sklearn.preprocessing import LabelEncoder

data1 = data.copy(deep = True)
le_color = LabelEncoder()
data1['Favorite Color'] = le_color.fit_transform(data['Favorite Color'])

le_genre = LabelEncoder()
data1['Favorite Music Genre'] = le_genre.fit_transform(data['Favorite Music Genre'])


le_beverage = LabelEncoder()
data1['Favorite Beverage'] = le_beverage.fit_transform(data['Favorite Beverage'])


le_drink = LabelEncoder()
data1['Favorite Soft Drink'] = le_drink.fit_transform(data['Favorite Soft Drink'])


le_gender = LabelEncoder()
data1['Gender'] = le_gender.fit_transform(data['Gender'])


# In[ ]:


data1.head()


# In[ ]:


X = data1.drop(['Gender'], axis = 1)
y = data1.Gender


# # Model

# In[ ]:


from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X, y)


# In[ ]:


clf.score(X, y)


# # Visualization

# # With tree.plot_tree

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize = (20,20))
_ = tree.plot_tree(clf.fit(X, y)) 


# # With GraphViz

# In[ ]:


import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data1.drop(['Gender'],axis=1).columns, filled=True) 
graph = graphviz.Source(dot_data) 


# In[ ]:


graph


# # Basic Interpretation

# #### 1. Each box represents a split.
# #### 2. In tree.plot_tree function we can see the condition for split.
# #### 3. Both graphs mention gini on each box. Gini is the default objective function of decision trees.
# #### 4. Number of samples used to make the split or leaf is also mentioned in each box.
# #### 5. Value = [] gives us the split count on all splits made
# #### 6. In graphviz function we can see the feature name used for split.

# # Hyper parameter tuning

# # min_samples_split

# In[ ]:


from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=6)
clf = clf.fit(X, y)


# In[ ]:


clf.score(X, y)


# In[ ]:


import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data1.drop(['Gender'],axis=1).columns, filled=True) 
graph = graphviz.Source(dot_data) 


# In[ ]:


graph


# # min_samples_leaf

# In[ ]:


from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=6, min_samples_leaf=5)
clf = clf.fit(X, y)


# In[ ]:


dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data1.drop(['Gender'],axis=1).columns, filled=True) 
graph = graphviz.Source(dot_data) 


# In[ ]:


graph


# ## Some splits are now not made due to the hyperparameters we put. This is a great way to study tree hyperparameters and its tuning on a decent scale

# In[ ]:




