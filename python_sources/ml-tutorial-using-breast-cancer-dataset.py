#!/usr/bin/env python
# coding: utf-8

# # Import Statements

# In[59]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# In[60]:


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[61]:


from xgboost.sklearn import XGBClassifier


# In[62]:


data = pd.read_csv("../input/data.csv")


# # Info given on data - 
# 
# #### Attribute Information:
# 
#     - column 1 : ID number
#     - column 2 : Diagnosis (M = malignant, B = benign)
#     - column 3-32 : Ten real-valued features are computed for each cell nucleus:
# 
#         a) radius (mean of distances from center to points on the perimeter)
#         b) texture (standard deviation of gray-scale values)
#         c) perimeter
#         d) area
#         e) smoothness (local variation in radius lengths)
#         f) compactness (perimeter^2 / area - 1.0)
#         g) concavity (severity of concave portions of the contour)
#         h) concave points (number of concave portions of the contour)
#         i) symmetry
#         j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

# In[63]:


data.sample(10)


# ###### Looking at the sample dataset, we can  infer -
# 
# 1. Result is the **diagnosis** column. It should be used as labels.
# 2. Columns **id**, **Unnamed: 32** don't have any significance in the dataset.
# 3. The other columns left can be used as features.

# In[64]:


del data['id']
del data['Unnamed: 32']


# In[65]:


data.sample(10)


# In[66]:


data.info()


# *Except for diagnosis column, every column is float64. Let's convert diagnosis too.*

# In[67]:


data['diagnosis'].unique()


# In[68]:


data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})


# In[69]:


data.sample(10)


# ## Frequency of different diagnosis

# In[70]:


sns.countplot(data.diagnosis, label='Count')


# # Further exploration in the feature set
# 
# So we know that there are 10 attributes for which **mean**, **standard error (se)**, **worst** have been calculated. These become the features. <br>
# Therefore, **10 x 3 = 30 Features**!
# 
# Let's divide the attributes into groups.

# In[71]:


print(list(data.columns))


# In[72]:


features_mean = list(data.columns[1:11])
features_se = list(data.columns[11:21])
features_worst =list(data.columns[21:31])
print("---------------- features_mean -------------------------------------------------------")
print(features_mean)
print("\n---------------- features_se (Standard Error) -------------------------------------------------------")
print(features_se)
print("\n---------------- features_worst --------------------------------------------------------")
print(features_worst)


# # Feature Selection
# 
# Using a correlation graph to see if we can remove any columns from the feature set.

# In[73]:


corr = data[features_mean].corr().abs()
lower_right_ones = np.tril(np.ones(corr.shape, dtype='bool'), k=-1)
correlations = corr.where(lower_right_ones)
correlations


# In[74]:


plt.figure(figsize=(12,12))
sns.heatmap(correlations, annot=True, cmap='RdBu_r', fmt= '.2f', vmax=1, vmin=-1)
plt.xticks(rotation=60)


# In[75]:


THRESHOLD_VALUE = 0.85
list(i for i in (correlations[correlations.gt(THRESHOLD_VALUE)].stack().index) if i[0] is not i[1])


# In[76]:


correlations[correlations.gt(THRESHOLD_VALUE)].stack().sort_values(ascending = False)


# ### We can conclude - 
# 
#  1. **radius_mean**, **perimeter_mean** and **area_mean** are highly correlated. Hence, we will use **radius_mean** only.
#  2. **concavity_mean**, **concave points_mean** and **compactness_mean** are highly correlated. Hence, we will use **concavity_mean** only.

# ------------------------------------
# 
# Let's do the same for features_se, features_worst too.

# In[77]:


corr = data[features_se].corr().abs()
lower_right_ones = np.tril(np.ones(corr.shape, dtype='bool'), k=-1)
correlations = corr.where(lower_right_ones)
plt.figure(figsize=(12,12))
sns.heatmap(correlations, annot=True, cmap='RdBu_r', fmt= '.2f', vmax=1, vmin=-1)
plt.xticks(rotation=60)


# In[78]:


THRESHOLD_VALUE = 0.85
correlations[correlations.gt(THRESHOLD_VALUE)].stack().sort_values(ascending = False)


# ### We can conclude - 
# 
#  1. **radius_se**, **perimeter_se** and **area_se** are highly correlated. Hence, we will use **radius_se** only.

# In[79]:


corr = data[features_worst].corr().abs()
lower_right_ones = np.tril(np.ones(corr.shape, dtype='bool'), k=-1)
correlations = corr.where(lower_right_ones)
plt.figure(figsize=(12,12))
sns.heatmap(correlations, annot=True, cmap='RdBu_r', fmt= '.2f', vmax=1, vmin=-1)
plt.xticks(rotation=60)


# In[80]:


THRESHOLD_VALUE = 0.85
correlations[correlations.gt(THRESHOLD_VALUE)].stack().sort_values(ascending = False)


# ### We can conclude - 
# 
#  1. **radius_worst**, **perimeter_worst** and **area_worst** are highly correlated. Hence, we will use **radius_worst** only.
#  2. **concavity_worst**, **concave points_worst** and **compactness_worst** are highly correlated. Hence, we will use **concavity_worst** only.

# ### Now we know what columns to use.

# In[81]:


to_remove = [
    'concave points_meanr', 'compactness_mean', 'perimeter_mea', 'area_mean',
    'concave points_worst', 'compactness_worst', 'perimeter_worst', 'area_worst',
    'perimeter_se', 'area_se'
]
to_use = [e for e in data.columns if e not in to_remove]
print(to_use)


# In[82]:


reduced_data = data[to_use]
reduced_data.sample(10)


# # Data Transformation

# In[83]:


X = reduced_data.loc[:, 'radius_mean': 'fractal_dimension_worst']
Y = reduced_data['diagnosis']


# In[84]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[85]:


sc = preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[86]:


pca = PCA(.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# # Using Models

# In[87]:


svc = SVC()
gaussian_nb = GaussianNB()
decision_tree_classifier = DecisionTreeClassifier()
random_forest_classifier = RandomForestClassifier()
logistic_regression = LogisticRegression()
k_neighbors_classifier = KNeighborsClassifier()


# In[88]:


svc.fit(X_train,Y_train)
gaussian_nb.fit(X_train,Y_train)
decision_tree_classifier.fit(X_train,Y_train)
random_forest_classifier.fit(X_train,Y_train)
logistic_regression.fit(X_train,Y_train)
k_neighbors_classifier.fit(X_train,Y_train)


# In[89]:


print("svc - {0:.3f}".format(svc.score(X_test, Y_test)))
print("gaussian_nb - {0:.3f}".format(gaussian_nb.score(X_test, Y_test)))
print("decision_tree_classifier - {0:.3f}".format(decision_tree_classifier.score(X_test, Y_test)))
print("random_forest_classifier - {0:.3f}".format(random_forest_classifier.score(X_test, Y_test)))
print("logistic_regression - {0:.3f}".format(logistic_regression.score(X_test, Y_test)))
print("k_neighbors_classifier - {0:.3f}".format(k_neighbors_classifier.score(X_test, Y_test)))


# In[ ]:




