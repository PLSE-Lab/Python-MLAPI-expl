#!/usr/bin/env python
# coding: utf-8

# ###Loading packages.

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error


# ###Loading dataset and inspecting it.

# In[ ]:


zoo = pd.read_csv('../input/zoo.csv', index_col='animal_name')


# In[ ]:


zoo.shape


# That's quite small, but for exercise purposes it should be okay.

# In[ ]:


zoo.head()


# In[ ]:


zoo.describe()


# All variables except for 'legs' are binary.

# In[ ]:


zoo.info()


# The variable type is int64 for all variables.

# ### Correlations

# In[ ]:


corr = zoo.corr()

sns.heatmap(corr, square=True, linewidths=.3,cmap="RdBu_r")
plt.show()


# In[ ]:


corr_filt = corr[corr != 1][abs(corr)> 0.7].dropna(how='all', axis=1).dropna(how='all', axis=0)
print(corr_filt)


# In[ ]:


print("Correlation of Milk and Hair: %1.3f" %corr.loc['milk', 'hair'])
print("-------------------------")
print("Correlation of Milk and Toothed: %1.3f" %corr.loc['milk', 'toothed'])
print("-------------------------")
print("Correlation of Tail and Backbone: %1.3f" %corr.loc['tail', 'backbone'])
print("-------------------------")
print("Correlation of Milk and Eggs: %1.3f" %corr.loc['milk', 'eggs'])
print("-------------------------")
print("Correlation of Hair and Eggs: %1.3f" %corr.loc['hair', 'eggs'])
print("-------------------------")
print("Correlation of Hair and Eggs: %1.3f" %corr.loc['feathers', 'eggs'])
print("-------------------------")


# What do we learn from this? 
# 
#  - If an animal lays eggs, it probably won't be able to produce milk and quite likely won't have hair (but the correlation with feathers isn't that big either
# - It might make sense to exclude at least eggs, maybe also hair. Just knowing that the animal is a mammal helps a lot in determining the species.

# In[ ]:


plt.hist(zoo.class_type, bins=7)
plt.show()


# In[ ]:


zoo.class_type.value_counts()


# It's tempting to just summarize this to 2 or 3 subclasses to have a balanced dataset, but for now we'll keep the imbalance and hope that the feature space has the power to distinguish them anyway.
# 
# 
# As mentioned before, 'eggs' and 'hair' will have to go the way of the dodo for now.

# In[ ]:


zoo_sel = zoo.drop(['eggs', 'hair'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(zoo_sel.drop('class_type', axis=1), zoo_sel['class_type'], test_size=0.33, random_state=42)


# ## Modeling

# ### Multinomial Logistic Regression

# In[ ]:


mult = LogisticRegression()


# In[ ]:


mult.fit(X_train, y_train)


# In[ ]:


mult_pred = mult.predict(X_test)


# In[ ]:


conf_matrix_mult = confusion_matrix(y_test,mult_pred)
print(conf_matrix_mult)


# The algorithm misclassified 4 out of 32 examples. That's not perfect but a decent benchmark.

# Playing around with the different settings yields no difference in results.

# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=7, leaf_size=50)


# In[ ]:


knn.fit(X_train, y_train)


# In[ ]:


knn_pred = knn.predict(X_test)


# In[ ]:


conf_matrix_knn = confusion_matrix(y_test,knn_pred)
print(conf_matrix_knn)


# So that did worse.

# ### Deep Learning
