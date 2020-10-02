#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Modules
from sklearn import datasets
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


iris_df = datasets.load_iris()


# In[ ]:


# Features
print(iris_df.feature_names)


# In[ ]:


# Targets
print(iris_df.target)


# In[ ]:


# Target Names
print(iris_df.target_names)
label = {0: 'red', 1: 'blue', 2: 'green'}


# In[ ]:


# Dataset Slicing
x_axis = iris_df.data[:, 0]  # Sepal Length
y_axis = iris_df.data[:, 2]  # Sepal Width


# In[ ]:


# Plotting
plt.scatter(x_axis, y_axis, c=iris_df.target)
plt.show()


# **Violet: Setosa, Green: Versicolor, Yellow: Virginica**

# **K-means Clustering**

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


# Declaring Model
model = KMeans(n_clusters=3)


# In[ ]:


# Fitting Model
model.fit(iris_df.data)


# In[ ]:


# Predicitng a single input
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])


# In[ ]:


# Prediction on the entire data
all_predictions = model.predict(iris_df.data)


# In[ ]:


predicted_label


# In[ ]:




