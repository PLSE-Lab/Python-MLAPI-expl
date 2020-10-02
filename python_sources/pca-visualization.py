#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets, decomposition, linear_model, preprocessing, model_selection
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Load Boston housing Dataset from sklearn 
data_boston = datasets.load_boston()
X = pd.DataFrame(data = data_boston.data, columns = data_boston.feature_names)
y = pd.DataFrame(data = data_boston.target, columns = ['target'])


# In[ ]:


# Preview the data
X.head()


# ### PCA Description and Visualization

# PCA is a method of representing data using fewer dimensions. If there is large collinearity in the data, PCA can be used to transform data to a feature space where there is no collinearity. However, PCA can be used to check if collinearity exists in data which will be explained later. 
# PCA is also used to shrink the feature space - sometimes by orders of magnitude. 
# 
# 

# First, we check how many dimensions we want to transform our data to. We plot the cumulative 
# variance to understand how many dimensions we need to retain. However first, it is important to scale the data. PCA gives directions in feature space in along which variance of the data exists. Scaling the data before performing PCA is important because, scaling ensures that all the features are of comparable magnitude; otherwise, if one feature has large values, PCA will be biased towards that feature and will show maximum variance of data along that feature.

# In[ ]:


# Scale data before performing PCA - to zero mean and unit variance
X_scaled = preprocessing.scale(X)

# Convert to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
X_scaled.describe()


# In[ ]:


# Initialize PCA transformer
pca = decomposition.PCA()

# Fit it to data
pca.fit(X_scaled)


# In[ ]:


# Plot the cumulative variance
fig = plt.figure(figsize = (10,4))
plt.plot(np.arange(1, X.shape[1]+1),100 * np.cumsum(pca.explained_variance_ratio_), marker = 'o',
         color = 'teal', alpha = .8)
plt.xticks(np.arange(1, X.shape[1]+1),np.arange(1, X.shape[1]+1))
plt.xlabel('Number of Components')
plt.ylabel('Expained Variance %')
plt.grid(linestyle = '--')


# 
# 
# 
# The above cumulative variance graph shows that 10 components end up explaining just a little more than 95% of the data. At this point, we need to decide if we want to retain the 11th, 12th, 13th component. 
# Usually, it is a good idea to keep the number of components that explain just more than 95% of data but another way could be to train a model on the selected features and retain the number of principal components that correspond to the lowest regression/classification error(using k fold crossvalidation). The reason PCA could reduce model prediction error is that sometimes, the components explaining least variance(the last components), are just noise - they are variations that do not matter.  

# In[ ]:


# Transform data
pca = decomposition.PCA().fit(X_scaled)
X_pca = pca.transform(X_scaled)


# So mathematically, each Principal Component is a linear combination of all the features. Let us now see how much each feature contributes to each PCA component.

# In[ ]:


# Save the pca weights in a dataframe
pca_component_directions = pd.DataFrame(pca.components_, columns = X.columns, 
                                        index = np.arange(1, X_pca.shape[1]+1))

# Make a heatmap to show the contribution of each feature to each principal component
fig = plt.figure(figsize = (12, 9))
sns.heatmap(pca_component_directions, linewidth = .2, annot = True, cmap = 'coolwarm',
            vmax = 1, vmin = -1)
plt.ylabel('Components', fontsize = 13)
plt.xlabel('Features', fontsize = 13)


# So, INDUS, NOX and TAX contribute the most to the first principal component. This is good for having a first understanding of the Principal Components. 

# Let us now look at how to visualize the entire data in 2 dimensions. For doing so, we will represent the data using its first 2 principal components and then make a scatter plot of the 2 components. Sometimes, in classification problems, a scatter plot of the first 2 principal components clearly shows the distinct classes. 
# 
# It is also possible in classification problems to colour code the scatter plot to show the data corresponding to each class in a different colour. But first, let us pick the first 2 principal components and plot them.

# In[ ]:


# Extract the first 2 principal components
X_2comps = decomposition.PCA(n_components = 2).fit_transform(X_scaled)

fig = plt.figure(figsize = (10,7))
plt.plot(X_2comps[:,0], X_2comps[:,1], marker = 'o', color = 'teal', alpha = .75, linewidth = 0)
plt.xlabel('Principal Component 1', fontsize = 14)
plt.ylabel('Principal Component 2', fontsize = 14)
plt.grid(linestyle = '--')


# In the beginning we had defined X and y. Let us Binarize y such that low values of y are mapped to 0 and high values to 1. Then, we will colour code the above scatter plot to show which data points correspond to zero and 1.

# In[ ]:


# Find the median value of y.
threshold = y.median()
threshold


# In[ ]:


# Store binary y and X in a dataframe 
plot_2comps = pd.DataFrame(X_2comps, columns = ['PC1', 'PC2'])

# Binarize y: replace y with 0 where y < threshold and with 1 where y>=threshold
y_copy = y.copy()
y_copy[y_copy < threshold] = 0
y_copy[y_copy >= threshold] = 1

plot_2comps['target'] = y_copy


# In[ ]:


# Colour code the 2 Principal Component plot depending on whether target is 1 or 0
X_2comps_0 = plot_2comps[plot_2comps.target == 0][['PC1', 'PC2']].copy()
X_2comps_1 = plot_2comps[plot_2comps.target == 1][['PC1', 'PC2']].copy()

# Teal Data points correspond to target < 21.2 and salmon points correspond to target >=21.2
fig = plt.figure(figsize = (10,7))
plt.plot(X_2comps_0['PC1'], X_2comps_0['PC2'], marker = 'o', color = 'teal', alpha = .75,
         linewidth = 0, label = 'target = 0')
plt.plot(X_2comps_1['PC1'], X_2comps_1['PC2'], marker = 'o', color = 'salmon', alpha = .75,
         linewidth = 0, label = 'target = 1')
plt.xlabel('Principal Component 1', fontsize = 14)
plt.ylabel('Principal Component 2', fontsize = 14)
plt.legend()
plt.grid(linestyle = '--')


# Such plots can be made for multiclass cases also; It is a good way to visualize which data points belong to which class. In the above plot, we can see that most of the high target values are on the left side(salmon colour)  and most of the low target values(teal colour) are on the right. We can see what kind of a decision boundary must be required to classify the data. We must remember, however, that the above figure is only in 2 dimensions where the original data was in 13 dimensions. So, we are missing higher dimensional interactions where a better decision boundary might be possible but it is good to draw the above plot for a visualization of the data. These plots are also good for explaining to someone the nature of the data in terms of the target.

# In[ ]:




