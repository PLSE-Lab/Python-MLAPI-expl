#!/usr/bin/env python
# coding: utf-8

# I am going to try to use a dimensionality reduction technique in combination with a classifier such as Random Forests to generate an output.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

# Import the 3 dimensionality reduction methods
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Load the MNIST data set and check visualization

# In[ ]:


train = pd.read_csv('../input/train.csv')
print(train.shape)


# The MNIST set has of 42,000 rows and 785 columns. There are 784 (28*28) columns corresponding to the pixels and 1 column that holds the "label" or the number that that image contains.

# Since we want to do dimensionality reduction on the pixels and remove the labels column, let us clean the data like so:

# In[ ]:


# save the labels to a Pandas series target
target = train['label']
# Drop the label feature
train = train.drop("label",axis=1)


# I'm going to try to use two dimensionality reduction techniques and compare their performance with a classifier. The two I've chosen for this problem are PCA and t-SNE. First, let us try PCA.
# # Note: t-SNE cannot be used in this fashion. See below for details.

# In[ ]:


# Standardize the data
from sklearn.preprocessing import StandardScaler
X = train.values
X_std = StandardScaler().fit_transform(X)


# In[ ]:


# Calculating Eigenvectors and eigenvalues of Cov matirx
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance


# In[ ]:


# Find the eigenvector beyond which 90% of the data is explained
[ n for n,i in enumerate(cum_var_exp) if i>90 ][0]


# So, we need 228 eigenvectors to explain 90% of the variance. So, let us reshape our training data matrix into a 42000*228 matrix with each column corresponding to the projections onto the eigenvectors.

# PCA Implementation via Sklearn
# 
# Now using the Sklearn toolkit, we implement the Principal Component Analysis algorithm as follows:

# In[ ]:


# Call the PCA method with 228 components. 
pca = PCA(n_components=228)
pca.fit(X_std)
X_228d = pca.transform(X_std)


# Let us do a sanity check to ensure that our data matrix is the size we were expecting:

# In[ ]:


print(X_228d.shape)


# Great! So now we have our dimensionality reduced training data set. Let us use this data to train our classifier of choice, in this case, a Random Forest.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier as RF
# Use 25 decision trees in our random forest and initialize
clf = RF(n_estimators = 500)

# Train the classifier
clf = clf.fit(X_228d,target)


# Ok, so we now have our trained classifier. So we load our test data set and obtain predictions. Note however that the test data first has to be converted to the same format as the train data was when we built the classifier - hence we use the same PCA axes and get a 228 dimensional data set first.

# In[ ]:


# read test data from CSV file 
test_images = pd.read_csv('../input/test.csv')

test_values = test_images.values
test_std = StandardScaler().fit_transform(test_values)
test_228d = pca.transform(test_std)

output_predictions = clf.predict(test_228d)


# Save the output file in a CSV format for submission as shown below:

# In[ ]:


np.savetxt('submission_rf_500.csv', 
           np.c_[range(1,len(test_images)+1),output_predictions], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')


# #2. T-SNE ( t-Distributed Stochastic Neighbour Embedding )
# 
# #Note: Simply using t-SNE makes the kernel crash instantly. 
# 
# And thereby lies t-SNE's greatest flaw - it is computationally heavy... But more importantly, it is not possible to "add" a new data point to computed t-SNE axes. If you have a new data point you want to plot in a t-SNE representation, you need to recompute the entire thing. This is due to how the "stochastic neighborhood embedding" works. The t-SNE representation has no meaning outside the data set for which it was computed. Since we cannot then do this for the test data set, this does not work. I'm leaving the code intact in the cells below in case you would like to run it.

# # Invoking the t-SNE method
# tsne = TSNE()
# X_37d_reduced = X_37d[:2000]
# tsne_results = tsne.fit_transform(X_37d_reduced) 

# Now, we train the RF classifier using the results obtained from t-SNE.

# from sklearn.ensemble import RandomForestClassifier as RF
# # Use 10 decision trees in our random forest and initialize
# clf = RF(n_estimators = 10, max_depth = 2)
# 
# train_reduced = target[:2000]
# # Train the classifier
# clf = clf.fit(tsne_results,train_reduced)

# # read test data from CSV file 
# test_images = pd.read_csv('../input/test.csv')
# tsne2_params = tsne.get_params()
# tsne2 = TSNE()
# tsne2.set_params = tsne2_params
# 
# 

# test_values = test_images.values
# test_std = StandardScaler().fit_transform(test_values)
# tsne_test = tsne2.fit_transform(test_std)
# 
# output_predictions = clf.predict(tsne_test)

# np.savetxt('submission_rf.csv', 
#            np.c_[range(1,len(test_images)+1),output_predictions], 
#            delimiter=',', 
#            header = 'ImageId,Label', 
#            comments = '', 
#            fmt='%d')
