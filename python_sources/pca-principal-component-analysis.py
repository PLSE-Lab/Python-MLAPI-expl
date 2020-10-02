#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Let us import the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # visualize the co-relation using this library
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler


# In[2]:


# Let us read the data from the file and see the first five rows of the data
data = pd.read_csv("../input/DS_BEZDEKIRIS_STD.data", header = None)
data.head()


# #### df.corr()
# will display the co-relation between features.
# +ve value denotes the positive relation ( if we increase value of one feature, other feature value also increases).
# -ve value denotes the negative relation ( if we increase value of one feature, other feature value also decreases).

# In[3]:


data.corr()


# In[4]:


correlation = data.corr()
plt.figure(figsize=(4,4))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different fearures')


# Seperating the features and classes from the dataset.

# In[5]:


X = data.iloc[:,0:4].values
y = data.iloc[:,-1].values
X


# In[6]:


y


# In[7]:


np.shape(X)


# In[8]:


np.shape(y)


# Standardize features by removing the mean and scaling to unit variance
# 
# Centering and scaling happen independently on each feature by computing
# the relevant statistics on the samples in the training set. Mean and
# standard deviation are then stored to be used on later data using the
# `transform` method.

# In[9]:


X_std = StandardScaler().fit_transform(X)


# In[10]:


X_std.shape


# In[11]:


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


# In[12]:


print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))


# In[13]:


plt.figure(figsize=(4,4))
sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different features')


# In[14]:


eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[15]:


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[16]:


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]


# In[17]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(4), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# The first two components are giving 90% of information. Hence choosing the top two vectors. i.e. `Feature Vector`

# In[18]:


feature_vector = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)
                    ))
print('Matrix W:\n', feature_vector)


# `Final data = Row feature vector * Row data adjust`
# 
# `Row feature vector` is the matrix with the eigenvectors in the columns transposed so that the eigenvectors are now in the rows, with the most significant eigenvector at the top. i.e. `Feature Vector`
# 
# `Row data adjust` is the mean adjusted data. i.e. `X_std`.

# In[19]:


final_data = X_std.dot(feature_vector)


# In[20]:


final_data


# ## PCA in scikit-learn

# In[21]:


from sklearn.decomposition import PCA
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,4,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# In[22]:


from sklearn.decomposition import PCA 
sklearn_pca = PCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)


# In[23]:


Y_sklearn


# References: 
#     http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
#     https://www.kaggle.com/nirajvermafcb/principal-component-analysis-explained
