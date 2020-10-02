#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd

df = pd.read_csv('../input/glass.csv', header=None, sep=',')

df.columns=['RI', 'Na', 'Mg', 'Al', 'Si','K', 'Ca', 'Ba', 'Fe', 'Type']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()


# In[73]:


X = df.ix[1:,0:9].values
y = df.ix[1:,9].values


# In[74]:


from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


# In[75]:


import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


# In[76]:


print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))


# In[77]:


cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[79]:


u,s,v = np.linalg.svd(X_std.T)
u


# In[80]:


for ev in eig_vecs.T:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')


# In[81]:


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[82]:


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[84]:


matrix_w = np.hstack((eig_pairs[0][1].reshape(9,1),
                      eig_pairs[1][1].reshape(9,1)))

print('Matrix W:\n', matrix_w)


# In[85]:


Y = X_std.dot(matrix_w)


# In[86]:


Y


# In[92]:


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6', 'Type 7'),
                        ('blue', 'red', 'green', 'yellow', 'black', 'violet','grey')):
        plt.scatter(Y[y==lab, 0],
                    Y[y==lab, 0],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# In[93]:


from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)


# In[95]:


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6', 'Type 7'),
                        ('blue', 'red', 'green', 'yellow', 'black', 'violet','grey')):
        plt.scatter(Y_sklearn[y==lab, 0],
                    Y_sklearn[y==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()


# In[ ]:




