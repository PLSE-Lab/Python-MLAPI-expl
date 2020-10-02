#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/HR_comma_sep.csv')


# In[ ]:


df.head()


# In[ ]:


df_drop=df.drop(labels=['sales','salary'],axis=1)
df_drop.head()


# In[ ]:


cols = df_drop.columns.tolist()
cols


# In[ ]:


cols.insert(0, cols.pop(cols.index('left')))


# In[ ]:


cols


# In[ ]:


df_drop = df_drop.reindex(columns= cols)


# In[ ]:


X = df_drop.iloc[:,1:8].values
y = df_drop.iloc[:,0].values
X


# In[ ]:


y


# In[ ]:


np.shape(X)


# In[ ]:


np.shape(y)


# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
import math

label_dict = {1: 'high',
              2: 'low',
              3: 'medium'}

feature_dict = {0: 'satisfaction_level',
1: 'last_evaluation',
2: 'number_project',
3: 'average_montly_hours',
4: 'time_spend_company',
5: 'Work_accident',
6: 'left',
7: 'promotion_last_5years'}


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(8, 6))
    for cnt in range(8):
        plt.subplot(4, 4, cnt+1)
        for lab in ('high', 'low', 'medium'):
            plt.hist(X[y==lab, cnt],
                     label=lab,
                     bins=10,
                     alpha=0.9,)
        plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)

    plt.tight_layout()
    plt.show()


# In[ ]:


from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


# In[ ]:


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


# In[ ]:


print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))


# In[ ]:


plt.figure(figsize=(8,8))
sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different features')


# In[ ]:


eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[ ]:


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i)


# In[ ]:


u,s,v = np.linalg.svd(X_std.T)
u


# In[ ]:


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]


# In[ ]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(7), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[ ]:


matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1), 
                      eig_pairs[1][1].reshape(7,1)
                    ))
print('Matrix W:\n', matrix_w)


# In[ ]:


Y = X_std.dot(matrix_w)
Y


# In[ ]:


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(12, 10))
    for lab, col in zip(('satisfaction_level',
 'last_evaluation',
 'number_project',
 'average_montly_hours',
 'time_spend_company',
 'Work_accident',
 'left',
 'promotion_last_5years'),
                        ('blue', 'red','green','blue', 'red','green','blue', 'red')):
        plt.scatter(Y[y==lab, 0],
                    Y[y==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# In[ ]:


from sklearn.decomposition import PCA 
sklearn_pca = PCA(n_components=6)
Y_sklearn = sklearn_pca.fit_transform(X_std)


# In[ ]:


print(Y_sklearn)


# In[ ]:


print(Y_sklearn)


# In[ ]:


Y_sklearn.shape

