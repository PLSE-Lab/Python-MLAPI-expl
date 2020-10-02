#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.


# 1. **age** age in years
# 2. **sex**(1 = male; 0 = female)
# 3. **cpchest** pain type
# 4. **trestbps** resting blood pressure (in mm Hg on admission to the hospital)
# 5. **chol** serum cholestoral in mg/dl
# 6. **fbs** (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7. **restecg** resting electrocardiographic results
# 8. **thalach** maximum heart rate achieved
# 9. **exang** exercise induced angina (1 = yes; 0 = no)
# 10. **oldpeak** ST depression induced by exercise relative to rest
# 11. **slope** the slope of the peak exercise ST segment
# 12. **ca** number of major vessels (0-3) colored by flourosopy
# 13. **thal** 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. **target**1 or 0

# In[ ]:


import pandas as pd
df = pd.read_csv('../input/heart-disease-uci/heart.csv', header=None)
df.head()


# In[ ]:


x = df.ix[1:,0:12].values
y = df.ix[1:,13].values


# In[ ]:


x


# In[ ]:


from sklearn.preprocessing import StandardScaler
x_std = StandardScaler().fit_transform(x)


# In[ ]:


x_std


# In[ ]:


x_std.shape[0]


# In[ ]:


import numpy as np
mean_vec = np.mean(x_std, axis=0)
cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec)) / (x_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)


# In[ ]:


cov_mat = np.cov(x_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[ ]:


for ev in eig_vecs.T:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')


# In[ ]:


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# In[ ]:


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[ ]:


cum_var_exp


# In[ ]:


eig_pairs


# In[ ]:


matrix_w = np.hstack((eig_pairs[0][1].reshape(13,1),
                      eig_pairs[1][1].reshape(13,1)))

print('Matrix W:\n', matrix_w)


# In[ ]:


Y = x_std.dot(matrix_w)
Y


# In[ ]:


y


# In[ ]:


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(12, 6))
    for lab, col in zip(('1','0'),
                        ('red', 'green')):
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


from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(x_std)


# In[ ]:


Y_sklearn


# In[ ]:


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('1', '0'),
                        ('red', 'green')):
        plt.scatter(Y_sklearn[y==lab,0],
                    Y_sklearn[y==lab,1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()

