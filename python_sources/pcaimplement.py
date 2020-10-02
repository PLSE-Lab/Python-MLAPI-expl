#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/german-credit/german_credit_data.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df.Housing = le.fit_transform(df.Housing)


# In[ ]:


df.head()


# In[ ]:


pd.value_counts(df.Housing)


# In[ ]:


pd.value_counts(df['Saving accounts'])


# In[ ]:


pd.value_counts(df['Checking account'])


# In[ ]:


df.isnull().sum()


# In[ ]:


data = df.copy(deep=True)
data = data.dropna(how='any')


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# In[ ]:


data.Duration.unique()


# In[ ]:


smalldata = data[['Age', 'Credit amount', 'Duration']]
smalldata.head()


# Now we have this small dataset where each rocord is a 3 dimensional vector

# 3d plot of the data

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
fig = plt.figure()
ax = Axes3D(fig)

# data for 3d plot
ax.scatter(smalldata['Age'], smalldata['Credit amount'], smalldata['Duration'])


# In[ ]:


#Standardizing data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
std_data = scaler.fit_transform(smalldata)
std_data


# In[ ]:


mean_vec = np.mean(std_data,axis=0)
cov_mat = (std_data - mean_vec).T.dot((std_data - mean_vec)) / (std_data.shape[0]-1)

#or we can directly use numpy library method
np.cov(std_data.T)


# In[ ]:


#eigen decompostion of the covariance matrix
eig_vals, eig_vectors = np.linalg.eig(cov_mat)

print("Eigen values:", eig_vals)
print("Eigen vectors:", eig_vectors)


# In[ ]:


# correlation matrix
# in most datasets especially financial datasets eigendecomposition on covariance matrix and
# correlation matrix yields the same results since correlation matrix can be
# understood as the normalized covariance matrix
# so here is the eigendecompostion on correlation matrix
corr_mat = np.corrcoef(std_data.T)
eig_vals, eig_vectors = np.linalg.eig(corr_mat)
print("Eigen Values:", eig_vals)
print("Eigen vectors:", eig_vectors)


# In[ ]:


# eigendecomposition of raw data based on correlation matri
corr_mat_raw = np.corrcoef(std_data.T)
eig_vals, eig_vecs = np.linalg.eig(corr_mat_raw)
print('Eigenvals are:', eig_vals)
print('eigenvectors are:', eig_vecs)

here we see that the three approaches give same eigen vectors and eigen values, these approaches are:
- Eigendecompostion of covariance  matrix after standardization of data
- Eigendecomppostion of correlation matrix after standardization of data
- Eigendecompostion of correlation matrix
# # Singular value decompostion

# In[ ]:


# while eigendecompostion of covariance matrix is more intutive, most implementations use singular value decompostion to increase computational efficiency
u,s,v = np.linalg.svd(std_data.T)
u


# # selecting principal components

# In[ ]:


# to see which compinents can be dropped we have to check eigen values
# eigenvectors with least eigen values have the least information about the distribution
# so we first rank the vectors from highest to lowest eigenvalues then select the first k


# In[ ]:


# make a list of (eigenvalue, eigenvectors) tuples
eig_pairs = [(eig_vals[i], eig_vecs[i]) for i in range(len(eig_vals))]
# eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:i]) for i in range(len(eig_vals))]

# print("Eig pairs are\n",eig_pairs )
# sort the eigenvalue, eigenvector tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

for i in eig_pairs:
    print(i[0])


# Estimating the importance of each feature vector using explined variance

# In[ ]:


import matplotlib.pyplot as plt
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

x_coordinates = ('PC1', 'PC2', 'PC3')
y_pos = np.arange(len(x_coordinates))

plt.bar(y_pos, var_exp, align='center')
plt.ylabel('Explained var in %')
plt.xticks(y_pos, x_coordinates)
plt.plot(cum_var_exp, 'r')
plt.show()


# This plot shows that around 58% variance can be explained by first component and around 30% variance is explained by second component. Thus they both cover around 88% of the variance and third component can be dropped without losing too much information.

# In[ ]:


eig_pairs


# In[ ]:


# making the new projection matrix 
pro_mat = np.hstack((eig_pairs[0][1].reshape(3,1),
                   eig_pairs[1][1].reshape(3,1)))

print('Projection Matrix :\n', pro_mat)


# # Projection onto new feature space

# In[ ]:


pro_mat.shape


# In[ ]:


std_data


# In[ ]:


new_mat = std_data.dot(pro_mat)
new_mat


# In[ ]:


X = [new_mat[i][0] for i in range(len(new_mat))]
y = [new_mat[i][1] for i in range(len(new_mat))]
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.scatter(X, y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




