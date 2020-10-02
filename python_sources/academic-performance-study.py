#!/usr/bin/env python
# coding: utf-8

# # Imlementing PCA without sckikit learn library function
# 
# Iplementing PCA on a student academic dataset [xAPI-Educational Mining Dataset](https://www.kaggle.com/aljarah/xAPI-Edu-Data)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 1. First importing the data

# In[ ]:


filepath = '../input/xAPI-Edu-Data/xAPI-Edu-Data.csv'
data = pd.read_csv(filepath)


# Let's see what does our data comprise of by selecting the first 5 rows

# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


dropped_data = data.drop(columns=['GradeID', 'SectionID', 'PlaceofBirth', 'Relation'])


# In[ ]:


for col in dropped_data.columns:
    print(col, '--')
    print(dropped_data[col].value_counts())
    print('---')


# In[ ]:


## Needed numerical Values : 
num_attributes = ['VisITedResources','AnnouncementsView', 'Discussion', 'raisedhands']

# If you want to visualize/work with 3 attributes uncomment the lines below 
# num_attributes = ['VisITedResources', 'Discussion', 'raisedhands']

num_data = dropped_data[num_attributes]
num_data.head()
print(num_data.info())


# # 2. Visualising the attributes
# Plotting 3 of the attributes in a 3D scatter plot

# In[ ]:


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.scatter3D(num_data['VisITedResources'], num_data['Discussion'], num_data['raisedhands'])

plt.show()


# # 3. Scaling the numerical attributes
# The values in the data need to be scaled around zero mean and unit variance

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaled_num_data = StandardScaler().fit_transform(num_data)                                                                        


# In[ ]:


mean_data = np.mean(scaled_num_data, axis=0)


# # 4. Computing the co-variance matrix 
# The formula for computing the covariance matrix :
# ![](https://education.howthemarketworks.com/wp-content/uploads/2013/09/Covariance.jpg)

# In[ ]:


# Computing the covariance matrix
cov_matrix = (scaled_num_data - mean_data).T.dot(scaled_num_data - mean_data) / (scaled_num_data.shape[0]-1)


# In[ ]:


# Computing eigen values and vectors 

eig_values, eig_vectors = np.linalg.eig(cov_matrix)

print('Eigen values: --', eig_values)
print('Eigen vectors: --', eig_vectors)


# In[ ]:


eig_pairs = [(eig_values[i], eig_vectors[:, i]) for i in range(len(eig_values))]
eig_pairs.sort(key = lambda x:x[0], reverse=True)
eig_pairs


# # 5. Estimating the importance of each feature vector using explined variance.
# 

# In[ ]:


total = sum(eig_values)
var_spread = [(i/total)*100 for i in sorted(eig_values, reverse=True)]
cum_var_spread = np.cumsum(var_spread)
print(cum_var_spread)

# x_coordinates = ['PC1', 'PC2', 'PC3'] for working with 3 attributes
x_coordinates = ['PC1', 'PC2', 'PC3', 'PC4']

y_pos = np.arange(len(x_coordinates))
plt.ylabel('Variance spread in %')
plt.xticks(y_pos, x_coordinates)
plt.bar(y_pos, var_spread)
plt.plot(cum_var_spread, 'r')
plt.show()


# This plot shows that around 63% variance can be explained by first component, around 21% variance is explained by second component and 9% variance by the third component. Thus the three cover around 92% of the variance and fourth component can be dropped without losing too much information. It's a good tradeoff considering we are reducing 25% of our computations. 

# In[ ]:


threshold = 90

# If you're working with three attributes
# threshold = 88 
 
keeping_vec = 0
for index, percentage in enumerate(cum_var_spread):
    if percentage > threshold:
        keeping_vec = index +1 
        break

print("We keep ", keeping_vec, " vectors")


# In[ ]:


num_features = scaled_num_data.shape[1]

projection_mat = eig_pairs[0][1].reshape(num_features, 1)


# # 6. Projection Matrix

# In[ ]:


for eig_vec_idx in range(1, keeping_vec):
    projection_mat = np.hstack((projection_mat, eig_pairs[eig_vec_idx][1].reshape(num_features, 1)))
    


# In[ ]:


PCA_data = scaled_num_data.dot(projection_mat)

PCA_data


# In[ ]:


plt.scatter(PCA_data[:, 0], PCA_data[:, 1])


# PCA reduces the 4 attribute data to 3 attributes
