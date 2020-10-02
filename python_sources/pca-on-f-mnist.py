#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
print(os.listdir("../input/fashionmnist"))


# In[ ]:


df = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
df.head()


# In[ ]:


label = df.label.astype(np.int)
df.drop("label", axis=1, inplace=True)
df.shape


# In[ ]:


def show_images(ids, data=df):
    pixels = np.array(data.iloc[ids])
    pixels = pixels.reshape((28,28))
    plt.imshow(pixels, cmap='gray')
    print("label: ", label.iloc[ids])
    


# In[ ]:


show_images(10)


# Standardize features with the help of standard scalar from sklearn library
# It will standardize features and make  mean = 0 and variance = 1.
# ![](http://3.bp.blogspot.com/_xqXlcaQiGRk/RpO4yR0oKtI/AAAAAAAAABM/7rUWCMwizus/s200/fig2.png)

# In[ ]:


from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(df)
standardized_data.shape


# In[ ]:


sample_data = standardized_data
covariance_matrix = np.matmul(sample_data.T, sample_data)
print(covariance_matrix.shape)


# Now we have covariance matrix with shape **dxd** where d is number of features. All we have to do is to find **eigen value** and corresponding **eigen vector ** . Eigen vector with maximum eigen value will be axis with maximum variance. 

# In[ ]:


from scipy.linalg import eigh

values, vectors = eigh(covariance_matrix)
print(vectors.shape)
print("Last 10 eigen values:")
print(values[:][-10:])
print("\nCorresponding vectors:")
print(vectors[-10:])


# We want resultant dimension as 2d, so we will select last two eigen vectors only.

# In[ ]:


values = values[-2:]
vectors = vectors[:,-2:]
vectors = vectors.T
print("Shape of eigen value: ", values.shape)
print("Shape of eigen vectors: ", vectors.shape)


# Multiply eigen vector with original data to get data with reduced dimension.

# In[ ]:


reduced_data = np.matmul(vectors, sample_data.T)
print("Reduced data shape: ", reduced_data.shape)


# In[ ]:


reduced_data = np.vstack((reduced_data, label))
reduced_data = reduced_data.T


# In[ ]:


reduced_df = pd.DataFrame(reduced_data, columns=['X', 'Y', 'label'])
reduced_df.label = reduced_df.label.astype(np.int)
reduced_df.head()


# In[ ]:


import seaborn as sns
reduced_df.dtypes


# In[ ]:


g = sns.FacetGrid(reduced_df, hue='label', size=12).map(plt.scatter, 'X', 'Y').add_legend()


# Converted 2 dimensions are **X** and **Y** and visualization is shown in 2-D.
