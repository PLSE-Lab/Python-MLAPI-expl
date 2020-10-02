#!/usr/bin/env python
# coding: utf-8

# # PCA Raw data and Normalized Data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

d0 = pd.read_csv('../input/Sales_Transactions_Dataset_Weekly.csv')
# save the labels into a variable l. Creating lables just for visulization purpose after doing PCA
l = d0['Product_Code']
# Drop the label feature and store the sales data in d. Separating the lable data
d = d0.drop("Product_Code",axis=1)
d.drop(d.columns[0:54], axis=1, inplace=True) # this is a normalized data


# In[ ]:


raw=d0[d0.columns[1:53]] # this is a raw data
raw.head() # how our data looks after seperating the lables


# In[ ]:


print(d.shape)
print(l.shape)


# In[ ]:


# initializing the pca
from sklearn import decomposition
pca = decomposition.PCA()


# ### Raw data

# In[ ]:


# configuring the parameteres
# the number of components = 2
pca.n_components = 2
pca_data = pca.fit_transform(raw)

# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)


# In[ ]:


pca.n_components = 52

pca_data_pca = pca.fit_transform(raw)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained,linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()


# a) we can use PCA for the raw data as we can see from the above graph,  the no. of componets are analyzed based on the business goal and how variance is explained for each principle componets. 
# 
# b) I Believe whether to use the data or not depends upon our business goal, yes for example, if we want to analyze the the sales data for the overall product then its is sufficient and reduced dimensions will be 10(approx.), however if want to analyze the sales and provide new product introduction or we want to generate business targets for the following year then we may look for 99% of the variance explained. In the latter case the reduced dimensions will be 36-38 approximately.

# ### Normalized Data

# In[ ]:


# configuring the parameteres
# the number of components = 2
pca.n_components = 2
pca_data = pca.fit_transform(d)

# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)


# In[ ]:


pca.n_components = 52

pca_data_pca = pca.fit_transform(d)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained,linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()


# The above graph is based on the normalized data from the sales data, where the variance for 2 principle components is approximaley 40 % which is quite low and 100% variance is explained only if we consider the data for all the given 52 weeks, its not feasible to consider normalized values rather than raw data. I mean raw data provides effective varaince and we can reduce diminiosns easily.
# 
# In raw data,for example consider 2 variables or components: The approximate variance is 93.2% which is best used for model
# In normalized data, for example consider 2 variables or components: The approximate varaince is 40% which is not good for effective results
# 
# __The best from raw data is  approx 95% variance explaination is captured with 10 variables and reduced dimensions in this case are 42.__
