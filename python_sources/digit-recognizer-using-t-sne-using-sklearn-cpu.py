#!/usr/bin/env python
# coding: utf-8

# # Working with MNIST data and Visualize using t-SNE
# 
# ## Background
# MNIST data is most famous dataset available for us which has 60,000 samples of hand-written digits (0 to 9). Out of which, 42,000 in the training set and 28,000 is testing test. 
# The digits are already converted to vectors of 784 data points. Each data point will  be considered as feature in this dataset
# 
# 
# ## Problem statement
# In this kernel I would like to focus on visualization of data points of these samples more than actual classifications. As we all know the task of representation of datapoints with features more than 2 or 3 is not new. It is commom problem to plot the graphs with higher dimentional features. Here we have 784 features. 
# 
# ## Available solutions 
# Best solution to represent the higher dimensions data is to perform Dimensionality reduction exercise. Following are solutions available
# 
# 1. **Feature Eliminations**: Each and every feature is removed and/or added to the solution and check the error results. Which ever feature had least positive impact on results or most negative impact, are eliminated from the set of features. 
# 
# 2. **Factor Analysis**: Among the features, check the correlations between the features themselves. If we find correlated features, we can keep one among them, rest can be removed 
# 
# 3. **Principal Component Analysis PCA**:  Most of common approach is PCA, where, we can project the data points orthogonally on principal component and reduce dimensions
# 
# 4. **T Distributed Stochastic Neighbourhood Embedding (T-SNE)**: This is most advanced and latest technology where data points are embedded on lower dimesion by keeping local topology and not concerning with gloabl configuration. Also this is similar to graph based technic. 
# 
# ## This Notebook
# Let us practice on TSNE techniche in this notebook and try to reduce 784 dimensions of MNIST to 2 dimensions and plot it on 2D graphics

# In[ ]:


from tsnecuda import TSNE as tsnecuda


# In[ ]:


import numpy as np 
import pandas as pd 
from  sklearn.manifold import TSNE as sktsne
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# ## Data Load and Acquaintance
# First of all, let us load the train and test data from Kaggle

# In[ ]:


df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_test = pd.read_csv('../input/digit-recognizer/test.csv')


# Let us try to understand the *train* and *test* datasets

# In[ ]:


print('The shape  of training Dataset: ', df_train.shape, '  The shape of testing Dataset:  ', df_test.shape)


# We can observe that Training Dataset has 42k records and 785 features, Testing Dataset has 28k data records with 784 fetures. We can intuite the additional feature in training dataset is the laballed class. 
# 
# Let us try to understand the data by displaying top few rows. We can see that, first Column, named **label** is the Y variable and rest 784 columns are independent variables, X. That means the first row corresponds to the digit 1 and pixel data is given by *pixel0* to *pixel783*

# Due to computational issues, taking only top 10000 digits

# In[ ]:


df_train = df_train.head(10000)


# Those X vaiables are nothing but the pixel data of handwritten characters of digits 0 to 9. 0 means pixel is OFF and 1 means pixel is ON. 
# 
# Let us now split the train data into label(Y) and pixels(X)
# 

# In[ ]:


Y = df_train[['label']]
X = df_train.drop('label', axis=1)


# Now let us try to plot the character from the pixel data provided. For reusability purpose, let us define a method

# In[ ]:


def plot_digit(digits):
    fig, axs = plt.subplots(1,len(digits),figsize=(2,2))
    for i, pixels in enumerate(digits):
        ax = axs[i]
        digit_data = pixels.values.reshape(28,28)
        ax.imshow(digit_data,interpolation=None, cmap='gray')
    plt.show()


# Plotting 3 digits for example in training set

# In[ ]:


plot_digit([X.iloc[0], X.iloc[20], X.iloc[201]])


# Before further anlaysis, let us preprocess the training data by using *StandardScaler*
# 
# *Update*: When I ran TSNE with and without scaller, I found without scaller the results was better

# In[ ]:


scaled_X = pd.DataFrame(StandardScaler().fit_transform(X))
scaled_X.head()


# Next step is to build the t-SNE model with following hyperparameter
# 
# Perplexity = 30 <br>
# No. of Iterations = 1000
# 

# In[ ]:


tsne_data = sktsne(n_components=2,random_state=42, perplexity=30.0, n_iter=1000).fit_transform(X)
# !export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:
# tsne_model = tsnecuda(n_components=2, perplexity=30.0).fit_transform(X)
# tsne_data = tsne_model.fit_transform(scaled_X)


# Once the model is ready and fit the data, it has produced the new dataset of only 2 dimesions

# Adding Label column for graphical representation in the next step

# In[ ]:


tsne_df = pd.DataFrame(tsne_data)
tsne_df = pd.concat([tsne_df,Y], axis=1)
# tsne_df


# Plotting the result of TSNE 

# In[ ]:


sns.FacetGrid(tsne_df, hue="label" , size=6).map(plt.scatter, 0, 1).add_legend()
plt.show()

