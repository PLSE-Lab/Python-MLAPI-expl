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
# 
# ### Higher Dimension
# In this kernel I would like to focus on visualization of data points of these samples more than actual classifications. As we all know the task of representation of datapoints with features more than 2 or 3 is not new. It is commom problem to plot the graphs with higher dimentional features. Here we have 784 features. 
# 
# ### Expensive TSNE Computation
# Whoever working with SKLearn might have already figured out, TSNE computation is very expensive. For example, fitting 10,000 MNIST data with perplexity=30 and 1000 iteration would take around 3/4 min to complete. 
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
# ### GPU Solution
# To tackle the problem of slow TSNE computation, we need to find a library which uses GPU effectively. One of the best available library which uses GPU for tsne computation is  ```tsnecuda``` https://github.com/CannyLab/tsne-cuda. However, Kaggle does not comes with tsnecuda and other pre-requisites installed in GPU enabled kernels. This notebook helps to install required libraries and pre-prequisites for running ```tsnecuda```
# 
# 
# ## Basic checks
# Let us practice on TSNE techniche in this notebook and try to reduce 784 dimensions of MNIST to 2 dimensions and plot it on 2D graphics. Also let us perform those tsne computations using GPU using ```tsnecuda``` library

# First of all, let us see if GPU is present

# In[ ]:


get_ipython().system('nvidia-smi')


# Next check what is the version of cuda in this GPU 

# In[ ]:


get_ipython().system('cat /usr/local/cuda/version.txt')


# ## Installation of Pre-requisites
# 
# 1. One of the pre-requisite library for tsnecuda to run is ```faiss```. Let us isntall the same for the version of *cuda*

# In[ ]:


## Passing Y as input while conda asks for confirmation, we use yes command
get_ipython().system('yes Y | conda install faiss-gpu cudatoolkit=10.0 -c pytorch')


# 2. Now let us install *tsnecuda* from the sources
# Also we found *libfaiss.so* was not found while running, but this file comes as part of sources. So, we move that to ```/usr/local/cuda/lib64```

# In[ ]:


# !wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2
# !tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2
# !cp -r site-packages/* /opt/conda/lib/python3.6/site-packages/
# # !export LD_LIBRARY_PATH="/kaggle/working/lib/" 
# !cp /kaggle/working/lib/libfaiss.so /usr/local/cuda/lib64/


# In[ ]:


get_ipython().system('wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2')
get_ipython().system("tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'lib/*'")
get_ipython().system("tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'site-packages/*'")
get_ipython().system('cp -r site-packages/* /opt/conda/lib/python3.6/site-packages/')
# !export LD_LIBRARY_PATH="/kaggle/working/lib/" 
get_ipython().system('cp /kaggle/working/lib/libfaiss.so /usr/local/cuda/lib64/')


# 3. We found another library missing, *openblas*, we install that now.

# In[ ]:


get_ipython().system('apt search openblas')
get_ipython().system('yes Y | apt install libopenblas-dev ')
# !printf '%s\n' 0 | update-alternatives --config libblas.so.3 << 0
# !apt-get install libopenblas-dev 
get_ipython().system('')


# ## Start the main objective of notebook

# In[ ]:


import faiss
from tsnecuda import TSNE
import pandas as pd
import numpy as np
from  sklearn.manifold import TSNE as sktsne
import matplotlib.pyplot as plt
import seaborn as sns


# ## Data Load and Acquaintance
# First of all, let us load the train and test data from Kaggle
# 
# We can observe that Training Dataset has 42k records and 785 features, Testing Dataset has 28k data records with 784 fetures. We can intuite the additional feature in training dataset is the laballed class. 
# 
# Let us try to understand the data by displaying top few rows. We can see that, first Column, named **label** is the Y variable and rest 784 columns are independent variables, X. That means the first row corresponds to the digit 1 and pixel data is given by *pixel0* to *pixel783*

# In[ ]:


df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_train = df_train.head(10000)
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


# Next step is to build the t-SNE model with following hyperparameter
# 
# Perplexity = 50 <br>
# No. of Iterations = 1000

# In[ ]:


tsne_model = TSNE(n_components=2, perplexity=40.0, n_iter=2000).fit_transform(X)


# Once the model is ready and fit the data, it has produced the new dataset of only 2 dimesions.
# Let us Label column for graphical representation in the next step

# In[ ]:


tsne_df = pd.DataFrame(tsne_model)
tsne_df = pd.concat([tsne_df,Y], axis=1)


# Plotting the result of TSNE 

# In[ ]:


sns.FacetGrid(tsne_df, hue="label" , size=6).map(plt.scatter, 0, 1).add_legend()
plt.show()

