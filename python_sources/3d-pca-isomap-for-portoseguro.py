#!/usr/bin/env python
# coding: utf-8

# I'd like to get a feeling for the data by running a dimensionality reduction on the data and plot it in 3D. I'll try principal component analysis and Isomap for dimensionality reduction.
# 
# Let's read in the data and print the column names.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[ ]:


X = pd.read_csv("../input/train.csv")
y = X["target"]
X.drop(["target", "id"], axis = 1, inplace = True)
X.columns.values


# As stated in the description of the data, the suffix "_cat" indicates categorical variables and the suffix "_bin" indicates binary variables. I'll throw out all categorical variables for this analysis. Here's why:
# 
# Let's say one variable would represent the parking opportunities for the car:
# + "0" = garage,
# + "1" = carport,
# + "2" = parking outside on your property,
# + "3" = parking on the street
# 
# PCA relies on distances and it is hard to tell what the distances between different categories should be. The difference between all numerical values is 1 but does the risk increase by the same amount from garage to carport as from carport to parking outside on your property? 
# 
# So to avoid this problem, we just ignore categorical variables for this analysis.
# 

# In[ ]:


def filter_cat(df):
    for x in df.columns.values:
        if x[-3:] == "cat":
            df.drop([x], axis = 1, inplace = True)
    return df

X_filt = filter_cat(X)


# Scaling the data is another thing we have to do because of PCA relying on distances. If you don't scale your features, those with a bigger magnitude will dominate the principal components.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

X_filt_scale = X_filt.copy()

for x in X_filt_scale.columns.values:
    if not x[-3:] == "bin":
        X_filt_scale[x] = scaler.fit_transform(X_filt[x].values.reshape(-1,1))


# As we see from the plots below the scaling adjusted every feature to range from 0 to 1.

# In[ ]:


fig, axs = plt.subplots(1,2, figsize=(10, 5))

plot_data = pd.DataFrame(X_filt.max()-X_filt.min(), columns = ["magnitude"]).sort_values(by = "magnitude")
plot_data.plot.area(ax=axs[0], title = "Before Scaling", use_index = False, colormap = "Blues_r")

plot_data_2 = pd.DataFrame(X_filt_scale.max()-X_filt_scale.min(), columns = ["magnitude"]).sort_values(by = "magnitude")
plot_data_2.plot.area(ax=axs[1], title = "After Scaling", use_index = False, colormap = "Blues_r")


# Now it's time to perform the PCA.

# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components = X_filt_scale.shape[1])
X_PCA = pca.fit_transform(X_filt_scale)
pca.explained_variance_ratio_.cumsum()[0:3]


# As we can see, the first 3 components explain 91 % of the variance, which is not bad.
# Edit: Oh - after following the hint from HDKIM, we see that the first 3 components explain only 27 % of the variance. So the 3D scatter plot is not that meaningful.
# 
# So now we can plot this as a 3D scatter plot. 
# 
# First I define a color for every data point. All samples with label "0" should be displayed in transparent purple and all samples with label "1" should be displayed in semi-transparent green. I use RGBA-coding for this. So the first three columns of my_color represent the amount of red, green and blue and the fourth column represents transparency. The colors look like this:
# 
# * transparent purple = (0.5, 0, 0.5, 0.05)
# * non-transparent yellow = (0, 0.5, 0, 0.6)
# 
# You can play around with the transparency by changing the values 0.8 and 0.05 in "my_color.iloc[:,3]".
# 
# If you know a way to build the dataframe "my_color" faster, please let me know in the comments.
# 
# The figure takes around one minute to load due to the amount of data points. 

# In[ ]:


def plot_pca(X, y, opacity_0, opacity_1):
    my_color = pd.DataFrame(np.zeros((len(y), 4)))
    my_color.iloc[:,0] = (1-y)*0.5
    my_color.iloc[:,1] = y*0.5
    my_color.iloc[:,2] = (1-y)*0.5
    my_color.iloc[:,3] = y*opacity_0 + (1-y)*opacity_1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=my_color)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('PCA of Data from PortoSeguro')

plot_pca(X_PCA, y, 0.6, 0.05)


# We see some nice clusters here and it seems clusters with high PC1-values have more samples with insurance claims (label 1, green) then clusters with low PC1-values.
# 
# Next we try Isomap. Isomap is a nonlinear dimensionality reduction method. The algorithm computes the nearest neighbors for every sample and builds a neighborhood-graph. So we should also use our scaled data, since Isomap relies on distances. In order to derive the lower-dimensional representation of the data, the algorithm calculates the shortest path between every pair of nodes in the neighborhood-graph. It then tries to find a lower-dimensional representation of the data that preserves these distances as good as possible.
# 
# Since Isomap is computationally heavy, we have to sample a subset of our data. My local maschine crashed when trying to apply Isomap on more than 7,000 samples so I'll use this as a limit to my subset. If you have more ressources at hand feel free to increase the number of samples in the subset.

# In[ ]:


from sklearn.model_selection import train_test_split

X_iso_train, forget1, y_iso_train, forget2 = train_test_split(X_filt_scale, y, train_size=7000, random_state=4)
y_iso_train.reset_index(drop= True, inplace = True)

print("size of dataset before the split: " + str(len(X_filt_scale)))
print("size of dataset after the split: " + str(len(X_iso_train)))


# I tried to find a seed for train_test_split that gives us a subset with a similar looking PCA figure in comparison to the full set. This is what PCA looks like on the subset.

# In[ ]:


X_PCA = pca.fit_transform(X_iso_train)
plot_pca(X_PCA, y_iso_train, 1, 0.5)


# So the subset seems to be a fair approximation of the whole data set at least in forms of the resulting principal components. Lets see how the embedding would look like with Isomap..

# In[ ]:


from sklearn.manifold import Isomap

iso = Isomap(n_neighbors = 5, n_components = 3)
X_iso = iso.fit_transform(X_iso_train)
plot_pca(X_iso, y_iso_train, 1, 0.5)


# The result of Isomap is not as clustered as the result of PCA. But even with a non-linear embedding of the data, the samples with insurance claims (green) are evenly distributed within the data set. So my interpretation of the result would be that seperating both classes should prove quite difficult.
