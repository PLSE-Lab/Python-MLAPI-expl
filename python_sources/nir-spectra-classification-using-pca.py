#!/usr/bin/env python
# coding: utf-8

# # Classificaiton of NIR spectra using Principal Component Analysis in Python
# PCA is nearly invariably used in the analysis of NIR data, and for a very good reason. Typical NIR spectra are acquired at many wavelengths. With our Brimrose Luminar 5030, we typically acquire 601 wavelength points with an interval of 2 nm.
# 
# 601 data points for each spectra are, in general, a very redundant set. NIR spectra are fairly information-poor, that is they never contains sharp features, such as absorption peaks. as it may be the case for Raman or MIR spectroscopy. For this reason, most of the features of NIR spectra at different wavelengths are highly correlated.
# 
# That is where PCA comes into play. PCA is very efficient at performing dimensionality reduction in multidimensional data which display a high level of correlation. PCA will get rid of correlated components in teh data by projecting the multidimensional data set (60 dimensions for our spectrometer) to a much lower dimensionality space, often a few, or even just two, dimensions.
# 
# These basic operations on the data are typically done using chemometric software, and there are many excellent choices on the market. If however you would ike to have some additional flexibility, for instance try out a few supervised or unsupervised learning techniques of your data, you might want to code the data analysis from scratch.
# 
# ## Reading and preparing the data for PCA analysis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA as sk_pca
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cluster import KMeans


# We are going to use NIR spectra from milk samples with a varying concentration of lactose. We prepared the samples by mixing . normal milk and lactose-free milk in different concentrations. The aim is to find out whether PCA applied to the spectrae can be useful to discriminate between the different concentrations.
# 
# The data is in a csv file ocntaining 450 scans. Each scan is identified with a label (from 1 to 9) identifying the sample that was scanned. Each sample was scanned 50 times and the difference between samples was in the relatvie content of milk and lactose-free milk.

# In[ ]:


# read the data
data = pd.read_csv("../input/milk.csv")


# In[ ]:


data.head()


# In[ ]:


# the first column is the labels
lab = data.values[:, 1].astype('uint8')
# read the features (scans) and transform data from relfectance to absorbance
feat = np.log(1.0/(data.values[:, 2:]).astype('float32'))


# In[ ]:


# Calcualte first derivative applying a Savitzky-Golay filter
dfeat = savgol_filter(feat, 25, polyorder = 5, deriv = 1)


# In[ ]:


dfeat


# In[ ]:


plt.plot(feat[0], label='feature')
plt.plot(dfeat[0], label='derrivative')
plt.legend()


# The last step is very important when analysing NIR spectroscopy data. Taking the first derivative of the data enables to correct for baseline differences in the scans, and highlight the major sources of variation between the different scans. Numerical derivatives are generally unstable, so we use the smoothing filter impleented in scipy.signal import [savgol_filter](https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.savgol_filter.html) to smooth the derivative data out.
# 
# **Why do we smooth? The reason is the wavelengths are in increasing oreder + the wavelength from one to the next are very much similar => so there shouldn't be sudden changes, if there is, then it woudl be due to noise**
# 
# **This link explained well about the [Savitzky-Golay filtering (smoothing)](https://www.youtube.com/watch?v=0TSvo2hOKo0)**
# 
# **Why do we use derrivatives instead of the actual values? The underlying changes in wavelengths (increasing order) is smooth => thus its reflectances to these wavelengths will increase/decrease smoothly too and this would not be helpful much (in this case if we use derrivative the derrivative will remained the same - though the reflectance value is increasing/decreasing smoothly - So, there if changes to the wavelengths are the same => then the derrivatives will be the same => meaning no differneces or no information gain, since that is what expected)**
# 
# 
# ## Picking the right number of principal components in PCA of NIR spectroscopy data
# As explained before, we are going to use PCA to reduce the dimensionality of our data, that is reduce the number of features from 601 (wavelengths0 to a much smaller number.
# 
# So here is the first big question: **How many featurs do we need?** Let's work through the answer to this question, as it may seem a bit at odd with what is usually done in some statistical analysis.
# 
# Each principal component explains some of the variation in the data. The first principal component will explain most of the variation, the second a little bit less, and so on. To get a bit more technical, we can talk of variance explained by each principal component. The sum of variances of all principal compoennts is called total variance.
# 
# The general advice is to choose the first n principal components that account for the large majority of the variance, typically 95% or 90% (or even 85%) depending on the problem. This approach is not always very insightful with NIR spectroscopy data, especially when dealing with derivative data. Let's spend some lines looking at this issue.
# 
# The previous sentence may sound a bit cryptic. To explain its meaning, here's the plot of the explained variance and the cumulative variance of the first 10 principal components extracted from our data.
# 
# We may applied these explaination to the raw data and the first derivative data.
# 
# When applied to the original data, we see that 3 principal components account for most of the variance in the data, 98.8% to be precise. The contribution of the subsequent principal components is negligible. The situlation is very different however when we run PCA on the first derivative data. Here is seems that each principal component adds a sizeable bit to the variance explained, and in fact it seems that many more principal components would be required to account for most of the variance.
# 
# So, what is going on here?
# 
# We believe that the culprit is the noise present in the data. Noise is amplified in the numerical derivative operation, and that is why we need to use smoothing filter. The filter however does not get rid of the noise completely. Risidual noise is uncorrelated form scan to scan, and it can't therefore be accounted for using only a small number of principal components.
# 
# The good news is hwoever that the most important variations in the data are in fact still described by a handful (3-4) principal components. The higher order principal components, found when decomposing first derivative data, and account mostly for noise in the data. In fact, see hwo the explained variance of the fist derrivaiteve data flats out after 4 principal components. After that, there is negligible informaiton in the first derivative data, just random noise variations.
# 
# So, a good rule of thumb is to choose the number of principal components by looking at the cumulative variance of the decomposed spectral data, even though we will be generally using the first derivative data for more accurate classification.

# In[ ]:


# Initialise
skpca1 = sk_pca(n_components = 10)
skpca2 = sk_pca(n_components = 10)

# Scale the features to have zero mean and standard deviation of 1
# This is important when correlating darta with very different variances
nfeat1 = StandardScaler().fit_transform(feat)
nfeat2 = StandardScaler().fit_transform(dfeat)


# In[ ]:


plt.plot(nfeat1[0], label='feature')
plt.plot(nfeat2[0], label='derrivative')
plt.legend()


# In[ ]:


# Fit the spectral data and extract the explained variance ratio
X1 = skpca1.fit(nfeat1)
expl_var_1 = X1.explained_variance_ratio_


# In[ ]:


print(expl_var_1)


# In[ ]:


# Fit the first derrivative data and extract the explained variance ratio
X2 = skpca2.fit(nfeat2)
expl_var_2 = X2.explained_variance_ratio_


# In[ ]:


print(expl_var_2)


# In[ ]:


# Plot data
with plt.style.context('ggplot'):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))
    fig.set_tight_layout(True)
    
    ax1.plot(expl_var_1, '-o', label = "Explained Variance %")
    ax1.plot(np.cumsum(expl_var_1),'-o', label='Cumulative variance %')
    ax1.set_xlabel("PC number")
    ax1.set_title("Absobance data")
    
    ax2.plot(expl_var_2, '-o', label="Explained Variance %")
    ax2.plot(np.cumsum(expl_var_2), '-o', label='Cumulative variance %')
    ax2.set_xlabel("PC number")
    ax2.set_title("First derivative data")
    
    plt.legend()
    plt.show()


# ## Running the Classification of NIR spectra using PCA in Python
# OK, now is the easy part. Once we established the number of principal components to use - let's say we go for 4 principal components - is just a matter of defining new transform and running the fit on the first derrivative data.
# 

# In[ ]:


skpca2 = sk_pca(n_components=4)

# Transform on the scaled features
Xt2 = skpca2.fit_transform(nfeat2)


# Finally we display the score plot of the first two principal components.

# In[ ]:


# Define the labels for the plot legend
labplot = [f'{i}/8 Milk' for i in range(9)]
# Scatter plot
unique = list(set(lab))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
with plt.style.context('ggplot'):
    for i, u in enumerate(unique):
        col = np.expand_dims(np.array(colors[i]), axis=0)
        xi = [Xt2[j, 0] for j in range(len(Xt2[:, 0])) if lab[j] == u]
        yi = [Xt2[j, 1] for j in range(len(Xt2[:, 1])) if lab[j] == u]
        plt.scatter(xi, yi, c=col, s = 60, edgecolors='k', label=str(u))
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(labplot, loc='upper right')
    plt.title('Principal Component Analysis')
    plt.show()


# In[ ]:




