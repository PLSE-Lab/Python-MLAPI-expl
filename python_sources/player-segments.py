#!/usr/bin/env python
# coding: utf-8

# # Segmenting FIFA 19 Players
# 
# In this analysis we will use clustering algorithms such as K-means Clustering to see if we can class players into positions based on their attributes. We will attempt to answer questions like:  
# How many player types are there really?  
# Can we accurately predict player types?

# In[ ]:


import numpy as np
import pandas as pd

import re

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Load the fifa19 player data set
data = pd.read_csv("../input/data.csv")
print("Fifa 19 player dataset has {} samples with {} features each.".format(*data.shape))
data.head()


# ## Data cleaning
# For this project I will be clustering the players into player segments. As such the position labels('ST', 'LS', 'RM',..) will be removed from the dataset.

# In[ ]:


remove = ['LS', 'ST', 'RS', 'LW', 
        'LF', 'CF', 'RF', 'RW',
        'LAM', 'CAM', 'RAM', 'LM', 
        'LCM', 'CM', 'RCM', 'RM', 
        'LWB', 'LDM','CDM', 'RDM', 
        'RWB', 'LB', 'LCB', 'CB', 
        'RCB', 'RB', 'Position', 
        'Unnamed: 0', 'Jersey Number', 'Loaned From', 
        'Contract Valid Until', 'Real Face',
       'Photo', 'Club Logo', 'Flag', 'ID', 'Work Rate', 
        'Joined','Release Clause','Nationality',
        'Club']

data.drop(remove, axis=1, inplace=True)
print(len(remove), "columns dropped.")
# Number of rows with na values.
print("Number of rows with null values: {}".format(len(data[data.isnull().any(axis=1)])))
features = list(data.keys()[6:])
for feat in features:
    print("null values in {}:".format(feat), len(data[data[feat].isnull()]))    
# Drop rows with Nan values.
data.dropna(inplace=True)
print("Shape of data: {}".format(np.shape(data)))


# In[ ]:


# Fix financial data
financial = ['Value', 'Wage']
for f in financial:
    # remove euro symbol
    data[f] = data[f].apply(lambda x: str(x)[1:])
    #data[f] = pd.to_numeric(data[f], errors='coerce')
    
# convert values with "M" and "K" to millions and thousands respectively.
def convert(value):
    regex = r'K|M'
    m = re.search(regex, value)
    if m:
        value = re.sub(regex, "", value)
        
        if m.group() == "M":
            value = pd.to_numeric(value) * 1e6
            value = value / 1000
        else:
            value = pd.to_numeric(value) * 1e3
            value = value / 1000
            
    return value
            
for f in financial:
    data[f] = data[f].apply(convert)

    
data.head()


# In[ ]:


# Fix height and weight
def height_convert(height):
    height_re = re.compile(r'\d\'\d')
    m = height_re.search(height)
    digits = m.group().split("\'")
    height = float(digits[0]) + (float(digits[1]) / 12.0)
    return round(height, 2)

data['Height'] = data['Height'].apply(lambda x: height_convert(height=x))

data['Weight'] = data['Weight'].apply(lambda x: float(re.sub(r'lbs','', x)))

data.head()


# In[ ]:


data.info()


# In[ ]:


indices = [1, 3, 431, 986, 544, 189, 107, 1009]

samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop = True)


# ### EDA

# In[ ]:


numerical = data.select_dtypes(include=['float64', 'int64']).keys().values
non_numerical = data.select_dtypes(exclude=['float64', 'int64']).keys().values
print( "Non-numerical features: " + str(len(non_numerical)), "\nNumerical features: "+ str(len(numerical)) )


# In[ ]:


# TODO: Create visualizations of distributions for numerical data.
fig = plt.figure(figsize=(20,20))
ax = fig.gca()
data[numerical].hist(ax = ax)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(14,8))
sns.set(font_scale=1.0)
sns.heatmap(data.corr())
plt.show()


# From the above correlation heatmap, we can see the relationships between the features. We can suspect that irrelevant features might show high correlation with others and conversely features that we think should be relevant to predicting the player type may not show any correlation between that features and the others. As we can expect we can identify the trend that offensive player attributes show higher correlation with other offensive player attributes, same to tend to also have high volley.) We can easily identify in the bottom right, where the goal keeping attributes lie all have very high correlation with one another. This can suggest that we may be able to reduce features further, possibly through PCA.

# In[ ]:


names = data['Name']
data = data.drop('Name', axis=1,errors='ignore')


# ## Data Preprocessing

# In[ ]:


np.log(data[numerical]+1).head()


# ### 1. Feature Scaling and Normalization

# In[ ]:


# Log-transform the skewed features
skewed = numerical
log_data = pd.DataFrame(data)
log_data[skewed] = log_data[skewed].apply(lambda x: np.log(x+1))

# Normalizing Numerical Features
# Initialize a scaler, then apply it to the features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_log_data = pd.DataFrame(log_data)
scaled_log_data[numerical] = scaler.fit_transform(scaled_log_data[numerical])

scaled_log_data.head()


# ### Outlier Detection

# In[ ]:


features = scaled_log_data[numerical].keys()
for feat in features:
    Q1, Q3 = np.percentile(scaled_log_data[numerical], [25, 75])
    step = (Q3 - Q1) * 1.5
    lower = Q1 - step
    upper = Q3 + step
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feat))
    scaled_log_data[~((scaled_log_data[feat] >= Q1 - step) & (scaled_log_data[feat] <= Q3 + step))]


# In[ ]:


# TODO: Convert non-numerical variables using the one-hot encoding scheme.
features_final = pd.get_dummies(scaled_log_data)

encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))
print(encoded)


# In[ ]:


features_final.head()


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(features_final)
first_pc = pca.components_[0]
second_pc = pca.components_[1]
transformed_data = pca.transform(features_final)
print(pca.explained_variance_ratio_)


# In[ ]:


pca.explained_variance_ratio_[:2].sum()


# In[ ]:


transformed_data = pd.DataFrame(transformed_data, columns= ['Dimension 1', 'Dimension 2'])
transformed_data.head()


# In[ ]:


transformed_data.plot.scatter(x='Dimension 1', y='Dimension 2')
plt.show()


# In[ ]:


transformed_data.shape


# In[ ]:


def biplot(good_data, reduced_data, pca):
    '''
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.
    
    good_data: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    return: a matplotlib AxesSubplot object (for any additional customization)
    
    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''

    fig, ax = plt.subplots(figsize = (14,8))
    # scatterplot of the reduced data    
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'], 
        facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
                  head_width=0.2, head_length=0.2, linewidth=2, color='red')
        #ax.text(v[0]*text_pos, v[1]*text_pos, good_data.columns[i], color='black', 
        #        ha='center', va='center', fontsize=18)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax

biplot(log_data, transformed_data, pca) # too many labels / arrows


# In[ ]:


reduced_data = pd.DataFrame(transformed_data, columns = ['Dimension 1', 'Dimension 2'])


# In[ ]:


reduced_data.head(50)


# In[ ]:


scaled_log_data.head()


# In[ ]:


biplot(scaled_log_data, reduced_data, pca)


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score

# TODO: Apply your clustering algorithm of choice to the reduced data 
if True:
    clusterer = KMeans(n_clusters=5, random_state=42).fit(transformed_data)
if False:
    clusterer = GaussianMixture(n_components=2, random_state=42).fit(transformed_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(transformed_data)

# TODO: Find the cluster centers
if True:
    centers = clusterer.cluster_centers_
if False:
    centers = clusterer.means_

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(transformed_data, preds)
print(score)


# Can the score be improved?

# In[ ]:


scores = {}
cluster_nums = range(2, 12)
for n in cluster_nums:
    clusterer_ = KMeans(n_clusters=n, random_state=42).fit(transformed_data)
    preds_ = clusterer_.predict(transformed_data)
    score_ = silhouette_score(transformed_data, preds_)
    scores[n] = score_
scores


# In[ ]:


import matplotlib.pyplot as plt

lists = sorted(scores.items())
n_clusters, scores = zip(*lists)
plt.plot(n_clusters, scores)
plt.xlabel("Number of K means clusters")
plt.ylabel("Silhouette score")

plt.show()


# In[ ]:


# Cluster Analysis.

