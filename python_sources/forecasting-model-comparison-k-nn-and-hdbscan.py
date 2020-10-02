#!/usr/bin/env python
# coding: utf-8

# # Comparitive Analysis of Weather Forecasting Models
# ###### an INFO 1998 final project by Ashley Jiang (yj387@cornell.edu) and Brady Sites (bas339@cornell.edu)

# **NOTE: This project was completed purely for educational purposes. We leave our work unlicensed, so you are free to do what you would like with it.**
# 
# As it stands, the weather forecasting paradigm has yet to be truly touched by machine learning. The Global Forecast System (GFS) used by the National Weather Service relies on mathematical models of the atmosphere, as does the forecasting model of the European Centre for Medium-Range Weather Forecasts (ECMWF). Physical models rule over computer-learned models. In fact, it would be redundant for a program to learn an atmopheric model if we already know the model. Why teach software to do something that we can already do with supercomputing power?
# 
# But as hinted at above, these physical models are computationally expensive. Companies like [Google](https://ai.googleblog.com/2020/03/a-neural-weather-model-for-eight-hour.html) have already produced work showing incredible speedups over state-of-the-art physical methods by engineering neural networks to predict local weather. The venture shows promise, but as of now, not enough is known about the efficacy of these methods for them to be implemented on a large scale.
# 
# So we posed the question to ourselves, "Can we predict the current amount of precipitation based on other initial conditions?" We hypothesize that we can, and that a proababilistic clustering model based on the HDBSCAN* algorithm will perform better than a k-nearest neighbors model at making predictions.
# 
# Before testing this, we will have to clean the data.

# In[ ]:


# imports!
import numpy as np
import pandas as pd
import xarray as xr  # easier handling of multi-dimensional arrays than numpy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import colors


#  ### Radar plotting

# In[ ]:


directory = '/kaggle/input/meteonet/'

zone = 'NW'
year = 2016
month = 1
submonth = 1
nan_val = -1

# rainfall files are located in a structured format, so we can easily locate them

def fetch_rainfall(directory, zone, year, month, submonth, ind=15):
    """
    Returns: rainfall data in xarray format, with coordinates.
    
    Precondition: directory is the directory to locate the file in, in string format
                  zone is a two-letter string corresponding to a zone of the dataset (default NW)
                  year is an int representing a valid year
                  month is an int representing a valid month
                  submonth is an int from 1 to 3, inclusive
                  ind is the 5-minute period to plot
    """
    
    s1 = f'{zone}_rainfall_{year}'
    s2 = f'rainfall-{zone}-{year}-' + f'{month}'.zfill(2)
    s3 = f'rainfall_{zone}_{year}_' + f'{month}'.zfill(2) + f'.{submonth}.npz'
    
    filepath = directory + s1 + '/' + s1 + '/' + s2 + '/' + s2 + '/' + s3
    coords_path = directory + 'Radar_coords/'*2 + f'radar_coords_{zone}.npz'
    
    # load the data
    data = np.load(filepath,allow_pickle=True)['data'][ind,:,:]
    coordinates = np.load(coords_path,allow_pickle=True)
    
    # uses spacial resolution of data (in degrees) to center coordinates
    res = 0.01
    lat = coordinates['lats']-res/2
    lon = coordinates['lons']+res/2
    
    data = xr.DataArray(data,coords=[lat[:,0],lon[0,:]],dims=['latitude','longitude'])
    radar = data.to_dataset(name = 'rainfall')
    
    return radar,lat,lon
    


# In[ ]:


# testing the function to see if we can plot the data
radar_test,lat_test,lon_test = fetch_rainfall(directory, zone, year, month, submonth)


# In[ ]:


fig = plt.figure()

if (np.max(radar_test['rainfall'].values) > 65):
    borne_max = np.max(radar_test['rainfall'].values)
else:
    borne_max = 65 + 10
cmap = colors.ListedColormap(['silver','white', 'darkslateblue', 'mediumblue','dodgerblue', 'skyblue','olive','mediumseagreen'
                              ,'cyan','lime','yellow','khaki','burlywood','orange','brown','pink','red','plum'])
bounds = [-1,0,2,4,6,8,10,15,20,25,30,35,40,45,50,55,60,65,borne_max]
norm = colors.BoundaryNorm(bounds, cmap.N)

# take out nan values
nan_radar_test = radar_test.where(radar_test["rainfall"]!=nan_val)

fig.set_size_inches(11,8)
plt.imshow(nan_radar_test["rainfall"].values,cmap=cmap, norm=norm)
fig.suptitle('Rainfall')


# ### Load our time series

# In[ ]:


ground2016 = pd.read_csv(directory + 'NW_Ground_Stations/'*2 + 'NW_Ground_Stations_2016.csv')
ground2016.describe()


# ### Manipulating the data

# Wow, this dataset has 22 MILLION rows in just one year! Since the dataset is so large, we need to take a random sample so that it is possible to take an exploratory approach to getting a better set of initial centroids. Running diagnostics on the entire dataset multiple times would become far too costly.

# In[ ]:


# select a random sample of sample_size rows
sample_size = 10000
sample_weather = ground2016.sample(n=sample_size, random_state=42)
sample_weather.isnull().sum()/sample_size


# Since we have so much initial data (in the whole dataset, not the sample), we can reduce our sample to only entries with an air pressure reading without worrying about making our feature space too small. ~80% of entries are missing a pressure measurement, but if we take 20% of 22 million, we still get over 4 million data points. And that is just for one year! We can safely assume that pressure readings are missing at random due to some equipment limitations; it is not like they only took pressure measurements when it was raining. Nevertheless, we can compare the distributions before and after dropping, just to be sure.

# In[ ]:


# these are essentially useless. we want to predict weather regardless of location!
sample_weather.drop(columns = ['lat', 'lon', 'number_sta', 'height_sta'], inplace=True)

# time is also not important
sample_weather.drop(columns = ['date'], inplace=True)
print(sample_weather.columns)


# In[ ]:


sample_filtered = sample_weather[sample_weather.isna()['psl'] == False]
print(sample_filtered.isnull().sum()/len(sample_filtered))
print(sample_filtered.describe()-ground2016.describe())


# What this tells us is this: the NaN columns were the ones we don't care about. Some boundary values have a seemingly significant difference, but this is a result of sampling. The difference in mean, quartiles, and standard deviation is so small compared to the actual values we saw for the dataset at the beginning that we don't have to worry about selection bias due to preserving the pressure measurement.

# In[ ]:


# interpolate missing values
# we don't have a lot, but we should still do it
sample_filtered = sample_filtered.interpolate(method='linear')
print(sample_filtered.isnull().sum())
print(sample_filtered.isnull())


# Successful imputation!

# ### t-SNE visualization

# The next step we want to take is visualizing the data that we will eventually fit a clustering model to. We can do this using T-distributed Stochastic Neighbor Embedding (t-SNE). We have high dimensional data that we would like to visualize in a low dimensional space. The reason we want to do this is so we can observe if the data naturally exhibits some degree of clustering.

# For effective use of t-SNE, we want to assure an adherence to tradition. We want the perplexity value to be less than the number of samples. The implementation will behave in an unpredictable way if we do not set the perplexity low enough. When van der Maaton & Hinton proposed this method, they recommended a typical range from 5 to 50 [[1]](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf). This will also depend on the sample size.

# In[ ]:


from sklearn.manifold import TSNE
import seaborn as sns
import time

feat_cols = ['dd','ff','hu','td','t', 'psl']

df_subset = sample_filtered.copy()

tsne_features = df_subset[feat_cols].values

perplexities = [2,3,5,10,15,30]

time_start = time.time()

for val in perplexities:
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=val, n_iter=5000)
    tsne_results = tsne.fit_transform(tsne_features)
    
    # organize results
    df_subset[f'tsne-2d-x-{val}'] = tsne_results[:,0]
    df_subset[f'tsne-2d-y-{val}'] = tsne_results[:,1]
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[ ]:


### Graphing and whatnot ###

fig = plt.figure()
fig.set_figheight(10*1.5)
fig.set_figwidth(16*len(perplexities)/3)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

cmap = sns.cubehelix_palette(dark=0.3, light=0.8, as_cmap=True)

for n in range(len(perplexities)):
    ax = fig.add_subplot(2, len(perplexities)/2, n+1)
    sns.scatterplot(
        x=f"tsne-2d-x-{perplexities[n]}", y=f"tsne-2d-y-{perplexities[n]}",
        palette=cmap,
        hue='precip',
        hue_norm=(0, 0.05),
        size='precip',
        data=df_subset,
        legend="full",
        alpha=0.6,
        ax=ax
    )
plt.show()


# Great, we now have a much better idea of what our data looks like, in a two dimensional space. Should we cluster on the output on one of these t-SNE models?
# 
# The answer is no! t-SNE does not preserve distances or densities, so performing any type of machine learning on t-SNE output would give  distorted results. Instead, we want to intepret these results within the specifications of t-SNE. Based on overall structure, some clusters naturally form within the data. However, these clusters are very strange-looking. Since t-SNE does not preserve distances, it only clusters neighbors, these clusters can definitely manifest in strange forms.
# 
# Therefore, we will train our clustering model on the original data, not the t-SNE output.

# ## k-NN

# But before we train our clustering model, let's see how a k-nearest neighbors model performs on the data.

# In[ ]:


# we can use a larger sample than we did with t-SNE
N = 200000
df = ground2016.sample(n=N, random_state=42)

df.drop(columns = ['lat', 'lon', 'number_sta', 'height_sta'], inplace=True)

df.drop(columns = ['date'], inplace=True)

df = df[df.isna()['psl'] == False]

df = df.interpolate(method='linear')

df_feats = df[feat_cols]


# To determine the ideal k value, we will check the coefficient of determination for models trained with a range of k values, and select the one that gives us the best results.

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

best_r = (0, -1)
r_squares = []

for k in range(1,201):
    X_train, X_test, Y_train, Y_test = train_test_split(df_feats, df['precip'], test_size=0.2, random_state=42)

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)
    r = r2_score(Y_test, Y_pred)
    r_squares.append(r)
    if r > best_r[0]:
        best_r = (r, k)

print(f'Greatest coef. of determination: {best_r[0]} occured at k = {best_r[1]}')


# In[ ]:


plt.plot(range(1,201), r_squares)
plt.title('R^2 vs. k')
plt.xlabel('k value')
plt.ylabel('coef. of determination')
plt.show()


# We ran an iterative diagnostic of the k-NN method, but it is easy to see that k-NN is not suited for this problem, under the current conditions. We observe a plateau around 0 in the coefficient of determination for k values higher than 50. This is not good! k-NN is having a hard time predicting the data better than a horizontal hyperplane of the dimension of the data.
# 
# But as discussed [here](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html), we have essentially broken the main assumption of the k-NN algorithm; our model has fallen victim to the "curse of dimensionality." Thus, we will retry this iterative approach after some Principal Component Analysis (PCA) to reduce the dimension of our data.

# In[ ]:


from sklearn.decomposition import PCA

# duplicate code, after PCA applied
best_r = (0, -1)
r_squares = []

pca = PCA(n_components=2)
pca_df = pca.fit_transform(X=df_feats)

for k in range(1,201):
    X_train, X_test, Y_train, Y_test = train_test_split(pca_df, df['precip'], test_size=0.2, random_state=42)

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)
    r = r2_score(Y_test, Y_pred)
    r_squares.append(r)
    if r > best_r[0]:
        best_r = (r, k)

print(f'Greatest coef. of determination: {best_r[0]} occured at k = {best_r[1]}')

plt.plot(range(1,201), r_squares)
plt.title('R^2 vs. k after PCA')
plt.xlabel('k value')
plt.ylabel('coef. of determination')
plt.show()


# Clearly, k-NN is not the correct model for forecasting with our given dataset; the coefficient of determination sticking around 0 means we are taking the wrong approach. Now, we will see how well clustering can predict weather.

# ### Determining optimal cluster count (supplemental)
# 
# ##### as we'll learn in the next section, we do not actually need to do this. but this provides us the opportunity to potentially improve our model based on what is known as the Elbow Method. if we had decided to create a Gaussian Mixture Model, for instance, this would be a necessary step.

# In[ ]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
ks = range(2,15)

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(df_feats)
    
    distortions.append(sum(np.min(cdist(df_feats, model.cluster_centers_, 
                      'euclidean'),axis=1)) / df_feats.shape[0])
    inertias.append(model.inertia_)
    
    mapping1[k] = sum(np.min(cdist(df_feats, model.cluster_centers_, 
                      'euclidean'),axis=1)) / df_feats.shape[0]
    mapping2[k] = model.inertia_
    


# In[ ]:


for k, dist in mapping1.items():
    print(str(k) + ': ' + str(dist))


# In[ ]:


plt.plot(ks, distortions, 'bx-')
plt.title('distortion elbow')
plt.xlabel('k = # of clusters')
plt.ylabel('distortion value')
plt.show()


# In[ ]:


plt.plot(ks, inertias, 'bx-')
plt.title('inertia elbow')
plt.xlabel('k = # of clusters')
plt.ylabel('inertia value')
plt.show()


# The value after which rate of change becomes linear occurs around k = 9. Subjectively, it could occur anywhere between 7 and 9 inclusive, but 9 would be the safest choice.

# # HDBSCAN*
# HDBSCAN* is a hierarchical clustering algorithm which extends from DBSCAN. It uses a technique to extract a flat clustering based in the stability of clusters. Due to its hierarchical nature, it is inevitably very sensitive to noise in the data. However, it is very neat for our case because it allows clustering without specifying the number of clusters. In addition, this density-based algorithm can be very powerful because it is indifferent to the shape of clusters and robust with respect to lcusters with different density.

# In[ ]:


# NOTE: the package for hdbscan is build ontop of sci-kit learn, but is NOT native to the library.
#       Thus, we will have to perform an inline import; comment this out if you are running on an environment with hdbscan installed.
get_ipython().system('pip install hdbscan')


# In[ ]:


import hdbscan

clusterer= hdbscan.HDBSCAN(min_cluster_size=30, prediction_data=True)
clusterer.fit(df)


# Let's first test the algorithm with the entire sampled dataset from before to understand better how it works and what our dataset looks like.

# In[ ]:


num_of_clusters=clusterer.labels_.max()
df_cols=clusterer.labels_
clusterValues, occurCount=np.unique(df_cols, return_counts=True)
print('cluster numbers are', clusterValues)
print('number of data in a cluster', occurCount)


# As we can see here, the model generates four clusters for us with reasonable sizes. However, we do see a great amount of outliers which we might have to adjust later when we try to use the model to actually predict. 

# In[ ]:


x=0
lst=[]
while x < num_of_clusters+1:
    precip=[]
    for i in range(0,len(df_cols)):
        if df_cols[i]==x:
            precip_amt=df['precip'].iloc[i]
            precip.append(precip_amt)
    mean=sum(precip)/len(precip)
    lst.append(mean)
    x=x+1
        
lst


# This allows us to find out the average precipitation for the datapoints that are in the same cluster. We can therefore predict the precipitation of a datapoint based on which cluster it falls into.

# In[ ]:


data=df['precip']
color_palette = sns.color_palette('deep',4)
cluster_colors = [color_palette[x] if x >= 0
                else (0.5,0.5,0.5)
                for x in clusterer.labels_]
cluster_member_colors = [sns.desaturate(x,p) for x, p in 
                        zip(cluster_colors, clusterer.probabilities_)]

#plt.scatter(data.T[0],data.T[1], s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)


# A good way to verify if our parameters are set properly is to check the condensed tree graph. For example, if  there is one cluster that takes up the majority of the color and space while we can barely see it for other clusters that means our parameters are not chosen properly.

# In[ ]:


clusterer.condensed_tree_
clusterer.condensed_tree_.plot()


# In[ ]:


clusterer.condensed_tree_.plot(select_clusters=True,
                              selection_palette=sns.color_palette('deep',11999))


# In[ ]:


clusterer.single_linkage_tree_
clusterer.single_linkage_tree_.plot()


# Now we have a better idea how our data fits into this specific type of model, we can now train and test our sample. 

# In[ ]:


clusterer=hdbscan.HDBSCAN(min_samples=3, prediction_data=True)

x=df.drop('precip', axis=1)
y=df['precip']
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)
clusterer.fit(x_train, y_train)
x=0
lst=[]
cols=clusterer.labels_
while x < clusterer.labels_.max()+1:
    precip=[]
    for i in range(0,len(cols)):
        if cols[i]==x:
            precip_amt=df['precip'].iloc[i]
            precip.append(precip_amt)
    mean=sum(precip)/len(precip)
    lst.append(mean)
    x=x+1
        
lst


# We modify our model a bit so that it will actually give us predicted values. Our features will be everything except the 'precip' column and our goal is to predict the amount of precipitation based on the other factors we have. After clustering, we calculate the average precipitation for each cluster.

# In[ ]:


test_labels, strengths=hdbscan.approximate_predict(clusterer, x_test)
predictions=[]
non_clustered = 0
for i in range(0,len(test_labels)):
    if test_labels[i]==0:
        predictions.append(0.0)
    elif test_labels[i]==1:
        predictions.append(0.014285714285714287)
    elif test_labels[i]==2:
        predictions.append(0.003250975292587777)
    else:
        predictions.append(y_test.iloc[i])
        if test_labels[i] == -1:
            non_clustered += 1

print(non_clustered/len(test_labels))
precip_predict=pd.Series(predictions)



# Luckily the hdbscan algorithm has its own prediction function; however, it only predicts which cluster a datapoint will fall into not its expected precipitation. Therefore, we have to assign the precipitation values for each datapoint. We also want to take into account how many datapoints were not able to fall in a cluster. 

# combine train_test_split with clustering allowed us to predict, now let's see how good it is

# In[ ]:


from sklearn.metrics import mean_absolute_error

print("sklearn's mean absolute error  for predicting precipitation is:", mean_absolute_error(y_test, precip_predict))
print("sklearn's R score for predicting precipitation is:", r2_score(y_test, precip_predict))


# it is safe for us to conclude that clustering is not a good way to predict precipitation since the 32 score is so low. One of the reasons can be our dataset is a lot smaller than what we need in order to provide accurate predictions. Other thing we've noticed is that since this algorithm only predicts the cluster a datapoint will fall into we have to kind of manually assign predicted values base on what we had before. As a consequence, the accuracy of our prediciton is largely compromised. 

# # Conclusion
# 

# 
