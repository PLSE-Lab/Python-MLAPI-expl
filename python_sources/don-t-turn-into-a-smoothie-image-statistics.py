#!/usr/bin/env python
# coding: utf-8

# # Uncover hidden image secrets! 
# 
# I used this notebook to generate datasets for the image statistics for the [SIIM-ISIC Melanoma Classification competition](https://www.kaggle.com/c/siim-isic-melanoma-classification). It's related to the EDA that I have done to highlight the differences and challenges in train and test data in my notebook [Don't turn into a Smoothie after the Shake-Up](https://www.kaggle.com/allunia/don-t-turn-into-a-smoothie-after-the-shake-up). 
# 
# The intend of this notebook is:
# 
# * to provide the code for generating image statistics
# * to extend the EDA to gain more insights 
# 
# Have fun! :-)
# 
# 
# <img src="https://cdn.pixabay.com/photo/2014/08/21/00/19/green-422995_1280.jpg" width="900px">
# 

# You are probably using external datasets and like to redo or complete the computation of image statistics. I have been asked to share my notebook for data generation. So here it is. ;-) 
# 
# It's very likely that I will work on this notebook in a few days to expand the EDA. Stay tuned! ;-)

# ## Table of contents
# 
# 1. [Prepare to start](#prepare)
# 2. [Generating image statistics data](#data_generation) 
# 3. [About the differences of dicom and jpeg](#dcmjpg)
# 4. [Explore the insights](#EDA) 
#     * [The image shape matters!](#image_shape)
#     * [Open your eyes with mean and std](#open_eyes) 
#     * [Can we group images by shape?](#shape_groups) 
# 5. [Clustering based on image statistics](#stats_groups) 
#     * [Starting easy with K-Means](#k_means)
#     * [Why should we bother about "how K-Means works"?](#preprocessing)
#     * [How many clusters should we choose?](#num_clusters)
#     * [Running K-means](#run_kmeans)
#     * [The cluster patchwork quilt](#patchwork_clusters)
# 6. [Clustering with GMM](#gmm)
#     * [Why are GMM and Kmeans similar but different?](#gmm_kmeans)
#     * [Running GMM on image statistics](#run_gmm)
#     * [Exploring clusters](#cluster_eda)
#     * [Exploring anomalies](#anomalies)
# 7. [Fitting catboost and submission](#catboost)
#     * [Data preparation](#data_prep)
#     * [Validation strategy](#validation)
#     * [Fitting](#fitting)
#     * [Feature importances](#shap_values)
#     * [Submission](#submission)
# 8. [Conclusion](#conclusion)

# # Prepare to start <a class="anchor" id="prepare"></a>

# Loading packages...

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objects as go

from PIL import Image
import pydicom
from skimage.io import imread

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import boxcox

from catboost import CatBoostClassifier, Pool, cv

import shap
# load JS visualization code to notebook
shap.initjs()


# What kind of datasets are given?

# In[ ]:


from os import listdir
listdir("../input/")


# To compute the statistics we need the dicom folder:

# In[ ]:


basepath = "../input/siim-isic-melanoma-classification/"
train_image_path = basepath + "/train/"
some_files = listdir(train_image_path)[0:5]
some_files


# In[ ]:


train_info = pd.read_csv(basepath + "train.csv")
train_info.head()


# In[ ]:


test_info = pd.read_csv(basepath + "test.csv")
test_info.head()


# In[ ]:


train_info["dcm_path"] = basepath + "/train/" + train_info.image_name + ".dcm"
test_info["dcm_path"] = basepath + "/test/" + test_info.image_name + ".dcm"


# In[ ]:


train_info.head(1)


# In[ ]:


train_info.shape


# In[ ]:


test_info.shape


# # Generating image statistics data <a class="anchor" id="data_generation"></a>
# 
# If you like to redo the computation of image statistics, here is the code. Otherwise you could also load the [dataset](https://www.kaggle.com/allunia/siimisic-melanoma-classification-image-stats) I have published. 

# In[ ]:


extract_train_0 = False
extract_train_1 = False
extract_train_2 = False
extract_test = False


# In[ ]:


from tqdm.notebook import tqdm 
from scipy.stats import skew


def extract_shapes(df):
    all_paths = df.dcm_path.values
    image_eda = pd.DataFrame(index=np.arange(len(df)),
                             columns=["path", "rows", "columns",
                                      "channels", "img_mean", "img_std",
                                      "img_skew", "red_mean", "green_mean",
                                      "blue_mean"])
    for i in tqdm(range(0, len(df))):
        path = all_paths[i]

        dcm_file = pydicom.dcmread(path)
        image = dcm_file.pixel_array

        image_eda.iloc[i]["path"] = path
        image_eda.iloc[i]["rows"] = image.shape[0]
        image_eda.iloc[i]["columns"] = image.shape[1]
        image_eda.iloc[i]["channels"] = image.shape[2]

        # some image stats
        image_eda.iloc[i]["img_mean"] = np.mean(image.flatten())
        image_eda.iloc[i]["img_std"] = np.std(image.flatten())
        image_eda.iloc[i]["img_skew"] = skew(image.flatten())

        image_eda.iloc[i]["red_mean"] = np.mean(image[:,:,0].flatten())
        image_eda.iloc[i]["green_mean"] = np.mean(image[:,:,1].flatten())
        image_eda.iloc[i]["blue_mean"] = np.mean(image[:,:,2].flatten())

    return image_eda


if extract_train_0:
    train_shapes = extract_shapes(train_info.iloc[0:10000])
    train_shapes.to_csv("train_image_stats_10000.csv", index=False)
elif extract_train_1:
    train_shapes = extract_shapes(train_info.iloc[10000:20000])
    train_shapes.to_csv("train_image_stats_20000.csv", index=False)
elif extract_train_2:
    train_shapes = extract_shapes(train_info.iloc[20000::])
    train_shapes.to_csv("train_image_stats_toend.csv", index=False)
elif extract_test:
    test_shapes = extract_shapes(test_info)
    test_shapes.to_csv("test_image_stats.csv", index=False)


# ## Loading stats and meta features for train and test
# 
# I have published a [dataset](https://www.kaggle.com/allunia/siimisic-melanoma-classification-image-stats) that contains the image statistics on jpeg-images and meta features for the original train and test data in one csv-file. Let's load these files instead of generating new one. You can also load the other files but they were computed using dicom!

# In[ ]:


listdir("../input/siimisic-melanoma-classification-image-stats")


# In[ ]:


train_image_stats = pd.read_csv("../input/siimisic-melanoma-classification-image-stats/train_stats_and_meta.csv")
test_image_stats = pd.read_csv("../input/siimisic-melanoma-classification-image-stats/test_stats_and_meta.csv")


# In[ ]:


train_image_stats.head()


# In[ ]:


train_image_names = train_image_stats.image_name.values
test_image_names = test_image_stats.image_name.values


# In[ ]:


train_image_stats["img_area"] = train_image_stats["rows"] * train_image_stats["columns"]
test_image_stats["img_area"] = test_image_stats["rows"] * test_image_stats["columns"]


# In[ ]:


train_image_stats.head()


# In[ ]:


test_image_stats.head()


# # About the differences of dicom and jpeg <a class="anchor" id="dcmjpg"></a>
# 
# 

# In[ ]:


jpeg_path = basepath + "/jpeg/train/"

fig, ax = plt.subplots(3,2,figsize=(20,15))
for n in range(3):
    dcm_file = pydicom.dcmread(train_info.dcm_path.values[n])
    pixel_array = dcm_file.pixel_array
    
    image_path = jpeg_path + train_info.image_name.values[n] + ".jpg"
    image = imread(image_path)
    
    sns.distplot(pixel_array.flatten(), ax=ax[n,0], color="sienna")
    ax[n,0].set_title("Distribution of values in dicom pixelarrays")
    
    sns.distplot(image.flatten(), ax=ax[n,1], color="deepskyblue")
    ax[n,1].set_title("Distribution of values in jpeg images");


# In this notebook I'm using statistics on jpeg files!

# # Explore the insights! <a class="anchor" id="EDA"></a>
# 
# Let's do some more EDA to gain more and more insights about the training and test images. 

# ## The image shape matters! <a class="anchor" id="image_shape"></a>
# 
# In the first notebook I wrote one can already see that there are some image groups in the test data that differ from the training images and one of this group holds ~15 % in test! These groups can be formed and summarized by the image shape. Images of the same shape show similar statistics. 
# 
# To uncover more groups, let's take a look at a scatter plot of row and column values in train and test:

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
ax[0].scatter(train_image_stats["rows"].values, train_image_stats["columns"].values, c="orangered")
ax[1].scatter(test_image_stats["rows"].values, test_image_stats["columns"].values, c="lightseagreen")

ax[0].set_title("Train images")
ax[1].set_title("Test images")

for n in range(2):
    ax[n].set_xlabel("row value")
    ax[n].set_ylabel("column value")
    ax[n].set_xlim([0,6500])
    ax[n].set_ylim([0,6500])


# ### Insights
# 
# Can you see it? There are two very interesting insights here:
# 
# 1. There is one very clear linear straight line of image shapes in train and test and 2-3 more could be imagined as well. Is there some relationship between images with shapes on the line? Can we use simple rescaling with images on the line or do they still differ too much in their color information?
# 2. There are several groups in train that are not present in test and vice versa there are also some groups in test that are not given in train!
# 
# Let's overlap the train spots with the test spots to see groups in test that are not given in train:

# In[ ]:


plt.figure(figsize=(20,5))
plt.scatter(test_image_stats["rows"].values, test_image_stats["columns"].values, c="lightseagreen")
plt.scatter(train_image_stats["rows"].values, train_image_stats["columns"].values, c="orangered");
plt.title("Uncover groups in test that are not present in train");
plt.xlabel("row value")
plt.ylabel("column value");


# There are at least 4 groups that are not present in train and do not live on the straight line.

# ## Open your eyes with mean and std <a class="anchor" id="open_eyes"></a>

# In[ ]:


from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("Train image stats", "Test image stats"))


trace0 = go.Scatter(
    x = train_image_stats.img_std.values,
    y = train_image_stats.img_mean.values,
    mode='markers',
    text=train_image_stats["rows"].values,
    marker=dict(
        color=train_image_stats["columns"].values,
        colorscale='Jet',
        opacity=0.4,
        size=4
    )
)

trace1 =go.Scatter(
    x = test_image_stats.img_std,
    y = test_image_stats.img_mean,
    mode='markers',
    text=test_image_stats["rows"],
    marker=dict(
        color=test_image_stats["columns"],
        colorscale='Jet',
        colorbar=dict(thickness=10, len=1.1, title="image columns"),
        opacity=0.4,
        size=4
    )
)

fig.add_trace(trace0, row=1, col=1)
fig.add_trace(trace1, row=1, col=2)

fig.update_xaxes(title_text="Image std", row=1, col=1)
fig.update_yaxes(title_text="Image mean", row=1, col=1)
fig.update_xaxes(title_text="Image std", row=1, col=2)
fig.update_yaxes(title_text="Image mean", row=1, col=2)

fig.update_layout(height=425, width=850, showlegend=False)
fig.show()


# ### Insights
# 
# * Plotting the image mean and standard deviation for each image in train and test we can easily see that there is a big group in test (~140 mean, ~60 std) that is not present in train. This is the group that holds ~15% of the test data.
# * Furthermore we can see that the mean and standard deviation depends on the row and column value. There are clear clusters and groups that are related to these quantities. The columns are given by the color and the rows by the text you can see when browsing over a spot. 

# ## Can we group images by shape? <a class="anchor" id="shape_groups"></a>
# 
# We have seen that there are several groups of images that show different image statistics like mean and standard deviation. As you can see by the colors and texts of the scatter plot above, these groups also depend on the shape of the image. A first good idea is therefore to group by shape. But some images have uncommon shapes and they need to be assigned to a similar group that shows same kind of image information. For this reason a clustering could help us more than just grouping by shape. ;-)

# In[ ]:


train_image_stats.head()


# How often can we find images whose shape is unique and not part of a bigger, more common group?

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(train_image_stats["rows"].value_counts().values,
             kde=False, ax=ax[0], color="magenta")
sns.distplot(train_image_stats["columns"].value_counts().values,
             kde=False, ax=ax[1], color="mediumvioletred")
ax[0].set_title("Distribution of row value counts")
ax[1].set_title("Distribution of column value counts")
for n in range(2):
    ax[0].set_xlabel("value counts")
    ax[0].set_ylabel("frequency")


# By looking at the value counts of row and column values we can see that most values are only present once. In contrast groups with a few hundereds to thousand images having the same row or column value are much smaller! If you are not sure how to interpret these plots take a look at some examples instead:

# In[ ]:


train_image_stats["rows"].value_counts().sort_values().iloc[0:5]


# There is only one image with a row value of 1955. In contrast there are 14703 images with a row value of 4000:

# In[ ]:


train_image_stats["rows"].value_counts().sort_values(ascending=False).iloc[0:5]


# If we would like to work with image groups we need to find a better method for grouping them then just using the shape as it is.

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(train_image_stats.img_area, ax=ax[0], kde=False, bins=30, color="orangered")
sns.distplot(test_image_stats.img_area, ax=ax[1], kde=False, bins=30, color="lightseagreen")

for n in range(2):
    ax[n].set_xlabel("image area")
    ax[n].set_ylabel("counts")

ax[0].set_title("Image area distribution in train")
ax[1].set_title("Image area distribution in test");


# In[ ]:


fig, ax = plt.subplots(2,3,figsize=(20,12))

sns.distplot(train_image_stats.img_mean, ax=ax[0,0], color="crimson", label="train")
sns.distplot(test_image_stats.img_mean, ax=ax[0,0], color="lightseagreen", label="test")
sns.distplot(train_image_stats.img_std, ax=ax[0,1], color="crimson", label="train")
sns.distplot(test_image_stats.img_std, ax=ax[0,1], color="lightseagreen", label="test")
sns.distplot(train_image_stats.img_skew, ax=ax[0,2], color="crimson", label="train")
sns.distplot(test_image_stats.img_skew, ax=ax[0,2], color="lightseagreen", label="test")

sns.distplot(train_image_stats.red_mean, ax=ax[1,0], color="crimson", label="train")
sns.distplot(test_image_stats.red_mean, ax=ax[1,0], color="lightseagreen", label="test")
sns.distplot(train_image_stats.green_mean, ax=ax[1,1], color="crimson", label="train")
sns.distplot(test_image_stats.green_mean, ax=ax[1,1], color="lightseagreen", label="test")
sns.distplot(train_image_stats.blue_mean, ax=ax[1,2], color="crimson", label="train")
sns.distplot(test_image_stats.blue_mean, ax=ax[1,2], color="lightseagreen", label="test")


for n in range(3):
    for m in range(2):
        ax[m,n].set_ylabel("density");
        ax[m,n].legend()
    
ax[0,0].set_title("Image means")
ax[0,1].set_title("Image stds")
ax[0,2].set_title("Image skewnesses");

ax[1,0].set_title("Red channel mean")
ax[1,1].set_title("Green channel mean")
ax[1,2].set_title("Blue channel mean");


# ### Insights
# 
# * All distributions are skewed somehow and show outliers. We need to perform some preprocessing to prevent a bad influence on the cluster center computation.
# * We can clearly see that there are differences in train and test image statistics.

# With preprocessing we should try to transform our distributions such that they look more normal. One way to do this is the [boxcox-transformation](https://en.wikipedia.org/wiki/Power_transform). The two parameters constant and lambda have to be set carefully and one needs to play a bit with them to find a good choice:

# In[ ]:


def preprocess_k_means(train_data, test_data, feature, constant, lam):
    min_max_scaler = MinMaxScaler()
    scaled_train_feature = min_max_scaler.fit_transform(train_data[feature].values.reshape(-1, 1))
    scaled_test_feature = min_max_scaler.transform(test_data[feature].values.reshape(-1,1))
    
    boxcox_train_feature = boxcox(scaled_train_feature[:,0] + constant, lam)
    boxcox_test_feature = boxcox(scaled_test_feature[:,0] + constant, lam)

    scaler = StandardScaler()
    preprocessed_train_feature = scaler.fit_transform(boxcox_train_feature.reshape(-1,1))
    preprocessed_test_feature = scaler.fit_transform(boxcox_test_feature.reshape(-1,1))
    
    train_data.loc[:, "preprocessed_" + feature] = preprocessed_train_feature
    test_data.loc[:, "preprocessed_" + feature] = preprocessed_test_feature
    return train_data, test_data


# Let's preprocess the skewness as well as the red and green mean:

# In[ ]:


train_image_stats, test_image_stats = preprocess_k_means(train_image_stats,
                                                         test_image_stats, 
                                                         "red_mean",
                                                         constant=1, lam=10)
train_image_stats, test_image_stats = preprocess_k_means(train_image_stats,
                                                         test_image_stats, 
                                                         "green_mean",
                                                         constant=0.5,lam=2)
train_image_stats, test_image_stats = preprocess_k_means(train_image_stats,
                                                         test_image_stats, 
                                                         "img_skew",
                                                         constant=0.05,lam=2)


# The preprocessed distributions are less skewed now and the effect of outliers is suppressed:

# In[ ]:


fig, ax = plt.subplots(1,3,figsize=(20,5))

sns.distplot(train_image_stats.preprocessed_red_mean, ax=ax[0], color="crimson", label="train")
sns.distplot(test_image_stats.preprocessed_red_mean, ax=ax[0], color="lightseagreen", label="test")
sns.distplot(train_image_stats.preprocessed_green_mean, ax=ax[1], color="crimson", label="train")
sns.distplot(test_image_stats.preprocessed_green_mean, ax=ax[1], color="lightseagreen", label="test")
sns.distplot(train_image_stats.preprocessed_img_skew, ax=ax[2], color="crimson", label="train")
sns.distplot(test_image_stats.preprocessed_img_skew, ax=ax[2], color="lightseagreen", label="test")


for n in range(3):
    ax[n].set_ylabel("density");
    ax[n].legend()
    
ax[0].set_title("Image means")
ax[1].set_title("Image stds")
ax[2].set_title("Image skewnesses");


# Now, let's take a look at all basic statistic features at once in a scatterplot:

# In[ ]:


fig = make_subplots(rows=1, cols=2, subplot_titles=("Preprocessed train image stats", "Preprocessed test image stats"))


trace0 = go.Scatter(
    x = train_image_stats.img_mean.values,
    y = train_image_stats.img_std.values,
    mode='markers',
    text=train_image_stats["columns"].values,
    marker=dict(
        color=train_image_stats.preprocessed_img_skew.values,
        colorbar=dict(thickness=10, len=1.1, title="preprocessed skewness"),
        colorscale='Jet',
        opacity=0.4,
        size=2
    )
)

trace1 = go.Scatter(
    x = test_image_stats.img_mean.values,
    y = test_image_stats.img_std.values,
    mode='markers',
    text=test_image_stats["columns"].values,
    marker=dict(
        color=test_image_stats.preprocessed_img_skew.values,
        colorscale='Jet',
        opacity=0.4,
        size=2
    )
)

fig.add_trace(trace0, row=1, col=1)
fig.add_trace(trace1, row=1, col=2)

fig.update_xaxes(title_text="Image std", row=1, col=1)
fig.update_yaxes(title_text="Image mean", row=1, col=1)
fig.update_xaxes(title_text="Image std", row=1, col=2)
fig.update_yaxes(title_text="Image mean", row=1, col=2)

fig.update_layout(height=425, width=850, showlegend=False)
fig.show()


# We can observe that the differences in the train and test distributions with mean values (70) and std (150) belongs to a hidden group of images!

# ## How many clusters should we choose? <a class="anchor" id="num_clusters"></a>
#     
# We have already found that the image statistics depend on the shape of the images. For this reason we could set the number of clusters equal to the number of the most common groups find in train and test:

# In[ ]:


train_shapes = train_image_stats.groupby(
    ["rows", "columns"]).size().sort_values(ascending=False) / train_image_stats.shape[0] * 100
test_shapes = test_image_stats.groupby(
    ["rows", "columns"]).size().sort_values(ascending=False) / test_image_stats.shape[0] * 100


# Let's take all groups into account that hold at least 0.2 % of all images. For example for the training data this would mean at least this number of images in one cluster:

# In[ ]:


train_image_stats.shape[0] * 0.2/100


# In[ ]:


common_train_shapes = set(list(train_shapes[train_shapes > 0.3].index.values))
common_test_shapes = set(list(test_shapes[test_shapes > 0.3].index.values))


# In[ ]:


common_shape_groups = common_train_shapes.union(common_test_shapes)
common_shape_groups


# In[ ]:


num_clusters = len(common_shape_groups)
num_clusters


# ## Running K-Means <a class="anchor" id="run_kmeans"></a>

# In[ ]:


combined_stats = train_image_stats.append(test_image_stats)
combined_stats.head(1)


# In[ ]:


kmeans = KMeans(n_clusters=num_clusters, 
                random_state=0)

x = combined_stats.loc[:, ["img_mean", "img_std", "preprocessed_img_skew",
                           "preprocessed_red_mean", "preprocessed_green_mean", "blue_mean"]].values #,
                           #"img_area", "rows", "columns"]].values
cluster_labels = kmeans.fit_predict(x)
combined_stats["cluster_label"] = cluster_labels


# In[ ]:


train_image_stats = combined_stats.iloc[0:train_image_stats.shape[0]]
test_image_stats = combined_stats.iloc[train_image_stats.shape[0]::]


# In[ ]:


fig = make_subplots(rows=1, cols=2, subplot_titles=("Train image stats", "Test image stats"))


trace0 = go.Scatter(
    x = train_image_stats.img_std.values,
    y = train_image_stats.img_mean.values,
    mode='markers',
    text=train_image_stats["cluster_label"].values,
    marker=dict(
        color=train_image_stats.cluster_label.values,
        colorbar=dict(thickness=10, len=1.1, title="cluster label"),
        colorscale='Jet',
        opacity=0.4,
        size=2
    )
)

trace1 = go.Scatter(
    x = test_image_stats.img_std.values,
    y = test_image_stats.img_mean.values,
    mode='markers',
    text=test_image_stats["cluster_label"].values,
    marker=dict(
        color=test_image_stats.cluster_label.values,
        colorscale='Jet',
        opacity=0.4,
        size=2
    )
)

fig.add_trace(trace0, row=1, col=1)
fig.add_trace(trace1, row=1, col=2)

fig.update_xaxes(title_text="Image std", row=1, col=1)
fig.update_yaxes(title_text="Image mean", row=1, col=1)
fig.update_xaxes(title_text="Image std", row=1, col=2)
fig.update_yaxes(title_text="Image mean", row=1, col=2)

fig.update_layout(height=425, width=850, showlegend=False)
fig.show()


# ### Insights
# 
# * We have obtained a patchwork quilt of cluster points. This is a drawback of K-Means. It's not able to adapt well to the shape of elliptical clusters. This won't improve with further preprocessing and we need a better clustering algorithm to cover the shapes better! ;-)

# In[ ]:


combined_stats = train_image_stats.append(test_image_stats)


# ## The cluster patchwork quilt <a class="anchor" id="patchwork_clusters"></a>

# Let's take a look at some example images within these 8 patchwork clusters. 

# In[ ]:


fig, ax = plt.subplots(num_clusters,8, figsize=(20, 2.5*num_clusters))

for cluster in range(num_clusters):
    selection = np.random.choice(combined_stats[combined_stats.cluster_label==cluster].image_path.values,
                                 size=8, replace=False)
    m=0
    for path in selection:
        image = imread(path)
        ax[cluster, m].imshow(image)
        ax[cluster, m].set_title("K-Means cluster {}".format(cluster))
        ax[cluster, m].axis("off")
        m+=1


# ### Insights
# 
# 1. With K-Means we already found some cluster that contain similar images. 
# 3. Furthermore some images seem to fit not so well in their group.
# 
# In the next step I like to improve the clustering using a Gaussian Mixture Model. It has the advantage that it can adapt better to the elliptical cluster shapes and it's also powerful to detect anomalies in the data.  

# # Clustering with GMM <a class="anchor" id="gmm"></a>

# ## Why are GMM and Kmeans similar but different? <a class="anchor" id="gmm_kmeans"></a>
# 
# By clustering with gaussian mixture models we still follow an interative approach of updating cluster centers and assigning data spots to them but in a so called "soft" manner. This means now that each cluster is responsible for each data point but with different strenghts. The cluster that holds the highest responsibility for one data point can then be hard assigned as well. 
# 
# In the case of **mixture models it's assumed that there exist some hidden latent variables that generate the distribution of data you observe**. In our case we hope that these latent variables walk along with different cameras that have taken the images at different conditions. Some cameras might always produce dark, small images whereas other ones might take large, but bright images. Do you get the idea? We don't know which camera was used and we don't know how the conditions looked like. This information is hidden.  
# 
# 
# Now the gaussian mixture model descripes the following: **Each of our latent variable, each camera, causes one multivariate gaussian**, this is a normal distribution over more than one dimension. In our case we like to use the statistics per image like mean, standard deviation and skewness. Doing so our feature space is 3-dimensional and by looking at the 3d-scatterplot we can observe ellipsoids with different densities, locations and expansions. Then our distribution $p(x)$ of data we observe is said to be generated by these gaussian ellipsoids:
# 
# $$ p(x) = \sum_{k=1}^{K} \pi_{k} \cdot N(x|\mu_{k}, \Sigma_{k}) $$
# 
# 
# Each gaussian $k$ is somehow responsible that a single data spot is placed where it is. **One gaussian may be more responsible than the others but they are all working in a mixture to explain your data**. During learning all of these ellipsoid clusters are moving in space and varying their shape trying to match the distributed data $p(x)$ perfectly. **The learning procedure for models with latent variables is the expectation maximization**. This algorithm works in two consequtive steps that are repeated until all gaussians only have found their place and would only show slight changes in further steps. 
# 
# #### E-Step (expectation):  
# 
# At the start of the model are gaussians are placed and shaped randomly (or pretrained by k-means locations). Then for each gaussian, that represents one cluster, the model calculates how responsible the cluster $z_{k}$ is for one data spot $x_{n}$.
# 
# $$\gamma_{nk}  = \frac {\pi_{k} \cdot N(x_{n}|\mu_{k}, \Sigma_{k})} {\sum_{j=1}^{K} \pi_{j} \cdot N(x_{n}|\mu_{j}, \Sigma_{j})}$$
# 
# These responsibilities sum up to 1 over all clusters for one data spot. Here we can read out the winner at the end of the learning procedure that is assigned as the predicted cluster for that data spot $x_{n}$.
# 
# #### M-Step (maximization):
# 
# During the maximization step, all parameters of the gaussians, the locations given by the cluster center and the shapes covered by the covariance matrices are updated:
# 
# $$\mu_{k} = \frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{nk} x_{n} $$
# 
# $$\Sigma_{k} =  \frac{1}{N_{k}}  \sum_{n=1}^{N}  \gamma_{nk} (x_{n} - \mu_{k}) (x_{n} - \mu_{k})^{T}$$ 
# 
# 
# You can see that we obtain each cluster location $\mu_{k}$ by looking and summing over all data spots. This is a big difference between K-Means and GMM as the further only calculates the cluster centers using cluster members. But in our case, as already told, each gaussian is responsibile for all data points. Hence the new location is calculated by using them all, but weighting with the responsibility. Thisway points that could be well explained by this cluster contribute much more to the new location (mean) than the other data points. The same holds for the cluster shapes aka covariances.  
# 
# 
# But nontheless there is also one more similarity: We are taking a weighted mean and for this reason we also need to take about outliers. We need to preprocess our data by rescaling and trying to transform the features such that they look normally distributed to suppress their effect. 

# ## Running GMM on image statistics <a class="anchor" id="run_gmm"></a>

# Let's now run the Gaussian Mixture Model to find clusters that suite better to the elliptical shape of the groups we can observe. Instead of using a number of clusters that roughly equals the most common shapes, let's use a number with one more cluster. I have often experienced that there will be at least once cluster that holds many outliers and for this reason it could be fruitful to have one more "outlier-cluster" as well:

# In[ ]:


num_clusters = num_clusters + 1


# In[ ]:


gmm = GaussianMixture(n_components=num_clusters, 
                      max_iter=100, 
                      n_init=10,
                      random_state=0)

features = ["img_mean", "img_std", "preprocessed_img_skew",
            "preprocessed_red_mean", "preprocessed_green_mean", "blue_mean"]#,
            #"img_area", "rows", "columns"]

x = combined_stats.loc[:, features].values
cluster_labels = gmm.fit_predict(x)
combined_stats["gmm_cluster_label"] = cluster_labels
combined_stats["gmm_logL"] = gmm.score_samples(x)


# In[ ]:


gmm.converged_


# Beside the cluster label we can obtain a measure on how anomalistic an image is by computing the per sample log likelihood. This is a nice feature to find outliers in the data!

# In[ ]:


train_image_stats = combined_stats.iloc[0:train_image_stats.shape[0]]
test_image_stats = combined_stats.iloc[train_image_stats.shape[0]::]


# Ok, now let's take a look at the cluster assigments:

# In[ ]:


fig = make_subplots(rows=1, cols=2, subplot_titles=("Train image stats", "Test image stats"))


trace0 = go.Scatter(
    x = train_image_stats.img_std,
    y = train_image_stats.img_mean,
    mode='markers',
    text=train_image_stats["gmm_cluster_label"],
    marker=dict(
        color=train_image_stats.gmm_cluster_label,
        colorbar=dict(thickness=10, len=1.1, title="cluster label"),
        colorscale='Jet',
        opacity=0.4,
        size=2
    )
)

trace1 = go.Scatter(
    x = test_image_stats.img_std,
    y = test_image_stats.img_mean,
    mode='markers',
    text=test_image_stats["gmm_cluster_label"],
    marker=dict(
        color=test_image_stats.gmm_cluster_label,
        colorscale='Jet',
        opacity=0.4,
        size=2
    )
)

fig.add_trace(trace0, row=1, col=1)
fig.add_trace(trace1, row=1, col=2)

fig.update_xaxes(title_text="Image std", row=1, col=1)
fig.update_yaxes(title_text="Image mean", row=1, col=1)
fig.update_xaxes(title_text="Image std", row=1, col=2)
fig.update_yaxes(title_text="Image mean", row=1, col=2)

fig.update_layout(height=425, width=850, showlegend=False)
fig.show()


# ### Insights
# 
# * The patchwork quilt of clusters has been removed. 
# * The shapes of the clusters suite better to what we can observe with our own eyes.
# * Furthermore the test data seems to have one cluster more than the training data.

# ## Exploring clusters <a class="anchor" id="cluster_eda"></a>

# ### Looking at examples

# I'm very curious how these clusters look like with GMM:

# In[ ]:


fig, ax = plt.subplots(num_clusters,8, figsize=(20, 2.5*8))

for cluster in range(num_clusters):
    selection = np.random.choice(combined_stats[combined_stats.gmm_cluster_label==cluster].image_path.values,
                                 size=8, replace=False)
    m=0
    for path in selection:
        image = imread(path)
        ax[cluster, m].imshow(image)
        ax[cluster, m].set_title("Cluster {}".format(cluster))
        ax[cluster, m].axis("off")
        m+=1


# ### Insights
# 
# * Even though we are only using some basic statistics like the mean, std and skewness of the overall pixel values of an image as well as color channel means, we can find pretty nice clusters.
# * The result looks way better and more ordered than with K-Means.
# * We can clearly see that one cluster, that holds the big hidden image group in test that is not really present in train, differs a lot in the way the images look like compared to all other clusters. 

# ### Cluster image shapes & statistics

# In my first notebook for this competition ["Don't turn into a Smoothie after the Shape-Up!"](https://www.kaggle.com/allunia/don-t-turn-into-a-smoothie-after-the-shake-up) one can already see by the 3d-scatter-plot of image statistics and colored by column, texted by row that the statistics depend on the image shape. For this reason we can expect that the clusters found by GMM based on image statistics should cover specific shapes as well: 

# In[ ]:


fig, ax = plt.subplots(1,3,figsize=(20,5))
sns.boxplot(combined_stats.gmm_cluster_label, combined_stats["rows"], ax=ax[0], palette="Greens")
ax[0].set_title("Rows in train and test")
sns.boxplot(combined_stats.gmm_cluster_label, combined_stats["columns"], ax=ax[1], palette="Blues")
ax[1].set_title("Columns in train and test")
sns.boxplot(combined_stats.gmm_cluster_label, combined_stats["img_area"], ax=ax[2], palette="Reds")
ax[2].set_title("Areas in train and test");


# ### Insights
# 
# * Some clusters hold very specific image shapes and this is great to see. It confirms the assumption that our different image types show different shapes. 
# * Some clusters hold a broad range of rows, columns and areas. Here we have found images that look similar but show very different shapes. One way to find out wheather these are already nice groups or should be improved, is to add more features to the clustering process or to add the column and row values themselves.

# In[ ]:


fig, ax = plt.subplots(2,3,figsize=(20,10))
sns.violinplot(combined_stats.gmm_cluster_label, combined_stats.img_mean, ax=ax[0,0], palette="Purples")
sns.violinplot(combined_stats.gmm_cluster_label, combined_stats.img_std, ax=ax[0,1], palette="Oranges")
sns.violinplot(combined_stats.gmm_cluster_label, combined_stats.preprocessed_img_skew, ax=ax[0,2], palette="Greys");
sns.violinplot(combined_stats.gmm_cluster_label, combined_stats.preprocessed_red_mean, ax=ax[1,0], palette="Reds")
sns.violinplot(combined_stats.gmm_cluster_label, combined_stats.preprocessed_green_mean, ax=ax[1,1], palette="Greens")
sns.violinplot(combined_stats.gmm_cluster_label, combined_stats.blue_mean, ax=ax[1,2], palette="Blues");


# ### Insights
# 
# * These are the inner cluster distributions of the preprocessed features we used for clustering: the mean, the standard deviation and the skewness as well as the red, green and blue color channel means.
# * Some clusters show clearly defined scopes of the features, but there are some with broad ranges as well. 
# * Furthermore some clusters hold many outlier values.

# ## Cluster interactions <a class="anchor" id="interactions"></a>

# Another great feature of mixture models is the possibility to obtain the responsibilities of each cluster per data point (in this case per image). This way we get some insight, which cluster may be an alternative for the hard assigned one. Looking at THE alternative cluster for all data points we might get an impression which image group is close to the group of interest - for example close to our hidden test group. :-)

# In[ ]:


responsibilities = gmm.predict_proba(x)
combined_stats["alternative_cluster"] = np.argsort(responsibilities, axis=1)[:,-2]


# In[ ]:


competition = combined_stats.groupby("gmm_cluster_label").alternative_cluster.value_counts().unstack()
competition.fillna(0, inplace=True)
plt.figure(figsize=(15,15))
sns.heatmap(competition, cmap="Greens", annot=True, fmt="g", cbar=False);


# ### Insights
# 
# * Using only image statistics (without further shape information) we can see that the cluster 7 with the hidden test images seems to be close to cluster 0 and 5. If you compare these clusters above you can see that only cluster 0 is really similar as we find more microscopes or light loss at image boundaries. In addition you can see that only the microscopes have shifted the test group to darker image mean values and higher standard deviations. If more groups would have microscopes we would be better able to simulate this hidden test group in our training data.
# * Furthermore we can find some clusters that have only one close cluster whereas other cluster have multiple counterparts. This could be interesting information if we would like to perform cluster-dependent augmentation. 

# ## Exploring anomalies <a class="anchor" id="anomalies"></a>

# One very nice feature of GMM is the possibility to detect samples that live in regions with low density. For this purpose we need to compute the log likelihood values per sample. If you like to know more about anomaly detection with Gaussian Mixture Models, I would like to guide you to [my data science trainee notebook on GMMS](https://www.kaggle.com/allunia/hidden-treasures-in-our-groceries) that explains a bit more on "how it works". ;-)

# In[ ]:


plt.figure(figsize=(20,5))
sns.distplot(combined_stats.gmm_logL, color="gold");
plt.title("Sample log-likelihoods in train and test")
plt.ylabel("Density");


# At the moment it's sufficient to know that low logL numbers are related to points in low dense, "outlier"-regions. If we would like a hard assignment or decision what is meant by an outlier, we could use a quantile measurement like this one:

# In[ ]:


np.quantile(combined_stats.gmm_logL.values, q=0.05)


# In[ ]:


combined_stats["is_outlier"] = np.where(combined_stats.gmm_logL <= np.quantile(
    combined_stats.gmm_logL.values, q=0.05), 1, 0)


# In[ ]:


train_image_stats = combined_stats.iloc[0:train_image_stats.shape[0]]
test_image_stats = combined_stats.iloc[train_image_stats.shape[0]::]


# Let's try to find out if there is a cluster full of outliers in our data:

# In[ ]:


outliers_in_cluster = combined_stats[combined_stats.is_outlier==1].groupby("gmm_cluster_label").size() * 100
outliers_in_cluster /= combined_stats.groupby("gmm_cluster_label").size() 

fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.boxplot(x=combined_stats.gmm_cluster_label, y=combined_stats.gmm_logL, ax=ax[1])
sns.barplot(x=outliers_in_cluster.index, y=outliers_in_cluster.values, ax=ax[0])
ax[0].set_ylabel("% outlier in cluster");
ax[0].set_title("How many outliers can be found per cluster?")
ax[1].set_title("How is logL distributed within the clusters?");


# ### Insights
# 
# * Given our quantile-definition there are only 2 clusters that are occupied with > 20% outliers - cluster 5 and 6. 
# * If we compare their logL-distributions with all other clusters, we can see that they show the lowest values and even their medians are lower than of all other groups. 
# * Furthermore cluster 0 and 4 hold a lot of outliers as well!
# 
# Let's look at some example images:

# In[ ]:


fig, ax = plt.subplots(num_clusters,8, figsize=(20, 2.5*num_clusters))

for cluster in range(num_clusters):
    outlier_selection = np.random.choice(combined_stats[
        (combined_stats.gmm_cluster_label==cluster) & (combined_stats.is_outlier==1)
    ].image_path.values, size=8, replace=True)
    
    m=0
    for path in outlier_selection:
        image = imread(path)
        ax[cluster, m].imshow(image)
        ax[cluster, m].set_title("Outlier in \n cluster {}".format(cluster))
        ax[cluster, m].axis("off")
        m+=1


# ### Insights
# 
# * Some cluster have only a few or one outliers like cluster 9, 8 and 1. Their images are often repeated only to yield this plot. 
# * In contrast cluster 0, 4, 5 and 6 yield an impression of how anomalies in the data look like and for which kind of images our model may start to struggle in making good predictions.

# In[ ]:


train_image_stats.loc[:, "image_name"] = train_image_names
test_image_stats.loc[:, "image_name"] = test_image_names


# In[ ]:


train_image_stats.to_csv("train_stats_meta_cluster.csv", index=False)
test_image_stats.to_csv("test_stats_meta_cluster.csv", index=False)


# # Fitting catboost & submission <a class="anchor" id="catboost"></a>

# ## Data preparation <a class="anchor" id="data_prep"></a>

# In[ ]:


test_image_stats.columns.values


# In[ ]:


to_drop = ["image_path", "channels", "image_name", "patient_id"]
test_image_stats = test_image_stats.drop(to_drop, axis=1)
train_image_stats = train_image_stats.drop(to_drop, axis=1)
test_image_stats = test_image_stats.drop("target", axis=1)


# In[ ]:


use_features = test_image_stats.columns.values


# In[ ]:


targets = train_image_stats.target.values
train_df = train_image_stats[use_features].copy()
test_df = test_image_stats[use_features].copy()
train_df["target"] = targets


# In[ ]:


for df in [train_df, test_df]:
    df["rows"] = df["rows"].astype(np.str)
    df["columns"] = df["columns"].astype(np.str)
    df["img_area"] = df["img_area"].astype(np.str)
    df["gmm_cluster_label"] = df["gmm_cluster_label"].astype(np.str)
    df["cluster_label"] = df["cluster_label"].astype(np.str)
    df["age_approx"] = df["age_approx"].astype(np.str)
    df["is_outlier"] = df["is_outlier"].astype(np.str)

cat_features = np.where(test_df.dtypes=="object")[0]
cat_features


# In[ ]:


test_df.columns.values[cat_features]


# In[ ]:


train_df = train_df.fillna("NaN")
test_df = test_df.fillna("NaN")


# ## Validation strategy <a class="anchor" id="validation"></a>

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)


# ## Fitting <a class="anchor" id="fitting"></a>

# In[ ]:


class_weights = [1, train_df[train_df.target==0].shape[0] / train_df[train_df.target==1].shape[0]]
class_weights


# In[ ]:


params = {
    'iterations': 4000,
    'learning_rate': 0.01,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': True,
    'loss_function': 'Logloss',
    'od_type': 'Iter',
    'od_wait': 1500,
    'l2_leaf_reg': 100,
    'depth': 5,
    'rsm': 0.6,
    'random_strength': 2,
    'bagging_temperature': 10,
    'class_weights': class_weights,
    'random_seed': 0
}


# In[ ]:


test_pred = np.zeros(len(test_df))
oof = pd.DataFrame(data=np.zeros(len(train_df)), index=train_df.index.values, columns=["pred"])
feature_importance_df = pd.DataFrame(index=use_features)

m = 0
for train_idx, dev_idx in skf.split(train_df.drop("target", axis=1).index.values, train_df.target.values):
    
    x_train, x_dev = train_df.loc[train_idx].drop("target", axis=1), train_df.loc[dev_idx].drop("target", axis=1)
    y_train, y_dev = train_df.loc[train_idx].target, train_df.loc[dev_idx].target
    
    train_pool = Pool(x_train, y_train, cat_features=cat_features)
    dev_pool = Pool(x_dev, y_dev, cat_features=cat_features)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=dev_pool, plot=True)
    
    oof.loc[dev_idx, "pred"] = model.predict_proba(x_dev)[:,1]
    test_pred += model.predict_proba(test_df)[:,1]/N_SPLITS
    feature_importance_df.loc[:, "fold_" + str(m)] = model.get_feature_importance(train_pool)
    m+=1


# In[ ]:


feature_importance_df["mean"] = feature_importance_df.mean(axis=1)
feature_importance_df["std"] = feature_importance_df.std(axis=1)
feature_importance_df = feature_importance_df.sort_values(by="mean", ascending=False)


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(x=feature_importance_df["mean"].values,
            y=feature_importance_df.index.values, palette="Greens_r");
plt.title("Feature importances");
plt.show()


# ## Feature importances <a class="anchor" id="importances"></a>

# In[ ]:


kind=None
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(dev_pool)
if kind=="bar":
    shap.summary_plot(shap_values, x_dev, plot_type="bar")
else:
    shap.summary_plot(shap_values, x_dev)


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(test_pred, ax=ax[0], kde=False, color="mediumseagreen")
ax[0].set_title("Test predicted probability for having cancer")
sns.distplot(oof.pred, ax=ax[1], kde=False, color="red");
ax[1].set_title("Oof predicted probability for having cancer")
for n in range(2):
    ax[n].set_xlabel("predicted probability of class 1")
    ax[n].set_ylabel("count")


# In[ ]:


train_info["oof"] = oof.pred


# In[ ]:


train_info["gmm_cluster"] = train_image_stats.gmm_cluster_label
train_info["gmm_logL"] = train_image_stats.gmm_logL


# In[ ]:


spread_of_pos_targets = train_df.groupby("gmm_cluster_label").target.sum() / train_df.target.sum() * 100
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.barplot(spread_of_pos_targets.index, spread_of_pos_targets.values, ax=ax[0], palette="husl");
ax[0].set_title("How malignant cases are distributed over clusters");
ax[0].set_ylabel("% of 1-targets");
g = sns.countplot(train_df.gmm_cluster_label, hue=train_df.target, ax=ax[1], palette="Reds")
g.set_yscale("log")
ax[1].set_title("Class imbalance per cluster");


# In[ ]:


plt.figure(figsize=(20,5))
sns.violinplot(train_info.gmm_cluster, train_info.oof, palette="Blues")


# ## Submission <a class="anchor" id="submission"></a>

# In[ ]:


submission = pd.read_csv(basepath + "sample_submission.csv")


# In[ ]:


submission.target = test_pred


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index=False)


# ## What you can try out!
# 
# This is just a sketch how you can find image groups and it would probably lead to some ideas how you like to preprocess the images or to perform augmentations. Furthermore you have obtained more meta_features that could be useful for working with tabular data. I would have many ideas for this competition but I won't be able to try them out. I don't like to share them all as this would spoil the fun of this competition for all having similar ideas. But there are at least 2 very obvious experiments you could try out:
# 
# 1. Perform a proper preprocessing for all image stats features to obtain clusters of higher quality and play with hyperparameters.
# 2. Compute image statistics for external data and do some EDA and clustering with or/and without the competition data to understand why or if it's helpful to be added (not only for better class imbalance). 
# 
# Have a lot of fun! :-)
