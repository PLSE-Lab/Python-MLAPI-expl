#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))


# In[ ]:


import matplotlib as mpl

mpl.rcParams['figure.figsize'] = [15, 7]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100

mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'


# In[ ]:


get_ipython().system('head -n 5 ../input/train.csv')


# In[ ]:


get_ipython().system('wc -l ../input/train.csv')


# In[ ]:


data = pd.read_csv("../input/train.csv", nrows=100000)


# # Goals:
# - Hotel_clusters are dependent on which categorical variable?
#     - Graph the top 4
# - Hotel_clusters are dependent on which numerical variable?
# - How correlated are the user's country to the hotel's country? Continent?
# 
# # Models to check out
# - The closest 5 hotels in the latent space clicked by each person should be recommended at the top 5 (Content-based filtering)
# - Create a pairwise ranking matrix factorization model of user to hotel cluster (Collaborative filtering)
# - Factorization machine of the dependent categoricals and numericals (Hybrid)

# In[ ]:


CATEGORICALS = ["site_name", "posa_continent", "user_location_country", "user_location_region", "user_location_city", "is_mobile", "is_package", "channel", 
               "srch_destination_type_id", "hotel_continent", "hotel_country", "hotel_market", "srch_destination_id"]
NUMERICALS = ["orig_destination_distance", "srch_adults_cnt", "srch_children_cnt", "srch_rm_cnt", "cnt"]
USER_ID = "user_id"
IS_BOOKING = "is_booking"
HOTEL_CLUSTER = "hotel_cluster"


# In[ ]:


ax = data[HOTEL_CLUSTER].value_counts().plot.bar(color='dodgerblue')
ax.set_xticklabels([])
plt.title("Hotel clusters' clicks - Looks like the beginnings of a power-law distribution");


# In[ ]:


data_cat_dummies = pd.get_dummies(pd.get_dummies(data[CATEGORICALS].astype('category')))


# In[ ]:


def name_scores(featurecoef, col_names, label="Score", sort=False):
    df_feature_importance = pd.DataFrame([dict(zip(col_names, featurecoef))]).T.reset_index()
    df_feature_importance.columns = ["Feature", label]
    if sort:
        return df_feature_importance.sort_values(ascending=False, by=label)
    return df_feature_importance


# In[ ]:


from sklearn.feature_selection import chi2

sample_n = 10000
data_cat_dummies_sample = data_cat_dummies.sample(sample_n)
chi2_scores = chi2(data_cat_dummies_sample, data[HOTEL_CLUSTER].loc[data_cat_dummies_sample.index])
df_chi2_scores = name_scores(chi2_scores[0], data_cat_dummies_sample.columns)
df_chi2_scores = df_chi2_scores.sort_values(by="Score", ascending=False)


# In[ ]:


# get the top 100 features and graph the categorical
n = 100
top_n_features = df_chi2_scores[:n]["Feature"]


# In[ ]:


top_n_features.apply(lambda s : ' '.join(s.split("_")[:len(s.split("_"))-1])).value_counts()


# * The graph below gets the features whose dummy variables contain the most entries in the top 100 by chi2 scores.
# * Let's graph hotel market, country and user's city with respect to the hotel clusters. 
# * We'll dedicate a section for srch_destination_id, but that seems very relevant.
# * For ease of readability, we'll only get the top 5 hotel_clusters, their market, country and user location.

# ## Some basic findings
# - Looks like user city ~17 and ~90 is very active all throughout the different hotel clusters
# - Looks like hotel_market ~59 and 62 are related to many hotel_clusters
# - Looks like hotel_country ~8 is related to many hotel_clusters
# 

# In[ ]:


def create_matrix(data, group_column, val_column):
    grouped_data = data.groupby(group_column)[val_column].value_counts()
    grouped_data = grouped_data.groupby(level=0).nlargest(3)
    grouped_data.index = grouped_data.index.droplevel(0)
    # transfrom to a square matrix
    grouped_data_unstacked = grouped_data.unstack()
    return grouped_data_unstacked.fillna(0)


# In[ ]:


hotel_user_city_vc_matrix = create_matrix(data, "hotel_cluster", "user_location_city")
hotel_market_vc_matrix = create_matrix(data, "hotel_cluster", "hotel_market")
hotel_country_vc_matrix = create_matrix(data, "hotel_cluster", "hotel_country")
hotel_continent_vc_matrix = create_matrix(data, "hotel_cluster", "hotel_continent")

# hotel user city
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

axes[0][0].set_title("Looks like user city ~17 and ~90 is very active\n all throughout the different hotel clusters")
axes[0][0].imshow(hotel_user_city_vc_matrix, cmap='gray')

axes[0][1].set_title("Looks like hotel_market ~59 and 62 are\nrelated to many hotel_clusters")
axes[0][1].imshow(hotel_market_vc_matrix, cmap='gray')

axes[1][0].set_title("Looks like hotel_country ~8 is\nrelated to many hotel_clusters (France?)")
axes[1][0].imshow(hotel_country_vc_matrix, cmap='gray');

axes[1][1].set_title("I think hotel_continent 1 is Europe!")
axes[1][1].imshow(hotel_continent_vc_matrix, cmap='gray');


# # Numericals
# - Since orig_destination_distance has null values, let's see its distribution then decide on the imputation method.

# In[ ]:


sns.distplot(data["orig_destination_distance"].dropna())

plt.title("Majority of destination places are close to the origin. \nThus, let's just use the median for imputation.");


# In[ ]:


sample_n = 10000
data_numericals_sample = data[NUMERICALS].sample(sample_n)
chi2_scores = chi2(data_numericals_sample.fillna(data_numericals_sample.median()), data[HOTEL_CLUSTER].loc[data_numericals_sample.index])
df_chi2_scores = name_scores(chi2_scores[0], data_numericals_sample.columns)

df_chi2_pvalues = name_scores(chi2_scores[1], data_numericals_sample.columns)
df_chi2_pvalues.columns = ["Feature", "PValue"]

df_chi2_scores.merge(df_chi2_pvalues).sort_values(by="Score", ascending=False)


# ### Hypotheses:
# - Distance is important to travelers. Hotel clusters seem to be dependent on the distance.
# - Certain hotel clusters seem to be children friendly
# - Certain hotel_clusters seem also to rake in "bandwagoners" (thru cnt variable)

# In[ ]:


sns.boxplot(data=data, x=HOTEL_CLUSTER, y="orig_destination_distance")

plt.title("There's a lot of outlier distances for each hotel_cluster");


# In[ ]:


data_cluster = data[[HOTEL_CLUSTER]].copy()
data_cluster["cluster_num"] = pd.cut(data_cluster[HOTEL_CLUSTER], bins=4, labels=range(4))


# In[ ]:


clusters_1 = data[data[HOTEL_CLUSTER].isin(data_cluster.loc[data_cluster["cluster_num"] == 0, HOTEL_CLUSTER])]
clusters_1_unstacked = clusters_1.groupby(HOTEL_CLUSTER)["srch_children_cnt"].value_counts().unstack().fillna(0)

clusters_1_unstacked= clusters_1_unstacked.drop(0, axis=1).sort_values(by=2,)
ax = clusters_1_unstacked.plot.barh(stacked=True)
ax.legend(loc='upper right')

plt.title("Some hotel_clusters are more perceived to be family friendly", );


# # What is the relationship of srch_destination_id to hotel_clusters?

# In[ ]:


print("Search destination id's nunique: ", data["srch_destination_id"].nunique())
print("Search destination types nunique: ", data["srch_destination_type_id"].nunique())


# ## For each hotel cluster, how is srch type related?

# In[ ]:


hotel_search_type_matrix = create_matrix(data, HOTEL_CLUSTER, "srch_destination_type_id")
print("Looks like the search types tend around 1 and 6.")
display(hotel_search_type_matrix[:10])
plt.imshow(hotel_search_type_matrix, cmap='gray');


# ## For each hotel cluster, how is search_id related. How many unique search ids are there for each hotel cluster?
# 
# - Seems like there is a non-trivial amount of search id of just 1. These may be the "weird" searches that corresponded to some hotel clusters.
# - On the other hand, there's a lot of search ids per hotel cluster. Seems we can work with this for our content-based filtering algorithm. We can spend time tweaking embeddings for each hotel clsuter. Or we can just average them out.

# In[ ]:


hotel_to_search_n = data.groupby(HOTEL_CLUSTER)["srch_destination_id"].nunique()
ax = sns.distplot(hotel_to_search_n, bins=50, kde=False)
ax.set_xlabel("Number of unique search IDs per hotel cluster")

display(data.groupby(HOTEL_CLUSTER)["srch_destination_id"].nunique().describe().to_frame(
    "Stats of the number of unique search IDs per hotel cluster"))


# # Latent space variables look like a 10-20 'expressed' variables out of a hundred fifty

# In[ ]:


destinations = pd.read_csv("../input/destinations.csv")


# In[ ]:


def create_latent_search_img(index, ax):
    # to make the image 10x15, we create a 150th feature with the mean of the array
    img = np.array(destinations.loc[index].values[1:].tolist() + [destinations.loc[index].values[1:].mean()])
    img = img.reshape((15,10))
    sns.heatmap(img, cmap='gray', ax=ax)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax


# In[ ]:


ax1 = create_latent_search_img(0, plt.subplot(2, 3, 1))
ax2 = create_latent_search_img(1, plt.subplot(2, 3, 2))
ax3 = create_latent_search_img(2, plt.subplot(2, 3, 3))
ax4 = create_latent_search_img(3, plt.subplot(2, 3, 4))
ax5 = create_latent_search_img(3, plt.subplot(2, 3, 5))
ax6 = create_latent_search_img(3, plt.subplot(2, 3, 6))


# # Let's average out the search ids per hotel cluster
# - Keep in mind the minimum number of unique search ids per hotel_cluster is 5, so we should have unique "feature images" per hotel_cluster.
# - When we search, we may click or not click on a hotel cluster. We take into account that act of clicking, which means the more search ids associated to a hotel_cluster, the more expressed the variables of those ids should be.
#    - We can either make this a sum or weighted mean aggregation operation. This can be for exploration in the modeling stage.
#    - For now, we'll do a sum.

# In[ ]:


def get_latent_search_hotel_array(hotel_cluster_index):
    values = data.loc[data[HOTEL_CLUSTER] == hotel_cluster_index, "srch_destination_id"].to_frame().merge(destinations)
    values = values.drop("srch_destination_id", axis=1)
    values = values.sum()
    return values

def create_latent_search_hotel_image(hotel_cluster_index, ax):
    img = get_latent_search_hotel_array(hotel_cluster_index)
    img = np.array(img.tolist() + [img.mean()])
    img = img.reshape((15,10))
    
    ax.imshow(img, cmap='gray')
    ax.set_title("Cluster " + str(hotel_cluster_index))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax


# In[ ]:


hotel_arrays = []
for i in range(data[HOTEL_CLUSTER].nunique()):
    hotel_arrays.append(get_latent_search_hotel_array(i))
hotel_arrays = np.array(hotel_arrays)


# In[ ]:


from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

Z = hierarchy.linkage(hotel_arrays, 'single')
plt.figure(figsize=(20, 8))
dn = hierarchy.dendrogram(Z)

# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tick_params(axis='both', which='minor', labelsize=10)
plt.xticks(rotation=0)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(20, 20))

for i in range(data[HOTEL_CLUSTER].nunique()):
    create_latent_search_hotel_image(i, fig.add_subplot(10, 10, i+1))


# # Analysis:
# - There's some minute differences across clusters although there are the same latent variables that are always expressed.
# - From the dendrogram, some clusters are very close while those on the left seem very unique.
# 
# # Stats for Collaborative Filtering
# - Looks like we can do CF - Matrix Factorization here. Non-zero percentage is 28%, very trivial for CF.
# - We should use dimensionality lower than 100 as to not overfit.

# In[ ]:


user_id_col = "user_id"
item_id_col = HOTEL_CLUSTER
ratings = data[[user_id_col, item_id_col]]

num_users = ratings[user_id_col].nunique()
num_items = ratings[item_id_col].nunique()
possible_combinations = num_users * num_items
nnz = len(ratings)
nnz_percent = nnz / possible_combinations

print("Num Users:", num_users)
print("Num Items:", num_items)
print("Sparsity:", nnz_percent)
print("Not very sparse. CF will work wonders here.")

# average number of hotel_clusters per user
hotel_per_user = ratings.groupby(user_id_col)[item_id_col].nunique()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(211)
sns.distplot(hotel_per_user, kde=False, ax=ax)
ax.set_title("Mean number of clusters per user: {:.2f}".format(hotel_per_user.mean()))

# # average number of users per hotel
user_per_hotel = ratings.groupby(item_id_col)[user_id_col].nunique()
ax = fig.add_subplot(212)
sns.distplot(user_per_hotel, kde=False, ax=ax)
ax.set_title("Mean number of users per cluster: {:.2f}".format(user_per_hotel.mean()))

fig.tight_layout()

