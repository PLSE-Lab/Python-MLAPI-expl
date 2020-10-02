#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import os
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from covid19_functions import *


# <h1><ins>Task: Create Vector and Cluster Features</ins></h1><br>
# 
# Here we aim to use a pretrained doc2vec model to engineer features from medical documents that will allow a neural network to decide whether they contain information relating to coronavirus risk. 
# 
# The notebook runs as follows:
# 
# 1. Set up custom functions.
# 2. Load the data. 
# 3. Testing lookup function.
# 4. K-Means cluster evalutaion: Silhouette Score.
# 5. K-Means feature engineering.
# 6. Cluster contents.
# 7. Embedding visualisation coloured on cluster label. 
# 
# --------------------------------------

# <h3><ins>1. Custom Functions</ins></h3><br>
# 
# All custom functions imported from our utility script. 
# 
# --------------------------
# 
# 

# <h3><ins>2. Load The Data</ins></h3><br>
# 
# Data is loaded from the covid `.csv` file created in a previous kernel.
# 
# ---------------------------------------

# In[ ]:


corona_data = pd.read_csv("../input/kagglecovid19/kaggle_covid-19.csv")
corona_data = corona_data.drop(columns=['abstract'])
corona_data = corona_data.fillna("Unknown")
corona_data['risk_label'] = 'Unlabelled'


# ------------------------
# 
# Also loaded is the doc2vec model, trained on the whole corpus of documents. This model is used to create our vectored features. 
# 
# -----------------------
# 
# <i>To Note: Different embedding models may improve cluster silhouette score</i>
# 
# ------------------------
# 

# In[ ]:


coronas_d2v_model = Doc2Vec.load("../input/covidvectors/COVID_MEDICAL_DOCS_w2v_MODEL.model")


# <h3><ins>3. Lookup Function</ins></h3><br>
# 
# 
# As a means of assessing the clustering we have manually assigned labels to the data in a prevous notebook. These labels are based on the simple lookup function demonstrated below. 
# 
# Once vectorised and clustered, the instances will be plotted in 3d (based on their vectors, reduced using PCA). 
# 
# Two plots are created; one in which instances are coloured according to cluster label, and the other coloured according to the manually assigned label. 
# 
# Through comparison of these two plots we can assess how well K-Means has clustered our data. 
# 
# ------------------------------------------------------
# 

# -----------------------------------------
# 
# First, the lookup function finds all documents containing a keyword. 
# 
# 
# -------------------------------------

# In[ ]:


doc_folder = {"risk": return_doc_index("risk", corona_data),

              "preg": return_doc_index("pregnant", corona_data),

               "smoking": return_doc_index("smoking", corona_data),

               "co_infection": return_doc_index("co infection", corona_data),

                "neonates": return_doc_index("neonates", corona_data),

               "transmission": return_doc_index("transmission dynamics", corona_data),

                "high_risk": return_doc_index("high-risk patient", corona_data)
             }


# -----------------------------
# 
# A quick sanity check ensures the lookup function actually found some documents. 
# 
# -----------------------------------------

# In[ ]:


print(f"Number of Documents that Mention Risk: {len(doc_folder['risk'][0])}")

print(f"Number of Documents that Mention Pregnancy: {len(doc_folder['preg'][0])}")

print(f"Number of Documents that Mention Smoking: {len(doc_folder['smoking'][0])}")

print(f"Number of Documents that Mention Neonates: {len(doc_folder['neonates'][0])}")

print(f"Number of Documents that Mention Transmission Dynamics: {len(doc_folder['transmission'][0])}")

print(f"Number of Documents that Mention High Risk Patients: {len(doc_folder['high_risk'][0])}")


# In[ ]:


doc_folder['risk'][0]


# In[ ]:


corona_data = assign_label(doc_folder['risk'][0], corona_data, "risk")
corona_data = assign_label(doc_folder['preg'][0], corona_data, "preg")
corona_data = assign_label(doc_folder['smoking'][0], corona_data, "smoking")
corona_data = assign_label(doc_folder['neonates'][0], corona_data, "neonates")
corona_data = assign_label(doc_folder['transmission'][0], corona_data, "transmission")
corona_data = assign_label(doc_folder['high_risk'][0], corona_data, "high_risk")


# ----------------
# 
# Lastly, we encode the manually assigned labels to allow the 3d plot generator to colour instances based on this attribute.
# 
# ---------------

# In[ ]:


le = LabelEncoder()
corona_data['risk_label_encode'] = le.fit_transform(corona_data['risk_label'])


# <h3><ins>4. K-Means: Silhouette Score</ins></h3><br>
# 
# In order to properly conduct K-Means clustering we have to find out the optimum number of clusters. While the elbow method is one way to calculate this value, a better technique is to use the silhouette score. 
# 
# 
# The silhouette score of a model is the mean silhouette coefficient over all instances. An instances silhouette coefficient can be calucalted as:
# 
# `(b - a) / max(a, b)`
# 
# Where:
# 
# `a` = The intra-cluster distance (the mean distance to other instances in the same cluster).
# 
# `b` = The nearest-cluster distance (the mean distance to other instances in the next nearest cluster).
# 
# Silhouette coefficients can vary from -1 to 1, with a score closer to 1 meaning that instances are well within their own cluster and far from other clusters. A score closer to 0 means that it is close to the cluster boundary, while a score of -1 may mean an instance has been assigned to the wrong cluster. 
# 
# First, we create the vectors using the doc2vec model and build them into their own dataframe. 
# 
# ---------------------------------------------------

# In[ ]:


corona_data['title_vector'] = corona_data['title'].apply(create_body_vector, args=[coronas_d2v_model])


# In[ ]:


vectors = [x for x in corona_data['title_vector']]


# In[ ]:


vec_df = pd.DataFrame(vectors)


# -------------------
# 
# Once we're done with the `title vector` column we can just drop it. 
# 
# 
# ---------------------------
# 

# In[ ]:


corona_data = corona_data.drop(columns=['title_vector'])


# ------------------------
# 
# We use `MinMax Scaling` before clustering
# 
# ------------------------------

# In[ ]:


scaler = MinMaxScaler()
vec_df_s = scaler.fit_transform(vec_df)


# ---------------------------
# 
# <b><ins>Silhouette Plot</ins></b>
# 
# Then we call the function to start the silhouette analysis. 
# 
# We iteratively try a range of clusters. 
# 
# First it will output the silhouette score for the model, then two plots follow;
# 
# 1. <i>Silhouette Diagram:</i> The dashed red line represents the mean silhouette score for that model. When most instances have a lower coefficient than this line it indicates poor clustering, meaning the instances are too close together. 
# 
# 
# 2. <i>Visualisation Data:</i> Each instance visualised in cluster space, coloured on cluster label and highlighting each cluster centroid.
# 
# 
# Using these graphs it was determined that `8 clusters` was optimal. 
# 
# -------------------------------

# In[ ]:


# First we need to normalise the feature vectors before clustering,

silhouette_plot(vec_df_s, 2, 20)


# -------------------------
# 
# <b><ins>Optimal Weights</ins></b><br>
# 
# We find the optimal weights by using a similiar technique, only this time measuring the inertia of a small subset and saving the weight initialisations. Once an optimum weighting has been found, this is returned and used to train the final K-Means model on all the data.  
# 
# -------------------------------

# In[ ]:


best_cents = return_opt_weights(vec_df_s)


# In[ ]:


kmeans_optimised = KMeans(n_clusters=15, init=best_cents, max_iter=20)
kmeans_optimised.fit(vec_df_s)


# <h3><ins>5. K-Means feature engineering</ins></h3><br>
# 
# 
# Following from creating our cluster model we can now assign new features to the dataset. 
# 
# 1. <i>Cluster Label:</i> The cluster to which each instance belongs. 
# 
# 
# 2. <i>Distance To Cluser Centroid:</i> Distance of each cluster to every other cluster centroid.
# 
# 
# ---------------------------------------------

# <b><ins>Assign Cluster Labels</ins></b><br>
# 
# First we assign the labels to an attribute in the original dataframe. 
# 
# ----------------------

# In[ ]:


corona_data['cluster_labels'] = kmeans_optimised.labels_


# <b><ins>Testing The Distance Array</ins></b><br>
# 
# The K-Means object provides a handy method for returning an instances distance to each cluster centroid. To ensure this is working correctly I chose four random instances. 
# 
# Using `.transform()` I obtained an array with the distance to each centroid.
# 
# We then compared the index that held the lowest value (the cluster centroid closest to the instance) with the actual cluster label. This ensured that the list returned by `.transform()` was a true representation of the cluster space. 
# 
# -------

# In[ ]:


print(f"Instance 0 CLuster Label: {corona_data['cluster_labels'][0]}")
print(f"Instance 156 CLuster Label: {corona_data['cluster_labels'][9875]}")
print(f"Instance 5689 CLuster Label: {corona_data['cluster_labels'][5689]}")
print(f"Instance 12 CLuster Label: {corona_data['cluster_labels'][12]}")


# In[ ]:



reshaped_list = [
    
    (0, vec_df_s[0].reshape(-1, 1).T),
    (9875, vec_df_s[9875].reshape(-1, 1).T),
    (5689, vec_df_s[5689].reshape(-1, 1).T),
    (12, vec_df_s[12].reshape(-1, 1).T)
    
]



for r in reshaped_list:
    print('\n------------------------\n')
    ind_arr = list(kmeans_optimised.transform(r[1]))
    print(f"Instance {r[0]} Distance Array:\n\n{ind_arr}\n\n")
    print(f"Instance Actual CLuster Label: {corona_data['cluster_labels'][r[0]]}")
    print(f"Instance Lowest Distance Index: {np.argmin(ind_arr)}")
    print('\n-----------------------\n')


# <b><ins>Create Cluster Distance Features</ins></b>
# 
# Using the function we can now create a dataframe consisting of cluster distances and combine this with the vector dataframe for a full set of features. 
# 
# ----------------------

# In[ ]:


cluster_features = create_cluster_df(kmeans_optimised, vec_df_s)


# In[ ]:


cluster_features = rename_cluster_cols(cluster_features)


# In[ ]:


vec_df = rename_vec_df(vec_df)


# In[ ]:


num_covidDoc_repe = pd.concat([cluster_features, vec_df], axis=1)


# <h3><ins>Cluster Contents</ins></h3><br>
# 
# To get a rough idea of the contents of each cluster we grouped each of the instances together based on `cluster label`. Then, using the `print title` function above, we examined the subject of each document in the grouping. Rough topics are outlined below, however LDA or HDP will allow for automatic topic modelling and the assignment of meaningful labels to each grouping of documents. 
# 
# <ins>Estimated Topics</ins>
# 
# 1. <i>Virus Modelling</i>
# 
# 2. <i>Epidemiology</i>
# 
# 3. <i>Co-Infection</i>
# 
# 4. <i>Bio-Chemical Interactions</i>
# 
# 5. <i>Viral Evolution</i>
# 
# 6. <i>Unclear On This</i>
# 
# 7. <i>Virus Detection</i>
# 
# 
# <i>Please note these are rough estimations of the topics covered in each cluster.</i>
# 
# -----------------------------
# 
# 

# In[ ]:


doc_id_series = pd.Series(corona_data['doc_id'])
doc_source_series = pd.Series(corona_data['source'])

num_covidDoc_repe["doc_id"] = num_covidDoc_repe.insert(0, "doc_id", doc_id_series)
num_covidDoc_repe["source"] = num_covidDoc_repe.insert(0, "source", doc_id_series)
num_covidDoc_repe["cluster_label"] = num_covidDoc_repe.insert(0, "cluster_label", doc_id_series)


# In[ ]:


num_covidDoc_repe["doc_id"] = corona_data["doc_id"]
num_covidDoc_repe['source'] = corona_data['source']
num_covidDoc_repe['cluster_label'] = corona_data['cluster_labels']


# In[ ]:


clust_0_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==0]
clust_1_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==1]
clust_2_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==2]
clust_3_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==3]
clust_4_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==4]
clust_5_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==5]
clust_6_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==6]
clust_7_d = num_covidDoc_repe[num_covidDoc_repe['cluster_label']==7]

ind_0 = list(clust_0_d.index)
ind_1 = list(clust_1_d.index)
ind_2 = list(clust_2_d.index)
ind_3 = list(clust_3_d.index)
ind_4 = list(clust_4_d.index)
ind_5 = list(clust_5_d.index)
ind_6 = list(clust_6_d.index)
ind_7 = list(clust_7_d.index)


# In[ ]:


# We can see that cluster 1 documents deal with virus modelling. 

# print_doc_title(corona_data, ind_0)


# In[ ]:


# CLuster 1 appears to deal with EPidemiology 

# print_doc_title(corona_data, ind_1)


# In[ ]:


# Think cluister 2 appears to be about co-infection 

# print_doc_title(corona_data, ind_2)


# In[ ]:


# cLUSTER 3 appears to be about bio-checmical interactions

# print_doc_title(corona_data, ind_3)


# In[ ]:


# CLuster 4 appears to deal with viral evolution. 

# print_doc_title(corona_data, ind_4)


# In[ ]:


# Unsure about cluster 5, maybe some domain knowldge would help. 

# print_doc_title(corona_data, ind_5)


# In[ ]:


# Cluster 6 is mostly about detection 

# print_doc_title(corona_data, ind_6)


# In[ ]:


# CLuster 7 appears to be mostly about Transmission data. 

# print_doc_title(corona_data, ind_7)


# <h3><ins>7. Plot The Results</ins></h3><br>
# 
# As a means of visualising how well the algorithm has clustered the data, we reduced the dimensions to visualise the instance vectors. 
# 
# Using this approach we created two plots:
# 
# 1. <i>Coloured On Manual Label</i>
# 
# 
# 2. <i> Coloured On Cluster Label</i>
# 
# 
# As can be seen from the graph, compared with our manual approach, and the cursory glance at the cluster contents, the algorithm has grouped the documents quite well. 
# 
# Further analysis to automatically model the topics in these clusters is required. 
# 
# ------------------------------------

# In[ ]:


pca = PCA(n_components=3)
three_d_vectors = pca.fit_transform(vec_df_s)


# In[ ]:


pca_df = pd.DataFrame()
pca_df['pca_one'] = three_d_vectors[:, 0]
pca_df['pca_two'] = three_d_vectors[:, 1]
pca_df['pca_three'] = three_d_vectors[:, 2]


# In[ ]:


plot_vectors(pca_df, 'cluster', corona_data)


# In[ ]:


plot_vectors(pca_df, 'risk_label', corona_data)

