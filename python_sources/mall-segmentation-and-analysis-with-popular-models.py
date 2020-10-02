#!/usr/bin/env python
# coding: utf-8

# # Import and Functions

# In[ ]:


get_ipython().system('conda install -c conda-forge --yes hdbscan')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,MeanShift,estimate_bandwidth,AffinityPropagation,SpectralClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram,linkage,ward,complete,single,fcluster
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap
import hdbscan
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


def kmeans_cluster_decision(X,point,min_clusters=2,max_clusters=10):
    inertia = []
    score = []
    ticks = []
    for cluster in range(min_clusters,max_clusters):
        model = KMeans(n_clusters=cluster)
        model.fit(X)
        inertia.append(model.inertia_)
        score.append(silhouette_score(X,model.labels_))
        ticks.append(cluster)

    fig,ax = plt.subplots(1,2,figsize=(18,4))
    sns.lineplot(x=ticks,y=inertia,ax=ax[0]).set_title("Inertia vs Clusters")
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels(ticks)
    ax[0].plot(point,inertia[point - min_clusters],marker='x',c='r',markersize=6)
    sns.lineplot(x=ticks,y=score,ax=ax[1]).set_title("Sil. Score vs Clusters")
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(ticks)
    ax[1].plot(point,score[point - min_clusters],marker='x',c='r',markersize=6)
    plt.show()

def draw_analysis_graphs(labels):
    fig,ax=plt.subplots(2,3,figsize=(18,14))
    sns.scatterplot(x='age',y='income',data=dfdata,hue=labels,palette='Paired',ax=ax[0][0])
    sns.scatterplot(x='age',y='spending',data=dfdata,hue=labels,palette='Paired',ax=ax[0][1])
    sns.scatterplot(x='spending',y='income',data=dfdata,hue=labels,palette='Paired',ax=ax[0][2])
    sns.swarmplot(x=labels,y='age',data=dfdata,hue=labels,palette='Paired',ax=ax[1][0])
    sns.swarmplot(x=labels,y='spending',data=dfdata,hue=labels,palette='Paired',ax=ax[1][1])
    sns.swarmplot(x=labels,y='income',data=dfdata,hue=labels,palette='Paired',ax=ax[1][2])
    plt.show()

def scaler(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# # Get Data

# In[ ]:


dfdata = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
dfdata.head()


# In[ ]:


dfdata.describe()


# In[ ]:


dfdata.columns = ['id','gender','age','income','spending']
dfdata.info()


# ## Analysis of Data

# In[ ]:


plt.figure(figsize=(10,2))
sns.countplot(y = 'gender',data = dfdata)
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.pairplot(data = dfdata[['age','income','spending','gender']],hue='gender',diag_kind='kde')
plt.show()


# - Gender has no effect on distribution of other features. So, gender is excluded from our feature set.

# In[ ]:


plt.figure(figsize=(18,6))
sns.countplot(x='age',data=dfdata,hue='gender')
plt.title("Age vs Gender Distribution")
plt.show()


# - Distribution changes dramatically at ages 25,45 and 55 respectively.
# - Mid aged females are more frequent visitors with respect to mid aged males. But still the distinction is not exact with the graph

# In[ ]:


plt.figure(figsize=(18,6))
sns.countplot(x='income',data=dfdata,hue='gender')
plt.title("Income vs gender distribution")
plt.show()


# - Income distribution changes at 40 and 80. At all classes, females are more frequent visitors

# In[ ]:


plt.figure(figsize=(18,6))
sns.countplot(x='spending',data=dfdata,hue='gender')
plt.title("Spending score vs gender distribution")
plt.show()


# - Spending distribution changes at 40,65 and 80 respectively. Low spending males are more frequent visitors, while females are more frequent visitors at other spending values.
# - Assumptions of all clustering algorithms are scaling and normalization of data since we want to give equal chance to all features.
# - We will check the distribution of the data from TSNE graph to have an intuition for number of clusters.
# - After that, we will apply K-Means clustering algorithm to check clusters for all three features in addition to all combinations of two features
# - We will compare the results via inertia and silhouette score.

# # K-Means Algorithm<br>
# - Easy and simple algorithm
# - Captures only simple and convex shapes. Fails with dense distributions.
# - According to TSNE graph, it seems that we have convex and simple shape. Thats why we expect KMeans to be successful algorithm.
# - Kmodes for categorical data
# - MiniBatch K Means or BIRCH for big data

# ## Features : Age,Income,Spending

# In[ ]:


X_scaled = scaler(dfdata[['income','spending','age']])
kmeans_cluster_decision(X_scaled,6,2,10)


# In[ ]:


model = KMeans(n_clusters=6)
labels = model.fit_predict(X_scaled)
print("Inertia : ",round(model.inertia_,2),"Silhouette Score : ",round(silhouette_score(X_scaled,labels),2))
draw_analysis_graphs(labels)


# After KMeans clustering of all three features, with scatterplots and swarmplots above, we can we can easily conclude that our 6 clusters consist of the customers below.<br>Low-high values for all features are consistent with our previous graphs.
# - **label 0:** mid age - low spending  - high income customers
# - **label 1:** young   - high spending - low income customers
# - **label 2:** all age - low spending  - low income customers
# - **label 3:** mid age - high spending - high income customers
# - **label 4:** old     - mid spending  - mid income customers
# - **label 5:** young   - mid spending  - mid income customers

# ## Features : [Age,Income],[Age,Spending],[Income,Spending]

# In[ ]:


X_scaled_ai = scaler(dfdata[['age','income']])
X_scaled_as = scaler(dfdata[['age','spending']])
X_scaled_is = scaler(dfdata[['income','spending']])


# In[ ]:


kmeans_cluster_decision(X_scaled_ai,3,2,20)


# In[ ]:


model = KMeans(n_clusters=3)
labels_ai = model.fit_predict(X_scaled_ai)
print("Inertia : ",round(model.inertia_,2),"Silhouette Score : ",round(silhouette_score(X_scaled_ai,labels_ai),2))
draw_analysis_graphs(labels_ai)


# In[ ]:


kmeans_cluster_decision(X_scaled_as,6,2,20)


# In[ ]:


model = KMeans(n_clusters=6)
labels_as = model.fit_predict(X_scaled_as)
print("Inertia : ",round(model.inertia_,2),"Silhouette Score : ",round(silhouette_score(X_scaled_as,labels_as),2))
draw_analysis_graphs(labels_as)


# In[ ]:


kmeans_cluster_decision(X_scaled_is,5,2,20)


# In[ ]:


model = KMeans(n_clusters=5)
labels_is = model.fit_predict(X_scaled_is)
print("Inertia : ",round(model.inertia_,2),"Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_is),2))
draw_analysis_graphs(labels_is)


# After KMeans clustering according to **income** and **spending** features, we can we can easily conclude that our 6 clusters consist of the customers below.<br>Low-high values for all features are consistent with our previous graphs.<br>
# **label 0:** low spending  - high income customers <br>
# **label 1:** low spending  - low income customers<br>
# **label 2:** mid spending  - mid income customers<br>
# **label 3:** high spending - high income customers<br>
# **label 4:** high spending - low income customers

# After trials with 2 features, we conclude that clustering with respect to **income** and **spending** works better since inertia and silhouette score is by far the best scores. So, we will continue testing other algorithms by using only **income** and **spending**

# # Gaussian Mixture Clustering<br>
# - Normal distributed data
# - Consider mean and variance of data for clustering

# In[ ]:


score = []
xticks = []
for i in range(3,10):
    model = GaussianMixture(n_components=i).fit(X_scaled_is)
    labels = model.predict(X_scaled_is)
    xticks.append("i:" + str(i) + "c:" + str(len(np.unique(labels))))
    score.append(silhouette_score(X_scaled_is,labels))
plt.figure(figsize=(18,4))
sns.lineplot(x=range(len(score)),y=score)
plt.xticks(range(len(score)),xticks,rotation=45)
plt.show()


# In[ ]:


model = GaussianMixture(n_components=5).fit(X_scaled_is)
labels_gm = model.predict(X_scaled_is)
print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_gm),2))
draw_analysis_graphs(labels_gm)


# In[ ]:


pd.crosstab(labels_is,labels_gm)


# # Agglomerative Hierarchical Algorithm<br>
# - More successful than K-Means for complex shapes.
# - More informative with dendrograms
# - Still fails with complex density based distributions.
# - Connectivity and linkage parameters are important.
# - **Connectivity** : Points in clusters will be connected?
# - **linkage** : ward, single,average and complete. Ward is used generally

# In[ ]:


links = linkage(X_scaled_is,method='ward')
plt.figure(figsize=(20,6))
dendrogram(links,leaf_rotation=90,leaf_font_size=6)
plt.show()


# In[ ]:


score = []
for i in range(1,15):
    labels_dendro = fcluster(links,t=i,criterion='distance')
    score.append(silhouette_score(X_scaled_is,labels_dendro))
plt.figure(figsize=(18,5))
sns.lineplot(x=range(len(score)),y=score)
plt.show()


# In[ ]:


labels_dendro = fcluster(links,t=5,criterion = 'distance')
print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_dendro),2))
draw_analysis_graphs(labels_dendro)


# In[ ]:


score = []
for i in range(2,15):
    labels_agg = AgglomerativeClustering(n_clusters = i).fit(X_scaled_is)
    score.append(silhouette_score(X_scaled_is,labels_agg.labels_))
plt.figure(figsize=(18,5))
sns.lineplot(x=range(len(score)),y=score)
plt.show()


# In[ ]:


model = AgglomerativeClustering(n_clusters=5).fit(X_scaled_is)
labels_agg = model.labels_
print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_agg),2))
draw_analysis_graphs(labels_agg)


# # DBSCAN<br>
# - Good at density based distributions.
# - Hard to decide on correct eps and min_samples.
# - As eps increases, number of clusters decreases. As min_samples increases, number of clusters decreases.
# - Outlier values can be problem
# - Bad at varying density data. Use OPTICS instead.

# ![Density Based Distributions](https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png)

# In[ ]:


fig,ax=plt.subplots(3,3,figsize=(24,15))
for i,eps in enumerate([round(x*0.1,1) for x in range(4,7)]):
    for j,sample in enumerate(range(6,12,2)):
        model = DBSCAN(eps=eps,min_samples=sample).fit(X_scaled_is)
        labels = model.labels_
        sns.scatterplot(x = 'spending',y='income',hue=labels,data=dfdata,ax=ax[j][i],palette='Paired')
        title = "eps: " + str(eps) + " min_samples:" + str(sample)
        ax[j,i].set_title(title)
plt.show()


# In[ ]:


score = []
xlabels = []
for i,eps in enumerate([round(x*0.1,1) for x in range(3,9)]):
    for j,sample in enumerate(range(6,14)):
        model = DBSCAN(eps=eps,min_samples=sample).fit(X_scaled_is)
        labels = model.labels_
        if len(np.unique(labels)) > 1:
            score.append(silhouette_score(X_scaled_is,labels))
        else:
            score.append(0)
        xlabels.append('e: ' + str(eps) + ' s: ' + str(sample))
plt.figure(figsize=(18,5))
sns.lineplot(x=range(len(score)),y=score)
plt.xticks(range(len(xlabels)),xlabels,rotation = 90)
plt.show()


# # Mean Shift Algorithm <br>
# - Not scalable, not good at outliers
# - Number of clusters is not needed
# - Bandwidth is estimated via quantiles
# - Generally used in image classification
# - Used with kmeans clustering generally

# In[ ]:


score = []
for i in [x*0.01 for x in range(1,17)]:
    bandwidth = estimate_bandwidth(X_scaled_is, quantile=i)
    model = MeanShift(bandwidth).fit(X_scaled_is)
    score.append(silhouette_score(X_scaled_is,model.labels_))
plt.figure(figsize=(18,5))
sns.lineplot(x=range(len(score)),y=score)
plt.show()


# In[ ]:


bandwidth = estimate_bandwidth(X_scaled_is, quantile=0.15)
model = MeanShift(bandwidth).fit(X_scaled_is)
labels_ms = model.labels_
print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_ms),2))
draw_analysis_graphs(labels_ms)


#    # PCA Analysis<br>
#     - PCA is linear matrix factorization tool for feature elimination. Poor on sparse matrices. Need scaling of data
#     - While performing PCA, its important to hold high explained variance ratio, with experience its better to be between 80% - 90%.

# In[ ]:


selected_columns = ['age','income','spending','genderlabel']
dfdata['genderlabel'] = dfdata['gender'].replace(['Male','Female'],[0,1])
X_scaled_all = pd.DataFrame(scaler(dfdata[selected_columns]),columns = selected_columns)


# In[ ]:


score = []
for i in range(1,5):
    pca = PCA(n_components = i).fit(X_scaled_all)
    score.append(np.sum(pca.explained_variance_ratio_))
plt.figure(figsize=(10,5))
sns.lineplot(x=range(len(score)),y=score)
plt.xticks(range(len(score)),range(1,5))
plt.show()


# In[ ]:


score = []
pca_vars = []
for i in range(1,5):
    pca = PCA(n_components = i)
    X_pca = pca.fit_transform(X_scaled_all)
    for j in range(3,7):
        model = KMeans(n_clusters=j).fit(X_pca)
        score.append(silhouette_score(X_pca,model.labels_))
        pca_vars.append("pca:" + str(i) + "c:" + str(j))
plt.figure(figsize=(18,6))
sns.lineplot(x=range(len(score)),y=score)
plt.xticks(ticks=range(len(score)),labels=pca_vars,rotation=90)
plt.show()


# In[ ]:


pca = PCA(n_components = 3).fit(X_scaled_all)
X_pca = pca.transform(X_scaled_all)
sns.heatmap(pca.components_,annot=True,fmt='.1g')
plt.xticks(range(X_scaled_all.shape[1]),X_scaled_all.columns.values,rotation=45)
plt.show()


# In[ ]:


model = KMeans(n_clusters=6).fit(X_pca)
labels_pca = model.labels_
score.append(silhouette_score(X_pca,labels_pca))
print("Silhouette Score : ",round(silhouette_score(X_pca,labels_pca),2))
draw_analysis_graphs(labels_pca)


# # TSNE Analysis<br>
# - Graph based nonlinear feature elimination tool.
# - Calculate distance based on distance to neighbors.
# - Perplexity (btw 10 - 50) and learning rate (btw 10 - 200) are main hyper parameters. Difficult to find optimum parameters.

# In[ ]:


X_scaled = scaler(dfdata[['income','spending','age']])
tsne = TSNE(n_components=2,perplexity=15,learning_rate=450)
tsne_result=tsne.fit_transform(X_scaled)


# In[ ]:


sns.scatterplot(tsne_result[:,0],tsne_result[:,1])
plt.show()


# From 2d TSNE graph, it seems that we will have 4-6 clusters.<br>
# I tried perplexity and learning rate manually for perplexity values between 5-50 and learning rate 50-500.

# In[ ]:


fig,ax = plt.subplots(1,4,figsize=(24,6))
sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_is,palette = 'Paired',ax = ax[0])
sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_dendro,palette = 'Paired',ax = ax[1])
sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_agg,palette = 'Paired',ax = ax[2])
sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_ms,palette = 'Paired',ax = ax[3])
plt.show()


# With **Kmeans**, **Dendrogram**, **Agglomerative Clustering** and **Mean Shift Algorithm** we approximately found same silhouette score with same number of clusters. When we compare them on TSNE graphs, it seems that they are still approximately same. Now, its time to check results via crosstabs.

# In[ ]:


resultset = ['Kmeans','Dendro','Agg','MSA']
dfResult = pd.DataFrame(zip(labels_is,labels_dendro,labels_agg,labels_ms),columns=resultset)
for i in range(len(resultset)):
    for j in range(i + 1,len(resultset)):
        print("Results for " + resultset[i] + " and " + resultset[j])
        print(pd.crosstab(dfResult.iloc[:,i],dfResult.iloc[:,j]))


# - As expected, Dendro and Aggregated Clustering have same results, since they are same models. 
# - The difference between KMeans and Dendrogram are on clusters 0 and 4. 
# - Difference between KMeans and MSA is at clusters 1 and 2. 
# - Difference between MSA and Aggregated Clustering is at clusters 1 and 2 again.
# - According to scatter plots, we see that there are only 9 customers at cluster borders and they are the reason of difference between clustering algorithms. 

# # UMAP & HDBSCAN<br>
# - Non-linear graphical feature elimination tool. Distance is still calculated via neighbors.
# - n_neighbors is the main hyper parameter.
# - **HDBSCAN**: combine the power of agglomerated clustering and dbscan. More efficient in varying density data.

# In[ ]:


X_scaled = scaler(dfdata[['income','spending','age']])
umap_data = umap.UMAP(n_neighbors=20).fit_transform(X_scaled)
model = hdbscan.HDBSCAN(min_cluster_size=5).fit(umap_data)
labels_hdbscan = model.labels_
print("Silhouette Score : ",round(silhouette_score(X_scaled,labels_hdbscan),2))
fig,ax = plt.subplots(1,2,figsize=(18,5))
sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_hdbscan,palette = 'Paired',ax=ax[0])
sns.scatterplot(umap_data[:,0],umap_data[:,1],hue=labels_hdbscan,palette = 'Paired',ax=ax[1])
plt.show()


# In[ ]:


pd.crosstab(labels_is,labels_hdbscan)


# # Affinity Propagation<br>
# - Transformation of data to similarities matrix and clustering via similarities (availibilites and responsibilities).
# - damping between 0.5 and 1
# - Good at uneven cluster sizes, convex and non-conves geometry
# - Not scalable
# - Using Graph distance 
# - Used in image and biological clustering

# In[ ]:


score = []
for i in [round(x * 0.1,1) for x in range(5,10)]:
    model = AffinityPropagation(damping = i,random_state=12).fit(X_scaled_is)
    score.append(silhouette_score(X_scaled_is,model.labels_))
plt.figure(figsize=(18,5))
sns.lineplot(x=range(len(score)),y=score)
plt.xticks(range(len(score)),[round(x * 0.1,1) for x in range(5,10)])
plt.show()


# In[ ]:


model = AffinityPropagation(damping = 0.7,random_state=12).fit(X_scaled_is)
labels_ap = model.labels_
print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_ap),2))
sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_ap,palette = 'Paired')
plt.show()


# In[ ]:


pd.crosstab(labels_is,labels_ap)


# # Spectral Clustering<br>
# - Transformation of data to affinity matrix and clustering via affinity matrix.
# - Good at uneven cluster sizes, convex and non-conves geometry
# - Not scalable
# - Used in image clustering

# In[ ]:


score = []
for i in range(3,10):
    model = SpectralClustering(n_clusters=i).fit(X_scaled_is)
    score.append(silhouette_score(X_scaled_is,model.labels_))
plt.figure(figsize=(18,4))
sns.lineplot(x=range(len(score)),y=score)
plt.xticks(range(len(score)),range(5,10))
plt.show()


# In[ ]:


model = SpectralClustering(n_clusters = 8).fit(X_scaled_is)
labels_sc = model.labels_
print("Silhouette Score : ",round(silhouette_score(X_scaled_is,labels_sc),2))
sns.scatterplot(tsne_result[:,0],tsne_result[:,1],hue=labels_sc,palette = 'Paired')
plt.show()


# In[ ]:


pd.crosstab(labels_is,labels_sc)

