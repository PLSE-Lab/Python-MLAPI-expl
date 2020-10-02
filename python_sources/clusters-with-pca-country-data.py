#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv('/kaggle/input/Country-data.csv')


# In[ ]:


df


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize = (16,8))
sns.heatmap(df.corr(), annot = True)
plt.show()


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

col = df.columns 
dfnorm = scale.fit_transform(df.drop('country',1))


# In[ ]:


dfnorm = pd.DataFrame(dfnorm)


# In[ ]:


dfnorm.columns = df.drop('country',1).columns


# In[ ]:


dfnorm


# In[ ]:


from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


# In[ ]:


hopkins(dfnorm)


# In[ ]:


help(KMeans)


# In[ ]:


clust = KMeans(n_clusters= 5, max_iter= 50, random_state= 100)
clust.fit(dfnorm)


# In[ ]:


dfnorm['cluster']= pd.DataFrame(clust.labels_)


# In[ ]:


dfnorm


# In[ ]:


sns.barplot(x=dfnorm.cluster, y= dfnorm.income)
plt.show()


# In[ ]:


from sklearn.metrics import silhouette_score
sse = []

for k in range(2,15):
    km = KMeans(n_clusters= k, max_iter= 50, random_state= 100)
    km.fit(dfnorm)
    sse.append([k,silhouette_score(dfnorm, km.labels_)])
    


# In[ ]:


plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1])
plt.show()


# In[ ]:


ssd = []

for num in range(1,21):
    km2 = KMeans(n_clusters= num, max_iter= 50, random_state= 100)
    km2.fit(dfnorm)
    ssd.append(km2.inertia_)


# In[ ]:


plt.plot(ssd)


# In[ ]:


df1 = pd.DataFrame(dfnorm.groupby('cluster').child_mort.mean())


# In[ ]:


df2 = pd.DataFrame(dfnorm.groupby('cluster').exports.mean())
df3 = pd.DataFrame(dfnorm.groupby('cluster').health.mean())
df4 = pd.DataFrame(dfnorm.groupby('cluster').imports.mean())
df5 =pd.DataFrame(dfnorm.groupby('cluster').income.mean())
df6 =pd.DataFrame(dfnorm.groupby('cluster').inflation.mean())
df7 = pd.DataFrame(dfnorm.groupby('cluster').life_expec.mean())
df8 = pd.DataFrame(dfnorm.groupby('cluster').total_fer.mean())
df9 = pd.DataFrame(dfnorm.groupby('cluster').gdpp.mean())


# In[ ]:


final = pd.concat([pd.Series([0,1,2,3,4]), df1, df2, df3, df4, df5, df6, df7, df8 ,df9], axis = 1)


# In[ ]:


final.columns = ['cluster','child_mort', 'exports', 'health', 'imports', 'income', 'inflation',
       'life_expec', 'total_fer', 'gdpp']


# In[ ]:


final


# In[ ]:


plt.figure(figsize = (18,10))

plt.subplot(3,3,1)
sns.barplot(final.cluster, final.gdpp)
plt.subplot(3,3,2)

sns.barplot(final.cluster, final.total_fer)
plt.subplot(3,3,3)

sns.barplot(final.cluster, final.life_expec)
plt.subplot(3,3,4)

sns.barplot(final.cluster, final.inflation)
plt.subplot(3,3,5)

sns.barplot(final.cluster, final.income)
plt.subplot(3,3,6)

sns.barplot(final.cluster, final.imports)
plt.subplot(3,3,7)

sns.barplot(final.cluster, final.health)
plt.subplot(3,3,8)

sns.barplot(final.cluster, final.exports)
plt.subplot(3,3,9)

sns.barplot(final.cluster, final.child_mort)

plt.show()


# In[ ]:


sns.barplot(final.cluster, final.total_fer)


# In[ ]:


sns.barplot(final.cluster, final.life_expec)


# In[ ]:


sns.barplot(final.cluster, final.inflation)


# In[ ]:


sns.barplot(final.cluster, final.income)


# In[ ]:


sns.barplot(final.cluster, final.imports)


# In[ ]:


sns.barplot(final.cluster, final.health)


# In[ ]:


sns.barplot(final.cluster, final.exports)


# In[ ]:


sns.barplot(final.cluster, final.child_mort)


# In[ ]:





# In[ ]:


df['cluster']= pd.DataFrame(clust.labels_)


# In[ ]:


df[df.cluster == 2]


# In[ ]:


from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree


# In[ ]:


mergings = linkage(dfnorm, method = "single", metric='euclidean')
dendrogram(mergings)
plt.show()


# In[ ]:


plt.figure(figsize = (18,10))

mergings = linkage(dfnorm, method = "complete", metric='euclidean')
dendrogram(mergings)
plt.show()


# In[ ]:


dff = dfnorm.drop('cluster', 1)


# In[ ]:


clusterCut = pd.Series(cut_tree(mergings, n_clusters = 6).reshape(-1,))


# In[ ]:


hie = pd.concat([clusterCut,dff], 1)


# In[ ]:





# In[ ]:


hie.columns = ['cluster', 'child_mort', 'exports', 'health', 'imports', 'income',
       'inflation', 'life_expec', 'total_fer', 'gdpp' ]


# In[ ]:


dff2 = pd.DataFrame(hie.groupby('cluster').exports.mean())
dff3 = pd.DataFrame(hie.groupby('cluster').health.mean())
dff4 = pd.DataFrame(hie.groupby('cluster').imports.mean())
dff5 =pd.DataFrame(hie.groupby('cluster').income.mean())
dff6 =pd.DataFrame(hie.groupby('cluster').inflation.mean())
dff7 = pd.DataFrame(hie.groupby('cluster').life_expec.mean())
dff8 = pd.DataFrame(hie.groupby('cluster').total_fer.mean())
dff9 = pd.DataFrame(hie.groupby('cluster').gdpp.mean())
dff1 = pd.DataFrame(hie.groupby('cluster').child_mort.mean())


# In[ ]:


hie


# In[ ]:


final2  = pd.concat([pd.Series([0,1,2,3,4,5]), dff1, dff2, dff3, dff4, dff5, dff6, dff7, dff8 ,dff9], axis = 1)


# In[ ]:


final2.columns = ['cluster', 'child_mort', 'exports', 'health', 'imports', 'income',
       'inflation', 'life_expec', 'total_fer', 'gdpp' ]


# In[ ]:


final2


# In[ ]:


plt.figure(figsize = (18,10))

plt.subplot(3,3,1)
sns.barplot(final2.cluster, final2.gdpp)
plt.subplot(3,3,2)

sns.barplot(final2.cluster, final2.total_fer)
plt.subplot(3,3,3)

sns.barplot(final2.cluster, final2.life_expec)
plt.subplot(3,3,4)

sns.barplot(final2.cluster, final2.inflation)
plt.subplot(3,3,5)

sns.barplot(final2.cluster, final2.income)
plt.subplot(3,3,6)

sns.barplot(final2.cluster, final2.imports)
plt.subplot(3,3,7)

sns.barplot(final2.cluster, final2.health)
plt.subplot(3,3,8)

sns.barplot(final2.cluster, final2.exports)
plt.subplot(3,3,9)


sns.barplot(final2.cluster, final2.child_mort)

plt.show()


# In[ ]:


pdf = df.drop('cluster', 1)


# In[ ]:


pdf['cluster'] = clusterCut 


# In[ ]:


pdf[pdf.cluster == 5]


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


data = dfnorm.drop('cluster', axis = 1)


# In[ ]:


pca = PCA(svd_solver= 'randomized', random_state= 100, n_components=4)
pca.fit(data)


# In[ ]:


pca.components_


# In[ ]:


pca.n_components_


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


cols = data.columns


# In[ ]:


dd = pd.DataFrame(pca.components_)


# In[ ]:


dd


# In[ ]:


pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':cols})


# In[ ]:


pcs_df


# In[ ]:


plt.scatter(pcs_df.PC1, pcs_df.PC2)
for i, txt in enumerate(pcs_df.Feature):
    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))

plt.tight_layout()
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# In[ ]:


plt.figure(figsize = (12,8))
sns.heatmap(pd.DataFrame(pca.components_).corr(), annot =True)
plt.show()


# In[ ]:


pca_data = pca.fit_transform(data)


# In[ ]:


pca_data = pd.DataFrame(pca_data)


# In[ ]:


pca_data


# In[ ]:


plt.figure(figsize = (12,8))
sns.heatmap(pca_data.corr(), annot =True)
plt.show()


# In[ ]:


mod = KMeans(n_clusters= 5, max_iter=50, random_state=100)


# In[ ]:


mod.fit(pca_data)


# In[ ]:


mod.labels_


# In[ ]:


sse =[]

for k in range(2,15):
    mod2 = KMeans(n_clusters= k, max_iter=50, random_state=100)
    mod2.fit(pca_data)
    sse.append([k, silhouette_score(pca_data, mod2.labels_ )])


# In[ ]:


plt.plot(pd.DataFrame(sse)[0],pd.DataFrame(sse)[1] )


# In[ ]:


ssd = []

for num in range(1,21):
    mod3 = KMeans(n_clusters= num, max_iter= 50, random_state= 100)
    mod3.fit(pca_data)
    ssd.append(mod3.inertia_)


# In[ ]:


plt.plot(ssd)


# In[ ]:


pca_data['cluster'] = mod.labels_


# In[ ]:


pcca1 = pd.DataFrame(pca_data.groupby('cluster')[0].mean())
pcca2 = pd.DataFrame(pca_data.groupby('cluster')[1].mean())
pcca3 = pd.DataFrame(pca_data.groupby('cluster')[2].mean())
pcca4 = pd.DataFrame(pca_data.groupby('cluster')[3].mean())


# In[ ]:


final_data = pd.concat([pd.Series([0,1,2,3,4]), pcca1, pcca2, pcca3, pcca4], axis = 1)


# In[ ]:


final_data.columns = ['cluster', 'pc1', 'pc2', 'pc3', 'pc4']


# In[ ]:


final_data


# In[ ]:


plt.figure(figsize = (18,10))

plt.subplot(2,2,1)
sns.barplot(final_data.cluster, final_data.pc1)

plt.subplot(2,2,2)
sns.barplot(final_data.cluster, final_data.pc2)

plt.subplot(2,2,3)
sns.barplot(final_data.cluster, final_data.pc3)

plt.subplot(2,2,4)
sns.barplot(final_data.cluster, final_data.pc4)

plt.show()


# In[ ]:


pca_data['country'] = df['country']


# In[ ]:


pca_data[pca_data.cluster == 1]


# In[ ]:


pca_data.cluster.value_counts()


# In[ ]:


# Here the five clusters show urgent need of an aid, need aid in near future, moderate , doing about right and absolutely no need of aid.

