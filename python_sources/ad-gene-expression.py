# %% [markdown]
# "The study established two clones of induced pluripotent stem cells (iPSC) with the presenilin 2 mutation, N141 (PS2-1 iPSC and PS2-2 iPSC) by retroviral transduction of primary human fibroblasts. To show the similarity among 201B7 iPSC, PD01-25 iPSC(Sporadic Parkinson's disease patient derived iPSC), PS2-1 iPSC, PS2-2 iPSC, this experiment was designed. Undifferentiated 201B7 iPSC, PD01-25 iPSC, PS2-1 iPSC and PS2-2 iPSC were collected. Then, they were applied in this experiment.The columns in the dataset with samples and celllines,mutation,no mutation and foldchange listed as 
# ID_REF
# GSM701542 -iPSC cell line
# GSM701543 -Sporadic Parkinson's disease patient derived iPSC 
# GSM701544 -iPSC from primary fibroblast
# GSM701544 -\tiPSC from primary fibroblast
# no mutation
# mutation
# log 2 fold change
# fold change

# %% [code] {"id":"xEzN-n4ZiqTm"}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from mpl_toolkits.mplot3d import Axes3D
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# The gene expression details are given under the columns 'GSM701542', 'GSM701543', 'GSM701544', 'GSM701545' so, the columns log2foldchange and foldchange are redundant information of that so it has been planned to left log2foldchage and foldchange values. Also the same is for no mutation column,as the column mutation highly correlated with the gene expression of cell line samples.  

# %% [code] {"id":"hK98yvx4PCAc"}
collist=['ID_REF', 'GSM701542', 'GSM701543', 'GSM701544', 'GSM701545', 'mutation', 'log 2 fold change']
df=pd.read_csv("/kaggle/input/alzheimers-gene-expression-profiles/GSE28379 - GSE28379_series_matrix.csv",usecols=collist) 
ann=pd.read_csv("/kaggle/input/ad-annotation-for-genes-id/gene.csv",usecols=['ID','Gene.symbol'])

# %% [code] {"id":"g4LJD6FqiqUc","outputId":"b591ea22-0547-456c-b927-e010e139363e"}
df3=ann.merge(df, how='inner', left_on='ID', right_on='ID_REF')
df2=df[['GSM701542', 'GSM701543', 'GSM701544', 'GSM701545','mutation']]
df1=df2

# %% [markdown]
# The correlation between different feature help to make the relationship among gene expression and mutation of different samples

# %% [code] {"id":"zd2Iz5IriqVJ","outputId":"16946090-3f75-42e0-b476-537b3065d035"}
correlation = df2.corr()
plt.figure(figsize=(10,10))
sn.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')
plt.title('Correlation between different fearures')

# %% [code] {"id":"KS2xiDmviqVT"}
sn.pairplot(df2,kind='reg',diag_kind = 'kde',height= 4)

# %% [code] {"id":"t7vN0QRxiqVd","outputId":"1c31437f-43ab-4fe9-cc35-2a08188e8175"}
df2 = df2[1:-1]
len(df2) - df2.count()

# %% [code]
#To make assure about the null or missing values and to proceed with kmeans clustering

# %% [code] {"id":"xFr0sf0kiqWz","outputId":"a933b587-05ec-4ff3-9c96-2463b0989a09"}
# Prepare models
clus=[]
pred=[]
for i in range(1,11): 
  kmeans = KMeans(algorithm='auto',n_clusters=3, init ='k-means++',n_init=i, max_iter=300,tol=0.0001, random_state=0,verbose=0).fit(df2)
  clus.append(kmeans.inertia_)
min_samples = df2.shape[1]+1 
print('kmeans score: {}'.format(silhouette_score(df2, kmeans.labels_, metric='euclidean')))
pred=[kmeans.labels_]
print(pred)
print("Mininum number of samples=",min_samples)
p1=range(1,11)
plt.xlabel('Number of clusters')
plt.plot(p1,clus)
plt.title('The Elbow Method Graph')
plt.ylabel('clus')
plt.show()

# %% [code] {"id":"OFcP_1EMm1bM","outputId":"1974ba16-11a4-49c8-81e4-6c3d638a93bf"}
y_kmeans = kmeans.fit_predict(df2)
pred = pd.DataFrame(y_kmeans)
df3['predicted']=pred
#df3.shape
print(pred)


# %% [code] {"id":"ql-BJQtHuz5U","outputId":"ebfb1361-8f3c-45cf-e65a-a80ec329e35a"}
df3['predicted'].unique()
df3=df3.dropna(axis=0)
len(df3) - df3.count()

# %% [code] {"id":"EpzwvwriiqW6"}
from sklearn.decomposition import PCA
def prep_pca(components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = PCA(n_components=components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i:names[i] for i in range(components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels
    return df_matrix

# %% [code] {"id":"4RkL1UHbiqXC","outputId":"70dc1925-eb40-461b-c8a2-6d31b3a61f0b"}
pca_df = prep_pca(2, df2, kmeans.labels_)
sn.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette="Set1")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='k')
plt.show()

# %% [code] {"id":"AZQNXjKrE3LY","outputId":"12abe73a-81eb-43cf-e6e3-8a2201922358"}
fig=plt.figure(figsize=(10, 10)) 
ax = fig.add_subplot(111, projection = '3d') 
ax.scatter(pca_df.x, pca_df.y,pca_df.labels,s=20,c='b') 
   
str_labels = list(map(lambda label:'% s' % label, kmeans.labels_)) 
   
list(map(lambda data1, data2, data3, str_label: 
        ax.text(data1, data2, data3, s = str_label, size = 16, 
        zorder = 10, color = 'k'), pca_df.x, pca_df.y, 
        pca_df.labels, str_labels)) 
   
plt.show() 

# %% [code]
result=df3[['ID','predicted','Gene.symbol']]

result.to_csv('output.csv', index = False)