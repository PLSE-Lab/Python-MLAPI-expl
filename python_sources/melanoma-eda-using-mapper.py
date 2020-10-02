#!/usr/bin/env python
# coding: utf-8

# # Using a  Mapper algorithm for visualization of high-dimensional data****

# [Info on Keppler](http://https://github.com/scikit-tda/kepler-mapper) along with tutorials
# 
# [Kernel](https://www.kaggle.com/noelano/topological-analysis-of-premier-league-players/notebook) provides good information and uses on premier league dataset

# In[ ]:


get_ipython().system('pip install kmapper')


# In[ ]:


import pandas as pd
import cv2
from glob import glob
training_data_path = "../input/siic-isic-224x224-images/train/"


# In[ ]:




images_path = glob("../input/siic-isic-224x224-images/train/*")
df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")


# In[ ]:


df.head()


# In[ ]:


benign=df[df['benign_malignant']=='benign'].sample(600)
benign.shape


# In[ ]:


malignant=df[df['benign_malignant']=='malignant']
malignant.shape


# In[ ]:


Testdf=pd.concat([benign,malignant])
Testdf.shape
# df.merge(images_df, left_on='image_name')


# In[ ]:


image_list=Testdf['image_name']
image_list[0:5]


# In[ ]:


# labels_2 = []
images_df=pd.DataFrame()

for imageName in image_list:
#column_name=imagePath.split('/')[-1].split('.')[0]
  imagePath=training_data_path+imageName+'.png'
#   print(imagePath)
  image=cv2.imread(imagePath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # image=cv2.resize(image, (300, 300),interpolation=cv2.INTER_AREA)
#   print(imageName)
  images_df[imageName]=image.flatten()


# In[ ]:


images_df=images_df.transpose()
images_df.head()


# In[ ]:


image_name=images_df.index
images_df['image_name']=image_name


# In[ ]:


images_df['image_name'].head(10)


# In[ ]:


Testdf.head(10)


# In[ ]:


Testdf=Testdf.merge(images_df,right_on='image_name',left_on='image_name')


# In[ ]:


Testdf.head(10)


# # Using TSNE for dimensionality Reduction

# In[ ]:


import kmapper as km

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets


# In[ ]:


X=images_df.drop(['image_name'],axis=1)


# In[ ]:


# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper_full = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data_full = mapper_full.fit_transform(X,
                                      projection=sklearn.manifold.TSNE())


# In[ ]:


# Create the graph (we cluster on the projected data and suffer projection loss)
graph_full = mapper_full.map(projected_data_full,
#                    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
                   clusterer=sklearn.cluster.KMeans(n_clusters=2,
                                                    random_state=1618033),          
                   cover=km.Cover(15, 0.7))


# In[ ]:


Y=Testdf['benign_malignant']
##Tooltips with the target y-labels for every cluster member
mapper_full.visualize(graph_full,
                 title="Skin Cancer Mapper with  Y Labels ",
                 path_html="/kaggle/working/skin_cancer_tsne_ylabels_only.html",
                 custom_tooltips=Y)

# ##Tooltips with the target y-labels for every cluster member
# mapper_full.visualize(graph_full,
#                  title="Skin Cancer Mapper with  Image NAmes ",
#                  path_html="/kaggle/working/skin_cancer_image_name_tsne_image_names.html",
#                  custom_tooltips=Y)


# In[ ]:


# Matplotlib examples
km.draw_matplotlib(graph_full)
plt.show()


# In[ ]:


Y=Testdf['image_name']


# In[ ]:


##Tooltips with the target y-labels for every cluster member
mapper_full.visualize(graph_full,
                 title="Skin Cancer Mapper with  Image NAmes ",
                 path_html="/kaggle/working/skin_cancer_image_name_tsne_image_names.html",
                 custom_tooltips=Y)


# # Viewing Images from one cluster

# In[ ]:


Testdf[Testdf['image_name'].isin (['ISIC_6187353', 'ISIC_3593913', 'ISIC_2837876', 'ISIC_4301050', 'ISIC_4378851' ,
                                   'ISIC_0645454', 'ISIC_0961235', 'ISIC_1116483', 'ISIC_1975042', 'ISIC_3244067', 
                                   'ISIC_3253484', 'ISIC_3408231', 'ISIC_3993924', 'ISIC_4730066' ,'ISIC_7075474', 
                                   'ISIC_7181296', 'ISIC_8417873', 'ISIC_8872158' ,'ISIC_8882374', 'ISIC_9509757' ,
                                   'ISIC_9910791'])]


# Looking at  this cluster- It contains only malignant images

# In[ ]:


# img=images_df[images_df['image_name']=='ISIC_6187353']
# plt.imshow(np.array(img.iloc[:,0:50176]).reshape(224,224),cmap='gray')


# In[ ]:


# # plt.imshow(training_data_path+'ISIC_3593913.jpg')
# img=images_df[images_df['image_name']=='ISIC_3593913']
# plt.imshow(np.array(img.iloc[:,0:50176]).reshape(224,224),cmap='gray')


# In[ ]:


# # plt.imshow(training_data_path+'ISIC_6187353.png')
# img=images_df[images_df['image_name']=='ISIC_6187353']
# plt.imshow(np.array(img.iloc[:,0:50176]).reshape(224,224),cmap='gray')


# In[ ]:


# #ISIC_7075474
# img=images_df[images_df['image_name']=='ISIC_7075474']
# plt.imshow(np.array(img.iloc[:,0:50176]).reshape(224,224),cmap='gray')


# # Using MDS(Multi-dimensional Scaling)

# In[ ]:


X=images_df.drop(['image_name'],axis=1)


# In[ ]:


from sklearn.manifold import MDS

mapper = km.KeplerMapper(verbose=0)
lens = mapper.fit_transform(X, projection=MDS())


# In[ ]:



# Create the simplicial complex
graph = mapper.map(lens,
                   X,
                   cover=km.Cover(n_cubes=15, perc_overlap=0.7),
                   clusterer=sklearn.cluster.KMeans(n_clusters=2,
                                                    random_state=1618033))

y=Testdf['benign_malignant']
# Visualization
mapper.visualize(graph,
                 path_html="Skin-cancer_MDS.html",
                 title="Melanoma skin Cancer Dataset",
                 custom_tooltips=y)


# import matplotlib.pyplot as plt
km.draw_matplotlib(graph)
plt.show()


# In[ ]:


X=images_df.drop(['image_name'],axis=1)
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
X1_dist = pairwise_distances(X, metric= 'l2')

mapper = km.KeplerMapper(verbose=0)
lens2 = mapper.fit_transform(X1_dist, projection=MDS(dissimilarity='precomputed'))


# In[ ]:


# Create the simplicial complex
graph = mapper.map(lens2,
                   X,
                   cover=km.Cover(n_cubes=15, perc_overlap=0.7),
                   clusterer=sklearn.cluster.KMeans(n_clusters=2,
                                                    random_state=1618033))

y=Testdf['benign_malignant']
# Visualization
mapper.visualize(graph,
                 path_html="Skin-cancer_MDS_l2_20.html",
                 title="Melanoma skin Cancer Dataset",
                 custom_tooltips=y)


# import matplotlib.pyplot as plt
km.draw_matplotlib(graph)
plt.show()


# Increasing the resolution of the graph

# In[ ]:


# Create the simplicial complex
graph = mapper.map(lens2,
                   X,
                   cover=km.Cover(n_cubes=30, perc_overlap=0.7),
                   clusterer=sklearn.cluster.KMeans(n_clusters=2,
                                                    random_state=1618033))

y=Testdf['benign_malignant']
# Visualization
mapper.visualize(graph,
                 path_html="Skin-cancer_MDS_l2_30.html",
                 title="Melanoma skin Cancer Dataset",
                 custom_tooltips=y)


# import matplotlib.pyplot as plt
km.draw_matplotlib(graph)
plt.show()


# # Using PCA

# In[ ]:


X=images_df.drop(['image_name'],axis=1)


from sklearn.decomposition import PCA
mapper = km.KeplerMapper(verbose=0)
lens = mapper.fit_transform(X, projection=PCA(0.8))


# In[ ]:


# Create the simplicial complex
graph = mapper.map(lens,
                   X,
                   cover=km.Cover(n_cubes=15, perc_overlap=0.7),
                   clusterer=sklearn.cluster.KMeans(n_clusters=2,
                                                    random_state=1618033))

y=Testdf['benign_malignant']
# Visualization
mapper.visualize(graph,
                 path_html="Skin-cancer_PCA.html",
                 title="Melanoma skin Cancer Dataset",
                 custom_tooltips=y)


# import matplotlib.pyplot as plt
km.draw_matplotlib(graph)
plt.show()


# #  Using Anamoly Detection 

# In[ ]:


import sklearn
from sklearn import ensemble

# Create a custom 1-D lens with Isolation Forest
model = ensemble.IsolationForest(random_state=1729)
model.fit(X)
lens1 = model.decision_function(X).reshape((X.shape[0], 1))

# Create another 1-D lens with L2-norm
mapper = km.KeplerMapper(verbose=0)
lens2 = mapper.fit_transform(X, projection="l2norm")

# Combine both lenses to get a 2-D [Isolation Forest, L^2-Norm] lens
lens = np.c_[lens1, lens2]

# Define the simplicial complex
scomplex = mapper.map(lens,
                      X,
                      nr_cubes=15,
                      overlap_perc=0.7,
                      clusterer=sklearn.cluster.KMeans(n_clusters=2,
                                                       random_state=3471))


# In[ ]:


y=Testdf['benign_malignant']
# Visualization
mapper.visualize(graph,
                 path_html="Skin-cancer_IsolationForest.html",
                 title="Melanoma skin Cancer Dataset",
                 custom_tooltips=y)


# import matplotlib.pyplot as plt
km.draw_matplotlib(graph)
plt.show()

