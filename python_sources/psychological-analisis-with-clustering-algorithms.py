#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Aim of this research is segmenting people into clusters according to their sense of entertainment, social behaviour and moral values with <br>
# K-Means algorithm and visualising the results.
# 
# *Data which is used in this study is collected from GoogleDocs from real people. Their identities are not shared in order to keep their personal information safe.

# # Prepearing the Environment and Data Preprocessing 

# ## Step 1 - Importing Libraries
# **Here we import libraries that we will use in our research.**<br>
# Numpy: For linear algebra and array operations<br>
# Pandas: For structural data manipulation<br>
# Matplotlib: For data visualisation<br>
# Sklearn: For machine machine learning algorithms,data preprocessing and dimensionality reduction<br>
# Yellowbrick: For parametre analysis for our algorithm<br>
# os: For directory operations

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer

import os


# ## Step 2 - First Look To Our Data
# **Here we look into our data and examine it's general attributes.**<br><br>
# This is the name of our data file:

# In[ ]:


print(os.listdir("../input"))


# Now we read our dataset and have a look at it.

# In[ ]:


dataset=pd.read_excel("../input/dataset.xlsx")
dataset #this is our data 


# In[ ]:


dataset.isnull().sum() #Here we see if there are empty answers.


# In[ ]:


dataset.describe().T # Statistical properties of numeric data.


# In[ ]:


dataset.hist(figsize=(10,10),layout=(3,1)); #Histograms of our numeric data.


# In[ ]:


#number of unique values for each column
for col in dataset:
    print(len(dataset[col].unique()))


# ## Step 3 - Data Preprocessing
# **Here we manipulate our data for our algorithm.**<br><br>
# Machine learning algorithms work with numeric data so we have to preprocess our structural data before using it.

# In[ ]:


#Here we seperate our structural data from numeric data.
structural1=dataset.iloc[:,0:5]
structural2=dataset.iloc[:,8:]
numeric_data=dataset.iloc[:,5:8]
structural_data=pd.concat([structural1,structural2],axis=1)
structural_data.head()#first 5 rows of our structural data


# In[ ]:


numeric_data.head()#first 5 rows of our numeric data


# In[ ]:


#here we convert our structural data into numeric data
le=LabelEncoder()

for col in structural_data:
    structural_data[col]=le.fit_transform(structural_data[col])
structural_data


# In[ ]:


#Finally we merge our processed structural data with our numeric data.
final_data=pd.concat([structural_data,numeric_data],axis=1)
final_data


# And we will apply PCA to our data in order to visualise it approximately.

# In[ ]:


pca = PCA(n_components=2)

reduced_data=pca.fit_transform(final_data)

plt.scatter(reduced_data[:,0],reduced_data[:,1])
plt.show()


# ## Step 3 - Model Building
# Here we build our clustering model and evaluate it. First we will choose the appropriate cluster number with elbow method.

# In[ ]:


kmeans=KMeans() #this is our clustering algorithm ->KMeans

#We will determine our optimal number of clusters with elbow method from both our actual data and reduced data.

visualiser=KElbowVisualizer(kmeans,k=(2,10))
visualiser.fit(final_data)
visualiser.poof()


# In[ ]:


kmeans=KMeans()
visualiser=KElbowVisualizer(kmeans,k=(2,10))
visualiser.fit(reduced_data)
visualiser.poof()


# As we can see from the graphs incline suddenly changes and distortion score is not very high when our number of clusters is 4. So we will build a algorithm with 4 clusters.

# Now we will build our model and visualise it with our reduced data.

# In[ ]:


algorithm=KMeans(n_clusters=4)#our algorithm
k_fit=algorithm.fit(final_data)

clusters=k_fit.labels_
clusters #here are our clusters


# Now it's time to visualise the results.

# In[ ]:


plt.scatter(reduced_data[:,0],reduced_data[:,1],c=clusters,cmap="viridis")#we visualise the results
plt.show()


# Now let's add our results and have a final look.

# In[ ]:


final_data["cluster"]=clusters+1
final_data.head() #first 5 values of final data


# In[ ]:


final_data[final_data["cluster"]==1].head() #first 5 values of cluster 1


# In[ ]:


final_data[final_data["cluster"]==1].describe().T # Statistical properties of cluster 1


# In[ ]:


final_data[final_data["cluster"]==2].head() #first 5 values of cluster 2


# In[ ]:


final_data[final_data["cluster"]==2].describe().T # Statistical properties of cluster 2


# In[ ]:


final_data[final_data["cluster"]==3].head() #first 5 values of cluster 3


# In[ ]:


final_data[final_data["cluster"]==3].describe().T # Statistical properties of cluster 3


# In[ ]:


final_data[final_data["cluster"]==4].head() #first 5 values of cluster 4


# In[ ]:


final_data[final_data["cluster"]==4].describe().T # Statistical properties of cluster 4


# ## Result
# As we can see in the graph and samples at the clusters, approximately all participants are clustered with people who have tendency with them.
