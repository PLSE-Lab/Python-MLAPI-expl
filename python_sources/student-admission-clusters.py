#!/usr/bin/env python
# coding: utf-8

# ## Objective
# Using unsupervised machine learning techniques the idea is to identify different profiles into the people who applies to the graduate program.

# In[ ]:


import numpy as np                # linear algebra
import pandas as pd               # data frames
import seaborn as sns             # visualizations
import matplotlib.pyplot as plt   # visualizations
import scipy.stats                # statistics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/Admission_Predict.csv")

# Print the head of df
print(df.head())

# Print the info of df
print(df.info())

# Print the shape of df
print(df.shape)


# The dataset has 400 aspirants with 9 variables consider for its admission.

# ## Basic Exploratory Data Analysis
# 
# More about [preparation and exploratory analysis](https://www.kaggle.com/camiloemartinez/lucky-charms-lovers).

# In[ ]:


df.describe()


# In[ ]:


# Compute the correlation matrix
corr=df.iloc[:,1:9].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# It look like GRE and GPA are the most significant variables to be admitted into a graduate program, and maybe not coincidentally GRE and GPA area heavily correlated.

# In[ ]:


sns.jointplot(x="GRE Score", y="CGPA", data=df)


# Does all the features are important given different GPAs?

# In[ ]:


#Correlation for different deciles of the most important variable to be admitted
def corr_parts(data,x,y,z,z_cutoff):
    df_temp = data.loc[data[z] > z_cutoff]
    return df_temp[x].corr(df_temp[y])

dl_contrast = np.around(np.percentile(df['CGPA'], np.arange(0, 100, 10)),1)

corr_sop = []
for x in dl_contrast:
    corr_sop.append(corr_parts(df,'SOP','Chance of Admit ','CGPA', x ))
corr_lor = []
for x in dl_contrast:
    corr_lor.append(corr_parts(df,'LOR ','Chance of Admit ','CGPA', x ))
    
result = pd.DataFrame ({'decile': dl_contrast, 'sop': corr_sop, 'lor': corr_lor  })
result = result.melt('decile', var_name='vars',  value_name='corr')

# Set up the seaborn figure
sns.factorplot(x="decile", y="corr", hue='vars', data=result)


# The correlation of the statement of purpose and recommendation letters decay in the upper GPA deciles. It means that GPA is so strong that if you have a good one the rest of the variables do not matter that much, on the opposite if your GPA is not the best the these variables really influence the result of the admission.

# ## Data Transformations

# In[ ]:


#Scaling the continuos variables
df_scale = df.copy()
scaler = preprocessing.StandardScaler()
columns =df.columns[1:7]
df_scale[columns] = scaler.fit_transform(df_scale[columns])
df_scale.head()


# ## Clustering

# The first step is to find the number of clusters that minimize the variance but still are a practical number to analyze.

# In[ ]:


#Elbow graph
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(df_scale.iloc[:,1:])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[ ]:


# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(df_scale.iloc[:,2:9])

# Determine the cluster labels of new_points: labels
df_scale['cluster'] = model.predict(df_scale.iloc[:,2:9])

df_scale.head()


# Each register has a cluster now, lets visualized using [PCA](https://www.kaggle.com/camiloemartinez/is-the-human-freedom-index-a-good-index). 
# 

# In[ ]:


# Create PCA instance: model
model_pca = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model_pca.fit_transform(df_scale.iloc[:,2:9])

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
sns.scatterplot(x=xs, y=ys, hue="cluster", data=df_scale)


# The cluster works nice! The tree of them represent a significative population and different to the other populations.

# In[ ]:


sns.boxplot(x="cluster", y="Chance of Admit ", data=df_scale, palette="Set2" )


# In order to understand, explain and give a meaningful name we explore the centroids of each cluster.

# In[ ]:


centroids = model.cluster_centers_
df_scale.iloc[:,1:10].groupby(['cluster']).mean()


# In[ ]:


sns.heatmap(df_scale.iloc[:,1:10].groupby(['cluster']).mean(), cmap="YlGnBu")


# - Cluster 0: **Top** *students, higher score in all the variables than rest of the population.*
# - Cluster 1: **Average** *students, almost average in each variable but some of them have a good score in one variable in particular that make them more eligible in the admission.*
# - Cluster 2: **Aspirational** *student, below the average of the population. In limited cases eligible given extraordinary score in a particular variable.*
# 

# In[ ]:


pd.DataFrame(df_scale['cluster'].value_counts(dropna=False))


# In[ ]:


g = sns.PairGrid(df_scale.iloc[:,1:10], hue="cluster", palette="Set2")
g.map(plt.scatter);


# The scatter plots show the segmentation separate very well the population across all the variables. Which is the desire outcome of the exercise.

# 
