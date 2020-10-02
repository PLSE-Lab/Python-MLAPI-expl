#!/usr/bin/env python
# coding: utf-8

# **Importing Libraries:**

# In[ ]:


get_ipython().run_line_magic('reset', '-f')

import warnings
warnings.filterwarnings("ignore")

# 1.1 Data manipulation library
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')

# 1.2 OS related package

import os

# 1.3 Modeling librray
# 1.3.1 Scale data

from sklearn.preprocessing import StandardScaler

# 1.4 Plotting library

import seaborn as sns
import matplotlib.pyplot as plt

# 1.5 Import GaussianMixture class

from sklearn.mixture import GaussianMixture

# 1.6 TSNE
from sklearn.manifold import TSNE


# In[ ]:


# Display multiple outputs from a jupyter cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


# DateFrame object is created while reading file available at particular location given below

df=pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


# Rename Columns of DataFrame with Proper name

df.rename(columns = {'CustomerID': 'Customer_ID','Gender':'Gender','Age': 'Age','Annual Income (k$)' : 'Annual_Income',
                    'Spending Score (1-100)':'Spending_Score'},inplace=True)


# In[ ]:


df.info()


# In[ ]:


# Drop Customer Id column from DataFrame

df.drop(columns = "Customer_ID", inplace = True)


# In[ ]:


# Displaying the columns of DataFrame after dropping customer id column

df.columns


# In[ ]:


# Transforming Gender column to 0 and 1

df.Gender.replace(to_replace = ['Male','Female'],value = [0,1],inplace= True)


# In[ ]:


# Displaying first 5 rows after transforming gender to 0 and 1

df.head()


# In[ ]:


# Shows the relationship between annual income and spending score by scatter plot

sns.relplot(x="Annual_Income", y="Spending_Score", col="Gender", data=df)

# Shows the relationship between age and spending score by scatter plot

sns.relplot(x="Age", y="Spending_Score", col="Gender",data=df)


# In[ ]:


# Shows the relationship between annual income and spending score by line plot

sns.relplot(x="Annual_Income", y="Spending_Score", col="Gender",kind="line",data=df)

# Shows the relationship between age and spending score by line plot

sns.relplot(x="Age", y="Spending_Score", col="Gender",kind="line",data=df)


# In[ ]:


# Shows the relationship of each figure

sns.pairplot(hue="Gender",data=df)


# In[ ]:


# Shows the gender-wise count

sns.countplot(x="Gender",data=df,linewidth=5,edgecolor=sns.color_palette("dark", 1))


# In[ ]:


sns.distplot(df.Annual_Income,color='b')


# In[ ]:


sns.distplot(df.Spending_Score,color='b')


# In[ ]:


sns.regplot(x="Annual_Income", y="Spending_Score", data=df)


# In[ ]:


# Take the selected data from dataframe 

dfselecteddata = df[['Age','Annual_Income','Spending_Score']]

# Scale data using StandardScaler
    
ss = StandardScaler()                 # Create an instance of class
ss.fit(dfselecteddata)                # Train object on the data
X = ss.transform(dfselecteddata)      # Transform data
X[:5, :]                              # See first 5 rows


# In[ ]:


# Perform clsutering using Gaussian Mixture Modelling

gm = GaussianMixture(
                     n_components = 4,
                     n_init = 10,
                     max_iter = 100)

# Train the algorithm
gm.fit(X)

# Where are the clsuter centers
gm.means_

# Did algorithm converge?
gm.converged_

# How many iterations did it perform?
gm.n_iter_

# Clusters labels
gm.predict(X)


# In[ ]:


# Plot cluster and cluster centers from gmm

fig = plt.figure()

plt.scatter(X[:, 0], X[:, 1],
            c=gm.predict(X),
            s=2)

plt.scatter(gm.means_[:, 0], gm.means_[:, 1],
            marker='v',
            s=5,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )
plt.show()


# In[ ]:


# Anomaly detection
# Anomalous points are those that are in low-density region Or where density is in low-percentile of 4%


densities = gm.score_samples(X)       #score_samples() method gives score or density of a point at any location.
densities

density_threshold = np.percentile(densities,4)
density_threshold

anomalies = X[densities < density_threshold]
anomalies
anomalies.shape                                                    


# In[ ]:


# Show anomalous points

fig = plt.figure()

plt.scatter(X[:, 0], X[:, 1], c = gm.predict(X))
plt.scatter(anomalies[:, 0], anomalies[:, 1],
            marker='^',
            s=50,               # marker size
            linewidths=5,      # linewidth of marker edges
            color='red'
            )
plt.show()


# In[ ]:


# Understand differences in anomalous & unanomalous (ie normal)data.

# Get first unanomalous data

unanomalies = X[densities >= density_threshold]
unanomalies.shape    

# Transform both anomalous and unanomalous data to pandas DataFrame

df_anomalies = pd.DataFrame(anomalies, columns = ['x','y','z'])
df_anomalies['t'] = 'anomalous'   # Create a 4th constant column
df_normal = pd.DataFrame(unanomalies, columns = ['x','y','z'])
df_normal['t'] = 'unanomalous'    # Create a 4th constant column


# In[ ]:


# Let us see density plots

sns.distplot(df_anomalies['x'])
sns.distplot(df_normal['x'])


# In[ ]:


# Draw boxplots
# Ist stack two dataframes

df1 = pd.concat([df_anomalies,df_normal])

# 7.4.2 Draw featurewise boxplots

sns.boxplot(x = df1['t'], y = df1['x'])


# In[ ]:


# Use either AIC or BIC measures to discover ideal no of clusters
bic = []
aic = []
for i in range(4):
    gm = GaussianMixture(
                     n_components = i+1,
                     n_init = 10,
                     max_iter = 100)
    gm.fit(X)
    bic.append(gm.bic(X))
    aic.append(gm.aic(X))


# In[ ]:


fig = plt.figure()
plt.plot([1,2,3,4], aic)
plt.plot([1,2,3,4], bic)
plt.show()


# In[ ]:


gm = GaussianMixture(
                     n_components = 4,
                     n_init = 10,
                     max_iter = 100)
gm.fit(X)


# In[ ]:


#  Darw a 2-D t-sne plot and colour points by gmm-cluster labels

tsne = TSNE(n_components = 2)
tsne_out = tsne.fit_transform(X)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            #marker='x',
            s=50,              # marker size
            linewidths=5,      # linewidth of marker edges
            c=gm.predict(X)    # Colour as per gmm
            )

