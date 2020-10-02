#!/usr/bin/env python
# coding: utf-8
Created by : Ezhilarasan 
To analyse New York Stock Exchange fundamendals data and perform data visualization, Normalization, Clustering activities.
# In[ ]:


get_ipython().run_line_magic('reset', '-f')
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


# In[ ]:


from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer


# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#to display all the commands result (of a particular cell which we are running)
#otherwise it will display only the last result


# In[ ]:


np.set_printoptions(precision = 3,          # Display upto 3 decimal places
                    threshold=np.inf        # Display full array
                    )


# In[ ]:


#To get rid of scientific notations
pd.options.display.float_format = '{:.2f}'.format


# In[ ]:


#To reset the format back
#pd.reset_option('display.float_format')


# In[ ]:


pd.options.display.max_columns=100
#set maximum columns to display in the output

pd.options.display.max_rows=100
#set maximum rows to display in the output


# In[ ]:


#fun = pd.read_csv("G:\\Python\\Session11_31-May-2020\\Exercise\\854_1575_bundle_archive\\fundamentals.csv", index_col = 0, parse_dates=['Period Ending'])
fun = pd.read_csv("/kaggle/input/nyse/fundamentals.csv", index_col = 0, parse_dates=['Period Ending'])
#first column in the excel has been taken as index for the dataframe
#parse_dates will parse the mentioned column as datetime type


# Remove unwanted characters from column names

# In[ ]:


fun.columns = fun.columns.str.replace(" ", "_")
fun.columns = fun.columns.str.replace("_&_", "_")
fun.columns = fun.columns.str.replace("_/_", "_")
fun.columns = fun.columns.str.replace(".", "")
fun.columns = fun.columns.str.replace("-", "")
fun.columns = fun.columns.str.replace("'", "")
fun.columns = fun.columns.str.replace(",", "")
fun.columns = fun.columns.str.replace("/", "_")


# Drop columns which has null values

# In[ ]:


fun.shape
fun.dropna(axis=1,inplace=True)
fun.shape
fun.head()


# Predict a column (to cluster) based on Total_Revenue

# In[ ]:


fun[fun.Total_Revenue >10000000000.00 ].shape
fun[fun.Total_Revenue <10000000000.00 ].shape

fun['cluster_group'] = fun['Total_Revenue'].apply(lambda x : 1 if x > 10000000000 else 0)

fun.head()
fun.cluster_group.value_counts()


# Some Data Visualisation graphs

# In[ ]:


sns.barplot('cluster_group', 'Total_Revenue', estimator = np.mean, data=fun, ci=68)


# In[ ]:


sns.distplot(fun.Gross_Margin, bins=10)


# In[ ]:


sns.barplot('cluster_group', 'Gross_Profit', estimator = np.mean, data=fun, ci=68)


# Group by Ticker Symbol column and aggregare for average

# In[ ]:


fungroup = fun.groupby("Ticker_Symbol")
mean_df = fungroup.aggregate('mean')
mean_df.reset_index(inplace=True)
mean_df.head()


# Visualize Top 10 Performers based on Total Revenue

# In[ ]:


TopPerformer = mean_df.sort_values(by = 'Total_Revenue', ascending=False).head(10)
#TopPerformer
sns.barplot(x = 'Ticker_Symbol', y = 'Total_Revenue', data = TopPerformer)


# Bar chart based on Gross Profit

# In[ ]:


sns.barplot('cluster_group', 'Gross_Profit', estimator = np.mean, data=mean_df, ci=68)


# Remove Ticker Symbol (String Value) and Period (Date Value) to go for Scaling

# In[ ]:


#fun_remTick = fun.head() # Shallow Copy. It will remove column from original df also
fun_remTick = fun.copy() # Deep Copy. It will remove column from copied df only
fun_remTick.drop(["Ticker_Symbol","Period_Ending"], axis=1, inplace=True)
fun_remTick.head()
#fun.head()


# Separate the cluster group column which we created for clustering based on Total_Revenue

# In[ ]:


y = fun_remTick['cluster_group'].values
fun_remTick.drop(columns = ['cluster_group'], inplace = True)


# Scaling the final dataframe using StandardScaler

# In[ ]:


ss = StandardScaler()
ss.fit(fun_remTick)
X = ss.transform(fun_remTick)
X.shape


# Split the sample into Train & Test

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( X,y, test_size = 0.25)
X_train.shape
X_test.shape
y_train.shape
y_test.shape


# Draw Scree Plot (To find out No of Clusters)

# In[ ]:


sse1 = []
for k in range(1,10):
    km = KMeans(n_clusters = k)
    km.fit(X_train)
    sse1.append(km.inertia_)


# In[ ]:


plt.plot(range(1,10), sse1, marker='*')


# Calculate Silhouette score for Clusters 2,3 & 4

# In[ ]:


km2 = KMeans(n_clusters = 2)
km2.fit(X_train)
#km2.cluster_centers_
#km2.labels_
#km2.inertia_
silhouette_score(X_train, km2.labels_)


# In[ ]:


km1 = KMeans(n_clusters = 3)
km1.fit(X_train)
#km1.cluster_centers_
#km1.labels_
#km1.inertia_
silhouette_score(X_train, km1.labels_) 


# In[ ]:


km4 = KMeans(n_clusters = 4)
km4.fit(X_train)
#km4.cluster_centers_
#km4.labels_
#km4.inertia_
silhouette_score(X_train, km4.labels_)


# Based on the silhouette_score calculation, we can go for Cluster =2
# Draw cluster Visualisation Diagram using Cluster =2

# In[ ]:


kmeans = KMeans(n_clusters=2)
pred_y = kmeans.fit(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red')
plt.show()


# Check the prediction success rate

# In[ ]:


clf = KMeans(n_clusters = 2)
clf.fit(X_train)
y_pred = clf.predict(X_test)
#y_pred
np.sum(y_pred == y_test)/y_test.size


# SilhouetteVisualizer

# In[ ]:


visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')
visualizer.fit(X_train)        
visualizer.show()              


# In[ ]:


Draw InterclusterDistance diagram using yellowbrick


# In[ ]:


from yellowbrick.cluster import InterclusterDistance
visualizer = InterclusterDistance(clf)
visualizer.fit(X_train)        
visualizer.show()              


# In[ ]:




