#!/usr/bin/env python
# coding: utf-8

# # Clustering Card Usage with K Means

# > *Swetha Varikuti*

# > 2019-07-01

# ![Cards Image](https://static-news.moneycontrol.com/static-mcnews/2017/03/master-card-visa-debit-card-credit-card-digital-payment-transaction-shopping-770x433.jpg)

# * [1. Introduction](https://www.kaggle.com/swethavarikuti/clustering-card-usage-with-k-means#1-Introduction)
# * [2. About the data](https://www.kaggle.com/swethavarikuti/clustering-card-usage-with-k-means#2.-About-the-Data)
# * [3. Loading Data](https://www.kaggle.com/swethavarikuti/clustering-card-usage-with-k-means#3.-Loading-Data)
# * [4. Data Analysis](https://www.kaggle.com/swethavarikuti/clustering-card-usage-with-k-means#4.-Data-Analysis)
# * [5. Data Preparation](https://www.kaggle.com/swethavarikuti/clustering-card-usage-with-k-means#5.-Data-Preparation)
# * [6. K Means Modelling](https://www.kaggle.com/swethavarikuti/clustering-card-usage-with-k-means#6.-K-Means-Modelling)
# * [7. Optimal number of Clusters](https://www.kaggle.com/swethavarikuti/clustering-card-usage-with-k-means#7.-Optimal-number-of-clusters)
# * [8. Results](https://www.kaggle.com/swethavarikuti/clustering-card-usage-with-k-means#8.-Results)
# * [9. Summary](https://www.kaggle.com/swethavarikuti/clustering-card-usage-with-k-means#9.-Summary)

# # 1 Introduction

# k-means is an unsupervised machine learning algorithm used to find groups of observations (clusters) that share similar characteristics. What is the meaning of unsupervised learning? It means that the observations given in the data set are unlabeled, there is no outcome to be predicted. We are going to use a card usage data set to cluster different groups based on their card usage characteristics.

# # 2. About the Data

# * CUST_ID : Identification of Credit Card holder (Categorical) 
# * BALANCE : Balance amount left in their account to make purchases 
# * BALANCE_FREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated) 
# * PURCHASES : Amount of purchases made from account 
# * ONEOFF_PURCHASES : Maximum purchase amount done in one-go 
# * INSTALLMENTS_PURCHASES : Amount of purchase done in installment 
# * CASH_ADVANCE : Cash in advance given by the user 
# * PURCHASES_FREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased) 
# * ONEOFF_PURCHASES_FREQUENCY : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased) 
# * PURCHASES_INSTALLMENTS_FREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done) 
# * CASH_ADVANCE_FREQUENCY : How frequently the cash in advance being paid 
# * CASH_ADVANCE_TRX : Number of Transactions made with "Cash in Advanced" 
# * PURCHASES_TRX : Numbe of purchase transactions made 
# * CREDIT_LIMIT : Limit of Credit Card for user 
# * PAYMENTS : Amount of Payment done by user 
# * MINIMUM_PAYMENTS : Minimum amount of payments made by user 
# * PRC_FULL_PAYMENT : Percent of full payment paid by user 
# * TENURE : Tenure of credit card service for user

# # 3. Loading Data

# Let us first load libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="white")
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/CreditCardUsage.csv')


# # 4. Data Analysis

# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# In[ ]:


print("Number of unique id's in CUST_ID column : ",df.CUST_ID.nunique())
print("Number of rows in dataframe : ",df.shape[0])
print('This is to check if we have a single row for each unique ID. We can drop customer id since we do not get any information from it.')


# In[ ]:


df.drop(columns='CUST_ID',inplace=True)


# In[ ]:


df.columns


# In[ ]:


f=plt.figure(figsize=(20,20))
for i, col in enumerate(df.columns):
    ax=f.add_subplot(6,3,i+1)
    sns.distplot(df[col].ffill(),kde=False)
    ax.set_title(col+" Distribution",color='Blue')
    plt.ylabel('Distribution')
f.tight_layout()


# In[ ]:


corr = df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(200, 50, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap,)


# In[ ]:


import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
trace1 = go.Box(y = df['BALANCE'])
trace2 = go.Box(y=df['BALANCE_FREQUENCY'])
trace3=go.Box(y=df['PURCHASES'])
trace4=go.Box(y=df['ONEOFF_PURCHASES'])
trace5=go.Box(y=df['INSTALLMENTS_PURCHASES'])
trace6=go.Box(y=df['CASH_ADVANCE'])
trace7=go.Box(y=df['PURCHASES_FREQUENCY'])
trace8=go.Box(y=df['ONEOFF_PURCHASES_FREQUENCY'])
trace9=go.Box(y=df['PURCHASES_INSTALLMENTS_FREQUENCY'])
trace10=go.Box(y=df['CASH_ADVANCE_FREQUENCY'])
trace11=go.Box(y=df['CASH_ADVANCE_TRX'])
trace12=go.Box(y=df['PURCHASES_TRX'])
trace13=go.Box(y=df['CREDIT_LIMIT'])
trace14=go.Box(y=df['PAYMENTS'])
trace15=go.Box(y=df['MINIMUM_PAYMENTS'])
trace16=go.Box(y=df['PRC_FULL_PAYMENT'])
trace17=go.Box(y=df['TENURE'])

fig = tools.make_subplots(rows=3, cols=6, subplot_titles=('BALANCE', 'Balance_freq', 'PURCHASES', 'oneoff_purchases',
       'Installment_purchases', 'Cash_advance', 'Purchases_freq',
       'Oneoff_purchases_freq', 'Purchases_Installments_freq',
       'Cash_advance_freq', 'Cash_advance_trx', 'Purchases_trx',
       'Credit_Limit', 'Payments', 'Min_Payments', 'PRC_FULL_PAYMENT',
       'TENURE'))
fig['layout'].update(height=800, width=1420, title='Box Plots to visualize data distribution in each column')

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 1, 4)
fig.append_trace(trace5, 1, 5)
fig.append_trace(trace6, 1, 6)
fig.append_trace(trace7, 2, 1)
fig.append_trace(trace8, 2, 2)
fig.append_trace(trace9, 2, 3)
fig.append_trace(trace10, 2, 4)
fig.append_trace(trace11, 2, 5)
fig.append_trace(trace12, 2, 6)
fig.append_trace(trace13, 3, 1)
fig.append_trace(trace14, 3, 2)
fig.append_trace(trace15, 3, 3)
fig.append_trace(trace16, 3, 4)
fig.append_trace(trace17, 3, 5)
plt.tight_layout()
# data = [fig]
py.iplot(fig)


# # 5. Data Preparation

# In[ ]:


print('\n***************************************************    CHECK FOR NULL VALUES   ********************************************************* \n \n',df.isna().sum())


# CREDIT_LIMIT and MINIMUM_PAYMENTS have null values, both of which have outliers. So, let us fill null values using median.

# In[ ]:


df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median(),inplace=True)


# In[ ]:


df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(),inplace=True)


# **Normalizing over the standard deviation**
# 
# Now let's normalize the dataset. But why do we need normalization in the first place? K Means does clustering based on the spatial distance between each data point. We need to normalize our dataset so that one feature doesnot overweigh the other while calculating spatial distance.
# 
# We use standardScaler() to normalize our dataset.

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Scaled_df = scaler.fit_transform(df)


# In[ ]:


df_scaled = pd.DataFrame(Scaled_df,columns=df.columns)
df_scaled.head()


# In[ ]:


fig, ax=plt.subplots(1,2,figsize=(15,5))
sns.distplot(df['BALANCE'], ax=ax[0],color='#D341CD')
ax[0].set_title("Original Data")
sns.distplot(df_scaled['BALANCE'], ax=ax[1],color='#D341CD')
ax[1].set_title("Scaled data")
plt.show()


# Above plot shows that there is no difference in distribution of data before and after scaling. Only the magnitued is scaled about mean and standard deviation.

# # 6. K Means Modelling

# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6,random_state=0)
kmeans.fit(Scaled_df)


# In[ ]:


kmeans.labels_


# # 7. Optimal number of clusters

# #  7.1- ELBOW CURVE(inertia) to visualize optimal k value

# In[ ]:


Sum_of_squared_distances = []
K = range(1,21)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(Scaled_df)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# Above Elbow curve depicts sum of squared distances for each point from its respective centroid. Our goal is to check for a K value that has minimum sum of square distance.

# #  - 7.2 Silhouette Coefficient:

# Silhouette coefficients (as these values are referred to as) near +1 indicate that the sample is far away from the neighboring clusters. A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters and negative values indicate that those samples might have been assigned to the wrong cluster.

# Let's calculate and print Silhoutte score for each K value.

# In[ ]:


from sklearn.metrics import silhouette_score, silhouette_samples

for n_clusters in range(2,21):
    km = KMeans (n_clusters=n_clusters)
    preds = km.fit_predict(Scaled_df)
    centers = km.cluster_centers_

    score = silhouette_score(Scaled_df, preds, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))


# K=3 has maximum Silhoutte score. Let us visualize Silhouette score for each cluster at k=3.

# In[ ]:


from yellowbrick.cluster import SilhouetteVisualizer

# Instantiate the clustering model and visualizer
km = KMeans (n_clusters=3)
visualizer = SilhouetteVisualizer(km)

visualizer.fit(Scaled_df) # Fit the training data to the visualizer
visualizer.poof() # Draw/show/poof the data


# Cluster labelled 1 is clearly above zero and it doesnot overlap with other clusters. Clusters labelled 0 and 2 have slight overlap and may contain wrongly labelled data.

# In[ ]:


from yellowbrick.cluster import KElbowVisualizer
# Instantiate the clustering model and visualizer
km = KMeans (n_clusters=3)
visualizer = KElbowVisualizer(
    km, k=(2,21),metric ='silhouette', timings=False
)

visualizer.fit(Scaled_df) # Fit the training data to the visualizer
visualizer.poof() # Draw/show/poof the data


# Above plot represents mean of Silhouette coefficients for each K value. Higher the mean of Silhouette coefficient, bettter is the clustering. In below plot, at k=3, Silhouette mean is high.

# # 8. Results

# In[ ]:


km = KMeans(n_clusters=3)


# In[ ]:


km.fit(Scaled_df)


# In[ ]:


cluster_label = km.labels_


# In[ ]:


df['KMEANS_LABELS'] = cluster_label


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


f=plt.figure(figsize=(20,20))
scatter_cols =['BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
       'TENURE']
for i, col in enumerate(scatter_cols):
    ax=f.add_subplot(4,4,i+1)
    sns.scatterplot(x=df['BALANCE'],y=df[col],hue=df['KMEANS_LABELS'],palette='Set1')
    ax.set_title(col+" Scatter plot with clusters",color='blue')
    plt.ylabel(col)
f.tight_layout()


# # K Means on selected features

# In[ ]:


sample_df = pd.DataFrame([df['BALANCE'],df['PURCHASES']])
sample_df = sample_df.T
sample_df.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Sample_Scaled_df = scaler.fit_transform(sample_df)


# In[ ]:


km_sample = KMeans(n_clusters=4)
km_sample.fit(Sample_Scaled_df)


# In[ ]:


labels_sample = km_sample.labels_


# In[ ]:


sample_df['label'] = labels_sample


# In[ ]:


sns.set_palette('Set2')
sns.scatterplot(sample_df['BALANCE'],sample_df['PURCHASES'],hue=sample_df['label'],palette='Set1')


# # 9. Summary

# We used K Means to understand patterns based on clusters on scaled dataset. Later did analysis using inertia/elbow curve and Silhoutte Score to find optimal K value. But as we can see the scatterplots after applying (optimal k=3) clustering, still clusters are not as distinct as they are expected to be. This must be a result of many reasons. One such reason is 'CURSE OF DIMENSIONALITY'. Initially, when we applied KMeans on all the columns in dataset, we arrived at a point where we couldn't make clear statement even with optimal K value. Later when we tried KMeans on the selected columns based on which we want to cluster, we are able to see clear groups and label them as:
# 
# * label 0: Low balance and low purchases - Fine group
# * label 1: Moderate to high balance and low purchases - Saving group
# * label 2: Moderate balance and moderate purchases - choosy group
# * label 3: Low to moderate balance and high purchases - Carefree group
# 
# Making this kernel was a learning process and I enjoyed this process. Happy learning :)
