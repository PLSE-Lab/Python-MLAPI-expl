#!/usr/bin/env python
# coding: utf-8

# # TASK #1: UNDERSTAND THE PROBLEM STATEMENT AND BUSINESS CASE

# 
# <table>
#   <tr><td>
#     <img src="https://drive.google.com/uc?id=1OjWCpwRHlCSNYaJoUUd2QGryT9CoQJ5e"
#          alt="Fashion MNIST sprite"  width="1000">
#   </td></tr>
#   <tr><td align="center">
#     <b>Figure 1. Customers Segmentation
#   </td></tr>
# </table>
# 

# ![alt text](https://drive.google.com/uc?id=1Q43AkxxDy4g-zl5lIX4_PBJtTguh4Ise)

# ![alt text](https://drive.google.com/uc?id=1uS6vsccMt3koetsp3k9cAIfbpJw7Z1J8)

# ![alt text](https://drive.google.com/uc?id=1r1FjdO8duujUoI904Oy4vbza6KktxSXo)

# ![alt text](https://drive.google.com/uc?id=1vMr3ouoZ6Pc1mba1mBm2eovlJ3tfE6JA)

# ![alt text](https://drive.google.com/uc?id=1VvqzWWY8wFGeP4cl-rVtWVOg1P6saHfZ)

# ![alt text](https://drive.google.com/uc?id=1LpdL0-4E9lbc4s-x6eJ5zkyIVw_OpHuJ)

# Data Source: https://www.kaggle.com/arjunbhasin2013/ccdata

# # TASK #2: IMPORT LIBRARIES AND DATASETS

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[ ]:


df = pd.read_csv('../input/ccdata/CC GENERAL.csv')
df.head()


# In[ ]:


df.info()
df.describe()


# #### Mean balance is 1564 
# #### Balance frequency is frequently updated on average ~0.9
# #### Purchases average is 1000
# #### one off purchase average is 600
# #### Average purchases frequency is around 0.5
# #### average ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY, and CASH_ADVANCE_FREQUENCY are generally low
# #### Average credit limit ~ 4500
# #### Percent of full payment is 15%
# #### Average tenure is 11 years

# In[ ]:


# Let's see who made one off purchase of $40761!
df[df['ONEOFF_PURCHASES']==40761.250000]


# In[ ]:


df['CASH_ADVANCE'].max()


# In[ ]:


# Let's see who made cash advance of $47137!
# This customer made 123 cash advance transactions!!
# Never paid credit card in full
df[df['CASH_ADVANCE']==47137.211760000006]


# # TASK #3: VISUALIZE AND EXPLORE DATASET

# In[ ]:


# Let's see if we have any missing data, luckily we don't!
sns.heatmap(df.isnull(),yticklabels=False,cmap='Blues',cbar=False)


# In[ ]:


df.isnull().sum()


# In[ ]:


# Fill up the missing elements with mean of the 'MINIMUM_PAYMENT' 
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean())


# In[ ]:


# Fill up the missing elements with mean of the 'CREDIT_LIMIT' 
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean())


# In[ ]:


sns.heatmap(df.isnull(),cbar=False,cmap='Blues',yticklabels=False)


# In[ ]:


# Let's see if we have duplicated entries in the data
df.duplicated().sum()


# In[ ]:


# Let's drop Customer ID since it has no meaning here 
df.drop("CUST_ID",axis=1,inplace=True)
df.head(2)


# In[ ]:


n = len(df.columns)
n


# In[ ]:


df.columns


# In[ ]:


# distplot combines the matplotlib.hist function with seaborn kdeplot()
# KDE Plot represents the Kernel Density Estimate
# KDE is used for visualizing the Probability Density of a continuous variable. 
# KDE demonstrates the probability density at different values in a continuous variable. 
plt.figure(figsize=(10,60))
for i in range(n):
    plt.subplot(17,1,i+1)
    sns.distplot(df[df.columns[i]],kde_kws={'color':'b','bw': 0.1,'lw':3,'label':'KDE'},hist_kws={'color':'r'})
    plt.title(df.columns[i])
plt.tight_layout()


# #### Mean of balance is 1500
# #### 'Balance_Frequency' for most customers is updated frequently ~1
# #### For 'PURCHASES_FREQUENCY', there are two distinct group of customers
# #### For 'ONEOFF_PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY' most users don't do one off puchases or installment purchases frequently 
# #### Very small number of customers pay their balance in full 'PRC_FULL_PAYMENT'~0
# #### Credit limit average is around 4500
# #### Most customers are ~11 years tenure

# In[ ]:


correlations = df.corr()


# In[ ]:


plt.figure(figsize=(16,12))
sns.heatmap(correlations,annot=True)
# 'PURCHASES' have high correlation between one-off purchases, 'installment purchases, purchase transactions, credit limit and payments. 
# Strong Positive Correlation between 'PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY'


# # TASK #4: UNDERSTAND THE THEORY AND INTUITON BEHIND K-MEANS

# ![alt text](https://drive.google.com/uc?id=1EBCmP06GuRjVfPgTfH85Yhv9xIAZUj-K)

# ![alt text](https://drive.google.com/uc?id=1EYWyoec9Be9pYkOaJTjPooTPWgRlJ_Xz)

# ![alt text](https://drive.google.com/uc?id=1ppL-slQPatrmHbPBEaT3-8xNH01ckoNE)

# ![alt text](https://drive.google.com/uc?id=1Yfi-dpWW3keU5RLgwAT4YmQ2rfY1GxUh)

# ![alt text](https://drive.google.com/uc?id=1bLRDIZRda0NSTAdcbugasIjDjvgw4JIU)

# ![alt text](https://drive.google.com/uc?id=1rBQziDU0pS1Fz0m8VQRjQuBoGFSX1Spb)

# ![alt text](https://drive.google.com/uc?id=1BOX2q8R_8E4Icb4v1tpn1eymCTJY2b5o)

# ![alt text](https://drive.google.com/uc?id=1v7hJEPiigSeTTaYo0djbO-L4uEnTpcAU)

# # TASK #5: FIND THE OPTIMAL NUMBER OF CLUSTERS USING ELBOW METHOD

# In[ ]:


# Let's scale the data first
scaler = StandardScaler()


# In[ ]:


scaled_data = scaler.fit_transform(df)
scaled_data.shape


# In[ ]:


scores_1 = []

range_values = range(1,20)
for i in range_values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(scaled_data)
    scores_1.append(kmeans.inertia_)
plt.plot(scores_1, 'bx-')
plt.style.use('ggplot')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores') 
plt.show()


# In[ ]:


# From this we can observe that, 4th cluster seems to be forming the elbow of the curve. 
# However, the values does not reduce linearly until 8th cluster. 
# Let's choose the number of clusters to be 8.


# # TASK #6: APPLY K-MEANS METHOD

# In[ ]:


kmeans = KMeans(8)
kmeans.fit(scaled_data)
labels = kmeans.labels_


# In[ ]:


kmeans.cluster_centers_.shape


# In[ ]:


cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_,columns = [df.columns])
cluster_centers


# In[ ]:


# In order to understand what these numbers mean, let's perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers,columns = [df.columns])
cluster_centers

# First Customers cluster (Transactors): Those are customers who pay least amount of intrerest charges and careful with their money, Cluster with lowest balance ($104) and cash advance ($303), Percentage of full payment = 23%
# Second customers cluster (revolvers) who use credit card as a loan (most lucrative sector): highest balance ($5000) and cash advance (~$5000), low purchase frequency, high cash advance frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)
# Third customer cluster (VIP/Prime): high credit limit $16K and highest percentage of full payment, target for increase credit limit and increase spending habits
# Fourth customer cluster (low tenure): these are customers with low tenure (7 years), low balance 


# In[ ]:


labels.shape # Labels associated to each data point


# In[ ]:


labels.max()


# In[ ]:


labels.min()


# In[ ]:


y_kmeans = kmeans.fit_predict(scaled_data)
y_kmeans


# In[ ]:


# concatenate the clusters labels to our original dataframe
creditcard_df_cluster = pd.concat([df, pd.DataFrame({'cluster':labels})], axis = 1)
creditcard_df_cluster.head()


# In[ ]:


# Plot the histogram of various clusters
for i in df.columns:
  plt.figure(figsize = (35, 5))
  for j in range(8):
    plt.subplot(1,8,j+1)
    cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i,j))
  
  plt.show()


# # TASK 7: APPLY PRINCIPAL COMPONENT ANALYSIS AND VISUALIZE THE RESULTS

# ![alt text](https://drive.google.com/uc?id=1xDuvEnbuNqIjX5Zng39TCfGCf-BBDGf0)

# In[ ]:


# Obtain the principal components 
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(scaled_data)
principal_comp


# In[ ]:


# Create a dataframe with the two components
pca_df = pd.DataFrame(data=principal_comp,columns=['pca1','pca2'])
pca_df.sample(5)


# In[ ]:


# Concatenate the clusters labels to the dataframe
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()


# In[ ]:


plt.figure(figsize=(10,10))
plt.style.use('ggplot')
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink','yellow','gray','purple', 'black'])
plt.show()


# # TASK #8: UNDERSTAND THE THEORY AND INTUITION BEHIND AUTOENCODERS

# ![alt text](https://drive.google.com/uc?id=1g0tWKogvKaCrtsfzjApi6m8yGD3boy4x)

# ![alt text](https://drive.google.com/uc?id=1AcyUL_F9zAD2--Hmyq9yTkcA9mC6-bwg)

# ![alt text](https://drive.google.com/uc?id=1xk1D5uldId0DWywRJ3-OAVBcIr5NGCq_)

# # TASK #9: APPLY AUTOENCODERS (PERFORM DIMENSIONALITY REDUCTION USING AUTOENCODERS)

# In[ ]:


from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from keras.optimizers import SGD


# In[ ]:


encoding_dim = 7

input_df = Input(shape=(17,))


# Glorot normal initializer (Xavier normal initializer) draws samples from a truncated normal distribution 

x = Dense(encoding_dim, activation='relu')(input_df)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(x)

encoded = Dense(10, activation='relu', kernel_initializer = 'glorot_uniform')(x)

x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(encoded)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)

decoded = Dense(17, kernel_initializer = 'glorot_uniform')(x)

# autoencoder
autoencoder = Model(input_df, decoded)

#encoder - used for our dimention reduction
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer= 'adam', loss='mean_squared_error')


# In[ ]:


scaled_data.shape


# In[ ]:


autoencoder.fit(scaled_data,scaled_data,batch_size=128,epochs=25,verbose=1)


# In[ ]:


#autoencoder.save_weights('autoencoder.h5')


# In[ ]:


autoencoder.summary()


# In[ ]:


pred_ac = encoder.predict(scaled_data)


# In[ ]:


pred_ac.shape


# ### As seen that the shape of our input has reduced. Now using this as an input and repeting the whole process with the reduced input. 

# In[ ]:


scores_2 = []

range_values = range(1,20)
for i in range_values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(pred_ac)
    scores_2.append(kmeans.inertia_)
plt.plot(scores_2, 'bx-')
plt.style.use('ggplot')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores') 
plt.show()


# #### So as per my observations I find the optimal value of clusters to be 4 as the curve seems to be linear after 4

# ### Comparing both the results, using Autoencoders and by not using Autoencoders

# In[ ]:


plt.plot(scores_1, 'bx-', color = 'r',label='Without Autoencode')
plt.plot(scores_2, 'bx-', color = 'g',label='With Autoencode')


# ## Apply Kmeans algorithm

# In[ ]:


kmeans = KMeans(4)
kmeans.fit(pred_ac)
labels = kmeans.labels_
kmeans.cluster_centers_.shape


# In[ ]:


y_kmeans = kmeans.fit_predict(scaled_data)
y_kmeans


# In[ ]:


# concatenate the new reduced clusters labels to our original dataframe
creditcard_df_cluster_new = pd.concat([df, pd.DataFrame({'cluster':labels})], axis = 1)
creditcard_df_cluster_new.head()


# In[ ]:


# Plot the histogram of various clusters
for i in df.columns:
  plt.figure(figsize = (20, 5))
  for j in range(4):
    plt.subplot(1,4,j+1)
    cluster = creditcard_df_cluster_new[creditcard_df_cluster_new['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i,j))
  
  plt.show()


# ## APPLY PRINCIPAL COMPONENT ANALYSIS AND VISUALIZE THE RESULTS of the new encoded reduced data

# In[ ]:


# Obtain the principal components 
pca = PCA(n_components=2)
principal_comp_new = pca.fit_transform(pred_ac)
principal_comp_new


# In[ ]:


# Create a dataframe with the two components
pca_df = pd.DataFrame(data=principal_comp_new,columns=['pca1','pca2'])
pca_df.sample(5)


# In[ ]:


# Concatenate the clusters labels to the dataframe
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()


# In[ ]:


plt.figure(figsize=(10,10))
plt.style.use('ggplot')
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink'])
plt.show()


# ## Here we can see that by using autoencoders I was able to make clusters of data with very less overlapping. This is more meaningful clustering/segmentations of the customers. I will now be able to tell my clients that they have 4 different types of customers and each can be targeted in a different way. Autoencoding really helped in this case. 

# ## So to summarize all the steps:
# #### 1. Load the data & just have a brief look at it. Try to find information(.info), use .describe. By doing so you will be able to get good understanding of the data. Try to understand all the features and what do they mean as this is very important to understand which features are the most important or which are the least important. If possible try to ask questions to the team/person who has provided you the dataset. This step is important in a real world project. 
# 
# #### 2. Do some exploratory data analysis (EDA). Find missing values. Handling missing values is a critical step. You have to ask youe self this question. 
# 
# Is this value missing becuase it wasn't recorded or becuase it dosen't exist?
# 
# If a value is missing becuase it doens't exist (like the height of the oldest child of someone who doesn't have any children) then it doesn't make sense to try and guess what it might be. These values you probalby do want to keep as NaN. On the other hand, if a value is missing becuase it wasn't recorded, then you can try to guess what it might have been based on the other values in that column and row. (This is called "imputation") :)
# 
# #### Make some really good graphs by extracting information from the data. As a data scientist you might be asked for presentations of your work/product. Beautiful graphs really helps a lot.
# 
# #### 3. Now comes the Machine learning part. For this dataset I used unsupervised learning. Find the optimum number of clusters by using 'Elbow Method'. Apply Kmeans clustering and then use PCA dimensionality reduction technique to make a graph of your clusters.
# 
# #### 4. Use Autoencoding technique to encode the original data and reduce its dimensions. Then use the reduced encoded data as a new input and follow step 3 again.

# In[ ]:




