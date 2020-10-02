# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [code]
sm_data=pd.read_csv("../input/spencer-supermarket-data-east-india/Supermarket Purchase.csv")

# %% [markdown]
# Exploratory Data Analysis

# %% [code]
sm_data.shape

# %% [code]
sm_data.info()

# %% [markdown]
# No Null Values in the dataset

# %% [code]
#Rename column names for easy ledigibility
sm_data.rename(columns={'AVG_Actual_price_12': 'MRP', 'MONTH_SINCE_LAST_TRANSACTION': 'Month_of_Visit'}, inplace=True)
sm_data.columns

# %% [code]
sm_data.describe()

# %% [markdown]
# The data does seem to showcase some level of uneven distribution, which needs to be investigated along with negative values under the Total_Discount column

# %% [code]
#Analysing purchase behvaiour of customers
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
sns.countplot(x='Month_of_Visit', data=sm_data)

plt.subplot(2,2,2)
sns.barplot(x='Month_of_Visit', y='Purchase_Value', data=sm_data)

plt.subplot(2,2,3)
sns.barplot(x='Month_of_Visit', y='Total_Discount', data=sm_data)

plt.subplot(2,2,4)
sns.barplot(x='Month_of_Visit', y='No_of_Items', data=sm_data)

# %% [markdown]
# Observations:
# 1. The counts of visits is in January, which also translates to maximum items sold by value and volume
# 2. However, we also observe that the maximum discounts are also given in January
# 3. From initial observation, it does seem that discounts translate into more sales and in months of low total discount, the sale numbers (Purchase_Value) is also low. Another indication for this pattern is that the count is similar in the months of March, April and May and yet we see a differentual in the purchase value in these months, with variation following the discount pattern 

# %% [code]
#Investigating Discount value behvaiour
sm_data['Purchase Value']=sm_data['Purchase_Value']-sm_data['Total_Discount']  #Since discount is always given on MRP

# %% [code]
sm_data.head()

# %% [code]
sum(sm_data['Total_Discount']>=sm_data['Purchase_Value'])

# %% [code]
sum(sm_data['Total_Discount']<0)

# %% [markdown]
# There are 251 row counts wherein discount value is greater than Purchase Value which does not seem correct along with 18 negative values for Total_Discount. We will use the absolute value for Discount and ignore the Total_Discount and Purchase_Value due to less information available. It could be because of reward program or something similar for customers, hypothetically speaking.

# %% [code]
sm_data['Total_Discount']=abs(sm_data['Total_Discount'])
sum(sm_data['Total_Discount']<0)

# %% [code]
#Checking results post conversion
sm_data.describe()

# %% [code]
sns.barplot(x='Month_of_Visit', y='Total_Discount', data=sm_data)

# %% [markdown]
# Post conversion values of Discount, there is no change in the distribution. Hence, we can safely proceed with the approach

# %% [code]
plt.figure(figsize=(20,10))
sns.barplot(x='Cust_id',y='Purchase_Value', data=sm_data)
plt.show()

# %% [markdown]
# It does seem only limited customers are the high value consumers. Time to check this in detail

# %% [code]
Purchase_percentage=(sm_data.groupby('Cust_id').Purchase_Value.sum()/sm_data['Purchase_Value'].sum())*100

# %% [code]
Purchase_percentage.sort_values(ascending=False)

# %% [code]
Purchase_percentage.head(550).sum()

# %% [markdown]
# With the pervious observation, 550 customers (of a total of 702) contribute to 80% of the sales, hence the sales contribution of customer is low, with highest in the ~2% contribution range

# %% [code]
#Analyse data distribution
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
sns.distplot(sm_data['Purchase_Value'],color='blue')

plt.subplot(2,2,2)
sns.distplot(sm_data['MRP'], color='green')

plt.subplot(2,2,3)
sns.distplot(sm_data['Total_Discount'], color='red')

plt.subplot(2,2,4)
sns.distplot(sm_data['Purchase Value'])

plt.show()

# %% [markdown]
# The data distribution does show some degree of skewness and Kurtosis in the dataset. However, the calculated Purchase Value column does offset the Purchase_Value and Total_Discount entries. However, we will use the original columns to ensure unecessary dimensionality reduction which might lead to some unseen observations

# %% [code]
#Correlation Map
sm_corr=sm_data.corr()
sns.heatmap(sm_corr,annot=True)
plt.show()

# %% [code]
#Check distribution of entire dataset
X=sm_data.drop(['Purchase Value', 'Cust_id'], axis=1)

plt.figure(figsize=(15,10))
plt.hist(X)
plt.show()

# %% [code]
#Scale the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

y=scaler.fit_transform(X)

plt.figure(figsize=(15,10))
plt.hist(y)
plt.show()


# %% [markdown]
# The data distribution looks better after scaling however, is still left skewed. We will attempt a log transform in an attempt to normalize the data

# %% [code]
#Log transformation

X['Purchase_Value']=np.log1p(X['Purchase_Value'])
X['MRP']=np.log1p(X['MRP'])
X['No_of_Items']=np.log1p(X['No_of_Items'])
X['Total_Discount']=np.log1p(X['Total_Discount'])

# %% [code]
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
sns.distplot(X['Purchase_Value'],color='blue')

plt.subplot(2,2,2)
sns.distplot(X['MRP'], color='green')

plt.subplot(2,2,3)
sns.distplot(X['Total_Discount'], color='red')

plt.subplot(2,2,4)
sns.distplot(X['No_of_Items'])


plt.show()

# %% [code]
X.columns

# %% [code]
# Check distribution of entire data
plt.figure(figsize=(15,10))
plt.hist(X)
plt.show()

# %% [code]
#Scale the log transformed dataset
y1=scaler.fit_transform(X)
plt.hist(y1)
plt.show()

# %% [code]
#Log transformed dataset characteristics
X.describe()

# %% [markdown]
# The dataset seems to be scaled and normalized now. 

# %% [code]
#Pair Plot comparison of original and transformed dataset

sns.pairplot(sm_data)
plt.show()

# %% [code]
#Pair Plot comparison of original and transformed dataset

sns.pairplot(X)
plt.show()

# %% [markdown]
# Clustering Process

# %% [code]
cluster_feature=['MRP', 'Total_Discount','Month_of_Visit', 'No_of_Items']
cluster_data=X[cluster_feature]
cluster_data.head()


# %% [markdown]
# Dropping Purchase_Value due to high correlation with No. of Items purchased

# %% [code]
plt.hist(cluster_data)
plt.show()

# %% [code]
cluster_features = pd.pivot_table(X, values=['MRP', 'Purchase_Value','No_of_Items'],index='Total_Discount',aggfunc=np.mean)
X=cluster_features.values

# %% [markdown]
# K-means clustering

# %% [code]

from sklearn.cluster import KMeans
ssw=[]
cluster_range=range(1,25)
for i in cluster_range:
    model=KMeans(n_clusters=i, init="k-means++",n_init=10, max_iter=300, random_state=10)
    model.fit(X)
    ssw.append(model.inertia_)
    

# %% [code]
# Elbow method to determing ideal number of clusters
plt.figure(figsize=(12,7))
plt.plot(cluster_range, ssw, marker = "o",color="red")
plt.show()

# %% [code]
ssw_df=pd.DataFrame({"no. of clusters":cluster_range,"SSW":ssw})
print(ssw_df)

# %% [markdown]
# Basis the SSW values, and for ease of identification from a business perspective, we can have 6 clusters

# %% [code]
kmeans=KMeans(n_clusters=6, init="k-means++", n_init=10, random_state=10)
k_model=kmeans.fit(X)
cluster_km=kmeans.predict(X)

# %% [code]
plt.figure(figsize=(15,10))

#plt.subplot(2,2,1)
plt.title("K-means clustering")
plt.scatter(X[cluster_km==0, 0], X[cluster_km==0, 1], s=50, marker='o', color='red')
plt.scatter(X[cluster_km==1, 0], X[cluster_km==1, 1], s=50, marker='o', color='blue')
plt.scatter(X[cluster_km==2, 0], X[cluster_km==2, 1], s=50, marker='o', color='green')
plt.scatter(X[cluster_km==3, 0], X[cluster_km==3, 1], s=50, marker='o', color='purple')
plt.scatter(X[cluster_km==4, 0], X[cluster_km==4, 1], s=50, marker='o', color='orange')
plt.scatter(X[cluster_km==5, 0], X[cluster_km==5, 1], s=50, marker='o', color='brown')

plt.ylabel('No_of_Items')
plt.xlabel('MRP')





plt.show()

# %% [code]
#Hierarchical clustering
from scipy.cluster.hierarchy import linkage,dendrogram

merg=linkage(X,method="ward")
dendrogram(merg, leaf_rotation=90)

plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

# %% [markdown]
# Basis the dendogram, we also come up with a total of 6 clusters at an Euclidean distance of about 10

# %% [code]
#Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

Agg_cluster=AgglomerativeClustering(n_clusters=6, affinity="euclidean", linkage="ward")
cluster_agg=Agg_cluster.fit_predict(X)
cluster_agg

# %% [code]
plt.figure(figsize=(15,10))
plt.title("Hierarchical clustering")
plt.scatter(X[cluster_agg==0, 0], X[cluster_agg==0, 1], s=50, marker='o', color='red')
plt.scatter(X[cluster_agg==1, 0], X[cluster_agg==1, 1], s=50, marker='o', color='blue')
plt.scatter(X[cluster_agg==2, 0], X[cluster_agg==2, 1], s=50, marker='o', color='green')
plt.scatter(X[cluster_agg==3, 0], X[cluster_agg==3, 1], s=50, marker='o', color='purple')
plt.scatter(X[cluster_agg==4, 0], X[cluster_agg==4, 1], s=50, marker='o', color='orange')
plt.scatter(X[cluster_agg==5, 0], X[cluster_agg==5, 1], s=50, marker='o', color='brown')

plt.ylabel('No_of_Items')
plt.xlabel('MRP')
plt.show()

# %% [code]
#Gaussian Mixed model with random method
from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=6, init_params='random')
cluster_gmm=gmm.fit_predict(X)


# %% [code]
plt.figure(figsize=(15,10))

plt.title("GMM random clustering")
plt.scatter(X[cluster_gmm==0, 0], X[cluster_gmm==0, 1], s=50, marker='o', color='red')
plt.scatter(X[cluster_gmm==1, 0], X[cluster_gmm==1, 1], s=50, marker='o', color='blue')
plt.scatter(X[cluster_gmm==2, 0], X[cluster_gmm==2, 1], s=50, marker='o', color='green')
plt.scatter(X[cluster_gmm==3, 0], X[cluster_gmm==3, 1], s=50, marker='o', color='purple')
plt.scatter(X[cluster_gmm==4, 0], X[cluster_gmm==4, 1], s=50, marker='o', color='orange')
plt.scatter(X[cluster_gmm==5, 0], X[cluster_gmm==5, 1], s=50, marker='o', color='brown')

plt.ylabel('No_of_Items')
plt.xlabel('MRP')
plt.show()

# %% [code]
#Gaussian Mixed model with kmeans method
gmm_km=GaussianMixture(n_components=6, init_params='kmeans')
cluster_gmmk=gmm_km.fit_predict(X)


# %% [code]
cluster_gmmk=gmm_km.predict(X)


# %% [code]
plt.figure(figsize=(15,10))
plt.title("GMM kmeans clustering")
plt.scatter(X[cluster_gmmk==0, 0], X[cluster_gmmk==0, 1], s=50, marker='o', color='red')
plt.scatter(X[cluster_gmmk==1, 0], X[cluster_gmmk==1, 1], s=50, marker='o', color='blue')
plt.scatter(X[cluster_gmmk==2, 0], X[cluster_gmmk==2, 1], s=50, marker='o', color='green')
plt.scatter(X[cluster_gmmk==3, 0], X[cluster_gmmk==3, 1], s=50, marker='o', color='purple')
plt.scatter(X[cluster_gmmk==4, 0], X[cluster_gmmk==4, 1], s=50, marker='o', color='orange')
plt.scatter(X[cluster_gmmk==5, 0], X[cluster_gmmk==5, 1], s=50, marker='o', color='brown')

plt.ylabel('No_of_Items')
plt.xlabel('MRP')
plt.show()

# %% [markdown]
# Gaussian Mixed model algorithm was run with both ‘random’ and ‘kmeans’ method. However, we do observe that the cluster assignment in this plane is rather erratic hence does not seem to be an optimal clustering method for the dataset. 
# 