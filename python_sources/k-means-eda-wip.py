import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#exploratory
df = pd.read_csv('../input/Mall_Customers.csv')
df.head()
df.info()
df.describe()
df.isna().any()

#basic graphs
plt.title('Distplot of Age')
sns.distplot(df['Age'] , bins = 20)
plt.title('Distplot of Annual Income (k$)')
sns.distplot(df['Annual Income (k$)'] , bins = 20)
plt.title('Distplot of Spending Score (1-100)')
sns.distplot(df['Spending Score (1-100)'] , bins = 20)

# counts
sns.countplot(y='Gender', data=df)

# groups
bins = [0, 15, 40, 90, 120, np.inf]
names = ['<15', '15-40', '40-90', '90-120', '120+']
df['AnnualRange'] = pd.cut(df['Annual Income (k$)'], bins, labels=names)

bins = [17, 20, 25, 35, 45, np.inf]
names = ['<20', '20-25', '25-30', '35-45', '45+']
df['AgeRange'] = pd.cut(df['Age'], bins, labels=names)


bins = [0, 20, 40, 70, 90, np.inf]
names = ['0-20', '20-40', '40-70', '70-90', '90+']
df['SpendingScoreRange'] = pd.cut(df['Spending Score (1-100)'], bins, labels=names)

# mass graphs
plt.figure(1 , figsize = (15 , 7))
n = 0 
for i in ['AgeRange' , 'AnnualRange' , 'SpendingScoreRange']:
    n += 1 
    plt.subplot(1 , 3 , n)
    sns.countplot(x="Gender", hue=i, data=df)
plt.show()
# counts #2 Gender
sns.countplot(x="Gender", hue="AgeRange", data=df)
sns.countplot(x="Gender", hue="AnnualRange", data=df)
sns.countplot(x="Gender", hue="SpendingScoreRange", data=df)
# counts #2 AgeRange
sns.countplot(x="AgeRange", hue="AnnualRange", data=df)
sns.countplot(x="AgeRange", hue="SpendingScoreRange", data=df)


X = df.iloc[:, [3, 4]].values
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'cyan', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()