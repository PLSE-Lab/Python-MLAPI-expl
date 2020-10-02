#!/usr/bin/env python
# coding: utf-8

# ### Importing the required packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics,preprocessing
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA


# ### Loading the dataset

# In[ ]:


pro = pd.read_csv('../input/Bank_Personal_Loan_Modelling-1.xlsx')
pro.head()


# ### From the description provided in the dataset, we can consider 'Personal Loan' as the target variable

# In[ ]:


## Checking the shape of the dataset

pro.shape


# In[ ]:


pro.columns


# In[ ]:


pro.rename(columns = {'Personal Loan':'Personal_Loan','ZIP Code':'ZIP_Code',
                      'Securities Account':'Securities_Account','CD Account':'CD_Account'},inplace=True)


# ### Descriptive Statistics

# In[ ]:


pro.isnull().sum()


# ###### There are no null values present in the given dataset

# In[ ]:


## Checking for outliers using describe

pro.describe().transpose()


# ###### From the above table, we can infere that income and Mortgage features have outliers in them.

# In[ ]:


sns.boxplot(pro['Age'])   # No outliers in Age Feature


# In[ ]:


sns.boxplot(pro['Experience'])    # No outliers in this feature


# In[ ]:


sns.boxplot(pro['Income'])   # There are few outliers present in the Income feature. We will treat them later


# In[ ]:


sns.boxplot(pro['Family'])   # No outliers


# In[ ]:


sns.boxplot(pro['CCAvg'])    # Outliers are present


# In[ ]:


sns.boxplot(pro['Mortgage'])   # Many outliers are present


# ###### From the above graphs, we can how many outliers are present in each feature of the dataset. Most of the outliers are in Income, Mortgage, and CCAvg

# We will now treat the outliers present in income feature. Since Mortgage and CCAvg are important variables related to our dataset, we are not going to remove the outliers in those two features. We will treat the outliers in Income feature by capping them with upper whisker.

# In[ ]:


up_whisk=pro["Income"].quantile(0.75)+1.5*(pro["Income"].quantile(0.75) - pro["Income"].quantile(0.25))

for i in pro['Income']:
    if(i > up_whisk):
        pro['Income']=pro['Income'].replace(i,up_whisk)


# In[ ]:


sns.boxplot(pro['Income'])  # From the box plot we can now see that the outliers from Income feature has been treated.


# In[ ]:


# Now we will drop the ID feature and ZIP Code feature as they are unrelated to our dataset

pro.drop(['ID','ZIP_Code'],axis = 1, inplace=True)


# In[ ]:


sns.countplot(x = pro['Family'])  # From the count plot, all families have almost equal members in it.


# In[ ]:


sns.countplot(x = pro['Education'])  # Most of the persons in the dataset are undergrads


# In[ ]:


sns.countplot(x = pro['Securities_Account'])  # More number of people do not have securities account at the bank


# In[ ]:


sns.countplot(x = pro['CD_Account']) # more customers have no CD account


# In[ ]:


sns.countplot(x = pro['Online'])  # More number of customers use online transaction compared to offline transactions


# In[ ]:


sns.countplot(x = pro['CreditCard']) # Not many people use credit card issued by the bank


# In[ ]:


sns.countplot(x = pro['Personal_Loan'])  # From the graph, it is very clear that only very few customers get personal loan from the bank.
plt.title('Personal Loan Details')


# In[ ]:


sns.boxplot(y='Income',x='Personal_Loan',data = pro)


# In[ ]:


sns.boxplot(y='CCAvg',x='Personal_Loan',data = pro)


# In[ ]:


# We will now plot a pair plot for the remaining features in the dataset

sns.pairplot(pro)


# In[ ]:


# Since the pairplot shows the correlation among all the features in a graphical way, it becomes difficult to interpret using it.
# So, we will check on the correlation using heatmap and correlation matrix


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(pro.corr(),annot = True)


# In[ ]:


# From the perspective of Personal Loan accpeted by the customer, we can filter the corrmap with only those who accepted the
# personal loan

pro.corr().loc['Personal_Loan']


# #### From the correlation map and pairplot graph, we can infere that Income, CCAvg and CD Account are more corelated to Personal Loan and hence they are the most important features for this dataset

#  

# # Unsupervised  Learning - Without Scaling Data

# In[ ]:


# Dropping the dependant variable from dataset for unsupervised learning

x = pro.drop('Personal_Loan',axis=1)
y_original = pro['Personal_Loan']


# In[ ]:


# First we will proceed with unsupervised learning using K-means clustering method
# For that we need to determine the number of clusters using the elbow graph

cluster_range = range(1,11)
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans(num_clusters)
    clusters.fit(x)
    cluster_errors.append(clusters.inertia_)


# In[ ]:


clusters_df = pd.DataFrame({'num_clusters':cluster_range,'cluster_errors':cluster_errors})
clusters_df[0:11]


# In[ ]:


sns.pointplot(x=clusters_df.num_clusters,y=clusters_df.cluster_errors)


# #### From the Elbow graph we can see that the number of clusters for unscaled data is 2.

# In[ ]:


# Now we will fir the kmean with the newly found number of clusters.

kmean = KMeans(n_clusters=2)
kmean.fit(x)

centers = kmean.cluster_centers_


# In[ ]:


y_kmean = list(kmean.labels_)


# In[ ]:


# Now we will check the unsupervised learning for the same dataset with Hierarchial clustering method.
# We will find the no of clusters using dendrogram

z = linkage(x,'ward')
dendrogram(z)
plt.show()


# In[ ]:


plt.figure(figsize=(25,10))
plt.title('Hierarchial Clustering Dendrogram')
dendrogram(z, leaf_rotation=90,leaf_font_size=8)
plt.show()


# In[ ]:


# We will truncate the dendrogram for easier visibility.

plt.figure(figsize=(25,10))
plt.title('Hierarchial Clustering Dendrogram (Truncated)')
dendrogram(z,truncate_mode='lastp',p = 16,show_leaf_counts=True, leaf_rotation=90,leaf_font_size=12,show_contracted=True)
plt.show()


# #### From the dendrogram it is clear that the number of clusters in the given dataset is 2.

# #### We will now perform Agglomerative Clustering using the number of clusters obtained above

# In[ ]:


cluster = AgglomerativeClustering(n_clusters = 2, linkage = 'ward')
y_hier = cluster.fit_predict(x)


# In[ ]:


print(metrics.accuracy_score(y_original,y_kmean))


# In[ ]:


print(metrics.accuracy_score(y_original,y_hier))


# ## By comparing the accuracy scores obtained from K-mean and Hierarical with original target, we can see that using K-Means provide better accuracy. Hence we will use K-means Clustering from hereon.

# In[ ]:


# First we will perform PCA using all the available features in the dataset. We will find the required number of components using
# elbow graph and we will fit it using the found number of clusters


# In[ ]:


pca = PCA(n_components=11)
pca.fit_transform(x)


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(pca.explained_variance_ratio_,marker='o')
plt.xlabel('number_of_components')
plt.ylabel('cumulative explained variance')
plt.xticks(range(12))
plt.show()


# ### From the elbow graph it is clear that the number of components required for PCA is 2

# In[ ]:


pca_new = PCA(n_components=2)


# In[ ]:


# We will split the train test data with x as unscaled x, y as y obtained from k_mean, and we will fit x data with pca

x_train1,x_test1,y_train1,y_test1 = train_test_split(x,y_kmean,test_size = 0.3, random_state = 0)
x_train1 = pca_new.fit_transform(x_train1)
x_test1 = pca_new.transform(x_test1)


# In[ ]:


rf = RandomForestClassifier()
rf.fit(x_train1,y_train1)
y_pred1_rf = rf.predict(x_test1)
random_forest_unscaled = metrics.accuracy_score(y_test1,y_pred1_rf)
print(metrics.accuracy_score(y_test1,y_pred1_rf))


# In[ ]:


log = LogisticRegression()
log.fit(x_train1,y_train1)
y_pred1_log = log.predict(x_test1)
log_unscaled = metrics.accuracy_score(y_test1,y_pred1_log)
print(metrics.accuracy_score(y_test1,y_pred1_log))


# In[ ]:


tree = DecisionTreeClassifier(max_depth = 4)
tree.fit(x_train1,y_train1)
y_pred1_tree = tree.predict(x_test1)
tree_unscaled = metrics.accuracy_score(y_test1,y_pred1_tree)
print(metrics.accuracy_score(y_test1,y_pred1_tree))


# In[ ]:


knn = KNeighborsClassifier()
knn.fit(x_train1,y_train1)
y_pred1_knn = knn.predict(x_test1)
knn_unscaled = metrics.accuracy_score(y_test1,y_pred1_knn)
print(metrics.accuracy_score(y_test1,y_pred1_knn))


# In[ ]:


nb = GaussianNB()
nb.fit(x_train1,y_train1)
y_pred1_nb = nb.predict(x_test1)
nb_unscaled = metrics.accuracy_score(y_test1,y_pred1_nb)
print(metrics.accuracy_score(y_test1,y_pred1_nb))


# In[ ]:


svm = SVC()
svm.fit(x_train1,y_train1)
y_pred1_svm = svm.predict(x_test1)
svm_unscaled = metrics.accuracy_score(y_test1,y_pred1_svm)
print(metrics.accuracy_score(y_test1,y_pred1_svm))


# #### Thus the accuracy scores for all 6 models using Unscaled data is shown above

# ## Now we will scale the data using Standard Scalar

# In[ ]:


x_std = preprocessing.StandardScaler().fit_transform(x)


# #### Now we will again repeat the steps above starting from fitting the data in PCA

# In[ ]:


pca = PCA(n_components=11)
pca.fit_transform(x_std)


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot(pca.explained_variance_ratio_,marker='o')
plt.xlabel('number_of_components')
plt.ylabel('cumulative explained variance')
plt.xticks(range(12))
plt.show()


# #### From the above elbow graph, the number of components in scaled data is 3

# In[ ]:


pca_new_scaled = PCA(n_components=3)


# In[ ]:


# We will split the train test data with x as scaled x, y as y obtained from k_mean, and we will fit x data with pca

x_train2,x_test2,y_train2,y_test2 = train_test_split(x_std,y_kmean,test_size = 0.3, random_state = 0)
x_train2 = pca_new.fit_transform(x_train2)
x_test2 = pca_new.transform(x_test2)


# In[ ]:


rf2 = RandomForestClassifier()
rf2.fit(x_train2,y_train2)
y_pred2_rf = rf2.predict(x_test2)
random_forest_scaled = metrics.accuracy_score(y_test2,y_pred2_rf)
print(metrics.accuracy_score(y_test2,y_pred2_rf))


# In[ ]:


log2 = LogisticRegression()
log2.fit(x_train2,y_train2)
y_pred2_log = log2.predict(x_test2)
log_scaled = metrics.accuracy_score(y_test2,y_pred2_log)
print(metrics.accuracy_score(y_test2,y_pred2_log))


# In[ ]:


tree2 = DecisionTreeClassifier(max_depth = 4)
tree2.fit(x_train2,y_train2)
y_pred2_tree = tree2.predict(x_test2)
tree_scaled = metrics.accuracy_score(y_test2,y_pred2_tree)
print(metrics.accuracy_score(y_test2,y_pred2_tree))


# In[ ]:


knn2 = KNeighborsClassifier()
knn2.fit(x_train2,y_train2)
y_pred2_knn = knn2.predict(x_test2)
knn_scaled = metrics.accuracy_score(y_test2,y_pred2_knn)
print(metrics.accuracy_score(y_test2,y_pred2_knn))


# In[ ]:


nb2 = GaussianNB()
nb2.fit(x_train2,y_train2)
y_pred2_nb = nb2.predict(x_test2)
nb_scaled = metrics.accuracy_score(y_test2,y_pred2_nb)
print(metrics.accuracy_score(y_test2,y_pred2_nb))


# In[ ]:


svm2 = SVC()
svm2.fit(x_train2,y_train2)
y_pred2_svm = svm2.predict(x_test2)
svm_scaled = metrics.accuracy_score(y_test2,y_pred2_svm)
print(metrics.accuracy_score(y_test2,y_pred2_svm))


# ##### Thus the accuracy of all 6 models using scaled data is shown above

#  

# ## Now we will verify the above accuracy scores of unscaled data and scaled data using target as kmean with using original y as target as done in supervised learning.

# In[ ]:


# We will start with train test split of x and y with y as y_original from the original dataset

x_train3,x_test3,y_train3,y_test3 = train_test_split(x,y_original,test_size = 0.3, random_state = 0)


# In[ ]:


rf3 = RandomForestClassifier()
rf3.fit(x_train3,y_train3)
y_pred3_rf = rf3.predict(x_test3)
random_forest_original = metrics.accuracy_score(y_test3,y_pred3_rf)
print(metrics.accuracy_score(y_test3,y_pred3_rf))


# In[ ]:


log3 = LogisticRegression()
log3.fit(x_train3,y_train3)
y_pred3_log = log3.predict(x_test3)
log_original = metrics.accuracy_score(y_test3,y_pred3_log)
print(metrics.accuracy_score(y_test3,y_pred3_log))


# In[ ]:


tree3 = DecisionTreeClassifier(max_depth = 4)
tree3.fit(x_train3,y_train3)
y_pred3_tree = tree3.predict(x_test3)
tree_original = metrics.accuracy_score(y_test3,y_pred3_tree)
print(metrics.accuracy_score(y_test3,y_pred3_tree))


# In[ ]:


knn3 = KNeighborsClassifier()
knn3.fit(x_train3,y_train3)
y_pred3_knn = knn3.predict(x_test3)
knn_original = metrics.accuracy_score(y_test3,y_pred3_knn)
print(metrics.accuracy_score(y_test3,y_pred3_knn))


# In[ ]:


nb3 = GaussianNB()
nb3.fit(x_train3,y_train3)
y_pred3_nb = nb3.predict(x_test3)
nb_original = metrics.accuracy_score(y_test3,y_pred3_nb)
print(metrics.accuracy_score(y_test3,y_pred3_nb))


# In[ ]:


svm3 = SVC()
svm3.fit(x_train3,y_train3)
y_pred3_svm = svm3.predict(x_test3)
svm_original = metrics.accuracy_score(y_test3,y_pred3_svm)
print(metrics.accuracy_score(y_test3,y_pred3_svm))


# In[ ]:


accuracy = pd.DataFrame([[random_forest_unscaled,random_forest_scaled,random_forest_original],
                         [log_unscaled,log_scaled,log_original],
                         [tree_unscaled,tree_scaled,tree_original],
                         [knn_unscaled,knn_scaled,knn_original],
                         [nb_unscaled,nb_scaled,nb_original],
                         [svm_unscaled,svm_scaled,svm_original]],
                       columns = ['Unscaled Data with PCA','Scaled Data with PCA','Supervised Learning'],
                       index = ['Random Forest Model','Logistic Regression Model','Decision Tree Model','KNN Model',
                               'Naive Bayes Model','Support Vector Machine Model'])
accuracy


# ## From the above comparisons of the accuracy scores, we can conclude that the accuracy of Random Forest Model and Decision Tree Model using unscaled data with PCA is very much higher than the rest of the models and also higher than scaled data and using original target variable.
