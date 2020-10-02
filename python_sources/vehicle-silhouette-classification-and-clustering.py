#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Reading and understanding the data-set

# In[ ]:


data=pd.read_csv("/kaggle/input/vehicle.csv")


# In[ ]:


#Checking the head of the data
data.head(10)


# In[ ]:


#Checking the dtypes of the data
data.dtypes


# #### All the attributes excpet class is of integer or float, class will be dropped as it defeats the purpose of an unsupervised learning model

# In[ ]:


#Checking the information of the data
data.info()


# #### We can infer that there are some null values in the following attributes,
# ##### Circularity, distance_circularity, radius_ratio, pr.axis_aspect_ratio, scatter_ratio, elongatedness,pr.axis_rectangularity, scaled_variance, scaled_variance.1, scaled_radius_of_gyration, scaled_radius_of_gyration.1, skewness_about, skewness_about.1, skewness_about.2 

# In[ ]:


# Checking the shape of the data
data.shape


# #### We can infer that we have 846 records with 18 independent attributes excluding class attribute

# In[ ]:


#To check if there are any null values present
nulllvalues=data.isnull().sum()
print(nulllvalues)


# #### We can infer that there are some null values in the following attributes,
# ##### Circularity, distance_circularity, radius_ratio, pr.axis_aspect_ratio, scatter_ratio, elongatedness,pr.axis_rectangularity, scaled_variance, scaled_variance.1, scaled_radius_of_gyration, scaled_radius_of_gyration.1, skewness_about, skewness_about.1, skewness_about.2 

# In[ ]:


# Filling the Null values with the median of the column
data=data.fillna(data.median())
data.head()


# In[ ]:


#To check if there are any NaN values present
NaNvalues=data.isna().sum()
print(NaNvalues)


# #### All null values have been replaced with their median

# In[ ]:


# To describe the data- Five point summary
data.describe().T


# #### Mean of scatter_ratio, elongatedness, scaled_variance, scaled_variance.1 is not in sync with the median which infers that they have outliers

# ## Exploratory Data Analytics

# ### Univariate Analysis

# In[ ]:


#Distribution of continous data

plt.figure(figsize=(30,6))

#Subplot 1
plt.subplot(1,3,1)
plt.title('compactness')
sns.distplot(data['compactness'],color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('circularity')
sns.distplot(data['circularity'],color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('distance_circularity')
sns.distplot(data['distance_circularity'],color='green')



plt.figure(figsize=(30,6))

#Subplot 1- Boxplot
plt.subplot(1,3,1)
plt.title('compactness')
sns.boxplot(data['compactness'],orient='horizondal',color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('Circularity')
sns.boxplot(data['circularity'],orient='horizondal',color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('distance_circularity')
sns.boxplot(data['distance_circularity'],orient='horizondal',color='green')


# #### Average compactness is between 85 and 100
# #### Average Circularity is between 40 and 48
# #### Average distance_circularity is between 70 and 100
# #### There are no outliers

# In[ ]:


#Distribution of continous data

plt.figure(figsize=(30,6))

#Subplot 1
plt.subplot(1,3,1)
plt.title('radius_ratio')
sns.distplot(data['radius_ratio'],color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('max.length_aspect_ratio')
sns.distplot(data['max.length_aspect_ratio'],color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('scatter_ratio')
sns.distplot(data['scatter_ratio'],color='green')



plt.figure(figsize=(30,6))

#Subplot 1- Boxplot
plt.subplot(1,3,1)
plt.title('radius_ratio')
sns.boxplot(data['radius_ratio'],orient='horizondal',color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('max.length_aspect_ratio')
sns.boxplot(data['max.length_aspect_ratio'],orient='horizondal',color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('scatter_ratio')
sns.boxplot(data['scatter_ratio'],orient='horizondal',color='green')


# #### Average radius_ratio is between 140 and 200 and there are some outliers
# #### Max_length_aspect_ratio is between 7 and 10 with some amount of outliers
# #### scatter_ratio is between 150 and 200

# In[ ]:


#Distribution of continous data

plt.figure(figsize=(30,6))

#Subplot 1
plt.subplot(1,3,1)
plt.title('elongatedness')
sns.distplot(data['elongatedness'],color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('pr.axis_rectangularity')
sns.distplot(data['pr.axis_rectangularity'],color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('max.length_rectangularity')
sns.distplot(data['max.length_rectangularity'],color='green')



plt.figure(figsize=(30,6))

#Subplot 1- Boxplot
plt.subplot(1,3,1)
plt.title('elongatedness')
sns.boxplot(data['elongatedness'],orient='horizondal',color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('pr.axis_rectangularity')
sns.boxplot(data['pr.axis_rectangularity'],orient='horizondal',color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('max.length_rectangularity')
sns.boxplot(data['max.length_rectangularity'],orient='horizondal',color='green')


# #### Average elongatedness is between 35 and 45
# #### Average pr.axis_regtangularity is between 19 and 23
# #### Average max_length_rectangularity between 140 and 160
# #### There are no outliers

# In[ ]:


#Distribution of continous data

plt.figure(figsize=(30,6))

#Subplot 1
plt.subplot(1,3,1)
plt.title('pr.axis_aspect_ratio')
sns.distplot(data['pr.axis_aspect_ratio'],color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('scaled_variance')
sns.distplot(data['scaled_variance'],color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('scaled_variance.1')
sns.distplot(data['scaled_variance.1'],color='green')



plt.figure(figsize=(30,6))

#Subplot 1- Boxplot
plt.subplot(1,3,1)
plt.title('pr.axis_aspect_ratio')
sns.boxplot(data['pr.axis_aspect_ratio'],orient='horizondal',color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('scaled_variance')
sns.boxplot(data['scaled_variance'],orient='horizondal',color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('scaled_variance.1')
sns.boxplot(data['scaled_variance.1'],orient='horizondal',color='green')
    


# #### Average pr.axis_aspect_ratio is between 55 and 65 and there are some outliers
# #### scaled_variance is between 170 and 275 with some amount of outliers
# #### scaled_variance1 is between 375 and 600 with some amount of outliers

# In[ ]:


#Distribution of continous data

plt.figure(figsize=(30,6))

#Subplot 1
plt.subplot(1,3,1)
plt.title('scaled_radius_of_gyration')
sns.distplot(data['scaled_radius_of_gyration'],color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('scaled_radius_of_gyration.1')
sns.distplot(data['scaled_radius_of_gyration.1'],color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('skewness_about')
sns.distplot(data['skewness_about'],color='green')



plt.figure(figsize=(30,6))

#Subplot 1- Boxplot
plt.subplot(1,3,1)
plt.title('scaled_radius_of_gyration')
sns.boxplot(data['scaled_radius_of_gyration'],orient='horizondal',color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('scaled_radius_of_gyration.1')
sns.boxplot(data['scaled_radius_of_gyration.1'],orient='horizondal',color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('skewness_about')
sns.boxplot(data['skewness_about'],orient='horizondal',color='green')
    


# #### scaled_radius_of_gyration is between 145 and 200
# #### scaled_radius_of_gyration.1 is between 65 and 75 with some amount of outliers
# #### skewness_about is between 3 and 10 with some outliers

# In[ ]:


#Distribution of continous data

plt.figure(figsize=(30,6))

#Subplot 1
plt.subplot(1,3,1)
plt.title('skewness_about.1')
sns.distplot(data['skewness_about.1'],color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('skewness_about.2')
sns.distplot(data['skewness_about.2'],color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('hollows_ratio')
sns.distplot(data['hollows_ratio'],color='green')



plt.figure(figsize=(30,6))

#Subplot 1- Boxplot
plt.subplot(1,3,1)
plt.title('skewness_about.1')
sns.boxplot(data['skewness_about.1'],orient='horizondal',color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('skewness_about.2')
sns.boxplot(data['skewness_about.2'],orient='horizondal',color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('hollows_ratio')
sns.boxplot(data['hollows_ratio'],orient='horizondal',color='green')
    


# #### skewness_about_1 is between 5 and 20 with some amount of outliers
# #### kewness_about_2 is between 185 and 195 
# #### hollows_ratio is between 190 and 200

# In[ ]:


sns.countplot(data['class'])


# #### Cars are more in this data-set compared to van and bus

# ### Multi-variate analyisis

# In[ ]:


sns.pairplot(data, hue='class',palette="Set1", diag_kind="kde", height=2.5)


# In[ ]:


#To find the correlation between the continous variables
correlation=data.corr()
correlation.style.background_gradient(cmap='coolwarm')


# In[ ]:


df=data.drop(['class'],axis=1)
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=20):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 20))


# #### These pairs of independednt attibutes have strong correlation

# In[ ]:


sns.heatmap(correlation)


# #### There is strong correlation between the independednt variables

# In[ ]:


#Dropping the class attribute
new_data=data.drop(['class'],axis=1)


# In[ ]:


new_data.head()


# In[ ]:


#Import neccessary libraries
from scipy.stats import zscore


# ## Applying z-score to scale the data and standardize the data

# In[ ]:


new_data=new_data.apply(zscore)


# In[ ]:


new_data.head()


# In[ ]:


new_data.info()


# In[ ]:


new_data=new_data.join(data['class'])


# ## Removing all columns with z-score greater and lesser than 3 and -3 respectivley as the values are outliers
# 

# In[ ]:


floats = new_data.columns[new_data.dtypes == 'float64']
for x in floats:
    indexNames_larger = new_data[ new_data[x]>3].index
    indexNames_lesser = new_data[ new_data[x]<-3].index
    # Delete these row indexes from dataFrame
    new_data.drop(indexNames_larger , inplace=True)
    new_data.drop(indexNames_lesser , inplace=True)
    data.drop(indexNames_larger , inplace=True)
    data.drop(indexNames_lesser , inplace=True)
new_data.head()


# In[ ]:


data.shape


# In[ ]:


new_data=new_data.drop(['class'],axis=1)


# In[ ]:


new_data.shape


# #### 22 records have been removed as they are outliers

# In[ ]:


new_data.head()


# In[ ]:


data.head()


# ## Principal Component Analysis

# ### Covariance Matrix

# In[ ]:


cov_matrix=np.cov(new_data,rowvar=False)
print(cov_matrix)


# In[ ]:


#Importing necessary libraries
from sklearn.decomposition import PCA


# In[ ]:


#Creating the model and fitting the model
pca_model= PCA(n_components=18)


# In[ ]:


#Fitting the model
pca_model.fit(new_data)


# ### Eigen Values

# In[ ]:


pca_model.explained_variance_


# ### Eigen Vectors

# In[ ]:


print(pca_model.components_)


# ### % of variation

# In[ ]:


print(pca_model.explained_variance_ratio_)


# In[ ]:


plt.bar(list(range(1,19)),pca_model.explained_variance_ratio_,alpha=0.5, align='center')
plt.ylabel('Variation explained')
plt.xlabel('eigen Value')
plt.show()


# In[ ]:


plt.step(list(range(1,19)),np.cumsum(pca_model.explained_variance_ratio_), where='mid')
plt.ylabel('Cum of variation explained')
plt.xlabel('eigen Value')
plt.show()


# In[ ]:


plt.figure()
plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()


# ## Dimensionality Reduction
# 
# ### Now 6 dimensions seems good. With 6 variables we can explain over 95% of the variation in the original data

# In[ ]:


PCA_6dim= PCA(n_components=6)


# In[ ]:


PCA_6dim.fit(new_data)


# ### Eigen Vectors

# In[ ]:


print(PCA_6dim.components_)


# ### Eigen values

# In[ ]:


print(PCA_6dim.explained_variance_)


# ### % of variance

# In[ ]:


print(PCA_6dim.explained_variance_ratio_)


# In[ ]:


# Transforming the data
PCA_6dimT=PCA_6dim.transform(new_data)


# In[ ]:


PCA_6dimT


# In[ ]:


sns.pairplot(pd.DataFrame(PCA_6dimT))


# ### Now the independent variables are not corrleated and are truly independednt

# ### Created PCA data frame

# In[ ]:


PCA_data=pd.DataFrame(PCA_6dimT)
PCA_data.head()


# # Support Vector Classifier- without PCA, Hyper-parameterization and Cross-validation

# ### Defining X(Independednt Attributes) and Y(Dependednt Attributes) 

# In[ ]:


#importing necessary libraries
from sklearn.model_selection import train_test_split


# In[ ]:


# Deternmining the indepedent and dependent variales (X and Y)
X=new_data


# In[ ]:


#Changing class to numerical representation 
Target_dict={'class':{'car':0,'van':1,'bus':2}}

data.replace(Target_dict, inplace=True)

data.head()


# In[ ]:


Y=data['class']


# In[ ]:


# Checking the shape of X and Y
X.shape


# In[ ]:


Y.shape


# #### Splitting the data to 70% training data and 30% test data

# In[ ]:


Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.3,random_state=1)
Xtrain.head()


# In[ ]:


#importing necessary libraries
from sklearn.svm import SVC


# In[ ]:


model_svc=SVC()


# In[ ]:


#fitting the data
model_svc.fit(Xtrain,Ytrain)


# In[ ]:


Ypred=model_svc.predict(Xtest)


# In[ ]:


#Checking the score for SVC
SVC_Trainscore=model_svc.score(Xtrain,Ytrain)
print("The score for SVC-Training Data is {0:.2f}%".format(SVC_Trainscore*100))
SVC_Testscore=model_svc.score(Xtest,Ytest)
print("The score for SVC-Test Data is {0:.2f}%".format(SVC_Testscore*100))


# In[ ]:


#Misclassification error
SVC_MSE=1-SVC_Testscore
print("Misclassification error of SVC model is {0:.1f}%".format(SVC_MSE*100))


# In[ ]:


from sklearn import metrics
accuracy_score=metrics.accuracy_score(Ytest,Ypred)
percision_score=metrics.precision_score(Ytest,Ypred,average='micro')
recall_score=metrics.recall_score(Ytest,Ypred,average='micro')
f1_score=metrics.f1_score(Ytest,Ypred,average='micro')
print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))
print("The Percission of this model is {0:.2f}%".format(percision_score*100))
print("The Recall score of this model is {0:.2f}%".format(recall_score*100))
print("The f1 score of this model is {0:.2f}%".format(f1_score*100))


# # Support Vector Classifier- with PCA, Hyper-parameterization and Cross validation

# ### Defining X(Independednt Attributes) and Y(Dependednt Attributes) 

# In[ ]:


# Deternmining the indepedent and dependent variales (X and Y)
X=PCA_data


# In[ ]:


Y=data['class']


# In[ ]:


# Checking the shape of X and Y
X.shape


# In[ ]:


Y.shape


# ## Splitting the data with k-fold cross-validation

# In[ ]:


# simulate splitting a dataset of 25 observations into 5 folds
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
kf.get_n_splits(X)
print(kf)


for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]


# ## Using Grid Search  for hyper-parameterization

# In[ ]:


from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.05, 0.5, 1]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train,y_train)
clf.best_params_


# ## The above is the best parameter from Grid Search

# In[ ]:


#fitting the data
model_svc_PCA=SVC(C=1, kernel= 'rbf',gamma=0.5)
model_svc_PCA.fit(X_train,y_train)
scores = cross_val_score(model_svc_PCA, X, Y, cv=5,scoring='f1_macro')
scores


# ## The above are the cross validation score and our accuracy should be in the above range

# In[ ]:


Ypred=model_svc_PCA.predict(X_test)


# In[ ]:


#Checking the score for SVC
SVC_Trainscore=model_svc_PCA.score(X_train,y_train)
print("The score for SVC-Training Data is {0:.2f}%".format(SVC_Trainscore*100))
SVC_Testscore=model_svc_PCA.score(X_test,y_test)
print("The score for SVC-Test Data is {0:.2f}%".format(SVC_Testscore*100))


# In[ ]:


#Misclassification error
SVC_MSE=1-SVC_Testscore
print("Misclassification error of SVC model is {0:.1f}%".format(SVC_MSE*100))


# In[ ]:


accuracy_score=metrics.accuracy_score(y_test,Ypred)
percision_score=metrics.precision_score(y_test,Ypred,average='macro')
recall_score=metrics.recall_score(y_test,Ypred,average='macro')
f1_score=metrics.f1_score(y_test,Ypred,average='macro')
print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))
print("The Percission of this model is {0:.2f}%".format(percision_score*100))
print("The Recall score of this model is {0:.2f}%".format(recall_score*100))
print("The f1 score of this model is {0:.2f}%".format(f1_score*100))


# ## Verdict:
# ### 1. Data was read and understood and missing values were filled with their respective median
# ### 2. Exploratory Data analytics was performed- univariate and multivariate
# ### 3. Target was dropped 
# ### 4. Data was standardized using z-score
# ### 5. Outliers were removed from the data-set
# ### 6. PCA was performed and 6 dimensions was selected that explained 95% of the variance in the data
# ### 7. Dimensions were reduced to 6 from 18 and PCA data-set was created
# ### 8. Support vector classifier was performed with the original data and accuracy and other performance metrics were identified
# ### 9. Support vector classifier was performed with PCA data with K-fold cross validation and Hyper parameterization using GridSearchCV and accuracy other performance metrics were identified
# ### 10. Accuracy other performance metrics are lower with PCA data that with scaled data but this is with just 6 dimensions as opposed to 18

# ## K-Means Clustering with original data- Unsupervised model

# In[ ]:


#Finding optimal no. of clusters
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
clusters=range(1,10)
meanDistortions=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(new_data)
    prediction=model.predict(new_data)
    meanDistortions.append(sum(np.min(cdist(new_data, model.cluster_centers_, 'euclidean'), axis=1)) / new_data.shape[0])


plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')


# ### 3 clusters seems optimal when looking into the elbow point

# In[ ]:


kmeans_model=KMeans(3)
kmeans_model.fit(new_data)
prediction=kmeans_model.predict(new_data)

#Append the prediction 
#new_data["GROUP"] = prediction
data["GROUP"] = prediction
print("Groups Assigned : \n")
data.head(10)


# In[ ]:


data.groupby(['GROUP'])


# In[ ]:


data.boxplot(by='GROUP', layout = (4,6),figsize=(15,10))


# ### Evaluating the K-means model using mallows_score and Silhoutte_score

# In[ ]:


metrics.fowlkes_mallows_score(data['class'],data['GROUP'])


# In[ ]:


metrics.silhouette_score(new_data, kmeans_model.labels_, metric='euclidean')


# In[ ]:


data=data.drop(['GROUP'],axis=1)
data.head()


# ## K-Means Clustering with PCA data

# In[ ]:


#Finding optimal no. of clusters
clusters=range(1,7)
meanDistortions=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(PCA_data)
    prediction=model.predict(PCA_data)
    meanDistortions.append(sum(np.min(cdist(PCA_data, model.cluster_centers_, 'euclidean'), axis=1)) / PCA_data.shape[0])


plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')


# ### 3 clusters seems optimal when looking into the elbow point

# In[ ]:


kmeans_model=KMeans(3)
kmeans_model.fit(PCA_data)
prediction=kmeans_model.predict(PCA_data)

#Append the prediction 
data["GROUP"] = prediction
print("Groups Assigned : \n")
data.head(10)


# In[ ]:


data.groupby(['GROUP'])


# In[ ]:


data.boxplot(by='GROUP', layout = (4,6),figsize=(15,10))


# ### Evaluating the K-means model using mallows_score and Silhoutte_score

# In[ ]:


metrics.fowlkes_mallows_score(data['class'],data['GROUP'])


# In[ ]:


metrics.silhouette_score(PCA_data, kmeans_model.labels_, metric='euclidean')


# In[ ]:


data=data.drop(['GROUP'],axis=1)
data.head()


# In[ ]:




