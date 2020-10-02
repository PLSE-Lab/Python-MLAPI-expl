#!/usr/bin/env python
# coding: utf-8

# In this analysis I will be using K Nearest Neighbors to build the model and as the dataset: Credit Card Fraud Detection dataset provided by Kaggle. For this analysis I focused more on data preprocessing because it helps to gain a better accuracy rather than just sending raw data and telling to build a model. 
# 
# # Data Preprocessing
# What is data preprocessing?  It is a data mining technique that transforms raw data into understandable format. Raw data(real world data) is always incomplete and that data cannot be send through a model. Yhay would cause certain errors. That is why we need to preprocess data before sending through a model.
# 
# I have used severel preprocessing techniques in this analysis.
# Without further a do let's get started!!!

# # Steps in Data Preprocessing
# Here iare the steps I have followed;
# 1. Import libraries
# 2. Read data
# 3. Checking for missing values
# 4. Checking for categorical data
# 5. Standardize the data
# 6. PCA transformation
# 7. Data splitting

# # 1. Import libraries
# As the main libraries, I am using Pandas, Numpy and time; 
# * Pandas : Use for data manipulation and data analysis. 
# * Numpy : fundamental package for scientific computing with Python. 
# 
# As for the visualization I am using Matplotlib and Seaborn.
# 
# For the data preprocessing techniques and algorithms I used Scikit-learn libraries.

# In[ ]:


# main libraries
import pandas as pd
import numpy as np
import time

# visual libraries
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
plt.style.use('ggplot')

# sklearn libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,roc_curve
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# # 2. Read data
# You can find more details on dataset here: 

# In[ ]:


# Read the data in the CSV file using pandas
df = pd.read_csv('../input/creditcard.csv')
df.head()


# In[ ]:


df.shape


# # 3. Checking for missing values
# In this data set, there are no missing values. So we don't need to handle missing values in the dataset.

# In[ ]:


df.isnull().sum()


# ## Let's look at the data. 
# So the dataset is labeled as 0s and 1s. 0 = non fraud and 1 = fraud. 
# 

# In[ ]:


All = df.shape[0]
fraud = df[df['Class'] == 1]
nonFraud = df[df['Class'] == 0]

x = len(fraud)/All
y = len(nonFraud)/All

print('frauds :',x*100,'%')
print('non frauds :',y*100,'%')


# In[ ]:


# Let's plot the Transaction class against the Frequency
labels = ['non frauds','fraud']
classes = pd.value_counts(df['Class'], sort = True)
classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), labels)
plt.xlabel("Class")
plt.ylabel("Frequency")


# # 4. Checking for categorical data
# the only categorical cariable we have in this data set is the target variable.  Other features are already in numerical format, so no need of converting to categorical data.
# 
# ## Let's plot the distribution of features 
# I used seaborn distplot() to visualize the distribution of features in the dataset.
# We have 30 features and target variable  in the dataset. 

# In[ ]:


# distribution of Amount
amount = [df['Amount'].values]
sns.distplot(amount)


# In[ ]:


# distribution of Time
time = df['Time'].values
sns.distplot(time)


# In[ ]:


# distribution of anomalous features
anomalous_features = df.iloc[:,1:29].columns

plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(df[anomalous_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[cn][df.Class == 1], bins=50)
    sns.distplot(df[cn][df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()


# In this analysis I will not be dropping any features looking at the distribution of features, because I am still in the learning process of working with data preprocessing in numarous ways.So I would like to experiment  step by step on data.
# 
# Instead all the features will be tranformed to scaled variables.

# In[ ]:


# heat map of correlation of features
correlation_matrix = df.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,vmax=0.8,square = True)
plt.show()


# # 5. Standardize the data
# The dataset is  contains only numerical input variables which are the result of a PCA transformation. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. So PCA is effected by scale so we need to scale the features in the data before applying PCA. For the scaling I am using Scikit-learn's StandardScaler(). In order to fit to the scaler the data should be reshaped within -1 nad 1. 

# In[ ]:


# Standardizing the features
df['Vamount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df['Vtime'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1,1))

df = df.drop(['Time','Amount'], axis = 1)
df.head()


# Now all the features are standardize into unit sclae (mean = 0 and variance = 1)

# # 6. PCA transformation
# PCA (Principal Component Analysis) mainly using to reduce the size of the feature space while retaining as much of the information as possible.  In here all the features transformed into 2 features using PCA.

# In[ ]:


X = df.drop(['Class'], axis = 1)
y = df['Class']

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X.values)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[ ]:


finalDf = pd.concat([principalDf, y], axis = 1)
finalDf.head()


# In[ ]:


# 2D visualization
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# Since the data is highly imbalanced, I am only taking 492 rows from the non_fraud transactions.

# In[ ]:


# Lets shuffle the data before creating the subsamples
df = df.sample(frac=1)

frauds = df[df['Class'] == 1]
non_frauds = df[df['Class'] == 0][:492]

new_df = pd.concat([non_frauds, frauds])
# Shuffle dataframe rows
new_df = new_df.sample(frac=1, random_state=42)

new_df.head()


# In[ ]:


# Let's plot the Transaction class against the Frequency
labels = ['non frauds','fraud']
classes = pd.value_counts(new_df['Class'], sort = True)
classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), labels)
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[ ]:


# prepare the data
features = new_df.drop(['Class'], axis = 1)
labels = pd.DataFrame(new_df['Class'])

feature_array = features.values
label_array = labels.values


# # 7. Data splitting

# In[ ]:


# splitting the faeture array and label array keeping 80% for the trainnig sets
X_train,X_test,y_train,y_test = train_test_split(feature_array,label_array,test_size=0.20)

# normalize: Scale input vectors individually to unit norm (vector length).
X_train = normalize(X_train)
X_test=normalize(X_test)


# For the model building I am using K Nearest Neighbors. So we need find an optimal K to get the best out of it. 

# In[ ]:


neighbours = np.arange(1,25)
train_accuracy =np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))

for i,k in enumerate(neighbours):
    #Setup a knn classifier with k neighbors
    knn=KNeighborsClassifier(n_neighbors=k,algorithm="kd_tree",n_jobs=-1)
    
    #Fit the model
    knn.fit(X_train,y_train.ravel())
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train.ravel())
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test.ravel()) 


# In[ ]:


#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbours, test_accuracy, label='Testing Accuracy')
plt.plot(neighbours, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


idx = np.where(test_accuracy == max(test_accuracy))
x = neighbours[idx]


# In[ ]:


#k_nearest_neighbours_classification
knn=KNeighborsClassifier(n_neighbors=x[0],algorithm="kd_tree",n_jobs=-1)
knn.fit(X_train,y_train.ravel())


# In[ ]:


# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(knn, filename)


# In[ ]:


# load the model from disk
knn = joblib.load(filename)


# In[ ]:


# predicting labels for testing set
knn_predicted_test_labels=knn.predict(X_test)


# In[ ]:


from pylab import rcParams
#plt.figure(figsize=(12, 12))
rcParams['figure.figsize'] = 14, 8
plt.subplot(222)
plt.scatter(X_test[:, 0], X_test[:, 1], c=knn_predicted_test_labels)
plt.title(" Number of Blobs")


# In[ ]:


#scoring knn
knn_accuracy_score  = accuracy_score(y_test,knn_predicted_test_labels)
knn_precison_score  = precision_score(y_test,knn_predicted_test_labels)
knn_recall_score    = recall_score(y_test,knn_predicted_test_labels)
knn_f1_score        = f1_score(y_test,knn_predicted_test_labels)
knn_MCC             = matthews_corrcoef(y_test,knn_predicted_test_labels)


# In[ ]:


#printing
print("")
print("K-Nearest Neighbours")
print("Scores")
print("Accuracy -->",knn_accuracy_score)
print("Precison -->",knn_precison_score)
print("Recall -->",knn_recall_score)
print("F1 -->",knn_f1_score)
print("MCC -->",knn_MCC)
print(classification_report(y_test,knn_predicted_test_labels))


# In[ ]:


import seaborn as sns
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, knn_predicted_test_labels)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# ## Conclusion
# I tried without standardizing the data to get a better accuracy. But after I learnt this method and applied it, it gave a promising result. I am still doing experiments and still learning about data preprocessing techniques. I only used KNN algorithm for this dataset.  
# Feel free to comment and give an Upvote if you find this kernel helpful.
# 
