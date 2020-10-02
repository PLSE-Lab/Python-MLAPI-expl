#!/usr/bin/env python
# coding: utf-8

# My Aim is to find the best possible models that give a high prediction accuracy in predicting the quality of White wine
# This also happens to be my first indendent project , so any feedback is highly appreciated 

# 1. **Data pre processing **
# 1. Lets start with importing the necessary libraries and try to understand the data set 

# In[2]:



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


# In[3]:



dataset = pd.read_csv('../input/winequality_white.csv',header = 0)
dataset.head()


# we can see that the features are continous variables and in different scales , lets try a few visualizations to understand the data set further 

# In[4]:


dataset.describe()


# In[5]:


#checking for null values
dataset.isnull().sum()


# Clearly there are no null values , so there we wont have any problems associated with missing values
# 
# scatter plots helps us to understand how features vary for different class and the correlation between the features  

# In[6]:


sns.pairplot(dataset,vars = ['fixed acidity','volatile acidity','citric acid'],hue = 'quality')
sns.pairplot(dataset,vars = ['residual sugar','chlorides','free sulfur dioxide'],hue = 'quality')
sns.pairplot(dataset,vars = ['total sulfur dioxide','density','pH'],hue = 'quality')


# another way of chekcing correlation when having many variables is correlation matric. when using regression models we might need to drop variables which have high correlation. 

# In[81]:



# Compute the correlation matrix
corr = dataset.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
#for custom colors
cmap = sns.diverging_palette(350, 69, as_cmap=True)
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap,center=0,square=True, linewidths=.5) 
plt.show()


# 
# We see a high positive correlation between residual sugar and density.
# Lets us check the count for each class variable to see if there is any imbalance with the data set

# In[7]:



sns.set_style("whitegrid")
ax = sns.countplot(x="quality", data=dataset)


# **Building Model**
# 1. We can clearly see that the quality 5,6 and 7 are the majority and 3,4,8,9 are very negligble.
# 1. We will proceed with this imbalance for now , so lets do some feature scaling and proceed with building the models

# In[53]:


# seperate features from class variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#test train split to split the data set 
from sklearn.model_selection import train_test_split
# 10 % of data is my test set 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10, random_state=120)

# feature scaling to normalize all the features in one scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_sc = sc_X.fit_transform(X_train)
X_test_sc = sc_X.transform(X_test)


# In[58]:


# Got a warning for setting cv value to 5 since i have only 3 values for class variable 3 , so had to reduce the k-fold value to 3 or less 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {"max_depth": [25,26,27,28,30,32],
              "max_features": [3, 5,10],
              "min_samples_split": [2, 3,7, 10],
              "min_samples_leaf": [6,7,10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

extra_clf = ExtraTreesClassifier()
grid_search = GridSearchCV(extra_clf,param_grid=param_grid,cv = 2)
grid_search.fit(X_train_sc, y_train)
result_Train_et = grid_search.predict(X_train_sc)
result_Test_et = grid_search.predict(X_test_sc)


# In[57]:


# train accuracy is good but the test accuracy is very poor
from sklearn.metrics import accuracy_score
score_et_test = accuracy_score(y_test,result_Test_et)
score_et_train = accuracy_score(y_train,result_Train_et)
print("train accuracy",score_et_train)
print ("test accuracy",score_et_test )


# In[64]:


# lets try to balance the class variable and see if there is any change in accuracy 
# thanks to elite data science https://elitedatascience.com/imbalanced-classes
from sklearn.utils import resample
# Separate majority and minority classes

df_majority = dataset[dataset.quality==6]
df_minority1 = dataset[dataset.quality==9]
df_minority2 = dataset[dataset.quality==5]
df_minority3 = dataset[dataset.quality==7]
df_minority4 = dataset[dataset.quality==8]
df_minority5 = dataset[dataset.quality==4]
df_minority6 = dataset[dataset.quality==3]
 
# Upsample minority class

df_minority1_upsampled = resample(df_minority1, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible results
df_minority2_upsampled = resample(df_minority2, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible resu
df_minority3_upsampled = resample(df_minority3, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible results                                 
df_minority4_upsampled = resample(df_minority4, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible results
df_minority5_upsampled = resample(df_minority5, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible results
df_minority6_upsampled = resample(df_minority6, 
                                 replace=True,     # sample with replacement
                                 n_samples=2198,    # to match majority class
                                 random_state=123) # reproducible results 
                                  
                                  
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority1_upsampled,df_minority2_upsampled,df_minority3_upsampled,df_minority4_upsampled,
                           df_minority5_upsampled,df_minority6_upsampled])
sns.set_style("whitegrid")
ax = sns.countplot(x="quality", data=df_upsampled)


# In[61]:


# seperate features from class variable
X = df_upsampled.iloc[:, :-1].values
y = df_upsampled.iloc[:, -1].values
#test train split to split the data set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10, random_state=123)

# feature scaling once again for our upsampled dataset 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_sc = sc_X.fit_transform(X_train)
X_test_sc = sc_X.transform(X_test)


# In[68]:


# now we can mention CV values greater than 3 cause all class variables now are greater than 2000
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {"max_depth": [25,26,27,28,30,32],
              "max_features": [3, 5,10],
              "min_samples_split": [2, 3,7, 10],
              "min_samples_leaf": [6,7,10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

extra_clf = ExtraTreesClassifier()
grid_search = GridSearchCV(extra_clf,param_grid=param_grid,cv = 5,n_jobs = -1)
grid_search.fit(X_train_sc, y_train)
result_Train_et = grid_search.predict(X_train_sc)
result_Test_et = grid_search.predict(X_test_sc)


# In[69]:


from sklearn.metrics import accuracy_score
score_et_test = accuracy_score(y_test,result_Test_et)
score_et_train = accuracy_score(y_train,result_Train_et)
print("train accuracy",score_et_train)
print ("test accuracy",score_et_test )


# Now we can see immediate accuracy improvements for both train and test sets , lets try some other algorithms to see how they work

# In[74]:


from sklearn.neighbors import KNeighborsClassifier

param_grid = {"n_neighbors": [2,5,8],
              "algorithm": ["ball_tree", "kd_tree", "brute"],
           "leaf_size":[10,20,30 ]}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn,param_grid=param_grid,cv = 5)
grid_search_knn.fit(X_train_sc, y_train)

result_Train_knn = grid_search_knn.predict(X_train_sc)
result_Test_knn = grid_search_knn.predict(X_test_sc)


# In[75]:


#calaculate accuracy
score_Knn_test = accuracy_score(y_test,result_Test_knn)
score_Knn_train = accuracy_score(y_train,result_Train_knn)
print("train accuracy",score_Knn_train)
print ("test accuracy",score_Knn_test )


# In[78]:


# lets see how it works with SVM
from sklearn import svm
param_grid = {"C":[0.8,1.0,2.0],"kernel" :["linear","rbf"],"gamma" :[0.60, 0.75],"class_weight":["balanced"],"probability":[False,True]}
svm= svm.SVC()
grid_search_svm = GridSearchCV(svm,param_grid=param_grid,cv = 5,n_jobs = -1)
grid_search_svm.fit(X_train_sc,y_train)
result_Train_svm = grid_search_svm.predict(X_train_sc)
result_Test_svm = grid_search_svm.predict(X_test_sc)


# In[80]:


score_Svm_test = accuracy_score(y_test,result_Test_svm)
score_Svm_train = accuracy_score(y_train,result_Train_svm)
print("train accuracy",score_Svm_train)
print ("test accuracy",score_Svm_test )


# In[85]:


# lets try logistic regression 

from sklearn.linear_model import LogisticRegression
param_grid = {"penalty":["l2"],"solver":["newton-cg", "sag" ],"multi_class" : ["ovr", "multinomial"]}
regr = LogisticRegression()
grid_search_regrr = GridSearchCV(regr,param_grid=param_grid,cv = 5,n_jobs = -1)
                            
grid_search_regrr.fit(X_train_sc,y_train)
results_Train = grid_search_regrr.predict(X_train_sc)
results_Test = grid_search_regrr.predict(X_test_sc)


# In[86]:


#calculate accuracy
from sklearn.metrics import accuracy_score
score_regression_test = accuracy_score(y_test,results_Test)
score_regression_train = accuracy_score(y_train,results_Train)
print("train accuracy",score_regression_train)
print ("test accuracy",score_regression_test )


# **Conclusion :**
# 1. Feature scaling imporve accuracy for classifiers. 
# 1. Class imbalance gives lower accuracy values as algorithm wont be trained with all the classes 
# 1. Grid search helps to get the best possible parameters for different algorithms.
# 1. Regression doesnt seem to be good for our dataset 
# 1. Can try and see if parameters like F1-score and Recall vary before and after dataset is balanced.

# 
