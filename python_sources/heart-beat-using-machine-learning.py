#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.


# > Lets Import the heart.csv file into a pandas dataframe
# 

# In[ ]:


dataset = pd.read_csv('../input/heart.csv')


# First of all we need an insight of the data what it contains and how to interpret the data into a more meaningful statistics. So we will first check the contents using head. Head() will give 5 rows as default from the top. We can also use tail, which provides data from the bottom. For now we will stick to head(). 

# In[ ]:


dataset.head()


# 
# age :            age in years
# 
# sex:            (1 = male; 0 = female)
# 
# cp:              chest pain type
# 
# trestbps       resting blood pressure (in mm Hg on admission to the hospital)
# 
# cholserum   cholestoral in mg/dl
# 
# fbs(fasting blood sugar > 120 mg/dl): (1 = true; 0 = false)
# 
# restecg       resting electrocardiographic results
# 
# thalachmaximum heart rate achieved
# 
# exangexercise induced angina (1 = yes; 0 = no)
# 
# oldpeakST depression induced by exercise relative to rest
# 
# slope        the slope of the peak exercise ST segment
# 
# ca             number of major vessels (0-3) colored by flourosopy
# 
# thal           3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# target       1 or 0
# 

# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()


# Let's check the shape of the data, i.e count and columns available
# 
# 

# In[ ]:


dataset.shape


# **Gender distribution in the file using Seaborn**

# In[ ]:


sns.countplot(x='sex',data=dataset)


# **Gender Ratio:**
# 
# Lets see percentage wise ratio of dataset for gender.

# In[ ]:


plt.figure(figsize=(8,6))
explode =[0.1,0]
labels='Male','Female'
plt.pie(dataset['sex'].value_counts(),explode=explode,autopct='%1.1f%%',labels=labels,shadow=True,startangle=140)


# **Chest Pain** : We can see there are different pain type, so lets build a pie chart which will show the data distribution.
# 

# In[ ]:


plt.figure(figsize=(10,6))
explode=[0.1,0,0,0]
labels='Pain-Type 0','Pain Type-1','Pain-Type2','Pain-Type3'
plt.pie(dataset['cp'].value_counts(),explode=explode,labels=labels,autopct='%1.1f%%',shadow=True,startangle=140)


# In[ ]:


sns.boxplot(dataset['trestbps'],orient='v',color='Magenta')


# In[ ]:


sns.boxplot(dataset['chol'],orient='v',color='Magenta')


# In[ ]:



#dataset.plot.scatter(x='age',y='trestbps')
plt.figure(figsize=(20,10))
sns.boxplot(x='age',y='trestbps',data=dataset)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x='age',y='thalach',data=dataset)


# In[ ]:


sns.set()
col=['age','trestbps','chol','thalach']
sns.pairplot(dataset[col])
plt.show()


# Lets build a heat map to check the co relation between variables. From the below it is evident that hardly strong co realtion exists between variables. 

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(dataset.corr(),annot=True,cmap='YlGnBu')


# **Creating Dummy Variables:**
# 
# From the above we can see there are categorical values, which includes: sex,cp,fbs etc.
# 
# So we will create dummy variables. We will also use prefix so that categorical columns when converted are recognized properly.

# In[ ]:


sex = pd.get_dummies(dataset['sex'],prefix='sex',drop_first=True)
fbs = pd.get_dummies(dataset['fbs'],prefix='fbs',drop_first=True)
restecg = pd.get_dummies(dataset['restecg'],prefix='restecg',drop_first=True)
exang = pd.get_dummies(dataset['exang'],prefix='exang',drop_first=True)
cp = pd.get_dummies(dataset['cp'],prefix='cp',drop_first=True)
slope = pd.get_dummies(dataset['slope'],prefix='slope',drop_first=True)
thal = pd.get_dummies(dataset['thal'],prefix='thal',drop_first=True)

dataset = pd.concat([dataset,sex,fbs,restecg,exang,cp,slope,thal],axis=1)



#Will do a quick check if it worked or not :P
dataset.head()


# Dropping the columns since we have already converted the categorical data and taken care the dummy trap above

# In[ ]:


dataset = dataset.drop(columns=['sex','fbs','restecg','exang','cp','slope','thal'])
dataset.head()


# **Making Predictions**
# 
# 

# Extracting the dependent (Y) and X variables.
# 
# 

# In[ ]:


X= dataset.drop('target',axis=1)
y = dataset['target'].values


# **Train Test Splitting**
# 
# We will split the data into train test based on 80:20 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# **Standard Scaler**
# 
# Lets Standarize the data before fitting the data into the model.Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data.
# 
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)


# **PCA component **
# 
# 

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=None,random_state=0)
X_train = pca.fit_transform(X_train)
X_test =pca.transform(X_test)

pca.explained_variance_ratio_


# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
lr_score = lr.score(X_test,y_test)


# **Support Vector **

# In[ ]:


from sklearn.svm import SVC
sv = SVC(kernel ='rbf',random_state=0)
sv.fit(X_train,y_train)
sv_pred = sv.predict(X_test)
sv_score = sv.score(X_test,y_test)


# **Random Forest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_regressor = RandomForestClassifier(n_estimators = 1000, random_state = 0)
rf_regressor.fit(X_train, y_train)
rf_pred = rf_regressor.predict(X_test)
rf_score = rf_regressor.score(X_test,y_test)


# **KNN**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn_score = knn.score(X_test,y_test)


# **Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nv = GaussianNB()
nv.fit(X_train,y_train)
nv_sc = nv.score(X_test,y_test)


# **Model Score**

# In[ ]:



print("Logistic Regression Model Score is ",round(lr_score*100))
print("SVC Model Score is ",round(sv_score*100))
#print("Decision tree  Regression Model Score is ",round(tr_regressor.score(X_test,y_test)*100))
print("Random Forest Regression Model Score is ",round(rf_score*100))

print("KNeighbors Classifiers Model score is",round(knn_score*100))
print("Naive Bayes model score is",round(nv_sc*100))


# **Cross Validation Score with 10 iteration**

# In[ ]:




from sklearn.model_selection import cross_val_score
accuracies_lr = cross_val_score(estimator = lr,X = X_train,y = y_train,cv = 10)
accuracies_sv = cross_val_score(estimator = sv,X = X_train,y = y_train,cv = 10)
accuracies_rf = cross_val_score(estimator = rf_regressor,X = X_train,y = y_train,cv = 10)

accuracies_knn = cross_val_score(estimator = knn,X = X_train,y = y_train,cv = 10)
accuracies_nv = cross_val_score(estimator = nv,X = X_train,y = y_train,cv = 10)

print("Mean Accuracies based on cross val score for logistic regression",round(accuracies_lr.mean()*100))
print("Mean Accuracies based on cross val score for SVM ",round(accuracies_sv.mean()*100))
print("Mean Accuracies based on cross val score for Random Forest",round(accuracies_rf.mean()*100))

print("Mean Accuracies based on cross val score for KNN",round(accuracies_knn.mean()*100))
print("Mean Accuracies based on cross val score for Naive Bayes",round(accuracies_nv.mean()*100))


# **Confusion Matrix:**
# 
# Logistic Regression and Random Forest since this performs a better model in comparison to other
# 
# 

# In[ ]:



cm_lr = confusion_matrix(y_test,y_pred)
cm_lr


# **Confusion Matrix** for Random Forest is as below
# 

# In[ ]:


cm_rf = confusion_matrix(y_test,rf_pred)
cm_rf


# **Conclusion**
# 
# Though there are weak co relation between variables and also exists other model, but Logistic Model and Random Forest much better than other model. 
# 
# 
# Please ****Upvote**** my work if you like it :)
# 
# 

# In[ ]:




