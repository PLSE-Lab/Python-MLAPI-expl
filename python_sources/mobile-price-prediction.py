#!/usr/bin/env python
# coding: utf-8

#  <font size="5">Please do Vote up if you like my work,any help to upgrade this is appreciated.</font>
# * Linkedin : https://www.linkedin.com/in/pratikrandad/**

# <font size="3">AIM:</font>
# 
# In this Project,On the basis of the mobile Specification like Battery power, 3G enabled , wifi ,Bluetooth, Ram etc we are predicting Price range of the mobile

# <font size="3">Data:</font>
# 
# * id:ID
# * battery_power:Total energy a battery can store in one time measured in mAh
# * blue:Has bluetooth or not
# * clock_speed:speed at which microprocessor executes instructions
# * dual_sim:Has dual sim support or not
# * fc:Front Camera mega pixels
# * four_g:Has 4G or not
# * int_memory:Internal Memory in Gigabytes
# * m_dep:Mobile Depth in cm
# * mobile_wt:Weight of mobile phone
# * n_cores:Number of cores of processor
# * pc:Primary Camera mega pixels
# * px_height:Pixel Resolution Height
# * px_width:Pixel Resolution Width
# * ram:Random Access Memory in Megabytes
# * sc_h:Screen Height of mobile in cm
# * sc_w:Screen Width of mobile in cm
# * talk_time:longest time that a single battery charge will last when you are
# * three_g:Has 3G or not
# * touch_screen:Has touch screen or not
# * wifi:Has wifi or not
# 

# <font size="3"> USE:</font>
# 
# * This kind of prediction will help companies estimate price of mobiles to give tough competion to other mobile manufacturer
# * Also it will be usefull for Consumers to verify that they are paying best price for a mobile.
# <font size="3">Applied Models:</font>
# 
# * KNN Classifier
# * SVM(kernel=linear)
# * SVM(kernel=rbf)
# * Decision tree
# * Random forest

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os


# **<font size="4">Load Data</font>**

# In[ ]:


dataset=pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')


# **<font size="4">Data Analysis</font>**

# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.columns


# In[ ]:


dataset.describe()


# **<font size="4">Data Visualisation</font>**

# In[ ]:


corr=dataset.corr(method='pearson')
plt.figure(figsize=(19, 6))
sns.heatmap(corr,cmap="YlGnBu",annot=True)


# <font size="3">RAM has the highest co-relation with phone price</font>

# <font size="4">RAM vs Price</font>

# In[ ]:


sns.catplot(x='price_range',y='ram',data=dataset)


# <font size="4">Internal_memory vs Price</font>

# In[ ]:


sns.catplot(x='price_range',y='int_memory',kind='swarm',data=dataset)


# <font size="4">3G Phones vs Non 3G Phones</font>

# In[ ]:


from matplotlib.pyplot import pie
values=dataset['three_g'].value_counts().values
pie(values,labels=['3G Supported','3G Not Supported'],autopct='%1.1f%%' ,shadow=True,startangle=90)


# <font size="4">4G Phones vs Non 4G Phones</font>

# In[ ]:


values=dataset['four_g'].value_counts().values
pie(values,labels=['4G Supported','4G Not Supported'],autopct='%1.1f%%' ,shadow=True,startangle=90)


# <font size="4">Battery Power vs Price Range</font>

# In[ ]:


sns.boxplot(x='price_range',y='battery_power',data=dataset)


# <font size="4">Data Pre-processing<font>

# <font size="3">X & Y matrix</font>

# In[ ]:


X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# <font size="4">Training and Test Data Split</font>

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# <font size="4">Most Models like KNN work on Euclidean Distance, larger values will impact the result,hence scaling is required.</font>

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
X_test


# <font size="4">KNN</font>

# <font size="3">KNN with K=10</font>

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
KNNclassifier=KNeighborsClassifier(n_neighbors=10)
KNNclassifier.fit(X_train,y_train)
y_pred = KNNclassifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100


# <font size="4">Finding Optimum value of K</font>

# <font size="4">K-fold Cross Validation</font>

# In[ ]:


from sklearn.model_selection import cross_val_score
# Creating odd list K for KNN
neighbors = list(range(1,30))
# empty list that will hold cv scores
cv_scores = [ ]
#perform 10-fold cross-validation
for K in neighbors:
    knn = KNeighborsClassifier(n_neighbors = K)
    scores = cross_val_score(knn,X_train,y_train,cv = 10,scoring =
    "accuracy")
    cv_scores.append(scores.mean())


# In[ ]:


# Changing to mis classification error
mse = [1-x for x in cv_scores]
# determing best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal no. of neighbors is {}".format(optimal_k))


# <font size="3">K value using Elbow Method</font>
# * Finding K value so that mis match between actual and predicted values is least

# In[ ]:


mismatch=[]
for i in range(1,30):
    classifier=KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    mismatch.append(np.sum(y_pred != y_test))
    


# In[ ]:


plt.plot(range(1,30),mismatch)
plt.show()


# <font size="3">KNN with K=22, accuracy increased by 4%</font>

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
KNNclassifier=KNeighborsClassifier(n_neighbors=22)
KNNclassifier.fit(X_train,y_train)
y_pred = KNNclassifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100


# <font size="4">SVM Model(kernel=linear)</font>

# In[ ]:


from sklearn.svm import SVC
SVMlinear=SVC(kernel='linear')
SVMlinear.fit(X_train,y_train)
SVMlinear_predict=SVMlinear.predict(X_test)
y_pred = SVMlinear.predict(X_test)
accuracy_score(y_test,y_pred)*100


# <font size="4">SVM Model(kernel=rbf)</font>

# In[ ]:


from sklearn.svm import SVC
SVMrbf=SVC(kernel='rbf')
SVMrbf.fit(X_train,y_train)
SVMrbf_predict=SVMrbf.predict(X_test)
y_pred = SVMrbf.predict(X_test)
accuracy_score(y_test,y_pred)*100


# <font size="4">Naive Bayes Model</font>

# In[ ]:


from sklearn.naive_bayes import GaussianNB
NB=GaussianNB()
NB.fit(X_train,y_train)
NB_predict=NB.predict(X_test)
y_pred = NB.predict(X_test)
accuracy_score(y_test,y_pred)*100


# <font size="4">Decision Tree Model</font>

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
DecisionTree=DecisionTreeClassifier(criterion='entropy',random_state=0)
DecisionTree.fit(X_train,y_train)
DecisionTree_predict=DecisionTree.predict(X_test)
y_pred = DecisionTree.predict(X_test)
accuracy_score(y_test,y_pred)*100


# <font size="4">Finding Optimum value for No. of trees using K-fold cross validation</font>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
trees = list(range(1,20))
# empty list that will hold cv scores
cv_scores = [ ]
#perform 10-fold cross-validation
for n in trees:
    RFC = RandomForestClassifier(n_estimators = n,criterion='entropy',random_state=0)
    scores = cross_val_score(RFC,X_train,y_train,cv = 10,scoring =
    "accuracy")
    cv_scores.append(scores.mean())


# In[ ]:


# Changing to mis classification error
mse = [1-x for x in cv_scores]
# determing best n
optimal_n = trees[mse.index(min(mse))]
print("The optimal no. of trees is {}".format(optimal_n))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(n_estimators=19,criterion='entropy',random_state=0)
RFC.fit(X_train,y_train)
RFC_predict=RFC.predict(X_test)
y_pred = RFC.predict(X_test)
accuracy_score(y_test,y_pred)*100


# <font size="5">Conclusion:</font>
# <font size="5">Linear SVM Classifier fits best for this model with 96% Accuracy</font>
