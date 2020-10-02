#!/usr/bin/env python
# coding: utf-8

# **Import necessary modules**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics


# **Importing data**

# In[ ]:


data=pd.read_csv("../input/data.csv",header=0)
print(data.head(2))


# In[ ]:


data.info()


# Now we can see Unnamed:32 have 0 non null object it means the all values are null in this column so we cannot use this column for our analysis, therefore, we drop those columns.

# In[ ]:


data.drop("Unnamed: 32",axis=1,inplace=True)


# In[ ]:


data.drop("id",axis=1,inplace=True)


# Here, we divided the data into three categories.

# In[ ]:


features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:20])
features_worst=list(data.columns[21:31])


# In[ ]:


data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
data.describe()


# **Getting into the data and exploring the aspects**

# In[ ]:


sns.countplot(data['diagnosis'],label="Count")


# In[ ]:


corr=data[features_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,cbar=True,square=True,annot=True,fmt='.2f',annot_kws={'size': 15},xticklabels=features_mean,yticklabels=features_mean,cmap='coolwarm')


# **Observation:**
# 
# * Radius,parameter and area are highly correlated as expected from their relation, therefore we use any one of them.
# * Compactness_mean, concavity_mean and concavepoint_mean are highly correlated so we will use compactness_mean from here.
# * Therefore,the selected parameter for use is perimeter_mean, texture_mean, compactness_mean and symmetry_mean. 

# In[ ]:


prediction_var=['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
train,test=train_test_split(data,test_size=0.3)


# In[ ]:


#Splitting the data into train and test.
train_X=train[prediction_var]
train_y=train.diagnosis
test_X=test[prediction_var]
test_y=test.diagnosis


# In[ ]:


model=RandomForestClassifier(n_estimators=100)


# In[ ]:


model.fit(train_X,train_y)


# In[ ]:


prediction=model.predict(test_X)


# In[ ]:


metrics.accuracy_score(prediction,test_y)


# **Now we try with Support Vector Machine Algorithm  **

# In[ ]:


model=svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


# **SVM is giving only 0.8538 which we can improve by using different techniques, we will get the overview of Machine Learning**
# Now we will for all feature_mean, where we will get the important features using Random Forest Classifier

# In[ ]:


prediction_var=features_mean #with all features


# In[ ]:


train_X=train[prediction_var]
train_y=train.diagnosis
test_X=test[prediction_var]
test_y=test.diagnosis


# In[ ]:


model=RandomForestClassifier(n_estimators=100)


# In[ ]:


model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


# * Taking all features in consideration, accuracy increased but not so much so according to Razor's rule.
# * Let's check the important features in the prediction.

# In[ ]:


featimp=pd.Series(model.feature_importances_,index=prediction_var).sort_values(ascending=False)
print(featimp)


# **Let's do it first using Support Vector Machine Algorithm(SVM) **

# In[ ]:


model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


# We can observe that accuracy has dropped down while using SVM, then we'll take only important features using Random Forest Classifier

# In[ ]:


prediction_var=['concave points_mean','perimeter_mean' , 'concavity_mean' , 'radius_mean','area_mean']      


# In[ ]:


train_X= train[prediction_var]
train_y= train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis


# In[ ]:


model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


# In[ ]:


model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


#  We observe that multi collinearity effecting our SVM part a lot.
#  Converslely, it is not affecting so much of Random Forest because for random forest we don't need to make so much effort for our analysis part.
#  
#  Now, we have to deal with the worst part of the data, let's start with features_worst

# In[ ]:


prediction_var=features_worst


# In[ ]:


train_X= train[prediction_var]
train_y= train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis


# In[ ]:


model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


# Same problem will continue ith SVM, as we observe low accuracy, therefore we will ture it's parameters.
# Again, we will extract out the important features using Random Forest Classifier 

# In[ ]:


model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


# **Getting important features**

# In[ ]:


featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
print(featimp)


# **Let's take top five features**

# In[ ]:


prediction_var = ['concave points_worst','radius_worst','area_worst','perimeter_worst','concavity_worst'] 


# In[ ]:


train_X= train[prediction_var]
train_y= train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis


# In[ ]:


model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


# In[ ]:


#check for SVM
model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


# As per the consistency and simplicity, we will use Random Forest Classifier for prediction.

# From the feature mean, we will indentify the variables which can be used for the classification the two class of the cancer. Therefore, we will plot a scatter plot from where we can visualise the distinguishable boundary between the two class of the cancer.

# In[ ]:


color_function = {0: "blue", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B
colors = data["diagnosis"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column
pd.scatter_matrix(data[features_mean], c=colors, alpha = 0.5, figsize = (18, 18)); # plotting scatter plot matrix


# **Observation from the above scatter plot**
# 
# Radius, area and perimeter have a strong linear correlation as expected and the above scatter plot shows the features texture_mean, smoothness_mean, symmetry_mean and fractal_dimension_mean can't be used for distinguishing the two class of the cancer because both classes are scattered and mixed within and there is no distinguishable plane.
# 
# There we remove them from our prediction_var.

# In[ ]:


features_mean


# In[ ]:


predict_var=['radius_mean','perimeter_mean','compactness_mean','area_mean','concave points_mean']


# In[ ]:


#Checking the accuracy of the model
def model(model,data,prediction,outcome):
    kf=KFold(n_splits=10)


# In[ ]:


#Cross validation using different model
def classification_model(model,data,prediction_input,output):
    model.fit(data[prediction_input],data[output]) 
    predictions = model.predict(data[prediction_input])
    accuracy = metrics.accuracy_score(predictions,data[output])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    kf = KFold(n_splits=10,random_state=42,shuffle=False)
    error = []
    for train, test in kf.split(data):
        train_X = (data[prediction_input].iloc[train,:])
        train_y = data[output].iloc[train]
        model.fit(train_X, train_y)
        
        test_X=data[prediction_input].iloc[test,:]
        test_y=data[output].iloc[test]
        error.append(model.score(test_X,test_y))
        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))


# **Now, we will use different models, using the concept of Machine Learning.**
# * We will compute the accuracy of the each model along with the algorithm with cross validation, for the approval of the consideration of the model.

# In[ ]:


model = DecisionTreeClassifier()
predict_var=['radius_mean','perimeter_mean','compactness_mean','area_mean','concave points_mean']
outcome_var= "diagnosis"
classification_model(model,data,prediction_var,outcome_var)


# Since, we observe that the model is overfitting as it's accuracy is 100%, moreover the cross-validation scores are not good, so accuracy cannot be considered is this senario.

# In[ ]:


model = svm.SVC()
classification_model(model,data,prediction_var,outcome_var)


# In[ ]:


model = KNeighborsClassifier()
classification_model(model,data,prediction_var,outcome_var)


# Again, the cross-validation scores are not good.

# In[ ]:


model = RandomForestClassifier(n_estimators=100)
classification_model(model,data,prediction_var,outcome_var)


# **Here, we can observe that RandomForestClassifier have good cross-validation score.**

# In[ ]:


model=LogisticRegression()
classification_model(model,data,prediction_var,outcome_var)


# Above, we compared the different models of Machine Learning and extract out the better one of them.
# **Now, we will tune the parameters for the different models**
# 
# Tuning parameters using GridSearchCV for every model used.

# In[ ]:


data_X=data[prediction_var]
data_y=data['diagnosis']


# In[ ]:


#GridSearchCV function
def Classification_model_gridsearchCV(model,param_grid,data_X,data_y):
    clf = GridSearchCV(model,param_grid,cv=10,scoring="accuracy")
    clf.fit(train_X,train_y)
    print("The efficient parameter to be used is")
    print(clf.best_params_)
    print("the efficient estimator is ")
    print(clf.best_estimator_)
    print("The efficient score is ")
    print(clf.best_score_)


# In[ ]:


#Here, we have to use the parameters which we used in Decision Tree Classifier
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2,3,4,5,6,7,8,9,10], 
              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }
model= DecisionTreeClassifier()
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)


# In[ ]:


#Here, we wil use the parameters use in KNN
model = KNeighborsClassifier()
k_range = list(range(1, 30))
leaf_size = list(range(1,30))
weight_options = ['uniform', 'distance']
param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options}
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)


# In[ ]:


#Here, we try it with SVM 
model=svm.SVC()
param_grid = [
              {'C': [1, 10, 100, 1000], 
               'kernel': ['linear']
              },
              {'C': [1, 10, 100, 1000], 
               'gamma': [0.001, 0.0001], 
               'kernel': ['rbf']
              },
 ]
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)


# **Conclusion:**
# 
# The SVM was initally giving bad accuracy which was improved after tuning the parameters, there we understand the importance of classifier.
