#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Table Of Content:
# 
# # **1. Feature Enineering/Data Pre Processing**
# 
# * 1(a). Import Dataset
# * 1(b). Describing Descriptive Statistics
# * 1(c). Visualising Descriptive Statistics
# * 1(d). Checking Null or Empty Values (Data Cleaning)
# * 1(e). Label Encoder/One Hot Encoder
# * 1(f). Handle Outliers
# * 1(g). Feature Split
# * 1(h). Resample Evaluate performance model
# # **2. Modeling**
# 
# * 2(a). Classification Models Without Feature Scale.
# * 2(b). Classification Models With Feature Scale.
# * 2(c). Regularisation Tuning For Top 2 Classification Algorithms.
# * 2(d). Ensemble and Boosting Classification Algorithms With Feature Scale.
# * 2(e). Regularisation Tuning For Top 2 Ensemble and Boosting Classification Algorithms.
# * 2(f). Compare All 4 Tunned Algorithms And Selecting The Best Algorithm
# * 2(g). Fit and Predict The Best Algorithm.
# * 2(h). Accuracy Of An Algorithm.

# # 1. Feature Enineering/Data Pre Processing
# 
# 
# # 1(a). Import Dataset

# In[ ]:


# importing dataset
import pandas as pd
dataset = pd.read_csv("../input/detection-of-parkinson-disease/parkinsons.csv")


# # 1(b). Describing Descriptive Statistics

# In[ ]:


# Displaying the head and tail of the dataset

dataset.head()


# In[ ]:


dataset.tail()


# In[ ]:


# Displaying the shape and datatype for each attribute
print(dataset.shape)
dataset.dtypes


# In[ ]:


# Dispalying the descriptive statistics describe each attribute

dataset.describe()


# # 1(c). Visualising Descriptive Statistics
# 
# Histogram plot visualisation for each attribute will be so diffcult because we have high dimensional column 23.
# 
# So Better, we can use heat map to find the correlations coefficient values. we will remove the less correlation coefficient columns. We can remove the irrelavant features it will minimize the accuracy of an algorithm.
# 
# It will be better if we take relavent features columns then we can achive to get good accuracy..

# In[ ]:


# Heatmap visulisation for each attribute coefficient correlation.
import seaborn as sb
corr_map=dataset.corr()
sb.heatmap(corr_map,square=True)


# In[ ]:


# Now visualise the heat map with correlation coefficient values for pair of attributes.
import matplotlib.pyplot as plt
import numpy as np

# K value means how many features required to see in heat map
k=10

# finding the columns which related to output attribute and we are arranging from top coefficient correlation value to downwards.
cols=corr_map.nlargest(k,'status')['status'].index

# correlation coefficient values
coff_values=np.corrcoef(dataset[cols].values.T)
sb.set(font_scale=1.25)
sb.heatmap(coff_values,cbar=True,annot=True,square=True,fmt='.2f',
           annot_kws={'size': 10},yticklabels=cols.values,xticklabels=cols.values)
plt.show()


# Well as u saw above heatmap plot it looks like we did. We got coerrelation coefficient values for each pair of values.
# 
# But we just visualized top 10 coefficient values.
# 
# Now we need to print all the coefficient values in each attribute,later we can decide which attribute have relavant and irrelavant features.

# In[ ]:


# correlation coefficient values in each attributes.
correlation_values=dataset.corr()['status']
correlation_values.abs().sort_values(ascending=False)


# Above is the correlation values in descending order, we have correaltion values in each attribute so we are going to drop from MDVP:RAP column to MDVP:Fhi(Hz) because it have less correlation with other columns.
# 
# If we decrease the column count then accuracy will increase gradually because we are not keeping the irrelevant features.

# # 1(d). Checking Null or Empty Values (Data Cleaning)

# In[ ]:


# Checking null values
dataset.info()


# In[ ]:


# Checking null value sum
dataset.isna().sum()


# we dont have any null values so now we can safely go ahead...

# # 1(e). Label Encoder/One Hot Encoder
# 
# Encoding the Categorical values into numerical values is not required in this dataset. Because all values we have floating type only. we have name column as a categorical values but we are not going to use that column in model prediction.
# 
# So no need to apply label encoding...

# # 1(f). Handle Outliers
# 
# We didn't find any outliers in our dataset so we can safely go ahead.

# # 1(g). Feature Split
# 
# Splitting the dataset into input and output attributes.....
# 
# we are going to drop irrelavant column values from our dataset so that we can get better accuracy...

# In[ ]:


# split the dataset into input and output attribute.

y=dataset['status']
cols=['MDVP:RAP','Jitter:DDP','DFA','NHR','MDVP:Fhi(Hz)','name','status']
x=dataset.drop(cols,axis=1)


# # 1(h). Resample Evaluate performance model
# 
# Splitting the dataset into training and test set...

# In[ ]:


# Splitting the dataset into trianing and test set

train_size=0.80
test_size=0.20
seed=5

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=train_size,test_size=test_size,random_state=seed)


# # 2. Modeling
# 
# # 2(a). Classification Models Without Feature Scale.

# In[ ]:


# Spotcheck and compare algorithms with out applying feature scale.......

n_neighbors=5
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# keeping all models in one list
models=[]
models.append(('LogisticRegression',LogisticRegression()))
models.append(('knn',KNeighborsClassifier(n_neighbors=n_neighbors)))
models.append(('SVC',SVC()))
models.append(("decision_tree",DecisionTreeClassifier()))
models.append(('Naive Bayes',GaussianNB()))

# Evaluating Each model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
names=[]
predictions=[]
error='accuracy'
for name,model in models:
    fold=KFold(n_splits=10,random_state=0)
    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)
    predictions.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)
    

# Visualizing the Model accuracy
fig=plt.figure()
fig.suptitle("Comparing Algorithms")
plt.boxplot(predictions)
plt.show()


# Prediction we got without applying feature scaling
# 
# 1. Logistic Regression Classification Algorithm : 0.859583 (0.114429)
# 2. K-Nearest Neighbors classification Algorithm : 0.834167 (0.118714)
# 3. Support Vector Machine classification Algorithm : 0.821667 (0.117951)
# 4. Decision Tree Classification Algorithm : 0.840000 (0.106771)
# 5. Naive bayes Classification Algorithm : 0.735833 (0.071715)

# # 2(b). Classification Models With Feature Scale.

# In[ ]:


# Spot Checking and Comparing Algorithms With StandardScaler Scaler
from sklearn.pipeline import Pipeline
from sklearn. preprocessing import StandardScaler
pipelines=[]
pipelines.append(('scaled Logisitic Regression',Pipeline([('scaler',StandardScaler()),('LogisticRegression',LogisticRegression())])))
pipelines.append(('scaled KNN',Pipeline([('scaler',StandardScaler()),('KNN',KNeighborsClassifier(n_neighbors=n_neighbors))])))
pipelines.append(('scaled SVC',Pipeline([('scaler',StandardScaler()),('SVC',SVC())])))
pipelines.append(('scaled DecisionTree',Pipeline([('scaler',StandardScaler()),('decision',DecisionTreeClassifier())])))
pipelines.append(('scaled naive bayes',Pipeline([('scaler',StandardScaler()),('scaled Naive Bayes',GaussianNB())])))

# Evaluating Each model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
names=[]
predictions=[]
for name,model in models:
    fold=KFold(n_splits=10,random_state=0)
    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)
    predictions.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)
    

# Visualizing the Model accuracy
fig=plt.figure()
fig.suptitle("Comparing Algorithms")
plt.boxplot(predictions)
plt.show()


# Prediction we got without applying feature scaling
# 
# 1. Logistic Regression Classification Algorithm : 0.859583 (0.114429)
# 2. K-Nearest Neighbors classification Algorithm : 0.834167 (0.118714)
# 3. Support Vector Machine classification Algorithm : 0.821667 (0.117951)
# 4. Decision Tree Classification Algorithm : 0.840000 (0.106771)
# 5. Naive bayes Classification Algorithm : 0.735833 (0.071715)
# 
# 
# Prediction we got with applying feature scaling
# 
# 1. Logistic Regression Classification Algorithm : 0.859583 (0.114429)
# 2. K-Nearest Neighbors classification Algorithm : 0.834167 (0.118714)
# 3. Support Vector Machine classification Algorithm : 0.821667 (0.117951)
# 4. Decision Tree Classification Algorithm : 0.865833 (0.076508)
# 5. Naive bayes Classification Algorithm : 0.735833 (0.071715)

# # 2(c). Regularisation Tuning For Top 2 Classification Algorithms.

# As per above accuracy we are going to pickup top 2 best performance algorithms.
# 
# 1. Decision Tree Classification Algorithm
# 2. Logistic Regression Classification Algorithm

# In[ ]:


# Decision Tree Tunning Algorithms
import numpy as np
from sklearn.model_selection import GridSearchCV
scaler=StandardScaler().fit(x_train)
rescaledx=scaler.transform(x_train)
param_grid=dict()
model=DecisionTreeClassifier()
fold=KFold(n_splits=10,random_state=5)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)
grid_result=grid.fit(rescaledx,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# In[ ]:


# Logistic Regression Tuning Algorithm
import numpy as np
from sklearn.model_selection import GridSearchCV
scaler=StandardScaler().fit(x_train)
rescaledx=scaler.transform(x_train)
c=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
param_grid=dict(C=c)
model=LogisticRegression()
fold=KFold(n_splits=10,random_state=5)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)
grid_result=grid.fit(rescaledx,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# After Applying Tuning to top 2 algorithms.
# 
# 1. Decision Tree Classification Algorithm Best: 0.852500 using {} 
# 2. Logistic Regression Classification Algorithm Best: 0.853333 using {'C': 0.1} 

# # 2(d). Ensemble and Boosting Classification Algorithms With Feature Scale.

# In[ ]:


# Ensemble and Boosting algorithm to improve performance

#Ensemble
# Boosting methods
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Bagging methods
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
ensembles=[]
ensembles.append(('scaledAB',Pipeline([('scale',StandardScaler()),('AB',AdaBoostClassifier())])))
ensembles.append(('scaledGBC',Pipeline([('scale',StandardScaler()),('GBc',GradientBoostingClassifier())])))
ensembles.append(('scaledRFC',Pipeline([('scale',StandardScaler()),('rf',RandomForestClassifier(n_estimators=10))])))
ensembles.append(('scaledETC',Pipeline([('scale',StandardScaler()),('ETC',ExtraTreesClassifier(n_estimators=10))])))

# Evaluate each Ensemble Techinique
results=[]
names=[]
for name,model in ensembles:
    fold=KFold(n_splits=10,random_state=5)
    result=cross_val_score(model,x_train,y_train,cv=fold,scoring=error)
    results.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)
    
# Visualizing the compared Ensemble Algorithms
fig=plt.figure()
fig.suptitle('Ensemble Compared Algorithms')
plt.boxplot(results)
plt.show()


# We got accuracy for ensemble algorithms likewise...
# 
# 1. Ada Boost Classification Algorithm : 0.854167 (0.073409)
# 2. Gradient Boosting Classification Algorithm : 0.916667 (0.029226)
# 3. Random Forest Classification Algorithm : 0.884583 (0.084411)
# 4. Extra Trees Classification Algoriothm : 0.897917 (0.080434)

# # 2(e). Regularisation Tuning For Top 2 Ensemble and Boosting Classification Algorithms.

# Now we are going to tuning to top 2 ensemble algorithms.
# 
# 1. Gradient Boosting Classification Algorithm
# 2. Extra Trees Classification Algoriothm

# In[ ]:


# GradientBoosting ClassifierTuning
import numpy as np
from sklearn.model_selection import GridSearchCV
scaler=StandardScaler().fit(x_train)
rescaledx=scaler.transform(x_train)
n_estimators=[10,20,30,40,50,100,150,200,250,300]
learning_rate=[0.001,0.01,0.1,0.3,0.5,0.7,1.0]
param_grid=dict(n_estimators=n_estimators,learning_rate=learning_rate)
model=GradientBoostingClassifier()
fold=KFold(n_splits=10,random_state=5)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)
grid_result=grid.fit(rescaledx,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# In[ ]:


# Extra Trees Classifier Classifier Tuning
import numpy as np
from sklearn.model_selection import GridSearchCV
scaler=StandardScaler().fit(x_train)
rescaledx=scaler.transform(x_train)
n_estimators=[10,20,30,40,50,100,150,200]
param_grid=dict(n_estimators=n_estimators)
model=ExtraTreesClassifier()
fold=KFold(n_splits=10,random_state=5)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=error,cv=fold)
grid_result=grid.fit(rescaledx,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))


# After applying tuning to top 2 ensemble algorithms we got accuracy like
# 
# 1. Gradient Boosting Classification Algorithm  0.916667 using {'learning_rate': 0.1, 'n_estimators': 100} 
# 2. Extra Trees Classification Algoriothm 0.917083 using {'n_estimators': 30} 

# # 2(f). Compare All 4 Tunned Algorithms And Selecting The Best Algorithm
# 
# comparing all 4 algorithms top 2 algorithm and top 2 ensemble algorithms.
# 
# 1. Decision Tree Classification Algorithm Best: 0.852500 using {} 
# 2. Logistic Regression Classification Algorithm Best: 0.853333 using {'C': 0.1} 
# 3. Gradient Boosting Classification Algorithm  0.916667 using {'learning_rate': 0.1, 'n_estimators': 100} 
# 4. Extra Trees Classification Algoriothm 0.917083 using {'n_estimators': 30}

# Extra Trees Classification Algoriothm 0.917083 using {'n_estimators': 30} giving the best accuracy performance so we are going to use this ensemble algorithm to fit and predict our dataset

# # 2(g). Fit and Predict The Best Algorithm.

# In[ ]:


# Finalize Model
# we finalized the Extra Trees Classification Algoriothm and evaluate the model for Detection parkinsons disease

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
scaler=StandardScaler().fit(x_train)
scaler_x=scaler.transform(x_train)
model=ExtraTreesClassifier(n_estimators=30)
model.fit(scaler_x,y_train)

#Transform the validation test set data
scaledx_test=scaler.transform(x_test)
y_pred=model.predict(scaledx_test)
y_predtrain=model.predict(scaler_x)


# # 2(h). Accuracy Of An Algorithm.

# In[ ]:


accuracy_mean=accuracy_score(y_train,y_predtrain)
accuracy_matric=confusion_matrix(y_train,y_predtrain)
print("train set",accuracy_mean)
print("train set matrix",accuracy_matric)

accuracy_mean=accuracy_score(y_test,y_pred)
accuracy_matric=confusion_matrix(y_test,y_pred)
print("test set",accuracy_mean)
print("test set matrix",accuracy_matric)


# We got training accuracy of 100% and the test set accuracy 92.3% pretty good right....
# 
# our model not underfit or overfit it fitted perfectly..
# 
# if any suggesion please let me know.
