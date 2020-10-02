#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/train_real.csv')
test_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/test_real.csv')


# In[ ]:


train_df.head()


# Your EDA (Exploratory Data Analysis) goes here. Get a good feel of the data, look out for stuff that might help later.

# In[ ]:


''''f,ax=plt.subplots(1,2,figsize=(18,8))
train_df['label'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('label')
ax[0].set_ylabel('')
sns.countplot('label',data=train_df,ax=ax[1])
ax[1].set_title('label')
plt.show()'''


# In[ ]:


''''f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['Altitude','label']].groupby(['Altitude']).mean().plot.bar(ax=ax[0])
ax[0].set_title('altitude vs label')
sns.countplot('Altitude',hue='label',data=train_df,ax=ax[1])
ax[1].set_title('Altitude vs label')
plt.show()


# In[ ]:


''''f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['Azimuthal_angle','label']].groupby(['Azimuthal_angle']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Azimuthal_angle vs label')
sns.countplot('Azimuthal_angle',hue='label',data=train_df,ax=ax[1])
ax[1].set_title('Azimuthal_angle vs label')
plt.show()


# In[ ]:


''''f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['Incline','label']].groupby(['Incline']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Incline vs label')
sns.countplot('Incline',hue='label',data=train_df,ax=ax[1])
ax[1].set_title('Incline vs label')
plt.show()


# In[ ]:


''''f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['H_dist_Hydro','label']].groupby(['H_dist_Hydro']).mean().plot.bar(ax=ax[0])
ax[0].set_title('H_dist_Hydro vs label')
sns.countplot('H_dist_Hydro',hue='label',data=train_df,ax=ax[1])
ax[1].set_title('H_dist_Hydro vs label')
plt.show()


# In[ ]:


''''f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['V_dist_Hydro','label']].groupby(['V_dist_Hydro']).mean().plot.bar(ax=ax[0])
ax[0].set_title('V_dist_Hydro vs label')
sns.countplot('V_dist_Hydro',hue='label',data=train_df,ax=ax[1])
ax[1].set_title('V_dist_Hydro vs label')
plt.show()


# In[ ]:


''''f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['V_dist_Hydro','label']].groupby(['V_dist_Hydro']).mean().plot.bar(ax=ax[0])
ax[0].set_title('V_dist_Hydro vs label')
sns.countplot('V_dist_Hydro',hue='label',data=train_df,ax=ax[1])
ax[1].set_title('V_dist_Hydro vs label')
plt.show()


# In[ ]:


''''f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['H_dist_Road','label']].groupby(['H_dist_Road']).mean().plot.bar(ax=ax[0])
ax[0].set_title('H_dist_Road vs label')
sns.countplot('H_dist_Road',hue='label',data=train_df,ax=ax[1])
ax[1].set_title('H_dist_Road vs label')
plt.show()


# In[ ]:


''''f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['Hillshade_9am','label']].groupby(['Hillshade_9am']).mean().plot.bar(ax=ax[0])
ax[0].set_title('H_dist_Road vs label')
sns.countplot('Hillshade_9am',hue='label',data=train_df,ax=ax[1])
ax[1].set_title('Hillshade_9am vs label')
plt.show()


# In[ ]:


sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.5) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,16)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # Train-Test split
# 
# You are also expected to split train_df into train and validation dataframes (or else choose a cross validation scheme)
# 

# In[ ]:


#train_df.dtypes


# In[ ]:


X_train = train_df.drop(['Id', 'label', 'Soil'], axis=1)
Y_train = train_df['label']


# Dropping 'Soil' column for convenience. You should try to think of ways to generate features from these columns. Try seeing kernels from other Kaggle (Tabular data) competitions for inspirations for Feature Engineering.

# In[ ]:


#X_train.head()


# In[ ]:


#Y_train.head()


# In[ ]:


X_test = test_df.drop(['Id' ,'Soil'], axis=1)
X_test.head()


# # Basic Binary Logistic regression
# 
# You should obviously see the limitations of a model and familiarise yourselves with other models in sci-kit learn.

# In[ ]:


from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


# In[ ]:


train,test=train_test_split(train_df,test_size=0.3,random_state=0,stratify=train_df['label'])


# In[ ]:


train[train.columns[1:15]]


# In[ ]:


train_X=train[train.columns[1:15]]
train_Y=train[train.columns[16]]
test_X=test[test.columns[1:15]]
test_Y=test[test.columns[16]]
X=train_df[test_df.columns[1:15]]
Y=train_df['label']


# In[ ]:


train_X = train.drop(['Id', 'label', 'Soil'], axis=1)
test_X=test.drop(['Id', 'Soil','label'], axis=1)


# In[ ]:



'''from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

clf=LogisticRegressionCV(cv=5, max_iter = 1000).fit(train_X,train_Y)


# In[ ]:


''''train_res=clf.predict(train_X)
train_res


# In[ ]:


'''''print('The accuracy of the Logistic Regression is',metrics.roc_auc_score(train_res,train_Y))


# In[ ]:


''''model=svm.SVC(C=0.05, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
model.fit(train_X,train_Y)
prediction1=model.predict(test_X)


# In[ ]:


''''model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(train_X,train_Y)
prediction2=model.predict(test_X)
print('Accuracy for linear SVM is',metrics.roc_auc_score(prediction2,test_Y))


# In[ ]:


''''model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction4=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.roc_auc_score(prediction4,test_Y))


# In[ ]:


''''model=GaussianNB()
model.fit(train_X,train_Y)
prediction6=model.predict(test_X)
print('The accuracy of the NaiveBayes is',metrics.roc_auc_score(prediction6,test_Y))


# In[ ]:


model=RandomForestClassifier(bootstrap=True, class_weight= None, criterion='gini',                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=159,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
model.fit(train_X,train_Y)
prediction7=model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.roc_auc_score(prediction7,test_Y))


# In[ ]:


from progressbar import ProgressBar
pbar = ProgressBar()


# In[ ]:


''''from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Logistic Regression','KNN','Random Forest']
models=[LogisticRegression(),KNeighborsClassifier(n_neighbors=9),RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=159,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)]
for i in pbar(models):
    model = i
    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2


# In[ ]:


''''plt.subplots(figsize=(12,6))
box=pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()


# In[ ]:


from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=100,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=160,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[ ]:


''''#With Hyper Parameters Tuning
#2-3,SVM
#importing modules
from sklearn.model_selection import GridSearchCV
from sklearn import svm
#making the instance
model=svm.SVC()
#Hyper Parameters Set
params = {'C': [6,7,8,9,10], 
          'kernel': ['linear','rbf']}
#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)
#Learning
model1.fit(train_X,train_Y)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(test_X)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,test_Y))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_Y))


# In[ ]:


''''#With Hyper Parameters Tuning
#2-2,Randomforest
#importing modules
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#making the instance
model=RandomForestClassifier()
#hyper parameters set
params = {'criterion':['gini','entropy'],
          'n_estimators':range(155,165,1),
          'min_samples_leaf':[1,2,3],
          'min_samples_split':[3,4,5,6,7], 
          'random_state':[123],
          'n_jobs':[-1]}
#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)
#learning
model1.fit(train_X,train_Y)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(test_X)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,test_Y))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_Y))


# In[ ]:


''''model=RandomForestClassifier(bootstrap=True, class_weight= None, criterion='entropy',max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=7,
                       min_weight_fraction_leaf=0.0, n_estimators=160,
                       n_jobs=-1, oob_score=False, random_state=123, verbose=0,
                       warm_start=False)
model.fit(train_X,train_Y)
prediction7=model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.roc_auc_score(prediction7,test_Y))


# In[ ]:


''''model=KNeighborsClassifier('algorithm': 'auto', 'leaf_size': 1, 'n_jobs': -1, 'n_neighbors': 10, 'weights': 'distance')
model.fit(train_X,train_Y)
prediction5=model.predict(test_X)
print('The accuracy of the KNN is',metrics.roc_auc_score(prediction5,test_Y))


# In[ ]:


''''from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25]
gamma=[0.1,0.2,0.3,0.4]
kernel=['rbf']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)


# In[ ]:


''''from sklearn.model_selection import GridSearchCV
n_estimators=range(150,170,1)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)


# In[ ]:


''''from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=9)),
                                              ('RFor',RandomForestClassifier(n_estimators=150,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05))
                                                             
                                             ], 
                       voting='soft').fit(train_X,train_Y)
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))


# In[ ]:


''''from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
ada.fit(train_X,train_Y)
result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())


# In[ ]:


''''train_res=ada.predict(X_train)
train_res


# In[ ]:


test_res =model.predict(X_test)
test_res


# # Make a submission
# 

# In[ ]:


submission_df = pd.DataFrame()
submission_df['Id'] = test_df['Id']


# In[ ]:


submission_df['Predicted'] = test_res.tolist()


# In[ ]:


submission_df.tail()


# In[ ]:


submission_df.to_csv('timepass.csv',index=False)


# In[ ]:


get_ipython().system('ls')


# In[ ]:




