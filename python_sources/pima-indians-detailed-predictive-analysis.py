#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore')


# In[ ]:


data=pd.read_csv('../input/diabetes.csv')

# Pregnancies Number of times pregnant
#Glucose Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#BloodPressure Diastolic blood pressure (mm Hg)
#SkinThickness Triceps skin fold thickness (mm)
#Insulin 2-Hour serum insulin (mu U/ml)
#BMI Body mass index (weight in kg/(height in m)^2)
#DiabetesPedigreeFunction Diabetes pedigree function
#Age Age (years)
#Outcome Class variable (0 or 1) 268 of 768 are 1, the others are 0
# In[ ]:


data.isna().sum()


# In[ ]:


data.head(6)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.shape


# In[ ]:


data['Outcome'].value_counts()


# In[ ]:


#univariate Analysis
data.iloc[:,:-1].hist(bins=20, figsize=(20,10), grid=False, edgecolor='black', alpha=0.5, color='pink')


# In[ ]:


sns.countplot(data['Outcome'], palette='coolwarm')


# In[ ]:


###################################################################################################


# In[ ]:


#Upsampling
#As the data is less for outcome 1 so its better if we do upsampling for data
from sklearn.utils import resample
data_majority= data.loc[data['Outcome']==0]
data_minority= data.loc[data['Outcome']==1]


# In[ ]:


data_majority.shape


# In[ ]:


data_minority_resampled= resample(data_minority, replace=True, n_samples=500, random_state=0)


# In[ ]:


data1= pd.concat([data_majority, data_minority_resampled])


# In[ ]:


data1['Outcome'].value_counts()


# In[ ]:


#######################################################################################################


# In[ ]:


#BOXPlot #univariate analysis
fig,ax = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
for i in range(0,4):
    sns.boxplot(y=data1.iloc[:,i], ax=ax[0,i])
for j in range(4,8):
    sns.boxplot(y=data1.iloc[:,j], ax=ax[1,j-4])
#it seems that there are outliers in my dataset. But lets try to build model with outliers only            


# In[ ]:


#Bivariate Analysis
sns.pairplot(data1)


# In[ ]:


#Predictive Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


X= data1.iloc[:,:-1]
Y= data1.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, random_state=0, test_size=0.25)


# In[ ]:


#Logistic Regression
logreg= LogisticRegression()
logreg.fit(X_train, Y_train)
from sklearn import metrics
metrics.accuracy_score(Y_test, logreg.predict(X_test))


# In[ ]:


#SVM
svc= SVC(kernel='rbf', random_state=1)
svc.fit(X_train, Y_train)
from sklearn import metrics
metrics.accuracy_score(Y_test, svc.predict(X_test))


# In[ ]:


#Decisison tree
dtree= DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(X_train, Y_train)
metrics.accuracy_score(Y_test, dtree.predict(X_test))


# In[ ]:


#Random Forest
rforest= RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
rforest.fit(X_train, Y_train)
metrics.accuracy_score(Y_test, rforest.predict(X_test))


# In[ ]:


#naive Bayes
nb= GaussianNB()
nb.fit(X_train, Y_train)
metrics.accuracy_score(Y_test, nb.predict(X_test))


# In[ ]:


#KNN
#using optimum value of k to predict the accuracy 
accuracy=[]
for k in np.arange(2,20,1):
    knn= KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    knn.fit(X_train, Y_train)
    acc=metrics.accuracy_score(Y_test, knn.predict(X_test))  
    accuracy.append(acc)
    
print(accuracy)    


# In[ ]:


plt.plot(np.arange(2,20,1), accuracy)
plt.xlim(1,21)


# In[ ]:


#KNN
#taking sqrt of len of X_train values as k value
knn= KNeighborsClassifier(n_neighbors=int(np.sqrt(X_train.shape[0])), metric='minkowski', p=2)
knn.fit(X_train, Y_train)
metrics.accuracy_score(Y_test, knn.predict(X_test))


# In[ ]:


###############################################################################################################


# In[ ]:


#In a nutshell
ac= []
list=[LogisticRegression(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), GaussianNB()]
for i in list:
    model= i
    model.fit(X_train, Y_train)
    a=metrics.accuracy_score(Y_test, model.predict(X_test))
    ac.append(a)
    
print(pd.Series(data=ac, index=['LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'RandomForestClassifier', 'KNeighborsClassifier', 'NaiveBayes']))   


# In[ ]:


##########################################################################################################################


# In[ ]:


#Lets build each model in depth
#check the corr
data.corr()


# In[ ]:


#Heatmap
sns.heatmap(data.corr(), linecolor='black', linewidths=0.2, annot=True)


# In[ ]:


#Logistic regression
logreg1= LogisticRegression()
logreg1.fit(X_train, Y_train)


# In[ ]:


logreg1.predict(X_test)


# In[ ]:


metrics.confusion_matrix(Y_test, logreg1.predict(X_test))


# In[ ]:


metrics.accuracy_score(Y_test, logreg1.predict(X_test))


# In[ ]:


logreg1.coef_.ravel()


# In[ ]:


plot=sns.barplot(x=X_train.columns.tolist(),y=logreg1.coef_.ravel())
plot.set_xticklabels(plot.get_xticklabels(), rotation=90)


# In[ ]:


#as per coffecient using only 4 features to predict the linear model
a=pd.Series(logreg1.coef_.ravel(), index=X_train.columns.tolist()).sort_values(ascending=False)
a


# In[ ]:


var1= a.head(4).index.tolist()
var1


# In[ ]:


logreg2= LogisticRegression()
logreg2.fit(X_train[var1], Y_train)
metrics.confusion_matrix(Y_test, logreg2.predict(X_test[var1]))


# In[ ]:


print('The model accuracy for Log regression is {}'.format(metrics.accuracy_score(Y_test, logreg2.predict(X_test[var1]))))


# In[ ]:


#ROC-AUC curve
from sklearn.metrics import auc, roc_auc_score, roc_curve
fpr, tpr, thres= roc_curve(Y_test, logreg2.predict_proba(X_test[var1])[:,1])


# In[ ]:


roc_auc= roc_auc_score(Y_test, logreg2.predict(X_test[var1]))
print(roc_auc)
plt.plot(fpr, tpr, label='area={}'.format(roc_auc))
plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), linestyle='--')
plt.legend()
plt.show()


# In[ ]:


############################################################################################################################


# In[ ]:


#SVC with rbf kernel
svc1= SVC(kernel='rbf',random_state=1)
svc1.fit(X_train, Y_train)
metrics.accuracy_score(Y_test, svc1.predict(X_test))

######Performing K fold validations with diff kernels#######
# In[ ]:


from sklearn.model_selection import cross_val_score
svc1= SVC(kernel='rbf',random_state=1)
cv_score= cross_val_score(svc1, X, Y, cv=10)
cv_score


# In[ ]:


#lets try to find optimum value of C for Linear kernel
acc=[]
for c in np.arange(0.1,100,10).tolist():
    svc2= SVC(kernel='rbf',C=c,random_state=1)
    cv_score= cross_val_score(svc2, X, Y, cv=10)
    acc.append(cv_score.mean())
    
print(acc)    


# In[ ]:


plt.plot(np.arange(0.1,100,10).tolist(), acc)


# In[ ]:


# C value in between 1 to 10
acc=[]
for c in np.arange(1,10,1).tolist():
    svc2= SVC(kernel='rbf',C=c,random_state=1)
    cv_score= cross_val_score(svc2, X, Y, cv=10)
    acc.append(cv_score.mean())
    
print(acc)    
#lets take c=1


# In[ ]:


#optimal value of gaama
ac=[]
gaama=[0.0001, 0.001, 0.01, 0.1, 1, 10]
for g in gaama:
    svc2= SVC(kernel='rbf',gamma=g ,random_state=1)
    cvs_score=cross_val_score(svc2, X, Y, cv=10)
    ac.append(cvs_score.mean())
    
print(ac)    


# In[ ]:


plt.plot(gaama, ac)


# In[ ]:


#lets perform Grid search cv for optimum value of c and gaama
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1, 1, 10, 100], 'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}
svc_model=SVC()
grid= GridSearchCV(svc_model, param_grid=param_grid, cv=5, refit=True)


# In[ ]:


grid.fit(X_train, Y_train)
grid.predict(X_test)
metrics.accuracy_score(Y_test, grid.predict(X_test))


# In[ ]:


#Print Hyperparameter
print('Best parameter : {}'.format(grid.best_params_))
print('Best Score: {}'.format(grid.best_score_))


# In[ ]:


########################################################################################################


# In[ ]:


#Decision tree
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(criterion='entropy', random_state=1)


# In[ ]:


dt.fit(X_train, Y_train)
metrics.accuracy_score(Y_test, dt.predict(X_test))


# In[ ]:


from sklearn.metrics import roc_auc_score
dt_roc_auc= roc_auc_score(Y_test,dt.predict_proba(X_test)[:,1])
dt_roc_auc


# In[ ]:


from sklearn.metrics import roc_curve
fpr, tpr, thres= roc_curve(Y_test,dt.predict_proba(X_test)[:,1])


# In[ ]:


plt.plot(fpr, tpr)
plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), linestyle='--')


# In[ ]:


#lets draw the tree to analyse better
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn import tree
import pydot


# In[ ]:


#Create DOT data
dot_data= StringIO()
export_graphviz(dt, out_file=dot_data, feature_names=X_train.columns.tolist(), class_names=['0','1'], rounded=True, filled=True)

#Draw Graph
graph= pydot.graph_from_dot_data(dot_data.getvalue())

#Show Graph
Image(graph[0].create_png())


# In[ ]:


plot1=sns.barplot(X_train.columns.tolist(), dt.feature_importances_)
plot1.set_xticklabels(plot.get_xticklabels(), rotation=90)


# In[ ]:


#Now lets try to tune the hyper parameters in Decision tree. We have 4 parameters to tune
#1 max_depth
#2 min_samples_split
#3 min_samples_leaf
#4 max_features


# In[ ]:


#Lets keep max_depth from 1 to 10 and check accuracy
max_depth= np.linspace(1,20,20).tolist()
train_result=[]
test_result=[]
for max_depth in max_depth:
    dtr=DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    dtr.fit(X_train, Y_train)
    
    acc_train= metrics.accuracy_score(Y_train, dtr.predict(X_train))
    acc_test= metrics.accuracy_score(Y_test, dtr.predict(X_test))
    
    train_result.append(acc_train)
    test_result.append(acc_test)
    
print(train_result)
print('\n')
print(test_result) 
plt.bar(np.linspace(1,20,20).tolist(), train_result, )
plt.bar(np.linspace(1,20,20).tolist(), test_result, color='pink')


# In[ ]:


#Lets check min_samples_split to tune the model
#lets take it to 10 to 100%
min_samples_split= np.linspace(0.1,1,10).tolist()
train_result=[]
test_result=[]
for min_samples_split in min_samples_split:
    dtr=DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split)
    dtr.fit(X_train, Y_train)
    
    acc_train= metrics.accuracy_score(Y_train, dtr.predict(X_train))
    acc_test= metrics.accuracy_score(Y_test, dtr.predict(X_test))
    
    train_result.append(acc_train)
    test_result.append(acc_test)
    
print(train_result)
print('\n')
print(test_result)
plt.bar(np.arange(0.1, 1.1, 0.1).tolist(), train_result, edgecolor='black',linewidth=0.2, width=0.08)
plt.bar(np.arange(0.1, 1.1, 0.1).tolist(), test_result, color='pink', edgecolor='black',linewidth=0.2, width=0.08)


# In[ ]:


#Lets check max_features to tune the model

max_features= np.arange(1, (len(X.columns.tolist())+1)).tolist()
train_result=[]
test_result=[]
for max_features in max_features:
    dtr=DecisionTreeClassifier(criterion='entropy', max_features= max_features)
    dtr.fit(X_train, Y_train)
    
    acc_train= metrics.accuracy_score(Y_train, dtr.predict(X_train))
    acc_test= metrics.accuracy_score(Y_test, dtr.predict(X_test))
    
    train_result.append(acc_train)
    test_result.append(acc_test)
    
print(train_result)
print('\n')
print(test_result)
plt.bar(np.arange(1, (len(X.columns.tolist())+1)).tolist(), train_result, edgecolor='black',linewidth=0.2, )
plt.bar(np.arange(1, (len(X.columns.tolist())+1)).tolist(), test_result, color='pink', edgecolor='black',linewidth=0.2)


# In[ ]:


#Grid Search CV
from sklearn.model_selection import GridSearchCV
param={'max_depth':np.arange(1,21).tolist(), 'min_samples_split':np.linspace(0.1,1,10).tolist(),
       'max_features':["auto", "sqrt","log2"]}
model= DecisionTreeClassifier(criterion='entropy', )
grid= GridSearchCV(model, param_grid=param, refit=True)


# In[ ]:


grid.fit(X_train, Y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid_pred= grid.predict(X_test)
grid_pred


# In[ ]:


print('Tuned HyperParameter K : {}'.format(grid.best_params_))
print('Best Score : {}'.format(grid.best_score_))
print('Accuracy Score: {}'.format(metrics.accuracy_score(Y_test, grid_pred)))


# In[ ]:


##################################################################################################################


# In[ ]:


#Random forest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
acc=[]
for i in np.arange(10,300,10).tolist():
    rfr= RandomForestClassifier(n_estimators=i, criterion='entropy', random_state=1)
    rfr.fit(X_train, Y_train)
    a= metrics.accuracy_score(Y_test, rfr.predict(X_test))
    acc.append(a)

print(acc)    


# In[ ]:


plt.plot(np.arange(10,300,10), acc)
plt.show()
#Seems like n_estimator= 20 has max accuracy


# In[ ]:


#Grid Search CV
rfr= RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=1)
param_grid= {'max_depth':np.arange(1,21).tolist(), 'min_samples_split':np.linspace(0.1,1,10).tolist(), 
             'max_features':["auto", "sqrt","log2"]}
from sklearn.model_selection import GridSearchCV
grid_random= GridSearchCV(rfr, param_grid=param_grid, refit=True)


# In[ ]:


grid_random.fit(X_train, Y_train)


# In[ ]:


print('Tuned HyperParameter K : {}'.format(grid.best_params_))
print('Best Score : {}'.format(grid.best_score_))
print('Accuracy Score: {}'.format(metrics.accuracy_score(Y_test, grid_random.predict(X_test))))


# In[ ]:


###################################################################################################################


# In[ ]:


#Using boosting techniques
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn import model_selection


# In[ ]:


#Adaboost Classifier
kfold = model_selection.KFold(n_splits=10, random_state=5)
model1= AdaBoostClassifier(n_estimators=20, random_state=5)
cvscore= model_selection.cross_val_score(model1, X, Y, cv=kfold)
cvscore.mean()


# In[ ]:


#XGBoost Classifier
k= model_selection.KFold(n_splits=10, random_state=10)
model2= XGBClassifier(random_state=10)
cvs= model_selection.cross_val_score(model2, X, Y, cv=k)
cvs.mean()


# In[ ]:


#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
model3= GradientBoostingClassifier(n_estimators=150, random_state=1)
model3.fit(X_train, Y_train)
metrics.accuracy_score(Y_test, model3.predict(X_test))

