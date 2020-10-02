#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Datasets

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


wrng=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
wrng


# In[ ]:


df=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
d=df
df


# # EDA

# In[ ]:


df['Outcome'].value_counts()


# In[ ]:


plt.figure(figsize=(14,10))
sn.heatmap(df.corr(),annot=True)
plt.show()


# In[ ]:


r=df["Outcome"]
df=df.drop(["Outcome"],axis=1)
df.corr()


# In[ ]:


sn.set(font_scale=1.15)
plt.figure(figsize=(14, 10))
sn.heatmap(df.corr(), vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="black")
plt.title('Correlation between features');


# In[ ]:


d.columns


# In[ ]:


d.describe(include='all')


# In[ ]:


d.info()


# In[ ]:


d.isnull().sum()


# # EDA using Plotting

# In[ ]:


fig,ax=plt.subplots(4,2,figsize=(16,16))
sn.distplot(d.Age, bins = 20, ax=ax[0,0]) 
sn.distplot(d.Pregnancies, bins = 20, ax=ax[0,1]) 
sn.distplot(d.Glucose, bins = 20, ax=ax[1,0]) 
sn.distplot(d.BloodPressure, bins = 20, ax=ax[1,1]) 
sn.distplot(d.SkinThickness, bins = 20, ax=ax[2,0])
sn.distplot(d.Insulin, bins = 20, ax=ax[2,1])
sn.distplot(d.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 
sn.distplot(d.BMI, bins = 20, ax=ax[3,1]) 


# In[ ]:


sn.pairplot(d, x_vars=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'], y_vars='Outcome', height=7, aspect=0.7, kind='reg');


# In[ ]:


for x1 in df.columns:
    for y1 in df.columns:
        sn.lmplot(x=x1,y=y1,data=d,hue='Outcome')


# In[ ]:


for x2 in d.columns:
    print (x2)


# In[ ]:


for x2 in d.columns:
    sn.FacetGrid(d, hue = 'Outcome' , size = 5)      .map(sn.distplot , x2)      .add_legend()
    plt.show() 


# In[ ]:


sn.set_style("whitegrid")
sn.pairplot(d,hue="Outcome",size=3);
plt.show()


# In[ ]:


tmp=pd.cut(d['Age'],[18,30,42,54,66,78,80])


# In[ ]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['Glucose'],hue=d['Outcome'])


# In[ ]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['Pregnancies'],hue=d['Outcome'])


# In[ ]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['BMI'],hue=d['Outcome'])


# In[ ]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['BloodPressure'],hue=d['Outcome'])


# In[ ]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['Insulin'],hue=d['Outcome'])


# In[ ]:


plt.figure(figsize=(10,6))
sn.boxplot(x=tmp,y=d['DiabetesPedigreeFunction'],hue=d['Outcome'])


# In[ ]:


d['SkinThickness'].max()


# In[ ]:


tmp=pd.cut(d['SkinThickness'],[0,15,30,45,60,75,90,105])


# In[ ]:


for y1 in d.columns:
    plt.figure(figsize=(10,6))
    sn.boxplot(x=tmp,y=d[y1],hue=d['Outcome'])


# # Imputation

# In[ ]:


df[df['SkinThickness']>60]


# In[ ]:


d.loc[(d['SkinThickness'] == 0) , 'SkinThickness' ] = np.nan
d['SkinThickness'] = d['SkinThickness'].fillna(d['SkinThickness'].median())


# In[ ]:


d['BloodPressure'].max()


# In[ ]:


d['BloodPressure'].min()


# In[ ]:


tmp=pd.cut(d['BloodPressure'],[0,30,60,90,120,150])
for y1 in d.columns:
    plt.figure(figsize=(10,6))
    sn.boxplot(x=tmp,y=d[y1],hue=d['Outcome'])


# In[ ]:


d.loc[(d['BloodPressure'] < 30) | (d['BloodPressure']>120) , 'BloodPressure' ]


# In[ ]:


d.loc[ d['BloodPressure'] == 0 , 'BloodPressure' ] = np.nan
d['BloodPressure'] = d['BloodPressure'].fillna(d['BloodPressure'].median())


# In[ ]:


d.loc[(d['BloodPressure'] < 30) | (d['BloodPressure']>120) , 'BloodPressure' ]


# In[ ]:


d[d['BloodPressure']<35]


# In[ ]:


d.loc[d['BMI']==0,'BMI']= np.nan
d['BMI'] = d['BMI'].fillna(d['BMI'].median())


# In[ ]:


d['BMI'].min()


# In[ ]:


d['BMI'].max()


# In[ ]:


d.Age.max()


# In[ ]:


d.Age.min()


# In[ ]:


d.Glucose.min()


# In[ ]:


d.Glucose.max()


# In[ ]:


tmp=pd.cut(d['Glucose'],[0,30,60,90,120,150,180,210])
for y1 in d.columns:
    plt.figure(figsize=(10,6))
    sn.boxplot(x=tmp,y=d[y1],hue=d['Outcome'])


# In[ ]:


d.loc[d['Glucose']==0,'Glucose']=np.nan
d['Glucose'] = d['Glucose'].fillna(d['Glucose'].median())


# In[ ]:


d.loc[ d['Glucose'] <60 , 'Glucose' ]


# # Building Models and Evaluating

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

# split the data set into train and test
x_1, x_test, y_1, y_test = train_test_split(df, r, test_size=0.3, random_state=0)

# split the train data set into cross validation train and cross validation test
x_tr, x_cv, y_tr, y_cv = train_test_split(x_1, y_1, test_size=0.3)


# In[ ]:


for i in range(1,30,2):
    # instantiate learning model (k = 30)
    knn = KNeighborsClassifier(n_neighbors=i)

    # fitting the model on crossvalidation train
    knn.fit(x_tr, y_tr)

    # predict the response on the crossvalidation train
    pred = knn.predict(x_cv)

    # evaluate CV accuracy
    acc = accuracy_score(y_cv, pred, normalize=True) * float(100)
    print('\nCV accuracy for k = %d is %d%%' % (i, acc))


# In[ ]:


y_pred=knn.predict(x_test)
acc=accuracy_score(y_pred,y_test)*float(100)
acc


# In[ ]:


myList = list(range(0,50))
neighbors = list(filter(lambda x: x % 2 != 0, myList))

cv_scores=[]
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_tr, y_tr, cv=3, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is %d.' % optimal_k)

# plot misclassification error vs k 
plt.plot(neighbors, MSE)

for xy in zip(neighbors, np.round(MSE,3)):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

print("the misclassification error for each k value is : ", np.round(MSE,3))


# In[ ]:


knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
 
# fitting the model
knn_optimal.fit(x_tr, y_tr)

# predict the response
pred = knn_optimal.predict(x_test)

# evaluate accuracy
acc = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k, acc))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , pred)
cm


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test , pred))


# In[ ]:


def k_classifier_brute(X_train , Y_train):
    neighbors = list(range(5 , 51 , 2))
    cv_scores = []
    for i in neighbors:
        neigh = KNeighborsClassifier(n_neighbors = i,metric='correlation' )
        scores = cross_val_score(neigh , x_tr , y_tr , cv = 10 , scoring = 'accuracy')
        cv_scores.append(scores.mean())
    MSE = [1-x for x in cv_scores]
    optimal_k = neighbors[MSE.index(min(MSE))]
    print('Optimal k is {}'.format(optimal_k))
    print('Misclassification error for each k is {}'.format(np.round(MSE , 3)))
    plt.plot(neighbors , MSE)
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Misclassification Error')
    plt.title('Neighbors v/s Misclassification Error')
    plt.show()
    
    return optimal_k


# In[ ]:


optimal_k_pidd = k_classifier_brute(x_tr , y_tr)


# In[ ]:


knn_optimal_for_pidd = KNeighborsClassifier(n_neighbors = optimal_k_pidd , metric = 'correlation')
knn_optimal_for_pidd.fit(x_tr , y_tr)
pred = knn_optimal_for_pidd.predict(x_test)
accuracy_score(pred,y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , pred)
cm


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test , pred))


# In[ ]:





# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(df, r, random_state=1,test_size=0.2)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,Y_train)
predic=nb.predict(X_test)


# In[ ]:


accuracy_score(predic,Y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , predic)
cm


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test , predic))


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train)
predic=lr.predict(X_test)
accuracy_score(predic,Y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , predic)
cm


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test , predic))


# In[ ]:


from sklearn.naive_bayes import BernoulliNB
nb=BernoulliNB()
nb.fit(X_train,Y_train)
predic=nb.predict(X_test)
accuracy_score(predic,Y_test)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train,Y_train)
predic=model.predict(X_test)
accuracy_score(predic,Y_test)


# # Oversampling

# In[ ]:


from imblearn.over_sampling import SMOTE
smt = SMOTE()
X_train , Y_train = smt.fit_sample(X_train , Y_train)
print(Y_train.value_counts())


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.metrics import confusion_matrix


# In[ ]:


def k_classifier_brute(X_train , Y_train):
    neighbors = list(range(5 , 51 , 2))
    cv_scores = []
    for i in neighbors:
        neigh = KNeighborsClassifier(n_neighbors = i,metric='correlation')
        scores = cross_val_score(neigh , X_train , Y_train , cv = 10 , scoring = 'accuracy')
        cv_scores.append(scores.mean())
    MSE = [1-x for x in cv_scores]
    optimal_k = neighbors[MSE.index(min(MSE))]
    print('Optimal k is {}'.format(optimal_k))
    print('Misclassification error for each k is {}'.format(np.round(MSE , 3)))
    plt.plot(neighbors , MSE)
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Misclassification Error')
    plt.title('Neighbors v/s Misclassification Error')
    plt.show()
    
    return optimal_k


# In[ ]:


optimal_k_pidd = k_classifier_brute(X_train , Y_train)


# In[ ]:


knn_optimal_for_pidd = KNeighborsClassifier(n_neighbors = optimal_k_pidd , metric = 'correlation')
knn_optimal_for_pidd.fit(X_train , Y_train)
pred = knn_optimal_for_pidd.predict(X_test)


# In[ ]:


train_acc = knn_optimal_for_pidd.score(X_train , Y_train)
print('Training Accurcy = {}'.format(train_acc))

train_error = 1 - train_acc
print('Training Error = {}'.format(train_error))


# In[ ]:


accuracy = accuracy_score(Y_test , pred)
print('The accuracy of the model for k = {} is {}'.format(optimal_k_pidd , accuracy))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , predic)
cm


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test , predic))


# In[ ]:


nb1=GaussianNB()
nb1.fit(X_train,Y_train)
predic=nb1.predict(X_test)
accuracy_score(predic,Y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , predic)
cm


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test , predic))


# In[ ]:


nb1=BernoulliNB()
nb1.fit(X_train,Y_train)
predic=nb1.predict(X_test)
accuracy_score(predic,Y_test)


# In[ ]:


cm = confusion_matrix(Y_test , predic)
cm


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test , predic))


# In[ ]:


lr1=LogisticRegression()
lr1.fit(X_train,Y_train)
predic=lr1.predict(X_test)
accuracy_score(predic,Y_test)


# In[ ]:


cm = confusion_matrix(Y_test , predic)
cm


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test , predic))


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train,Y_train)
predic=model.predict(X_test)
accuracy_score(predic,Y_test)


# # SVM and XGBoost Classifiers

# In[ ]:


from sklearn import svm
clfr1=svm.SVC(kernel='linear')
clfr2=svm.SVC(kernel='rbf')
clfr1.fit(X_train,Y_train)
clfr2.fit(X_train,Y_train)
predic1=clfr1.predict(X_test)
predic2=clfr2.predict(X_test)
print("The accuracy for SVM model With linear Kernel is {} ", accuracy_score(predic1,Y_test))
print("The accuracy for SVM model With RBF Kernel is {} ", accuracy_score(predic2,Y_test))


# In[ ]:


cm1 = confusion_matrix(Y_test , predic1)
cm1


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm1 , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


cm2= confusion_matrix(Y_test , predic2)
cm2


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm2 , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test , predic1))
print(classification_report(Y_test , predic2))


# In[ ]:


import xgboost as xgb
xgb_model=xgb.XGBClassifier().fit(X_train,Y_train)
predictions=xgb_model.predict(X_test)
actuals=Y_test
print(accuracy_score(actuals,predictions))


# In[ ]:


cm4= confusion_matrix(actuals,predictions)
cm4


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm4 , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test , predic1))


# # Decision Trees and Random Forest

# In[ ]:


from sklearn import tree
clfr3=tree.DecisionTreeClassifier()
dt_model=clfr3.fit(X_train,Y_train)
predic3=clfr3.predict(X_test)
accuracy_score(predic3,Y_test)


# In[ ]:


cm3= confusion_matrix(Y_test , predic3)
cm3


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm3 , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test , predic3))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_mdl=RandomForestClassifier().fit(X_train,Y_train)
predic5=rf_mdl.predict(X_test)
accuracy_score(predic5,Y_test)


# In[ ]:


cm5= confusion_matrix(Y_test , predic5)
cm5


# In[ ]:


labels = ['negative' , 'positive']
df_cm = pd.DataFrame(cm5 , index = labels , columns = labels)

sn.heatmap(df_cm , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test , predic5))


# In[ ]:





# # Fine Tuning Hyperparameters using GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
tuned_param=[{'c':[10**-4,10**-2,10**0,10**2]}]
lr1=GridSearchCV(LogisticRegression(),tuned_param)
lr1.fit(X_train,Y_train)
predic=lr1.predict(X_test)
accuracy_score(predic,Y_test)


# In[ ]:


print(lr.best_estimator_)
print(model.score(X_test,Y_test))


# In[ ]:




