#!/usr/bin/env python
# coding: utf-8

# In[ ]:






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno


# In[ ]:


data=pd.read_csv("../input/sales-analysis/SalesKaggle3.csv")


# In[ ]:


data1=pd.read_csv("../input/sales-analysis/SalesKaggle3.csv")


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.shape


# In[ ]:


missingno.matrix(data)


# In[ ]:


data=data.drop(["SKU_number","Order","SoldCount","ReleaseYear"],axis=1)


# In[ ]:


data.head()


# In[ ]:


sns.distplot(data.StrengthFactor)


# In[ ]:


sns.distplot(data.PriceReg)


# In[ ]:


sns.distplot(data.LowUserPrice)


# In[ ]:


data.MarketingType.value_counts()


# In[ ]:


sns.countplot(data.SoldFlag)


# In[ ]:


data.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[ ]:


scdata=pd.DataFrame(sc.fit_transform(data.drop(["File_Type","SoldFlag","MarketingType","New_Release_Flag"],axis=1)),columns=data.drop(["File_Type","SoldFlag","MarketingType","New_Release_Flag"],axis=1).columns)


# In[ ]:


scdata.head()


# In[ ]:


scdata.isnull().sum()


# In[ ]:


scdata[["File_Type","SoldFlag","MarketingType","New_Release_Flag"]]=data[["File_Type","SoldFlag","MarketingType","New_Release_Flag"]]


# In[ ]:


scdata.head()


# In[ ]:


data=pd.get_dummies(scdata)


# In[ ]:


data.head()


# In[ ]:


sns.pairplot(data)


# In[ ]:


test=data[data.SoldFlag.isnull()]


# In[ ]:


test.SoldFlag.value_counts()


# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


train=data[data.SoldFlag.notnull()]


# In[ ]:


train.head()


# In[ ]:





# In[ ]:


train["SoldFlag"].unique()


# In[ ]:





# In[ ]:


test.head()


# In[ ]:


xtest11=test.drop("SoldFlag",axis=1)


# In[ ]:


xtrain1=train.drop("SoldFlag",axis=1)


# In[ ]:


ytrain1=train.SoldFlag


# In[ ]:


# Train Test Split


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(xtrain1,ytrain1,test_size=0.30,random_state=2)


# In[ ]:


xtrain.head()


# In[ ]:


ytrain.head()


# In[ ]:


# Decision Tree


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[ ]:


dt=DecisionTreeClassifier()


# In[ ]:


ypred=dt.fit(xtrain,ytrain).predict(xtest)


# In[ ]:


print (classification_report(ytest, ypred))
print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred) * 100))

sns.heatmap(confusion_matrix(ytest, ypred), annot=True, fmt='.2f')
plt.xlabel("Predicted")
plt.ylabel("Actal")
plt.show()


# In[ ]:


# Random Forest


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
batman=RandomForestClassifier()


# In[ ]:


ypred1=batman.fit(xtrain,ytrain).predict(xtest)


# In[ ]:


print (classification_report(ytest, ypred1))
print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred1) * 100))

sns.heatmap(confusion_matrix(ytest, ypred1), annot=True, fmt='.2f')
plt.xlabel("Predicted")
plt.ylabel("Actal")
plt.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_dist = {'max_depth': [2, 3, 4],'bootstrap': [True, False],'max_features': ['auto', 'sqrt', 'log2', None],'criterion': ['gini', 'entropy']}

cv_rf = GridSearchCV(batman, cv = 10,param_grid=param_dist)


# In[ ]:


cv_rf.fit(xtrain,ytrain)


# In[ ]:


cv_rf.best_params_


# In[ ]:


batman=RandomForestClassifier(bootstrap = True, criterion = 'gini', max_depth = 4, max_features = None)


# In[ ]:


ypred2=batman.fit(xtrain,ytrain).predict(xtest)


# In[ ]:


print (classification_report(ytest, ypred2))
print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred2) * 100))

sns.heatmap(confusion_matrix(ytest, ypred2), annot=True, fmt='.2f')
plt.xlabel("Predicted")
plt.ylabel("Actal")
plt.show()


# In[ ]:


# Logistic Regression


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[ ]:


ypred3=lr.fit(xtrain,ytrain).predict(xtest)


# In[ ]:


print (classification_report(ytest, ypred3))
print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred3) * 100))

sns.heatmap(confusion_matrix(ytest, ypred3), annot=True, fmt='.2f')
plt.xlabel("Predicted")
plt.ylabel("Actal")
plt.show()


# In[ ]:


# KNN Model


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[ ]:


ypred7=knn.fit(xtrain,ytrain).predict(xtest)


# In[ ]:


print (classification_report(ytest, ypred7))
print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred7) * 100))

sns.heatmap(confusion_matrix(ytest, ypred7), annot=True, fmt='.2f')
plt.xlabel("Predicted")
plt.ylabel("Actal")
plt.show()


# In[ ]:


# Applying Grid Search
l=[]
for i in range(1,10):
    l.append(i)
from sklearn.model_selection import GridSearchCV

param_dist = {"n_neighbors": l, "p":[1,2,3]}
cv_rf = GridSearchCV(knn, cv = 3,param_grid=param_dist)
cv_rf.fit(xtrain, np.ravel(ytrain))
print(cv_rf.best_params_)


# In[ ]:


knn=KNeighborsClassifier(n_neighbors= 8, p= 3)


# In[ ]:


ypred8=knn.fit(xtrain,ytrain).predict(xtest)


# In[ ]:


print (classification_report(ytest, ypred8))
print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred8) * 100))

sns.heatmap(confusion_matrix(ytest, ypred8), annot=True, fmt='.2f')
plt.xlabel("Predicted")
plt.ylabel("Actal")
plt.show()


# In[ ]:


# Trying out XGBoost


# In[ ]:


from xgboost import XGBRFClassifier
xg=XGBRFClassifier()


# In[ ]:


ypred9=xg.fit(xtrain,ytrain).predict(xtest)


# In[ ]:


print (classification_report(ytest, ypred9))
print ("Accuracy: {:.2f} %".format(accuracy_score(ytest, ypred9) * 100))

sns.heatmap(confusion_matrix(ytest, ypred9), annot=True, fmt='.2f')
plt.xlabel("Predicted")
plt.ylabel("Actal")
plt.show()


# In[ ]:


# Predicting the values of SaleFlag using out model using Random Forest Model


# In[ ]:


x2=test.drop("SoldFlag",axis=1)


# In[ ]:


ypred10=batman.predict(x2)


# In[ ]:


pd.DataFrame(ypred10).head()     # Predicted values


# In[ ]:


xyz=data1[data1.SoldFlag.isnull()]


# In[ ]:


xyz.head()


# In[ ]:


xyz.SoldFlag=ypred10


# In[ ]:


xyz.isnull().sum()


# In[ ]:


xyz.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




