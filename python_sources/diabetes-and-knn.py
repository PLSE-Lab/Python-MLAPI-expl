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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <h2>Introduction</h2>
# <div></div>
# <h3>Problem Statement</h3><div></div>
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# 
# <h3>My Model Approach</h3>
# Though it is abinary classification problem this kind of health related problem can be well surrounded by KNN as most of the health parameter of a victim belongs to more or less in same zone. So I am using KNN Algo here may be later we can try different Algo also like Neural Nets,Logits etc.
# 

# <h2>Modules Importing</h2>

# In[ ]:


import pandas as pd #to play with in memory dataset
import numpy as np #to do math
import matplotlib.pyplot as plt#general plot expert
import seaborn as sns #distribution expert
plt.style.use("ggplot")

import warnings
warnings.filterwarnings("ignore")


# <h2>Data Loading And Basic data utils </h2>

# In[ ]:


df=pd.read_csv("../input/diabetes.csv")


# In[ ]:


df.head()


# In[ ]:


x=df.columns.values
print("The variable/parameters are {}".format( x[:-1]))
print("The outcome or target is {}".format(x[-1]))


# In[ ]:


df.info()


# In[ ]:


df.describe()


# we know the dataset target variable is binary object but how the dataset is distributed if it is even or biased towards some class let's find out 

# In[ ]:


plt.figure(figsize=(16,9))
sns.countplot(df["Outcome"])


# Not even again not completely biased.
# <div></div>
# from .info() we can see there is no null hope for us also.

# In[ ]:


plt.figure(figsize=(16,16))
sns.heatmap(df.corr(),annot=True,vmin=0.2)


# <h3>Findings</h3>
# * outcome has a medium correlation with Glucose (true in sense of healthcare)
# * Glucose has also medium correlation with insulin and BMI (again true)
# * Age has a medium relation with pregneancy which is quite obvious as the older you are the more chance you have to commit mistake wow!

# In[ ]:


plt.figure(figsize=(16,9))
sns.pairplot(df)


# I genuinely ignore this plot in each of my kernel as it feels like devops so many things I won't understand

# In[ ]:


plt.figure(figsize=(16,9))
sns.distplot(df["Age"])


# In[ ]:


plt.figure(figsize=(16,9))
sns.distplot(df["BloodPressure"],color="darkblue",bins=20)


# In[ ]:


sns.scatterplot(x="Age",y="BloodPressure",data=df)


# <h2>Apply Knowledge Of outlier Prediction</h2>
# <div></div>There are one data point in Age range 20-30 this lady in particular must needs to be checked either her report is mis printed or she is seriously in trouble according to the data.

# In[ ]:


import cufflinks as cf 
cf.go_offline()


# In[ ]:


df["Glucose"].iplot(kind="hist",color="green")


# can someone's glucose level be zero I have no idea about that. 

# In[ ]:


df["Outcome"][df["Glucose"]<10]


# There are diabetes patients with glucose level less than 10 any doctors in the panel.

# In[ ]:


df.iloc[[349,502]]


# * with zero knowledge about doctoring I am going to discard this two row from dataset.
# 
# <div></div>
# <h2>Distribution Plots:</h2>

# In[ ]:


fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df.Age, bins = 20, ax=ax[0,0]) 
sns.distplot(df.Pregnancies, bins = 20, ax=ax[0,1]) 
sns.distplot(df.Glucose, bins = 20, ax=ax[1,0]) 
sns.distplot(df.BloodPressure, bins = 20, ax=ax[1,1]) 
sns.distplot(df.SkinThickness, bins = 20, ax=ax[2,0])
sns.distplot(df.Insulin, bins = 20, ax=ax[2,1])
sns.distplot(df.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 
sns.distplot(df.BMI, bins = 20, ax=ax[3,1]) 


# In[ ]:


plt.scatter(df["SkinThickness"],df["Insulin"])


# In[ ]:




fig,ax = plt.subplots(nrows=2, ncols=4, figsize=(18,18))
plt.suptitle('Box Plots',fontsize=24)
sns.boxplot(y="Pregnancies", data=df,ax=ax[0,0],palette='Set3')
sns.boxplot(y="Glucose", data=df,ax=ax[0,1],palette='Set3')
sns.boxplot (y ='BloodPressure', data=df, ax=ax[0,2], palette='Set3')
sns.boxplot(y='SkinThickness', data=df, ax=ax[0,3],palette='Set3')
sns.boxplot(y='Insulin', data=df, ax=ax[1,0], palette='Set3')
sns.boxplot(y='BMI', data=df, ax=ax[1,1],palette='Set3')
sns.boxplot(y='DiabetesPedigreeFunction', data=df, ax=ax[1,2],palette='Set3')
sns.boxplot(y='Age', data=df, ax=ax[1,3],palette='Set3')
plt.show()


# In[ ]:


new_df=df.copy()


# In[ ]:


new_df.head()


# In[ ]:


final_df=new_df.drop([349,502],axis=0)


# I Think we don't need pregnancy and skinthickness so drop it 

# In[ ]:


Feature=['Glucose', 'BloodPressure',
       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


# In[ ]:


X=df[Feature].values
y= df['Outcome'].values


# <h1>Model</h1>

# In[ ]:


#importing train_test_split
from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)


# In[ ]:


#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
     #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)


# In[ ]:


#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=7)


# In[ ]:


#Fit the model
knn.fit(X_train,y_train)


# In[ ]:


#Get accuracy. Note: In case of classification algorithms score method represents accuracy.
knn.score(X_test,y_test)


# In[ ]:


y_pred = knn.predict(X_test)


# In[ ]:


#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:


y_pred_proba = knn.predict_proba(X_test)[:,1]


# In[ ]:


from sklearn.metrics import roc_curve


# In[ ]:


fr, tr, thresholds = roc_curve(y_test, y_pred_proba)


# In[ ]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fr,tr, label='Knn')
plt.xlabel('fr')
plt.ylabel('tr')
plt.title('Knn(n_neighbors=7) ROC curve')
plt.show()


# In[ ]:


#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)


# <h2>Cross-Validation</h2>

# In[ ]:


#import GridSearchCV
from sklearn.model_selection import GridSearchCV


# In[ ]:


#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}


# In[ ]:


knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)


# In[ ]:


knn_cv.best_score_,knn_cv.best_params_


# In[ ]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=16)
#Fit the model
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:,1]


# In[ ]:


fr, tr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fr,tr, label='Knn')
plt.xlabel('fr')
plt.ylabel('tr')
plt.title('Knn(n_neighbors=7) ROC curve')
plt.show()


# In[ ]:


roc_auc_score(y_test,y_pred_proba)


# <h2>Thank You </h2>
