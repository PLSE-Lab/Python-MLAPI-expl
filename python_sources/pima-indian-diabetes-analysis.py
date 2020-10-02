#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[ ]:


data=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")


# In[ ]:


data.head()


# In[ ]:


data.isna().sum()


# In[ ]:


data.describe().T


# In[ ]:


corr=data.corr()


# In[ ]:


plt.figure(figsize=[10,7])
sns.heatmap(corr,annot=True)


# In[ ]:


k=sns.countplot(data["Outcome"])
for b in k.patches:
    k.annotate(format(b.get_height(),'.0f'),(b.get_x()+b.get_width() / 2.,b.get_height()))


# In[ ]:


sns.pairplot(data)


# In[ ]:


features = ['Pregnancies','Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


# In[ ]:


#replacing the 0s in the data
for feature in features:
    number = np.random.normal(data[feature].mean(), data[feature].std()/2)
    data[feature].fillna(value=number, inplace=True)


# In[ ]:


#Scaling the data
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(data.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])


# In[ ]:


X.head()


# In[ ]:


y = data.Outcome


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)


# In[ ]:


#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train,y_train)
acc = lr.score(X_test,y_test)*100
print("Logistic Regression Acc Score: ", acc)


# In[ ]:


#Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
model_nb=nb.predict(X_test)
print("Naive Bayes Acc Score",(accuracy_score(y_test,model_nb))*100)


# In[ ]:


#K-Nearest Neigbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
model_knn = knn.predict(X_test)
print("KNN Acc Score",(accuracy_score(y_test,model_knn))*100)

