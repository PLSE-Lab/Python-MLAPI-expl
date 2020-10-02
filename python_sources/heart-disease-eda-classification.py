#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


df= pd.read_csv('../input/heartdisease/heart.csv')
df.head(10)


# In[ ]:


df.target.value_counts()


# # **EDA**

# In[ ]:


import seaborn as sns
sns.countplot(x='target', data=df, palette='bwr')
plt.show


# In[ ]:


countNoDisease= len(df[df.target==0])
countHaveDisease= len(df[df.target==1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))


# In[ ]:


df.groupby('target').mean()


# In[ ]:


pd.crosstab(df.age,df.target).plot(kind='bar',figsize=(20,6))
plt.title('heart disease frequency for ages')
plt.xlabel('age')
plt.ylabel('frequency')
plt.show()


# In[ ]:


pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['blue','green'])
plt.title('heart disease frequency for sex')
plt.xlabel('sex(0=Female,1=Male)')
plt.xticks(rotation=0)
plt.legend(["havn't disease",'have disease'])
plt.ylabel('frequency')
plt.show()


# In[ ]:


plt.scatter(x=df.age[df.target==1],y=df.thalach[(df.target==1)],c='red')
plt.scatter(x=df.age[df.target==0],y=df.thalach[(df.target==0)],c='blue')
plt.legend(['disease','not disease'])
plt.xlabel('age')
plt.ylabel('max heart rate')
plt.show()


# In[ ]:


pd.crosstab(df.cp , df.target).plot(kind='bar',figsize=(15,6),color=['blue','green'])
plt.title('heart disease freq according tochest pain type')
plt.xlabel('chest pain type')
plt.xticks(rotation=0)
plt.ylabel('freq of disease or not')
plt.show()


# In[ ]:


a= pd.get_dummies(df['cp'],prefix='cp')
b= pd.get_dummies(df['thal'], prefix= "thal")
c= pd.get_dummies(df['slope'], prefix= "slope")


# In[ ]:


frames=[df,a,b,c]
df= pd.concat(frames, axis=1)
df.head()


# In[ ]:


df= df.drop(columns=['cp','thal','slope'])
df.head()


# In[ ]:


y= df.target.values
x= df.drop(['target'],axis=1)


# In[ ]:


# normalizing the data

#x= (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


# In[ ]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:


# transpose matrices
#x_train= x_train.T
#y_train= y_train.T
#x_test= x_test.T
#y_test= y_test.T


# # **MODELLING**

# In[ ]:


lr= LogisticRegression()
lr.fit(x_train,y_train)


# In[ ]:


y_pred= lr.predict(x_test)
y_pred


# In[ ]:


y_test


# #   **METRICS**

# In[ ]:


# confusion matrix
from sklearn.metrics import confusion_matrix
c_m= confusion_matrix(y_test,y_pred)
c_m


# In[ ]:


accuracy={}
acc= lr.score(x_test,y_test)*100
accuracy['logistic regression']= acc
print("Test Accuracy {:.2f}%".format(acc))


# In[ ]:


# from k nearest neighbours


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn= KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)


# In[ ]:


prediction= knn.predict(x_test)


# In[ ]:


prediction


# In[ ]:


y_test


# In[ ]:


c_mm= confusion_matrix(y_test,prediction)
c_mm


# In[ ]:


print("{} NN Score: {:.2f}%".format(2, knn.score(x_test, y_test)*100))


# In[ ]:


#try to find best k values

scorelist=[]
for i in range(1,20) :
    knn2= KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    scorelist.append(knn2.score(x_test,y_test))
plt.plot(range(1,20),scorelist)
plt.xticks(np.arange(1,20,1))
plt.xlabel(" K values")
plt.ylabel("score")
plt.show()

acc== max(scorelist)*100
accuracy['KNN']=acc
print("Maximum KNN Score is {:.2f}%".format(acc))


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svm = SVC(kernel='linear',random_state = 0)
svm.fit(x_train, y_train)

acc = svm.score(x_test,y_test)*100
accuracy['SVM'] = acc
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

acc = dtc.score(x_test, y_test)*100
accuracy['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc= RandomForestClassifier(n_estimators=100,criterion='entropy')
rfc.fit(x_train,y_train)
acc= rfc.score(x_test,y_test)*100
accuracy['random forest']=acc

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))


# In[ ]:


colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracy.keys()), y=list(accuracy.values()), palette=colors)
plt.show()


# CONCLUSION :
# From the result of each classifier we can see that , Random Forest Classifier gave us the best accuracy which is 90.16%.

# In[ ]:




