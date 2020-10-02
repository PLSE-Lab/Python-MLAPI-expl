#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


#DEMOGRAPHIC DATA DROPPED 
list1=["customerID",'gender','Partner',"Dependents",'SeniorCitizen','DeviceProtection','PaperlessBilling','PaymentMethod']
df.drop(list1,axis=1,inplace=True)
df.head()


# In[ ]:


#DEMOGRAPHIC PRESENTATION


# In[ ]:


sns.countplot(x="Churn",data=df)


# In[ ]:


#sns.countplot(x="Churn",hue="gender",data=df)


# In[ ]:


#sns.countplot(x="Churn",hue="SeniorCitizen",data=df)


# In[ ]:


#sns.countplot(x="Churn",hue="Partner",data=df)


# In[ ]:


#sns.countplot(x="Churn",hue="Dependents",data=df)


# In[ ]:


#sns.countplot(x="Churn",hue="DeviceProtection",data=df)


# In[ ]:


#sns.countplot(x="Churn",hue="PaperlessBilling",data=df)


# In[ ]:


#sns.countplot(x="Churn",hue="PaymentMethod",data=df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#Skewness of Data


# In[ ]:





# In[ ]:


sns.distplot(df["tenure"])


# In[ ]:


df["tenure"].skew()


# In[ ]:





# In[ ]:


df["MonthlyCharges"].skew()


# In[ ]:


df['TotalCharges'].replace(" ",np.NaN,inplace=True)
df['TotalCharges']=df['TotalCharges'].astype(float)
df['TotalCharges'].replace(np.NaN,np.mean(df["TotalCharges"]),inplace=True)


# In[ ]:


sns.distplot(df["TotalCharges"])


# In[ ]:


df["TotalCharges"].skew()


# In[ ]:


sns.distplot(np.log(df["TotalCharges"]))


# In[ ]:


np.log(df["TotalCharges"]).skew()


# In[ ]:


sns.distplot(np.sqrt(df["TotalCharges"]))


# In[ ]:


np.sqrt(df["TotalCharges"]).skew()


# In[ ]:


sns.distplot(np.cbrt(df["TotalCharges"]))


# In[ ]:


np.cbrt(df["TotalCharges"]).skew()


# In[ ]:


#AS WE CAN SEE,CUBE ROOT RETURNS THE MOST LESS SKEWED DATA HERE AND IS CLSER TO THE BELL CURVE THAN ANY OTHER METHOD LIKE SQRT AND LOG


# In[ ]:


df["TotalCharges"]=np.cbrt(df["TotalCharges"])
df["TotalCharges"].head()


# In[ ]:


df["TotalCharges"].skew()


# In[ ]:





# In[ ]:


#Null Value Treatment


# In[ ]:


df.isnull().sum()


# In[ ]:


#outlier treatment


# In[ ]:


sns.boxplot(x="tenure",data=df)
plt.show()


# In[ ]:


sns.boxplot(x="MonthlyCharges",data=df)
plt.show()


# In[ ]:


sns.boxplot(x="TotalCharges",data=df)
plt.show()


# In[ ]:


# No outlier found 


# In[ ]:


#Categorical Data TREATMENT


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()


# In[ ]:


df["PhoneService"]=labelencoder.fit_transform(df["PhoneService"])
df.head()


# In[ ]:


df["MultipleLines"]=labelencoder.fit_transform(df["MultipleLines"])


# In[ ]:


df["InternetService"]=labelencoder.fit_transform(df["InternetService"])


# In[ ]:


df["OnlineSecurity"]=labelencoder.fit_transform(df["OnlineSecurity"])


# In[ ]:


df["OnlineBackup"]=labelencoder.fit_transform(df["OnlineBackup"])


# In[ ]:


df["TechSupport"]=labelencoder.fit_transform(df["TechSupport"])


# In[ ]:


df["StreamingMovies"]=labelencoder.fit_transform(df["StreamingMovies"])


# In[ ]:


df["StreamingTV"]=labelencoder.fit_transform(df["StreamingTV"])


# In[ ]:


df["Contract"]=labelencoder.fit_transform(df["Contract"])


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


Churn_d=pd.get_dummies(df['Churn'],drop_first=True)
Churn_d.head(5)


# In[ ]:


df.head()


# In[ ]:


#FEATURE SCALING


# In[ ]:


from sklearn.model_selection import train_test_split
x=df.drop("Churn",axis=1)
y=Churn_d
#from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(x_train)
X_test=sc_X.fit_transform(x_test)
print(X_train)


# In[ ]:


#TRAINING OF DATA


# In[ ]:


#logistics regression


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


#EVALUATION OF THE MODEL


# In[ ]:


predictions=logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
res=confusion_matrix(y_test,predictions)
res


# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions)*100)


# In[ ]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)


# In[ ]:





# In[ ]:


#KNN


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[ ]:


#classofier
classifier=KNeighborsClassifier(n_neighbors=19,p=2,metric="euclidean")
classifier.fit(X_train,y_train)


# In[ ]:


y_pred=classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
res=confusion_matrix(y_test,y_pred)
res


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


from sklearn.metrics import classification_report
classification_report(y_test,y_pred)


# In[ ]:


#decision Tree


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3)
#create the model
classifier_entropy.fit(X_train,y_train)


# In[ ]:


#pediction
y_pred=classifier_entropy.predict(X_test)
print(y_pred)


# In[ ]:



print("accuracy is :",accuracy_score(y_test,y_pred)*100) 


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


from sklearn.metrics import classification_report
classification_report(y_test,y_pred)


# In[ ]:





# In[ ]:


#import pandas as pd
#WA_Fn_UseC__Telco-Customer-Churn = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

