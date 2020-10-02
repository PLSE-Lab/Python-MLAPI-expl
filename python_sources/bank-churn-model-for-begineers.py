#!/usr/bin/env python
# coding: utf-8

# **Bank churn modeling**
# * Data Exploration
# * Data Cleaning
# * Feature Engineering
# * Data Preprocessing
# * Apply ML Algorithim
# * Perfomance Analysis
# * Optimization and Tunning
# 1. 1.KNN
# 1. 2.Decision Tree
# 1. 3.Random Forest

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling


# In[ ]:


#data data
df =pd.read_csv('../input/bank-churn/Bank_churn_modelling.csv')


# In[ ]:


df.shape


# # 1.Data Exploration

# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.Gender.unique()


# In[ ]:


df.Geography.unique()


# In[ ]:


df.profile_report()


# # 2. data cleaning

# In[ ]:


#check for duplicate entries
df.duplicated().sum()


# In[ ]:


# check for missing values
df.isnull().sum()


# # 3. Feature Engineering
#  * Feature Extraction
#  * Feature Selection

# In[ ]:


df.columns


# In[ ]:


#df.drop(["Surname","RowNumber","CustomerId"],axis=1,inplace=True)
df.columns


# In[ ]:


x=df[['CreditScore', 'Geography', 'Gender', 'Age','Balance','NumOfProducts', 'IsActiveMember']]
y=df["Exited"]


# # Data Preprocessing

# In[ ]:


x.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lel = LabelEncoder()
x["Gender"]= lel.fit_transform(x["Gender"])
x.head()


# In[ ]:


y.head()


# In[ ]:


#here in Gender column female changed to 0 and male changed to 1
x.head(8)


# In[ ]:


#onehotencoding for geography
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("encoder",OneHotEncoder(),[1])],remainder="passthrough")# here 1 represents column geography
#here we can use remainder="drop" ,here it only represents geography
x = ct.fit_transform(x)


# In[ ]:


x.shape


# In[ ]:


x=pd.DataFrame(x)


# In[ ]:


x.head(10)


# In[ ]:


#here 0-france 1-germany 2-spain (in alaphabetical model)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x)


# In[ ]:


#splitting data info train and test set
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts = train_test_split(x,y,test_size=0.2)
print(x.shape)
print(xtr.shape)
print(xts.shape)


# # 5. Apply ML algorithim

# In[ ]:


from sklearn.linear_model import LogisticRegression
model =LogisticRegression()


# In[ ]:


#train the model -using training data -xtr,ytr
model.fit(xtr,ytr)


# # 6. Perfomance Analysis

# In[ ]:


#france, cs=580, age=58, Male,numofprod=3,isactmember=0,balance=456782
new_customer=[[1,0,0,580,1,58,456782,3,0]]
model.predict(new_customer)


# In[ ]:


#here [1] is exited


# # accuracy=(no of correct prediction/total no of prediction)*100

# In[ ]:


#check perfomance of model on test data
# getting prediction for test data
ypred = model.predict(xts)
from sklearn import metrics
metrics.accuracy_score(yts,ypred)


# In[ ]:


#here the accuracy which we got is 0.79 which is not good 
#after adding standardization it become 80% 0r 0.80


# #  therefore the model id 0.79% accurate and the perfomance is not so good or not so bad

# # 100 customer
# 1. actual-
#  * 90-0-notleaving
#  * 10-1.leaving
# 
# 2. ML model -1
#   * 95-0-notleaving -90
#   * 05-1-leaving -05
# 
# 3. Accuracy = 95%
#   * recall=50% (accuracy of class1)
#   * 95% is only solving 50% of the whole scenario
#   * this means 10 are leaving but model says 5 are leaving
#  
# 4. ML model -2 
#   * 70-0-notleaving -70
#   * 30-1-leaving -10
#   * accuracy = 80%
#   * recall= 100%
#  
#  

# In[ ]:


#calcuate recall
metrics.recall_score(yts,ypred)


# In[ ]:


#here we used feature scaling to get features in same range


# # 7.Optimization and tuning

# #above part where we did standardization is part of optimisation and tunning

# # KNN  ALGO

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors = 3)#no of neighbors is hpyer parameter
model2.fit(xtr,ytr)


# In[ ]:


ypred2=model2.predict(xts)
metrics.accuracy_score(yts,ypred2)#here it checks for both class 1 and class 0( here 0 and 1  are told as class)


# In[ ]:


metrics.recall_score(yts,ypred2)#here it ckecks only for class 1


# # Decision Tree Algorithim

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model3= DecisionTreeClassifier(criterion="gini")
#here we are facing the problem of overfitting
#train the model
model3.fit(xtr,ytr)


# In[ ]:


ypred3=model3.predict(xts)
metrics.accuracy_score(yts,ypred3)


# In[ ]:


metrics.recall_score(yts,ypred3)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model3= DecisionTreeClassifier(criterion="entropy")
#train the model
model3.fit(xtr,ytr)


# In[ ]:


ypred3=model3.predict(xts)
metrics.accuracy_score(yts,ypred3)


# In[ ]:


metrics.recall_score(yts,ypred3)


# In[ ]:


metrics.recall_score(ytr,model3.predict(xtr))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model3= DecisionTreeClassifier(criterion="gini",max_depth=8,min_samples_leaf=10)
#here max_depth and min_samples_leaf is used to control overfitting
#train the model
model3.fit(xtr,ytr)


# In[ ]:


ypred3=model3.predict(xts)
metrics.accuracy_score(yts,ypred3)


# In[ ]:


metrics.recall_score(yts,ypred3)


# In[ ]:


metrics.recall_score(ytr,model3.predict(xtr))


# In[ ]:


import graphviz
from sklearn import tree
fname=['France','Germany','Spain','CreditScore','Gender','Age','Balance','NumofProducts','IsActiveMember']
cname=['Not Excited','Excited']
graph_data = tree.export_graphviz(model3,out_file=None,feature_names=fname,class_names=cname,filled=True,rounded=True)
graph=graphviz.Source(graph_data)


# # Random forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators=50,criterion='gini',max_depth=10,min_samples_leaf=20)
model4.fit(xtr,ytr)


# In[ ]:


ypred4=model4.predict(xts)
metrics.accuracy_score(yts,ypred4)


# In[ ]:


metrics.recall_score(yts,ypred4)


# In[ ]:


metrics.recall_score(ytr,model4.predict(xtr))

