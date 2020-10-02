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


# # **Importing Libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,mean_squared_log_error,r2_score,roc_auc_score


# # **Reading Data**

# In[ ]:


data_mat = pd.read_csv('../input/student-mat.csv')
data_por = pd.read_csv('../input/student-por.csv')


# # **Preprocessing data**

# In[ ]:


data_mat['Dalc'] = data_mat['Dalc'] + data_mat['Walc']     #Alcohol consumption for all days
data_por['Dalc'] = data_por['Dalc'] + data_por['Walc']


# In[ ]:


data_mat=data_mat.drop(['Walc'],axis=1)             #We dont we Walc now
data_por=data_por.drop(['Walc'],axis=1)


# # **Define dependent variables**

# In[ ]:


y_train=data_por['G3'].values      #training on student-por 
y_test=data_mat['G3'].values       #testing on student-mat


# In[ ]:


x_train=data_por.drop(['G3'],axis=1)   #training on student-por 
x_test=data_mat.drop(['G3'],axis=1)    #testing on student-mat 


# In[ ]:


#Only 85 entries with different G1, G2 and paid are there 
'''df=pd.merge(data_mat,data_por, how='inner',on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet",'guardian',
 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'health', 'absences'])'''


# # **Dependent Variables**

# In[ ]:


x_train.columns


# # **Label Encoder**

# In[ ]:


le=LabelEncoder()
#x_train = x_train.apply(le.fit_transform)
col=['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
for i in col:
    #print(i)
    x_train[i]=le.fit_transform(x_train[i])
    x_test[i]=le.fit_transform(x_test[i])


# # **One Hot Encoder**

# In[ ]:


#Columns with more than 2 values after LabelEncoder need to be OneHotEncoded
col=['Mjob','Fjob','reason','guardian']      
onehot=OneHotEncoder(categorical_features=[8,9,10,11])
x_train=onehot.fit_transform(x_train).toarray()
x_test=onehot.fit_transform(x_test).toarray()


# # **Scaling **

# In[ ]:


'''#Scaling values
#col=['age','Medu','Fedu','traveltime','studytime','famrel','freetime','goout','Dalc','health','absences','G1','G2']
col=[2,6,7,12,13,23,24,25,26,27,28,29,30]
for i in col:
    x_train[i]=x_train[i]/x_train[i].max()
    x_test[i]=x_test[i]/x_test[i].max()'''


# In[ ]:


#Here it is not required
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
scy=MinMaxScaler()

y_train=scy.fit_transform(y_train.reshape(-1,1))
y_test=scy.transform(y_test.reshape(-1,1))


# In[ ]:


'''y_train=y_train/y_train.max()
y_test=y_test/y_test.max()  ''' 


# # **Model Creation**

# **SVM Regressor**

# In[ ]:


from sklearn import svm
svm_reg=svm.SVR(kernel='linear')
svm_reg.fit(x_train,y_train)
y_pred=svm_reg.predict(x_test)


mse2=mean_squared_error(y_test,y_pred)
var2=explained_variance_score(y_test,y_pred)
mae2=mean_absolute_error(y_test,y_pred)
r22=r2_score(y_test,y_pred)
print(mse2,var2,mae2,r22)
plt.scatter(np.arange(1,len(y_test)+1).tolist(), y_test, color = 'blue')
plt.scatter(np.arange(1,len(y_pred)+1).tolist(), y_pred, color = 'red')
plt.title("SVM")
plt.xlabel("Entry")
plt.ylabel("Values")
plt.show()


# **Linear Regressor**

# In[ ]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)


mse3=mean_squared_error(y_test,y_pred)
var3=explained_variance_score(y_test,y_pred)
mae3=mean_absolute_error(y_test,y_pred)
r23=r2_score(y_test,y_pred)
print(mse3,var3,mae3,r23)
plt.scatter(np.arange(1,len(y_test)+1).tolist(), y_test, color = 'blue')
plt.scatter(np.arange(1,len(y_pred)+1).tolist(), y_pred, color = 'red')
plt.title("Linear Regression")
plt.xlabel("Entry")
plt.ylabel("Values")
plt.show()


# **Decision Tree Regressor**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()
clf=clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)


mse4=mean_squared_error(y_test,y_pred)
var4=explained_variance_score(y_test,y_pred)
mae4=mean_absolute_error(y_test,y_pred)
r24=r2_score(y_test,y_pred)
print(mse4,var4,mae4,r24)
plt.scatter(np.arange(1,len(y_test)+1).tolist(), y_test, color = 'blue')
plt.scatter(np.arange(1,len(y_pred)+1).tolist(), y_pred, color = 'red')
plt.title("Decision tree")
plt.xlabel("Entry")
plt.ylabel("Values")
plt.show()


# **Random Forest Regressor**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=500)
rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)


mse5=mean_squared_error(y_test,y_pred)
var5=explained_variance_score(y_test,y_pred)
mae5=mean_absolute_error(y_test,y_pred)
r25=r2_score(y_test,y_pred)
print(mse5,var5,mae5,r25)
plt.scatter(np.arange(1,len(y_test)+1).tolist(), y_test, color = 'blue')
plt.scatter(np.arange(1,len(y_pred)+1).tolist(), y_pred, color = 'red')
plt.title("Random Forest")
plt.xlabel("Entry")
plt.ylabel("Values")
plt.show()


# **AdaBoost Regressor**

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
ad=AdaBoostRegressor()
ad.fit(x_train,y_train)

y_pred=ad.predict(x_test)

mse6=mean_squared_error(y_test,y_pred)
var6=explained_variance_score(y_test,y_pred)
mae6=mean_absolute_error(y_test,y_pred)
r26=r2_score(y_test,y_pred)
print(mse6,var6,mae6,r26)
plt.scatter(np.arange(1,len(y_test)+1).tolist(), y_test, color = 'blue')
plt.scatter(np.arange(1,len(y_pred)+1).tolist(), y_pred, color = 'red')
plt.title("Adaboost")
plt.xlabel("Entry")
plt.ylabel("Values")
plt.show()


# # **Visualizations and Evaluations**

# **Mean Squared Error**

# In[ ]:


x=['SVM','Linear Reg','DecsTree','Randforest','Adaboost']
y=[mse2,mse3,mse4,mse5,mse6]
fig, ax = plt.subplots()
plt.bar(x,y)
ax.set_ylabel('Mean sq error')
ax.set_xlabel('Model type')


# **Variance**

# In[ ]:


y=[var2,var3,var4,var5,var6]
fig, ax = plt.subplots()
plt.bar(x,y)
ax.set_ylabel('Variance')
ax.set_xlabel('Model type')


# **Mean Absolute Error**

# In[ ]:


y=[mae2,mae3,mae4,mae5,mae6]
fig, ax = plt.subplots()
plt.bar(x,y)
ax.set_ylabel('Mean abs error')
ax.set_xlabel('Model type')


# **R2 score**

# In[ ]:


y=[r22,r23,r24,r25,r26]
fig, ax = plt.subplots()
plt.bar(x,y)
ax.set_ylabel('r2 score')
ax.set_xlabel('Model type')


# # **Inferences**

# * Linear Regression and Random Forest Regressor are best models for this data with mean squared error of 0.0123 and 0.011 respectively .
# * Decision Tree performs worst with mean squared error of 0.019.
# 

# # **Reasoning**

# As we see the scatter plot between actual values and predicted values Linear regressor and Random Forest Regressor are most accurate among all other models and Adaboost and Decision Tree are least accurate.
