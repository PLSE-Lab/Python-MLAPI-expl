#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import the important libraries.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')
import pandas_profiling as pp


# ### Load the dataset.

# In[ ]:



data=pd.read_csv('../input/bank-additional-full.csv', sep = ';')
data.sample(5)


# In[ ]:


print("Shape of the data:",data.shape)
print("Columns Names are:\n",data.columns)


# In[ ]:


print("Data Types for all the columns of the data: \n",data.dtypes)


# In[ ]:


pp.ProfileReport(data)


# In[ ]:


data[data.duplicated(keep='first')]


# In[ ]:


data.drop_duplicates(keep='first',inplace=True)


# In[ ]:


print("Is there any null values in the data ? \n",data.isnull().values.any())


# In[ ]:


print("Total Null Values in the data = ",data.isnull().sum().sum())


# In[ ]:


print("Information about the dataframe : \n ")
data.info()


# #### Total numbers of missing values values in each column. ####

# In[ ]:


# Which columns have the most missing values?
def missing_data(df):
    total = df.isnull().sum()
    percent = total/df.isnull().count()*100
    missing_values = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    missing_values['Types'] = types
    missing_values.sort_values('Total',ascending=False,inplace=True)
    return(np.transpose(missing_values))
missing_data(data)


# In[ ]:


print('Discrption of Numeric Data : ')
data.describe()


# In[ ]:


print('Discrption of Object Data : ')
data.describe(include='object')


# In[ ]:


print("Histogram for the numerical features :\n")
data.hist(figsize=(15,15),edgecolor='k',color='skyblue')
plt.tight_layout()
plt.show()


# In[ ]:


print("Target values counts:\n",data['y'].value_counts())
data['y'].value_counts().plot.bar()
plt.show()


# In[ ]:


data.plot(kind='box',subplots=True,layout=(6,2),figsize=(15,15))
plt.tight_layout()


# In[ ]:


data.groupby(["contact"]).mean()


# In[ ]:


data.pivot_table(values="age",index="month",columns=["marital","contact"])


# In[ ]:


data.groupby("education").mean()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
cat_var=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week','poutcome','y']
for i in cat_var:
    data[i]=LE.fit_transform(data[i])
    
data.head()


# In[ ]:


X=data.iloc[:,0:7]
y=data.iloc[:,-1:]


# In[ ]:


#Now with single statement, you will be able to see all the variables created globally across the notebook, data type and data/information
get_ipython().run_line_magic('whos', '')


# #### Import machine learnig libraries

# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# In[ ]:


from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import math
import sklearn.model_selection as ms
import sklearn.metrics as sklm


# In[ ]:


sc=StandardScaler()
sc.fit_transform(X)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=2)


# In[ ]:


lr=LogisticRegression(penalty = 'l1',solver = 'liblinear')
lr.fit(X_train,y_train)
pred_lr=lr.predict(X_test)
confusion_matrix(y_test,pred_lr)
score_lr= accuracy_score(y_test,pred_lr)
print("Accuracy Score is: ", score_lr)


# In[ ]:


knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
confusion_matrix(y_test,pred_knn)


# In[ ]:


score_knn = cross_val_score(knn,y_test,pred_knn,cv=5)
print(score_knn)
print("Mean of the cross validation scores:",score_knn.mean())


# In[ ]:


dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
pred_dt=dt.predict(X_test)
confusion_matrix(y_test,pred_dt)


# In[ ]:


score_dt=cross_val_score(dt,y_test,pred_dt,cv=5)
print(score_dt)
print("Mean of the cross validation scores:",score_dt.mean())


# In[ ]:


rf=RandomForestClassifier()
rf.fit(X_train,y_train)
pred_rf=rf.predict(X_test)
confusion_matrix(y_test,pred_rf)


# In[ ]:


score_rf=cross_val_score(rf,y_test,pred_dt,cv=5)
print(score_rf)
print("Mean of the cross validation scores:",score_rf.mean())


# In[ ]:


xgb_clf= xgb.XGBClassifier()
xgb_clf.fit(X_train,y_train)
pred_xgb=xgb_clf.predict(X_test)
confusion_matrix(y_test,pred_xgb)


# In[ ]:


score_xgb = cross_val_score(xgb_clf,y_test,pred_xgb,cv=5)
print(score_xgb)
print("Mean of the cross validation scores:",score_xgb.mean())


# In[ ]:


print('Feature importances:\n{}'.format(repr(xgb_clf.feature_importances_)))


# In[ ]:


print("Accuracy Score of Logistic Regression",score_lr)
print("Accuracy Score of KNN",score_knn.mean())
print("Accuracy Score of Decision Tree",score_dt.mean())
print("Accuracy Score of Random Forest",score_rf.mean())
print("Accuracy Score of XGB",score_xgb.mean())


# In[ ]:


plt.bar(x=["LR","KNN","DT","RF","XGB"],height=[score_lr,score_knn.mean(),score_dt.mean(),score_rf.mean(),score_xgb.mean()])
plt.ylim(0.88,1)
plt.show()


# ### Thankyou for visit the kernel. If you have any suggustion please comment.if you feel the kernel helpful,
# 
# ### please upvote.
