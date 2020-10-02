#!/usr/bin/env python
# coding: utf-8

# # will it rain Tomorrow?

# ## Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# In[ ]:


data=pd.read_csv("../input/weatherAUS.csv",parse_dates=["Date"])


# In[ ]:





# In[ ]:





# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In[ ]:


data.isnull().sum()


# In[ ]:


data=data.drop(columns=['Date','Location','Sunshine','Cloud9am','Cloud3pm','RISK_MM','Evaporation'])


# In[ ]:


data.head()


# In[ ]:


data.corr()


# In[ ]:


data=data.dropna(how='any')
x=data.drop(columns=['RainTomorrow'])


# In[ ]:


x=pd.get_dummies(x)


# In[ ]:


y=data.RainTomorrow


# In[ ]:


print(y.shape,x.shape)


# In[ ]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = pd.DataFrame(min_max_scaler.fit_transform(x), columns = x.columns)
X_scaled.head()


# ## Feature selection using selectkBest

# In[ ]:


from sklearn.feature_selection import chi2,SelectKBest


# In[ ]:


xnew=SelectKBest(chi2,k=10) #choosing top 10 features.


# In[ ]:


xnew.fit(X_scaled,y)
x.columns[xnew.get_support()]


# In[ ]:


pd.Series(xnew.scores_,X_scaled.columns).sort_values(ascending=False)


# In[ ]:


X_final=x[['Rainfall', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm',
       'WindGustDir_E', 'WindDir9am_E', 'WindDir9am_N', 'WindDir9am_NNW',
       'RainToday_No', 'RainToday_Yes']]


# In[ ]:


X_final.head(5)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(X_final,y,test_size=0.30,random_state=10)


# # DecisionTreeClassifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


# In[ ]:


parameters={'max_depth':[1,2,3,4,5],'min_samples_split':[2,3,4,5],'min_samples_leaf':[1,2,3,4,5],'criterion':['gini','entropy']}
dt=DecisionTreeClassifier()


# In[ ]:


clf=GridSearchCV(dt,parameters,scoring='accuracy')


# In[ ]:


clf.fit(X_train,Y_train)


# In[ ]:


clf.best_score_


# In[ ]:


clf.best_params_


# In[ ]:


tree=DecisionTreeClassifier(criterion='entropy',
 max_depth= 5,min_samples_leaf=1,min_samples_split= 2)


# In[ ]:


tree.fit(X_train,Y_train)


# In[ ]:


Y_pred=tree.predict(X_test)


# In[ ]:


accuracy_score(Y_train,tree.predict(X_train))


# In[ ]:


accuracy_score(Y_test,Y_pred)


# In[ ]:


dict={'Y_test':Y_test,"Y_pred":Y_pred}


# In[ ]:


result=pd.DataFrame(dict)

result.head()
# In[ ]:


result.replace(['Yes','No'],[1,0])


# In[ ]:


#Creating Confusion MAtrix

conmat=confusion_matrix(result.Y_test,result.Y_pred)
conmat


# In[ ]:


plt.figure(figsize=(9,6))
plt.subplot(1,2,1)
sns.countplot(result.Y_test)
plt.title("Actual Value counts")
plt.subplot(1,2,2)
plt.title("Predicted Value counts")
sns.countplot(result.Y_pred)
plt.show()


# # KNN algorithm

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


KNN=KNeighborsClassifier()


# In[ ]:


para={'n_neighbors':range(1,15,1)}

knn=GridSearchCV(KNN,para,scoring="accuracy")


# In[ ]:


knn.fit(X_train,Y_train)


# In[ ]:


knn.best_estimator_


# In[ ]:


knn.best_score_


# In[ ]:


KNN=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=14, p=2,
           weights='uniform')


# In[ ]:


KNN.fit(X_train,Y_train)


# In[ ]:


ypredknn=KNN.predict(X_test)


# In[ ]:


print("accuracy score in Train:", accuracy_score(KNN.predict(X_train),Y_train))


# In[ ]:


print("accuracy score in Test:", accuracy_score(ypredknn,Y_test))


# In[ ]:


confusion_matrix(ypredknn,Y_test)

