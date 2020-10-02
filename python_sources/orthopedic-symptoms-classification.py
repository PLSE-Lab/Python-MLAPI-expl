#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/column_2C_weka.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df['class']=df['class'].apply(lambda x:1 if x=='Normal' else 0)


# In[ ]:


df.head()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(df['class'])


# In[ ]:


plt.figure(figsize=(14,6))
sns.heatmap(df.isnull(),cmap='viridis',yticklabels=False,cbar=False)


# In[ ]:


plt.figure(figsize=(14,6))
sns.heatmap(df.corr(),annot=True)


# In[ ]:


sns.lmplot('sacral_slope','pelvic_incidence',df,hue='class')


# In[ ]:


sns.pairplot(df,hue='class')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X=df.drop('class',axis=1)
y=df['class']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# <h1>Applying Logistic Regression</h1>

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lgr=LogisticRegression()


# In[ ]:


lgr.fit(X_train,y_train)


# In[ ]:


predictions=lgr.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# Hmmm ! 82% accuracy,not bad.
# Hey wait,can we try to improve our predictions.

# <h1>Let's try Random Forest Classification</h1>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier(n_estimators=100)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


rf_pred=rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,rf_pred))


# Wow ! seems its working..a slight creep though (0.01).
# What about keep going and try some more classification models

# <h1>Lets see what our KNN model has to say...</h1>

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()


# In[ ]:


scaler.fit(df.drop('class',axis=1))


# In[ ]:


scaled_features=scaler.transform(df.drop('class',axis=1))


# In[ ]:


df_scaled = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_scaled.head()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=10)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


knn_pred=knn.predict(X_test)


# In[ ]:


print(classification_report(y_test,knn_pred))


# In[ ]:


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[ ]:


knn_25=KNeighborsClassifier(n_neighbors=25)
knn_25.fit(X_train,y_train)
knn25_pred=knn_25.predict(X_test)
print(classification_report(y_test,knn25_pred))


# In[ ]:


knn_5=KNeighborsClassifier(n_neighbors=15)
knn_5.fit(X_train,y_train)
knn5_pred=knn_5.predict(X_test)
print(classification_report(y_test,knn5_pred))


# <h1>Now finally lets check SVM model...</h1>

# In[ ]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print(classification_report(y_test,grid_predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




