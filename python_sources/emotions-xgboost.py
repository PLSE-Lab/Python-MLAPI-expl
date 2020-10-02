#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/emotions.csv')


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


import seaborn as sns
sns.countplot(df['label'],color='lightblue')


# In[ ]:


dt = df['label']


# In[ ]:


df = df.drop('label',axis=1)


# In[ ]:


from xgboost.sklearn import XGBClassifier


# In[ ]:


from sklearn import model_selection


# In[ ]:


X_train,X_test,y_train,y_test = model_selection.train_test_split(df,dt,test_size=0.3,random_state=42)


# In[ ]:


params = {
    'objective': 'multi:softprob',
    'max_depth': 5,
    'learning_rate': 1.0,
    'n_estimators': 15
}


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = XGBClassifier(**params).fit(X_train, y_train)')


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


from sklearn import metrics


# In[ ]:


count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# In[ ]:


from sklearn import ensemble


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = ensemble.RandomForestClassifier(n_estimators=15,max_depth=4)')


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


X = scaler.fit_transform(df)


# In[ ]:


X_train,X_test,y_train,y_test = model_selection.train_test_split(X,dt,test_size=0.3,random_state=42)


# In[ ]:


from sklearn import linear_model


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = linear_model.LogisticRegression()')


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# In[ ]:




