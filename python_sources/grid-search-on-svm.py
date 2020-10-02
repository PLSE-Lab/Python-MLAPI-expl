#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/fish-market/Fish.csv')


# In[ ]:


df.head()


# In[ ]:


df.groupby('Species').mean()


# In[ ]:


df.info()


# # EDA

# In[ ]:


sns.countplot(x='Species',data=df)


# In[ ]:


# Then you map to the grid
g = sns.PairGrid(df)
g.map(plt.scatter)


# In[ ]:


sns.pairplot(df,hue='Species',palette='rainbow')


# In[ ]:


sns.boxplot(x='Species',y='Weight',data=df,palette='rainbow')


# In[ ]:


sns.boxplot(x='Species',y='Height',data=df,palette='rainbow')


# In[ ]:





# # data refining

# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


#no missing value


# # Standardizing the features and PCA
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


df.columns


# In[ ]:


scaler = StandardScaler()
scaler.fit(df[['Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']])


# In[ ]:


scaled_data = scaler.transform(df[['Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']])


# In[ ]:


scaled_df=pd.DataFrame(scaled_data,columns=['Weight', 'Length1', 'Length2', 'Length3', 'Height','Width'])


# In[ ]:


scaled_df['Species']=df['Species']


# In[ ]:


scaled_df.head()


# # splitting the data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X=scaled_df[['Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']]
y=scaled_df['Species']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # fitting decision tree

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print(confusion_matrix(y_test,rfc_pred))


# In[ ]:


print(classification_report(y_test,rfc_pred))


# # fitting SVM 

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model = SVC()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,predictions))


# In[ ]:


print(classification_report(y_test,predictions))


# # Grid Search for SVM

# In[ ]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,grid_predictions))


# In[ ]:


print(classification_report(y_test,grid_predictions))


# After grid search on SVM the accuracy of predictions of our model is increased.

# In[ ]:




