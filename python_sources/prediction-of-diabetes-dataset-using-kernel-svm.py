#!/usr/bin/env python
# coding: utf-8

# # Prediction of Diabetes using Kernel SVM and GridSearchCV
# 
# #### In this Analysis, we will using the PIMA Indian Diabetes dataset and model the prediction using Support Vector Machines (SVM) and optimize the results using GridSearchCV. 
# 
# ##### Libraries Used for this Analysis
# * Pandas
# * Numpy
# * Matplotlib
# * SciKit-Learn

# ### Import Libraries and Datasets

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/diabetes.csv",sep=',')


# ## Exploratory Data Analysis

# In[ ]:


data.head(10) 


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data['Outcome'].value_counts()


# In[ ]:


sns.countplot(x='Outcome',data=data, palette='Dark2')
plt.show()


# ### Replace zero values in BMI and Insulin columns with median values

# In[ ]:


bmi_median = data['BMI'].median()


# In[ ]:


data['BMI'].replace(0,bmi_median)


# In[ ]:


insulin_median = data['Insulin'].median()


# In[ ]:


data['Insulin'].replace(0,insulin_median)


# ### Correlation between features

# In[ ]:


plt.figure(figsize=(10,10))
plt.title('Pearson Correlation of Variables',y=1, size=15)
sns.heatmap(data.corr(),linewidths=0.1,vmax=0.1,square=True,linecolor='white',annot=True)


# In[ ]:


sns.boxplot(data.Outcome,data.BMI)


# In[ ]:


sns.boxplot(data.Outcome,data.Age)


# ## Train-test split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = data.drop('Outcome',axis=1)
y = data['Outcome']
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)


# ## Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc_X = StandardScaler(with_mean=False)


# In[ ]:


X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# ## Implement the Kernel SVM algorithm

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc_model = SVC(kernel='rbf',random_state=0)


# In[ ]:


svc_model.fit(X_train, y_train)


# In[ ]:


y_pred = svc_model.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(classification_report(y_test,y_pred))


# ## Calculate AUC score and plot ROC curve

# In[ ]:


from sklearn.metrics import roc_auc_score,roc_curve


# In[ ]:


auc = roc_auc_score(y_test,y_pred)
print("AUC %0.3f" %auc)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,y_pred)


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot([0,1],[0,1],linestyle="--")
plt.plot(fpr,tpr, label='SVM (AUC = %0.2f)'% auc)
plt.xlabel("1-Specificity",fontsize=12)
plt.ylabel("Sensitivity",fontsize=12)
plt.legend(loc='lower right')
plt.show()


# ## Model Cross-Validation using GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(X_train,y_train)


# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,grid_predictions))


# In[ ]:


print(classification_report(y_test,grid_predictions))


# In[ ]:


#Calculate AUC Score after GridSearchCV
auc_grid = roc_auc_score(y_test,grid_predictions)
print('AUC: %.3f' % auc_grid)


# In[ ]:


#Calculate ROC Curve after Grid Search CV
fpr , tpr , thresholds = roc_curve(y_test,grid_predictions)


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot([0,1],[0,1],linestyle="--")
plt.title('Receiver Operator Characteristic')
plt.plot(fpr,tpr, label='SVM (AUC = %0.2f)'% auc_grid)
plt.xlabel("1-Specificity",fontsize=12)
plt.ylabel("Sensitivity",fontsize=12)
plt.legend(loc='lower right')
plt.show()

