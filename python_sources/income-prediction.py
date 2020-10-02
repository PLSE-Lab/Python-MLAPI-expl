#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df= pd.read_csv("../input/adult.csv",na_values='#NAME?')


# In[ ]:


df.head()


# In[ ]:


print(df.head())


# In[ ]:


df.info()


# In[ ]:


df.income.unique()


# In[ ]:


df['income'].value_counts()


# In[ ]:





# In[ ]:


df['income'] = [0 if x == '<=50K' else 1 for x in df['income'] ]


# In[ ]:


print(df['income'].value_counts().sort_values(ascending=False).head())


# In[ ]:


print(df['native-country'].value_counts().sort_values(ascending=False).head(10))


# In[ ]:



df['native-country']=['United-States' if x == 'United-States' else 'others' for x in df['native-country']]
print(df['native-country'].value_counts().sort_values(ascending=False).head(10))


# In[ ]:


df.info()


# In[ ]:


#Assign X as a datafrsme of features and y as a series of the outcome variable
X= df.iloc[:,:-1]
y= df.iloc[:,14]


# In[ ]:


print(X)
print(y)


# In[ ]:


#df.info()
df.info()


# In[ ]:





# In[ ]:


cat_data= X[['workclass','education','marital-status','occupation','relationship','race','gender','native-country']]


# In[ ]:


cat_data.head()


# In[ ]:


X= X.drop(cat_data,1)


# In[ ]:


X.head()


# In[ ]:


D_data=pd.get_dummies(cat_data,drop_first=True)


# In[ ]:


D_data.head()


# In[ ]:


#Join Dummies data and Original Data
X= pd.concat([X,D_data], axis=1)


# In[ ]:


X.head()


# # Data Cleaning

# In[ ]:


X.columns


# In[ ]:





# In[ ]:


X.describe().transpose()


# In[ ]:


X.isnull().sum().sort_values(ascending=False)


# # Data Exploration
# Outliers Detection
# 
# 
# 
# 

# In[ ]:


X.corr()


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25, random_state=0);


# In[ ]:


print(X.head())
print(y.head())


# In[ ]:


X_train.info()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc_X=StandardScaler()


# In[ ]:


X_train= sc_X.fit_transform(X_train)


# In[ ]:


X_test=sc_X.transform(X_test)


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:





# In[ ]:





# In[ ]:


df.columns


# In[ ]:





# In[ ]:


df.groupby('age').mean()


# In[ ]:


df.groupby('workclass').mean()


# In[ ]:


df.groupby('education').mean()


# In[ ]:


df.groupby('marital-status').mean()


# In[ ]:


df.groupby('native-country').mean()


# In[ ]:





# In[ ]:


import scipy.stats
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


# # Logistic Reression

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[ ]:





# # Predicting the test set result

# In[ ]:


y_pred= classifier.predict(X_test)


# In[ ]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


# # Cross Validation

# In[ ]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# # Making the Condusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
confn_mtrx= confusion_matrix(y_test,y_pred)


# In[ ]:


print(confn_mtrx)


# In[ ]:


779   +438


# The result is telling us that we have 5721+1203= 6924 correct predictions and 6924+779 = 1217 incorrect predictions.

# # Compute precision, recall, F-measure and support

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:





# # ROC curve

# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:




