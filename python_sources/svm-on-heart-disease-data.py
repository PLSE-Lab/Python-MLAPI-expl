#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_palette('Set1')


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# # Reading Dataset

# In[ ]:


data = pd.read_csv('../input/heart.csv')


# In[ ]:


data.head()


# Just a sanity check for null values. 

# In[ ]:


data.isnull().sum()


# Based on the data types attached to the columns, it seems like the data got read into the environment accurately. 

# In[ ]:


data.info()


# # Univariate Data Analysis
# In this subsection, let us explore every attribute, within the data set, one by one. 
# <br>
# If the attribute into consideration possesses **continuous values**, then **distribution plots** provide a good summary of that attribute. If we have a **categorical quality**, then a **count plot** offers an excellent overview of that attribute. 
# <br>
# Later in bivariate data analysis, we will compare the attributes with the target attribute. 
# 

# In[ ]:


sns.distplot(data['age']);


# In[ ]:


# 1 = male; 0 = female
sns.countplot(data['sex']);


# In[ ]:


sns.countplot(data['cp']);


# In[ ]:


sns.distplot(data['trestbps'])


# In[ ]:


sns.distplot(data['chol'])


# In[ ]:


sns.countplot(data['fbs']);


# In[ ]:


sns.countplot(data['restecg']);


# In[ ]:


sns.distplot(data['thalach'])


# In[ ]:


sns.countplot(data['exang']);


# In[ ]:


sns.distplot(data['oldpeak']);


# In[ ]:


sns.countplot(data['slope']);


# In[ ]:


sns.countplot(data['ca']);


# In[ ]:


sns.countplot(data['thal']);


# # Bivariate Data Analysis
# In this subsection, let's go through the relationship shared by the target variable and the attributes. 

# In[ ]:


plt.figure(figsize=(14, 5))
sns.distplot(data[data['target'] == 1]['age'], label= "Disease - Yes")
sns.distplot(data[data['target'] == 0]['age'], label= "Disease - No")
plt.legend();


# In[ ]:


# 1 = male; 0 = female
sns.countplot(data['target'], hue = data['sex']);


# In[ ]:


sns.countplot(data['target'], hue = data['cp']);


# In[ ]:


plt.figure(figsize=(14, 5))
sns.distplot(data[data['target'] == 1]['trestbps'], label= "Disease - Yes")
sns.distplot(data[data['target'] == 0]['trestbps'], label= "Disease - No")
plt.legend();


# In[ ]:


plt.figure(figsize=(14, 5))
sns.distplot(data[data['target'] == 1]['chol'], label= "Disease - Yes")
sns.distplot(data[data['target'] == 0]['chol'], label= "Disease - No")
plt.legend();


# In[ ]:


sns.countplot(data['target'], hue = data['fbs']);


# In[ ]:


sns.countplot(data['target'] ,hue = data['restecg']);


# In[ ]:


plt.figure(figsize=(14, 5))
sns.distplot(data[data['target'] == 1]['thalach'], label= "Disease - Yes")
sns.distplot(data[data['target'] == 0]['thalach'], label= "Disease - No")
plt.legend();


# In[ ]:


sns.countplot(data['target'], hue = data['exang']);


# In[ ]:


plt.figure(figsize=(14, 5))
sns.distplot(data[data['target'] == 1]['oldpeak'], label= "Disease - Yes")
sns.distplot(data[data['target'] == 0]['oldpeak'], label= "Disease - No")
plt.legend();


# In[ ]:


sns.countplot(data['target'], hue = data['slope']);


# In[ ]:


sns.countplot(data['target'], hue = data['ca']);


# In[ ]:


sns.countplot(data['target'], hue = data['thal']);


# In[ ]:


sns.jointplot(x= 'oldpeak' , y= 'chol' ,data= data, kind= 'kde');


# # Trivariate Data Analysis
# The focus of this subsection would be to gather information between two categorical and one continuous attributes.

# In[ ]:


plt.figure(figsize=(8,6))
sns.violinplot(x = 'fbs',y= 'trestbps', data = data, hue = 'sex', split=True);


# In[ ]:


plt.figure(figsize=(8,6))
sns.boxplot(x = 'exang',y= 'trestbps', data = data, hue = 'sex');


# # Correlation Matrix

# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot= True, fmt='.2f')
plt.show();


# # Pairplot

# In[ ]:


sns.pairplot(data, hue = 'target');


# # Data Preprocessing
# In this subsection, we need to preprocess categorical attributes with the help of ```get_dummies``` from **Pandas**. <br> 
# If we skip this step, then the machine learning model will assume the numbers assigned to the attributes as a ranked variable, which it is not. 
# 

# In[ ]:


sex = pd.get_dummies(data['sex'])
cp = pd.get_dummies(data['cp'])
fbs = pd.get_dummies(data['fbs'])
restecg = pd.get_dummies(data['restecg'])
exang = pd.get_dummies(data['exang'])
slope = pd.get_dummies(data['slope'])
ca = pd.get_dummies(data['ca'])
thal = pd.get_dummies(data['thal'])


# In[ ]:


data = pd.concat([data, sex, cp, fbs, restecg, exang, slope, ca, thal], axis = 1)


# In[ ]:


data.head()


# In[ ]:


data.drop(['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], axis = 1, inplace= True)


# # Support Vector Machine (Without Scaler)

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model = SVC(probability=True)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = data.drop('target', axis = 1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc


# In[ ]:


print(classification_report(y_test, y_pred))


# # Support Vector Machine (With Scaler)
# Significant performance improvements with ```MinMaxScaler```. 
# The evaluation metric for this kernel will be **F1**, as it is the harmonic mean of precision and recall. <br>
# But the evaluation metric will switch from precision to recall and vice-versa, depending on the problem statement of the project. 
# 

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)


# In[ ]:


X_test = scaler.transform(X_test)


# In[ ]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# # Receiver Operating Characteristic (ROC) 1

# In[ ]:


y_prob = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Learning Curves
# We have achieved a commendable model above, but let us take this project one step further.<br> 
# Learning curves helps to detect if the model is affected by **bias or variance**.<br> 
# Let us plot learning curves based on a different set of hyperparameters. <br>
# 

# In[ ]:


from sklearn.model_selection import learning_curve


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(SVC(), X_train, y_train, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 20), cv = 3)


# In[ ]:


train_scores = np.mean(train_scores, axis = 1)
test_scores = np.mean(test_scores, axis = 1)


# In[ ]:


plt.plot(train_sizes, train_scores, 'o-', label="Training score")
plt.plot(train_sizes, test_scores, 'o-', label="Cross-validation score")
plt.legend();


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(SVC(C=2), X_train, y_train, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 20), cv = 3)
train_scores = np.mean(train_scores, axis = 1)
test_scores = np.mean(test_scores, axis = 1)
plt.plot(train_sizes, train_scores, 'o-', label="Training score")
plt.plot(train_sizes, test_scores, 'o-', label="Cross-validation score")
plt.legend();


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(SVC(C=3), X_train, y_train, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 20), cv = 3)
train_scores = np.mean(train_scores, axis = 1)
test_scores = np.mean(test_scores, axis = 1)
plt.plot(train_sizes, train_scores, 'o-', label="Training score")
plt.plot(train_sizes, test_scores, 'o-', label="Cross-validation score")
plt.legend();


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(SVC(C=3, gamma=0.1), X_train, y_train, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 20), cv = 3)
train_scores = np.mean(train_scores, axis = 1)
test_scores = np.mean(test_scores, axis = 1)
plt.plot(train_sizes, train_scores, 'o-', label="Training score")
plt.plot(train_sizes, test_scores, 'o-', label="Cross-validation score")
plt.legend();


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(SVC(C=3, gamma=0.01), X_train, y_train, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 20), cv = 3)
train_scores = np.mean(train_scores, axis = 1)
test_scores = np.mean(test_scores, axis = 1)
plt.plot(train_sizes, train_scores, 'o-', label="Training score")
plt.plot(train_sizes, test_scores, 'o-', label="Cross-validation score")
plt.legend();


# Now we can try every combination of hyperparameters and pick the best one. <br>
# Fortunately, ```scikit-learn``` comes with a ```GridSearchCV``` function to help us quickly come up with the best possible set of hyperparameters for the desired evaluation metric. 
# 

# # GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {'C':[1,2,3,4,5,6,7,8,14], 'gamma':[0.1, 0.01, 0.001, 0.0001], 'kernel':['linear', 'poly', 'rbf'], 'degree': [1,2,3,4,5]}
grid = GridSearchCV(param_grid= param_grid, estimator= SVC(), scoring='f1', refit= True, verbose=1)


# In[ ]:


grid.fit(X_train, y_train)


# In[ ]:


grid.best_params_


# Narrowing down further

# In[ ]:


param_grid = {'C':[6,7,8], 'gamma':np.linspace(0.01, 0.02, 10), 'kernel':['rbf'], 'degree': [1,2,3,4,5]}
grid = GridSearchCV(param_grid= param_grid, estimator= SVC(probability= True), scoring='f1', refit= True, verbose=1)
grid.fit(X_train, y_train)
grid.best_params_


# In[ ]:


y_pred = grid.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


y_prob = grid.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


result = pd.DataFrame({'Test':y_test, 'Prediction':y_pred, 'Probability': y_prob[:,1]})


# In[ ]:


result.to_csv('Result.csv')


# In[ ]:




