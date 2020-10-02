#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/heart-disease-uci/heart.csv')


# ## Getting some insight about our data

# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.hist(figsize=(15,15))


# We can see that the dataset contains few categorical features and few continuous features

# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn')


# There are few features that have negative correlation with target and few with positive correlation

# In[ ]:


plt.bar(df.target.unique(),df.target.value_counts(),color=['red','green'])
plt.xticks([0,1])
print('No disease:{}%\nDisease:{}%'.format(round(df.target.value_counts(normalize=True)[0],2)*100,
                                           round(df.target.value_counts(normalize=True)[1],2)*100))


# The two classes are not exactly 50% each but the ratio is good enough to continue without dropping/increasing our data.

# ## Data Preprocessing

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
column_trans = make_column_transformer(
                (OneHotEncoder(),['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']),
                (StandardScaler(),['age', 'trestbps', 'chol', 'thalach', 'oldpeak']),
                remainder = 'passthrough')


# Applying OneHotEncoder on the categorical features and StandardScalar on continuous features

# ## Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X = df.drop(['target'], axis = 1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# In[ ]:


column_trans.fit_transform(X_train)


# ## Classification Algorithms

# ### 1. Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
logreg = LogisticRegression(solver='lbfgs')
pipe = make_pipeline(column_trans,logreg)


# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()


# ### 2. K Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn_scores = []
for k in range(1,31):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    pipe = make_pipeline(column_trans,knn_classifier)
    knn_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())


# In[ ]:


plt.figure(figsize=(12,12))
plt.plot([k for k in range(1, 31)], knn_scores, color = 'red')
for i in range(1,31):
    plt.text(i, knn_scores[i-1], (i, round(knn_scores[i-1]*100,2)))
plt.xticks([i for i in range(1, 31)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[ ]:


knn_scores[25]


# ### 3. Support Vector Classifier (SVC)

# There are several kernels for Support Vector Classifier. I'll test some of them and check which has the best score.

# In[ ]:


from sklearn.svm import SVC
svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    pipe = make_pipeline(column_trans,svc_classifier)
    svc_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())


# In[ ]:


from matplotlib.cm import rainbow
colors = rainbow(np.linspace(0, 1, len(kernels)))
plt.figure(figsize=(10,10))
plt.bar(kernels, svc_scores, color = colors)
for i in range(len(kernels)):
    plt.text(i, svc_scores[i], svc_scores[i])
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier scores for different kernels')


# In[ ]:


svc_scores[0] #linear


# ### 4. Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    pipe = make_pipeline(column_trans,dt_classifier)
    dt_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())


# In[ ]:


plt.figure(figsize=(10,10))
plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')


# In[ ]:


dt_scores[4]


# ### 5. Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    pipe = make_pipeline(column_trans,rf_classifier)
    rf_scores.append(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())


# In[ ]:


plt.figure(figsize=(10,10))
colors = rainbow(np.linspace(0, 1, len(estimators)))
plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], round(rf_scores[i],5))
plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of estimators')


# In[ ]:


rf_scores[1]


# ### 6. Gaussian NB

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
pipe = make_pipeline(column_trans,nb)
cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()


# It is worth to test our model on 'testing dataset' using two classification models that are having high accuracy score than other classification models. They are (i) Logistic Regression and (ii) SVC using linear kernel.

# ## Make predictions on "unseen" data

# ### 1. Logistic Regression

# In[ ]:


pipe = make_pipeline(column_trans,logreg)
pipe.fit(X_train, y_train)


# In[ ]:


y_pred = pipe.predict(X_test)


# In[ ]:


from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)*100


# ### 2. SVC with linear kernel

# In[ ]:


svc_classifier = SVC(kernel = 'linear')
pipe = make_pipeline(column_trans,svc_classifier)
pipe.fit(X_train, y_train)


# In[ ]:


y_pred = pipe.predict(X_test)


# In[ ]:


from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)*100


# Suggestions are Welcome. Thank you!
