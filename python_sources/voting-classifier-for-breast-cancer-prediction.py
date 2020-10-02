#!/usr/bin/env python
# coding: utf-8

# ## Objective 
# Hi! Welcome to this kernel. Here, we'll explore a classification problem, try our hands on dimensionality reduction using Principal Component Analysis and some classification algorithms, try to improve these classifiers using hyperparameter tuning and in the end build an ensemble model to get a high accuracy score along with high precision and recall. 
# ### Importing modules
# Lets import the necessary modules.

# In[ ]:


import numpy as np 
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
data=pd.read_csv('../input/data.csv')
# Any results you write to the current directory are saved as output.


# Next, we check the data by looking for the features, its shape . 

# In[ ]:


data.columns


# In[ ]:


data.head(2)


# In[ ]:


data.shape


# As we will not need 'id' and 'Unnamed: 32', we drop these two columns.

# In[ ]:


data=data.drop(['id','Unnamed: 32'], axis=1)
data.columns


# Next, we check the distribution and statistics of data. As, can be observed many features have large varying scales.

# In[ ]:


data.describe()


# Surprisingly, this dataset doesn't have missing data. Cool! 

# In[ ]:


data.isnull().any()


# As disgnosis is our target variable, we check for the number of malignant and benign cases here.

# In[ ]:


M,B=data['diagnosis'].value_counts()
print('No. of malignant cases: ' ,M)
print('No. of benign cases: ' ,B)
sns.catplot(x='diagnosis',kind='count',data=data, palette="husl")


# Next, we make separate dataframes for malignant and benign cases (that will  be used for plotting purposes).

# In[ ]:


M=data.loc[data['diagnosis']=='M',:]
M.head()


# In[ ]:


B=data.loc[data['diagnosis']=='B',:]
B.head()


# In[ ]:


M=M.drop(['diagnosis'],axis=1)
B=B.drop(['diagnosis'],axis=1)


# ### Plots
# Next we make some Kernel Density Estimation (KDE) plots to check the distribution of malignant and benign cases for various features.

# In[ ]:


plt.subplots(figsize=(15,45))
sns.set_style('darkgrid')
plt.subplots_adjust (hspace=0.4, wspace=0.2)
i=0
for col in M.columns:
    i+=1
    plt.subplot(10,3,i)
    # first (0th) column of M is diagnosis, non-numerical
    sns.distplot(M[col],color='m',label='Malignant',hist=False, rug=True)
    sns.distplot(B[col],color='b',label='Benign',hist=False, rug=True)
    plt.legend(loc='right-upper')
    plt.title(col)
        


# From kde plots, it appears that radius_mean, perimeter_mean, area_mean, concavity_mean, concave_points_mean, compactness_mean, radius_worst, texture_worst, perimeter_worst, area_worst, concavity_worst, concave points_worst, compactness_worst show more difference between malignant and benign populations compared to others. Another way to check these differences can be by using boxplots.

# Next, we explore if and how these features are corrleted to one another. This is done using heatmap (as shown below). As here we have 30 features to look and compare, I will follow a suggestion given by  [Simon Bajew](https://www.kaggle.com/sbajew) in my first kernel to consider masking in the heat map.

# In[ ]:


sns.set(style="white")
fig,ax=plt.subplots(figsize=(16,16))
corr=data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,vmin=-1,vmax=1,fmt = ".1f",annot=True,cmap="coolwarm", mask=mask, square=True)


# As expected some of the features show complete correlation such as radius_mean, area_mean, perimeter_mean, radius_worst, area_worst and perimeter_worst. Also, texture_mean and texture_worst are correlated. concave points_mean and concavity_mean are very strongly correlated. radius_se, perimeter_se and area_se are strongly correlated. compactness_mean is correlated to concavity_mean, compactness_worst.

# In order to avoid high variance that can appear due to many correlated features, we drop some of the very highly correlated features. For example- we have dropped here  area_mean, perimeter_mean, radius_worst, area_worst and perimeter_worst while we kept radius_mean. The drop in shape is also checked here.  Also, we code Malignant and Benign cases with numbers- 1 and 0 respectively.

# In[ ]:


data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
data=data.drop(['area_mean', 'perimeter_mean', 'radius_worst', 'area_worst', 'perimeter_worst','texture_worst','concavity_mean','perimeter_se', 'area_se'],axis=1)
print(data.shape)


# ### Building model
# Before starting with bulding our model, we make test and train sets.

# In[ ]:


y=data['diagnosis'].values
X=data.drop(['diagnosis'],axis=1).values

X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=8)


# Before building model one thing that we must work on is to reduce or select the number of features. For this we will use Principal Component Analysis (PCA). PCA is an approach to obtain a few number of features from a large set. This aids in dimensionality reduction which can otherwise lead to high variance. Before performing PCA, it is crucial to standardize predictors to ensure all features are on the same scale. Otherwise, features that have high variance will influence the outcome of PCA. Here, we first use `StandardScaler` followed by PCA. Instead of arbitrarily choosing the number of dimension, we choose the number of dimensions that can explain most of the variance (usually 95% of the variance). To do that, we plot 'cumulative explained variance' vs 'number of components'. From this it seems that most of the variance can be explained by 10 components. So, we proceed with that number for PCA.

# In[ ]:


scaler=StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.transform(X_test)
pca=PCA().fit(X_train_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.xlim(0,22,2)


# In[ ]:


pca=PCA(n_components=10)
pca.fit(X_train_std)
X_train_pca=pca.transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
print(X_train_pca.shape)
print(X_test_pca.shape)


# We begin our best model search by starting with Logistic Regression.  We get a high accuracy of 97.5% with a high precision and recall rate of 97.9% and 95.2% respectively.

# In[ ]:


logreg=LogisticRegression(random_state=1)
score = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='accuracy'))
p_scores = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='precision'))
r_scores = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='recall'))
print("Accuracy: %s" % '{:.2%}'.format(score))
print ('Precision : %s' %'{:.2%}' .format(p_scores))
print ('Recall score: %s' % '{:.2%}'.format(r_scores))


# Next, we try KNeighborsClassifier. We obtain a high accuracy of 97.49% . Precision and recall are high as well, 98.5% and 89.7%. respectively. One can also check confusion matrix to check true positive/negative and false positive/negative (as shown below). But for rest of the analysis, I focus on the precision and recall score than confusion matrix.

# In[ ]:


knn=KNeighborsClassifier()
scores = np.mean(cross_val_score(knn,  X_train_pca, y_train, scoring='accuracy'))
p_scores = np.mean(cross_val_score(knn,  X_train_pca, y_train, scoring='precision'))
r_scores = np.mean(cross_val_score(knn,  X_train_pca, y_train, scoring='recall'))
print("Accuracy: %s" % '{:.2%}'.format(score))
print ('Precision : %s' %'{:.2%}' .format(p_scores))
print ('Recall score: %s' % '{:.2%}'.format(r_scores))

X1_train,X1_test,y1_train,y1_test= train_test_split(X_train_pca, y_train,test_size=0.3,random_state=21)
knn.fit(X1_train,y1_train)
y_pred=knn.predict(X1_test)
con=confusion_matrix(y1_test,y_pred)
print('Confusion matrix:')
print(con)


# Next, we perform some hyperparameter tuning using GridSearchCV to figure out the best parameters. For knn, we check for the number of neighbors which comes out to be 9. 

# In[ ]:


knn=KNeighborsClassifier()
param_grid = {"n_neighbors": np.arange(1,50)}
knn_cv = GridSearchCV(estimator = knn, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 0)
knn_cv.fit( X_train_pca, y_train)
print(knn_cv.best_params_)


# We then check how the KNeighborsClassifier (knn_cv) with hyperparameters tuning perform on new test sets and compare its performance with knn (without hyperparameter tuning). As, can be seen accuracy and precision improved for knn_cv.

# In[ ]:


knn_cv=KNeighborsClassifier(n_neighbors= 9)
score_knn_cv = np.mean(cross_val_score(knn_cv,  X_test_pca, y_test, scoring='accuracy'))
p_score_knn_cv = np.mean(cross_val_score(knn_cv,  X_test_pca, y_test, scoring='precision'))
r_score_knn_cv = np.mean(cross_val_score(knn_cv,  X_test_pca, y_test, scoring='recall'))
print("Accuracy for knn_cv: %s" % '{:.2%}'.format(score_knn_cv))
print ('Precision for knn_cv: %s' %'{:.2%}' .format(p_score_knn_cv))
print ('Recall score for knn_cv: %s' % '{:.2%}'.format(r_score_knn_cv))

score_knn = np.mean(cross_val_score(knn,  X_test_pca, y_test, scoring='accuracy'))
p_score_knn= np.mean(cross_val_score(knn,  X_test_pca, y_test, scoring='precision'))
r_score_knn = np.mean(cross_val_score(knn,  X_test_pca, y_test, scoring='recall'))
print("Accuracy for knn: %s" % '{:.2%}'.format(score_knn))
print ('Precision for knn: %s' %'{:.2%}' .format(p_score_knn))
print ('Recall score for knn: %s' % '{:.2%}'.format(r_score_knn))


# Next we try one of the most versatile of Supervised Learning Algorithms- Support Vector Machine. Accuracy is 96.23%...Not bad! Also, precision and recall score are high.

# In[ ]:


svc=SVC(random_state=1)
scores_svc = np.mean(cross_val_score(svc,  X_train_pca, y_train, scoring='accuracy'))
p_score_svc = np.mean(cross_val_score(svc,  X_test_pca, y_test, scoring='precision'))
r_score_svc = np.mean(cross_val_score(svc,  X_test_pca, y_test, scoring='recall'))
print("Accuracy for svc: %s" % '{:.2%}'.format(scores_svc))
print ('Precision for svc: %s' %'{:.2%}' .format(p_score_svc))
print ('Recall score for svc: %s' % '{:.2%}'.format(r_score_svc))


# Just like in case of KNN, we perform hyperparameter tuning for SVC. 

# In[ ]:


svc=SVC(random_state=1)
param_grid = {"C": [0.001,0.1,1,10], 'degree':[1,3,10]}
svc_cv=GridSearchCV(svc,param_grid=param_grid,cv = 3, n_jobs = -1, verbose = 0)
svc_cv.fit(X_train_pca, y_train)
svc_cv.best_params_


# svc_cv is then compared with svc. Both show the same accuracy, precision and recall score. I speculate that both are equally good for the tests set. The best parameters obtained are C=1, degree=1. The default setting for SVC has C=1, degree=3. Its likely that both degree=1 or 3 are equally good, they just return the lower number. If there is any other explaination then please let me know. For this exercise (for SVC), I stick to deafault settings.

# In[ ]:


svc_cv= SVC(random_state=1, C=1, degree=1)
score_svc_cv = np.mean(cross_val_score(svc_cv,  X_test_pca, y_test, scoring='accuracy'))
p_score_svc_cv = np.mean(cross_val_score(svc_cv,  X_test_pca, y_test, scoring='precision'))
r_score_svc_cv = np.mean(cross_val_score(svc_cv,  X_test_pca, y_test, scoring='recall'))
print("Accuracy for svc_cv: %s" % '{:.2%}'.format(score_svc_cv))
print ('Precision for svc_cv: %s' %'{:.2%}' .format(p_score_svc_cv))
print ('Recall score for svc_cv: %s' % '{:.2%}'.format(r_score_svc_cv))

score_svc = np.mean(cross_val_score(svc,  X_test_pca, y_test, scoring='accuracy'))
p_score_svc= np.mean(cross_val_score(svc,  X_test_pca, y_test, scoring='precision'))
r_score_svc = np.mean(cross_val_score(svc,  X_test_pca, y_test, scoring='recall'))
print("Accuracy for svc: %s" % '{:.2%}'.format(score_svc))
print ('Precision for svc: %s' %'{:.2%}' .format(p_score_svc))
print ('Recall score for svc: %s' % '{:.2%}'.format(r_score_svc))


# Next, we move to another most-versatile and easy to interpret and understand algorithm- Decision Trees. The result isn't as great as we have seen for SVC and logistic regression. 

# In[ ]:


dt=DecisionTreeClassifier(random_state=7)
score_dt = np.mean(cross_val_score(dt,  X_train_pca, y_train, scoring='accuracy'))
p_score_dt = np.mean(cross_val_score(dt,  X_train_pca, y_train, scoring='precision'))
r_score_dt = np.mean(cross_val_score(dt,  X_train_pca, y_train, scoring='recall'))
print("Accuracy for Decision Tree: %s" % '{:.2%}'.format(score_dt))
print ('Precision Decision Tree: %s' %'{:.2%}' .format(p_score_dt))
print ('Recall score Decision Tree: %s' % '{:.2%}'.format(r_score_dt))


# From Decision Tree we move to their forest -RandomForest. They show better accuracy, precision and recall score than Decision Trees.

# In[ ]:


rf=RandomForestClassifier(random_state=21)
score_rf = np.mean(cross_val_score(rf,  X_train_pca, y_train, scoring='accuracy'))
p_score_rf = np.mean(cross_val_score(rf,  X_train_pca, y_train, scoring='precision'))
r_score_rf = np.mean(cross_val_score(rf,  X_train_pca, y_train, scoring='recall'))
print("Accuracy for RandomForest: %s" % '{:.2%}'.format(score_rf))
print ('Precision RandomForest:: %s' %'{:.2%}' .format(p_score_rf))
print ('Recall score RandomForest:: %s' % '{:.2%}'.format(r_score_rf))


# Lets tune RandomForest's parameters and check if we get any improvement.

# In[ ]:


param_grid = {'max_depth': [80, 90, 100, 110],
              'max_features': [2, 3],
              'min_samples_leaf': [3, 4, 5],
              'min_samples_split': [8, 10, 12],
              'n_estimators': [100, 200, 300, 1000]}
# Create a basic model
rf = RandomForestClassifier(random_state=21)
# Instantiate the grid search model
rf_cv = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 0)
rf_cv.fit(X_train_pca, y_train)
print(rf_cv.best_params_)
score=rf_cv.best_score_
print("Accuracy: %s" % '{:.2%}'.format(score))


# We then compare the performance of rf_cv with rf on test set.  rf_cv show better accuracy than rf.

# In[ ]:


rf_cv=RandomForestClassifier(random_state=21,max_depth= 80, max_features= 3,min_samples_leaf= 5, 
                          min_samples_split=8,n_estimators= 100)
score_rf_cv = np.mean(cross_val_score(rf_cv,  X_test_pca, y_test, scoring='accuracy'))
print("Accuracy for rf_cv: %s" % '{:.2%}'.format(score_rf_cv))

score_rf = np.mean(cross_val_score(rf,  X_test_pca, y_test, scoring='accuracy'))
print("Accuracy for rf: %s" % '{:.2%}'.format(score_rf))


# ### Voting Classifier
# Next, we try if combining some/all of these algorithms can improve accuracy score. Here, in this ensemble method, we now aggregate the predictions from all the classifiers. This helps in predicting the class that gets the majority vote. Such way of voting is called 'Hard voting'.  The combined score is less than the accuracy score of LogisticRegression- 97.49% .  To get a better score, we move to 'Soft voting'.

# In[ ]:


logreg=LogisticRegression(random_state=1)
knn_cv=KNeighborsClassifier(n_neighbors= 9)
svc=SVC(random_state=1)
dt=DecisionTreeClassifier(random_state=7)
rf_cv=RandomForestClassifier(random_state=21,max_depth= 80, max_features= 3,min_samples_leaf= 5, 
                          min_samples_split=8,n_estimators= 100)

from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[('logreg',logreg),('knn_cv', knn_cv), ('rf_cv', rf_cv), ('dt',dt), ('svc', svc)], voting='hard')
score = np.mean(cross_val_score(voting_clf,  X_train_pca, y_train, scoring='accuracy'))
print("Accuracy : %s" % '{:.2%}'.format(score))


# 'Soft voting' is another way of voting classifier. In this case instead of prediting the class based on majority votes, class probabilities are aggregated and the class is predicted based on the class that gets highest class probability on averaged over all the classifiers. This type of voting usually achieves higher performance than hard voting. An important thing to remember is that soft voting only works with classifiers that can predict probabilities or have predict_proba function. To perform, soft voting we have to turn SVC's probability to True.   Accuracy reduces to 96.73% if we include all the classifiers. But it increases to 98.24% if we only consider LogisticRegression, SVC,RandomForest. FYI-if I use the same 3 classifiers and perform hard voting, accuracy remains 96.98%. 

# In[ ]:


svc=SVC(random_state=1,probability=True)

voting_clf = VotingClassifier(estimators=[('logreg',logreg), ('rf_cv', rf_cv),  ('svc', svc)], voting='soft')
score = np.mean(cross_val_score(voting_clf,  X_train_pca, y_train, scoring='accuracy'))
p_score = np.mean(cross_val_score(voting_clf,  X_train_pca, y_train, scoring='precision'))
r_score = np.mean(cross_val_score(voting_clf,  X_train_pca, y_train, scoring='recall'))
print("Accuracy : %s" % '{:.2%}'.format(score))
print ('Precision : %s' %'{:.2%}' .format(p_score))
print ('Recall :: %s' % '{:.2%}'.format(r_score))


# ### Conclusion
# Using feature selection and voting classifier we could achieve a high accuracy of 98.24%.  In addition, we achieved high Precision and Recall of 99.29% and 95.89%, respectively.
# #### Thank you for reading. As always, all your comments and suggestions are very welcome!
