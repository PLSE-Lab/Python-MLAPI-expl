#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection System

# Credit Card Fraud is a common form of theft in the finance sector. Unauthorized and suspicious transaction on bank statements have been an rising issue and these frauds constitute a significant part of the fiscal loss in the banking sector. Detecting a credit card fraud in real time is an uphill task primarily because of the fact that fraud transactions are very few in amount in comparison to the normal transactions. This Notebook follows all basic steps of a Machine Learning Classification lifecycle, and can be used as reference for the few steps involved in construction of a ML pipeline: <br>
# 1. Loading Data
# 2. Exploratory Data Analysis: Correlations, Distributions, Outlier detection
# 3. Feature Scaling: Standard Scaler
# 4. Data Segmentation
# 5. Safe Sampling to mitigate class imbalance: Near Miss and SMOTE
# 6. Model Training: using KNN, Logistic Regression, SVM, Logistic-KNN Ensemble with Hard Voting and Soft Voting, Tree Based methods, Bagging Classifier, SGD Classifier and LightGBM
# 7. Predictions and Evaluation
# 8. Cross Validation: Grid Search CV and KFold CV
# 9. Model Explanation
# 
# **Objective:** Detecting as many frauds possible and at same time minimizing the number of Non- Frauds categorized as Frauds in the process. <br>
# **Challenges:** Class Imbalance, No description available about the features <br>
# 
# For subsequent steps after model training that will help us in deployment, access notebook at: https://github.com/nayaksubhankar/CreditCardFraudDetection <br>
# --> Forming a sklearn pipeline <br>
# --> Serializing the pipeline and dumping it using Joblib <br>
# --> Inference Testing

# ### Imports

# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.utils import shuffle

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from lightgbm import record_evaluation

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.pipeline import Pipeline
import joblib
import pickle
import json

import random

import warnings
warnings.filterwarnings('ignore')


# ### Loading Data

# In[ ]:


ccfr= pd.read_csv('../input/creditcardfraud/creditcard.csv')
ccfr.head()


# In[ ]:


x= ccfr['Class']
plt.figure(figsize=(5,7))
sns.countplot(ccfr['Class'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[ ]:


fraud_df= ccfr[ccfr['Class']==1]
nonfraud_df= ccfr[ccfr['Class']==0]
fraudcount= fraud_df.count()[0]
nonfraudcount= nonfraud_df.count()[0]
print ("Frauds=",fraudcount)
print ("Non Frauds=",nonfraudcount)
print (ccfr.shape)


# ### EDA

# Describing the dataframe, we can see that all features are numeric but are not standardised. Hence we will standardise them using a Scaler in further steps.

# In[ ]:


ccfr.describe()


# There are no missing values in our data:

# In[ ]:


ccfr.isna().sum()


# Plotting the correlation heatmap shows that the independent features donot possess significant inter feature correlation. Hence, all of them can be a feature in our model as they are independent of each other. The pearson correlation indices tell us that the independent variables donot show significant linear relationship with the target variable. Note that the correlation coefficient is a way to represent linear relationships. However, the relationship may be of a higher degree (e.g. tertiary, quadratic and so on) which is not depicted by the correlation coefficient. To visualize such relations, more univariate and multivariate plots need to be checked. (I prefer doing this using Tableau)

# In[ ]:


plt.figure(figsize=(22,22))
sns.heatmap(ccfr.corr(),cmap="coolwarm", annot=True)
plt.show()


# Let's use seaborn distplot to see the distribution of each feature with respect to the target variable 'Class'. The distplots give us a visual presentation of how a particular feature's values can tell apart among Fraud and Non-fraud cases. The plots showing a clear divide can be considered as significant features that may have good impact on our target variable, which we will cross validate with other measures as well.

# In[ ]:


for c in ccfr.columns[0:30]:
    print ("******************** COLUMN ",c," ***********************")
    col= ccfr[c]
    col=np.array(col)
    col_mean= np.mean(col)
    col_median= np.median(col)
    col_std= np.std(col)
    col_var= np.var(col)
    col_range= col.max()-col.min()
    fig=sns.FacetGrid(ccfr,hue="Class",height=5,aspect=2,palette=["blue", "green"])
    fig.map(sns.distplot,c)
    fig.add_legend(labels=['Non Fraud','Fraud'])
    plt.axvline(col_mean,color='red',label='mean')
    plt.axvline(col_median,color='yellow',label='median')
    plt.legend()
    plt.show()


# In[ ]:


ccfr.columns


# There are plenty outliers in each feature. However, it is better not removing them as the minority class already has only few samples and there is good chance that these outliers impart some meaning to the problem:

# In[ ]:


for feature in ccfr.drop('Class', axis= 1).columns:
    sns.boxplot(x='Class', y= feature, data= ccfr)
    plt.show()


# ### Feature Scaling

# Using Sci-kit Learn's Standard Scaler

# In[ ]:


X = ccfr.drop('Class', axis= 1)
y = ccfr['Class']


# In[ ]:


scaler = StandardScaler()
scaled_features = scaler.fit_transform(X.values)
X_scaled = pd.DataFrame(scaled_features, columns= X.columns)


# In[ ]:


X_scaled.head()


# The features are now standardised and fit for modelling as all of them have mean almost equal to zero and standard deviation nearly equal to 1:

# In[ ]:


X_scaled.describe()


# In[ ]:


ccfr_scaled = pd.concat([X_scaled, y], axis= 1)


# ### Data Segmentation

# Train test split. Train set can be used for sampling purposes. The test set is kept aside for "real world" testing.

# In[ ]:


X = ccfr_scaled.drop('Class', axis= 1)
y = ccfr_scaled['Class']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.13, random_state= 48)


# In[ ]:


X_train.shape


# In[ ]:


y_train.value_counts()


# In[ ]:


y_test.value_counts()


# ### Sampling to deal with class imbalance

# 1. Undersampling the majority class using Near Miss strategy to get the samples from majority class that are nearest to the minority class records in terms of distance. The overall idea is to reduce information loss with undersampling. If our classifier trains on the samples closest to the decision boundary then it will be able to classify points that are farther apart.
# 2. Oversampling the minority class using SMOTE strategy that is common over sampling used in many class imbalance problems.
# 3. Aim is to sample to an extent without introducing many systhetic samples of the minority class nor losing out on significant majority class samples. The remaining imbalance can be handled by using class weights in our model parameters when and where required.

# In[ ]:


train = pd.concat([X_train, y_train], axis =1)


# In[ ]:


under_sampler = NearMiss(sampling_strategy= {0:100000, 1:410}) 
#under sampling the majority to 80000 records keeping minority as it is
X_train, y_train = under_sampler.fit_sample(X_train, y_train)


# In[ ]:


over_sampler = SMOTE(sampling_strategy= {0:100000, 1:10000}, random_state= 48)
#over sampling minority class to 20000 records
X_train, y_train = over_sampler.fit_sample(X_train, y_train)


# In[ ]:


train_sampled = pd.concat([X_train, y_train], axis= 1)
#it is good practice to shuffle the training set
train_sampled = train_sampled.sample(frac=1).reset_index(drop= True)
X_train = train_sampled.drop('Class', axis= 1)
y_train = train_sampled['Class']


# In[ ]:


X_train.shape


# In[ ]:


y_train.value_counts()


# In[ ]:


X_test.shape


# In[ ]:


y_test.value_counts()


# In[ ]:


sns.countplot(y_train)
plt.show()


# **Post Sampling**<br>
# Train Set: (110000, 30) -- 100000 Non Frauds, 10000 Frauds <br>
# Test Set: (37025, 30) -- 36943 Non Frauds, 82 Frauds

# ### Model Training and Evaluation 
# The motive of our classifier will be to predict maximum of the fraud transactions as fraud at the cost of some non fraud transactions being predicted as fraud. But we must not lose out on the fraud transactions! Hence we will aim for High recall on the frauds (that is the percentage of frauds correctly classified as frauds) but a low precision on the frauds is manageable. Thus, a good overall ROC Score and good recall score for the minority class will be apt.

# **KNN Classifier**

# In[ ]:


knn_model = KNeighborsClassifier(n_neighbors= 4, n_jobs= -1)
knn_model.fit(X_train, y_train)
pred = knn_model.predict(X_test)
print ("ROC AUC Score=",roc_auc_score(y_test, pred))
print ("Classification Report:")
print (classification_report(y_test, pred))


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test, pred))
plt.figure()
plt.title("Confusion Matrix for KNN")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# Hence, with KNN Classifier we are able to predict 93% (76 out of 82) Frauds correctly at a cost of 2% (766 out of 36943) of the Non Frauds being predicted as Frauds.

# **Logistic Regression**

# In[ ]:


lr_model= LogisticRegression(solver= 'liblinear')
lr_model.fit(X_train,y_train)
pred = lr_model.predict(X_test)
print ("ROC AUC Score=",roc_auc_score(y_test, pred))
print ("Classification Report:")
print (classification_report(y_test, pred))


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test, pred))
plt.figure()
plt.title("Confusion Matrix for LR")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# Hence, with Logistic Regression we are able to predict 94% (77 out of 82) Frauds correctly at a cost of 5% (1749 out of 36943) of the Non Frauds being predicted as Frauds.

# **SVM Classifier**

# In[ ]:


svc_model = svm.SVC(kernel='rbf', gamma= 0.03, C= 1.0)
svc_model.fit(X_train, y_train)
pred= svc_model.predict(X_test)


# In[ ]:


print ("ROC AUC Score=",roc_auc_score(y_test, pred))
print ("Classification Report:")
print (classification_report(y_test, pred))


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test, pred))
plt.figure()
plt.title("Confusion Matrix for SVM")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# Hence, Support Vector Classifier does great and is able to predict 95% (78 out of 82) Frauds correctly at a cost of 2% (893 out of 36943) of the Non Frauds being predicted as Frauds.

# **Logistic Regression + KNN Voting CLassifier**

# Voting Classifier combines predictions on the data set from two different classifiers and votes on the best prediction. Voting Classifier is a good means to combine predictions from two similar kind of classifiers and bring out the best parts of both the classifiers and combine them. We can see that KNN Classifier predicted 93% frauds compared to the 94% predicted by Logistic Regression. However, KNN predicted lesser no. of Non- Frauds as Frauds as compared to Logistic. (KNN predicted 766 non frauds as frauds compared to 1749 in case of LR.) So we will try and combine LR and KNN to see if we can get a better combination of results.

# In[ ]:


voting_clf_hard = VotingClassifier(estimators = [('lr', lr_model), ('knn', knn_model)], voting = 'hard')
voting_clf_hard.fit(X_train, y_train)
pred= voting_clf_hard.predict(X_test)
print ("ROC AUC Score=",roc_auc_score(y_test, pred))
print ("Classification Report:")
print (classification_report(y_test, pred))


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test, pred))
plt.figure()
plt.title("Confusion Matrix for LR + KNN Voting Classifier (Hard Voting)")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# The Hard voting classifier using LR and KNN gets 91% of Fraud predictions right at a reduced cost of just 2% (660 of 36943) of Non frauds getting classified as fraud. It reduces the no. of Non Frauds misclassified, but also hinders the fraud detection %. So, we will change the voting mechanism to Soft voting and provide weights to give more weights to probability calculated by the LR model:

# In[ ]:


voting_clf_soft = VotingClassifier(estimators = [('lr', lr_model), ('knn', knn_model)], voting = 'soft',
                                  weights= [0.7, 0.3])
voting_clf_soft.fit(X_train, y_train)
pred= voting_clf_soft.predict(X_test)
print ("ROC AUC Score=",roc_auc_score(y_test, pred))
print ("Classification Report:")
print (classification_report(y_test, pred))


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test, pred))
plt.figure()
plt.title("Confusion Matrix for LR + KNN Voting Classifier (Soft Voting)")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# The Soft Voting Classifier helps us combine the classifiers well. We have retained the better fraud detection % of the Logistic regression model i.e. 94% (77 of 82). We also managed to decrease the number of Non frauds misclassified as Frauds. (1432 of 36943)

# **Decision Tree Classifier**

# In[ ]:


dctree_model = DecisionTreeClassifier(random_state=0)
dctree_model.fit(X_train, y_train)
pred= dctree_model.predict(X_test)
print ("ROC AUC Score=",roc_auc_score(y_test, pred))
print ("Classification Report:")
print (classification_report(y_test, pred))


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test, pred))
plt.figure()
plt.title("Confusion Matrix for Decision Tree")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# Decision Tree classifier detects 90% of the frauds. We can do better by using the ensemble tree based approaches.

# **SGD Clasifier**

# SGD Classifier comes with set of hyperparameters that can be set accordingly to train a model similar to the working linear models like Logistic regression or SVMs while minimizing the loss using Stochastic Gradient Descent. SGD Classifier models do good on large sized data.

# In[ ]:


sgd_model = SGDClassifier(class_weight = 'balanced', learning_rate = 'adaptive', n_jobs = -1, eta0 = 0.001, 
                          max_iter = 100000)
sgd_model.fit(X_train, y_train)
pred = sgd_model.predict(X_test)
print ("ROC AUC Score=",roc_auc_score(y_test, pred))
print ("Classification Report:")
print (classification_report(y_test, pred))


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test, pred))
plt.figure()
plt.title("Confusion Matrix for SGD")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# Stochastic Gradient Descent Classifier detects 94% of the fraud cases but at a cost of 9% of Non Frauds being predicted as Fraud.

# **Bagging Classifier using Decision Tree**

# A Bagging Classifier can be used to build an ensemble of a number of similar model. Here we will build an ensemble by bagging 1000 Decision Tree Classifiers that we built earlier. RandomForest is an example of pre- available Bagging Tree classifiers present in Scikit Learn.

# In[ ]:


bagging_clf = BaggingClassifier(DecisionTreeClassifier(random_state = 0), n_estimators = 1000, bootstrap = True,
                               max_samples = 0.85, n_jobs = -1, oob_score = True)
#bootsrap True signifies sampling (max_samples) from the data without replacement for each estimator
#oob_score True enables training on set of samples chosen and test on the out-of-bag samples(samples not chosen for training)
bagging_clf.fit(X_train, y_train)
pred = bagging_clf.predict(X_test)
print ("ROC AUC Score=",roc_auc_score(y_test, pred))
print ("Classification Report:")
print (classification_report(y_test, pred))


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test, pred))
plt.figure()
plt.title("Confusion Matrix for Bagging Classifier")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# By using the Decision tree in a bagging classifier, the performance has improved significantly. The fraud prediction rate is now 94% compared to 90% achieved with decision trees classifier.

# **LightGBM**<br>
# LightGBM is a gradient boosting framework that is new and has been growing fast in terms of popularity. It promises great results similar to XGBoost and that to at extremely lesser training time. We will train the LightGBM model and track it's evaluation history using the eval_history callback to know the right no. of estimators to stop at. The evaluation metric chosen for the purpose is the roc auc score.

# In[ ]:


lgb_model = LGBMClassifier(boosting_type = 'gbdt', num_leaves = 32, max_depth = 7, learning_rate = 0.007, 
                           n_estimators = 3500, objective = 'binary', min_split_gain = 0.1, min_child_weight = 0.01,
                           class_weight= {0:0.2, 1:1},
                           min_child_samples = 20, subsample=0.6, colsample_bytree = 0.8, reg_alpha = 0.3, reg_lambda = 0.7,
                           n_jobs = -1, verbose = -1)
history = {}
eval_history = record_evaluation(history)
lgb_model.fit(X_train, y_train,
             eval_set = [(X_train, y_train), (X_test, y_test)],
             eval_metric = 'auc', verbose = 500,
             callbacks = [eval_history])
pred = lgb_model.predict(X_test)
print ("ROC AUC Score=",roc_auc_score(y_test, pred))
print ("Classification Report:")
print (classification_report(y_test, pred))


# In[ ]:


train_aucs = history['training']['auc']
test_aucs = history['valid_1']['auc']
plt.figure(figsize = (8,6))
plt.ylim([0,1.01])
plt.plot(train_aucs, color= 'r', label= 'training')
plt.plot(test_aucs, color= 'g', label= 'testing')
plt.xlabel("No. of Estimators")
plt.ylabel('AUC Scores')
plt.legend(loc= 'best')
plt.title("LightGBM Performance with n_estimators chosen")
plt.show()


# In[ ]:


train_logloss = history['training']['binary_logloss']
test_logloss = history['valid_1']['binary_logloss']
plt.figure(figsize = (8,6))
plt.ylim([0,1.01])
plt.plot(train_logloss, color= 'r', label= 'training')
plt.plot(test_logloss, color= 'g', label= 'testing')
plt.xlabel("No. of Estimators")
plt.ylabel('Binary Log Loss')
plt.legend(loc= 'best')
plt.title("LightGBM Loss Minimization")
plt.show()


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test, pred))
plt.figure()
plt.title("Confusion Matrix for LightGBM")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# LightGBM gives best overall performance by predicting 94% of the frauds at the cost of just 259 non frauds getting classified as frauds.

# **Evaluating LightGBM with K Fold CV**

# Let's see if KFold CV can help us run more estimators without overfitting on the train set and evaluate the predictions on the test set! <br>
# **Note:** KFold CV is more of an evaluation method to check if our classifier doesnot overfit on the training set and has good overall prediction on different testing samples. A KFold CV operation should be done on the parent data and not on the sampled set. The sampling should be done within the CV after attaining the train test split for each fold. Using sampled data for KFold splits in the Cross validation will not suffice the "real world" testing. <br>
# **Note 2:** For sampling the data splits during each Fold of the process, use the same sampling method used before training phase of the model. In this case it is Near Miss + Smote

# In[ ]:


lgb_model = LGBMClassifier(boosting_type = 'gbdt', num_leaves = 30, max_depth = 7, learning_rate = 0.007, 
                           n_estimators = 4500, objective = 'binary', min_split_gain = 0.1, min_child_weight = 0.01,
                           class_weight= {0:0.2, 1:1},
                           min_child_samples = 20, subsample=0.6, colsample_bytree = 0.8, reg_alpha = 0.3, reg_lambda = 0.7,
                           n_jobs = -1, verbose = -1)

cv = KFold(n_splits = 10, random_state = 48, shuffle = True)

TP = 0 #TruePositives
TN = 0 #TrueNegatives
FP = 0 #FalsePositives
FN = 0 #FalseNegatives
roc_auc_scores = []

x1 = X #taking the original scaled data post train test split
y1 = y

for train_ind, test_ind in cv.split(x1):
    xtrain, xtest, ytrain, ytest= x1.loc[list(train_ind)], x1.loc[list(test_ind)], y1.loc[list(train_ind)], y1.loc[list(test_ind)]
    
    under_sampler_cv = NearMiss(sampling_strategy= {0:100000, 1:410}) 
    xtrain, ytrain = under_sampler_cv.fit_sample(xtrain, ytrain)
    
    over_sampler_cv = SMOTE(sampling_strategy= {0:100000, 1:10000}, random_state= 48)
    xtrain, ytrain = over_sampler_cv.fit_sample(xtrain, ytrain)
    
    train_set = pd.concat([xtrain, ytrain], axis= 1)
    train_set = train_set.sample(frac=1).reset_index(drop= True)
    xtrain = train_set.drop('Class', axis= 1)
    ytrain = train_set['Class']
    
    lgb_model.fit(xtrain, ytrain)
    prd = lgb_model.predict(xtest)
    true = np.array(ytest)
    l = len(prd)
    for i in range (l):
        if true[i]==1 and prd[i]==1:
            TP+=1
        if true[i]==1 and prd[i]==0:
            FN+=1
        if true[i]==0 and prd[i]==1:
            FP+=1
        if true[i]==0 and prd[i]==0:
            TN+=1
    roc_auc_scores.append(roc_auc_score(true, prd))


# In[ ]:


cm = pd.DataFrame([[TN, FP], [FN, TP]], index= [0,1], columns= [0,1])
plt.figure()
plt.title("Confusion Matrix for LightGBM with 10 Fold CV")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# In[ ]:


print ("% of Frauds predicted correctly=",(426/(66+426))*100)
print ("Average ROC AUC Score=",np.average(roc_auc_scores))


# So, the average fraud prediction on entire data is 86.58% on 10 splits of the data which is a good overall result.

# **Grid Search CV on LightGBM**

# We can perform a Grid Search to get the best parameter list. We can use 'recall' metric for scoring as our model should be more recall oriented.

# In[ ]:


params = {'num_leaves' : [30, 40],
          'max_depth' : [7, 9],  
          'min_child_weight' : [0.01, 1],
          'subsample' : [0.6, 0.7],
          'colsample_bytree' : [0.8, 0.9],
          'reg_alpha' : [0.1, 0.3],
          'reg_lambda' : [0.1, 0.7]}
clf = GridSearchCV(lgb_model, params, scoring= 'recall', n_jobs= -1, cv= 2)
clf.fit(X_train, y_train)


# In[ ]:


clf.best_params_


# Now lets Redo 10 Fold CV again with the new params, and see if the prediction rate improves..

# In[ ]:


lgb_model = LGBMClassifier(boosting_type = 'gbdt', num_leaves = 30, max_depth = 9, learning_rate = 0.007, 
                           n_estimators = 3500, objective = 'binary', min_split_gain = 0.1, min_child_weight = 0.01,
                           class_weight= {0:0.1, 1:1},
                           min_child_samples = 20, subsample=0.6, colsample_bytree = 0.9, reg_alpha = 0.1, reg_lambda = 0.7,
                           n_jobs = -1, verbose = -1)

cv = KFold(n_splits = 10, random_state = 48, shuffle = True)

TP = 0 #TruePositives
TN = 0 #TrueNegatives
FP = 0 #FalsePositives
FN = 0 #FalseNegatives
roc_auc_scores = []

x1 = X #taking the original scaled data post train test split
y1 = y

for train_ind, test_ind in cv.split(x1):
    xtrain, xtest, ytrain, ytest= x1.loc[list(train_ind)], x1.loc[list(test_ind)], y1.loc[list(train_ind)], y1.loc[list(test_ind)]
    
    under_sampler_cv = NearMiss(sampling_strategy= {0:100000, 1:410}) 
    xtrain, ytrain = under_sampler_cv.fit_sample(xtrain, ytrain)
    
    over_sampler_cv = SMOTE(sampling_strategy= {0:100000, 1:10000}, random_state= 48)
    xtrain, ytrain = over_sampler_cv.fit_sample(xtrain, ytrain)
    
    train_set = pd.concat([xtrain, ytrain], axis= 1)
    train_set = train_set.sample(frac=1).reset_index(drop= True)
    xtrain = train_set.drop('Class', axis= 1)
    ytrain = train_set['Class']
    
    lgb_model.fit(xtrain, ytrain)
    prd = lgb_model.predict(xtest)
    true = np.array(ytest)
    l = len(prd)
    for i in range (l):
        if true[i]==1 and prd[i]==1:
            TP+=1
        if true[i]==1 and prd[i]==0:
            FN+=1
        if true[i]==0 and prd[i]==1:
            FP+=1
        if true[i]==0 and prd[i]==0:
            TN+=1
    roc_auc_scores.append(roc_auc_score(true, prd))


# In[ ]:


cm = pd.DataFrame([[TN, FP], [FN, TP]], index= [0,1], columns= [0,1])
plt.figure()
plt.title("Confusion Matrix for LightGBM with 10 Fold CV")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# In[ ]:


print ("% of Frauds predicted correctly=",(430/(62+430))*100)
print ("Average ROC AUC Score=",np.average(roc_auc_scores))


# With the New params, the average prediction rate is now 87.4% with ROC AUC score of 0.9326

# **LightGBM Explanation**

# In[ ]:


import lightgbm
lightgbm.plot_importance(lgb_model, figsize= (12, 10))


# **Results**

# In[ ]:


pred = lgb_model.predict(X_test)
print ("ROC AUC Score=",roc_auc_score(y_test, pred))
print ("Classification Report:")
print (classification_report(y_test, pred))


# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test, pred))
plt.figure()
plt.title("Confusion Matrix for LightGBM")
sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.show()


# In[ ]:


from sklearn.metrics import roc_curve
fpr , tpr, threshold = roc_curve(y_test, pred)
roc_auc = roc_auc_score(y_test, pred)

plt.title('Receiver Operating Characteristics')
plt.plot(fpr, tpr, 'b', label= "AUC = %0.2f" % roc_auc)
plt.legend(loc = 'best')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("False Postive Rate")
plt.ylabel("True Positive Rate")
plt.show()


# Thus, with the final trained LGBMClassifier is able to achieve 95% fraud prediction rate at the cost of a meagre 132 non frauds being classified as fraud, with an ROC AUC score of 0.9738

# **Thanks for the Read!**
