#!/usr/bin/env python
# coding: utf-8

# **Project title :- Santander customer transaction prediction using Python**

# **Problem statement :-**
# 
# In this challenge, we need to identify which customers will make a specific transaction in
# the future, irrespective of the amount of money transacted.
# 

# **Contents:**
# 
#  1. Exploratory Data Analysis
#            * Loading dataset and libraries
#            * Data cleaning
#            * Typecasting the attributes
#            * Target classes count        
#            * Missing value analysis
#         2. Attributes Distributions and trends
#            * Distribution of train attributes
#            * Distribution of test attributes
#            * Mean distribution of attributes
#            * Standard deviation distribution of attributes
#            * Skewness distribution of attributes
#            * Kurtosis distribution of attributes      
#            * Outliers analysis
#         4. Correlation matrix 
#         5. Split the dataset into train and test dataset
#         7. Modelling the training dataset
#            * Logistic Regression Model
#            * SMOTE Model
#            * LightGBM Model
#         8. Cross Validation Prediction
#            * Logistic  Regression CV Prediction
#            * SMOTE CV Prediction
#            * LightGBM CV Prediction
#         9. Model performance on test dataset
#            * Logistic Regression Prediction
#            * SMOTE Prediction
#            * LightGBM Prediction
#         10. Model Evaluation Metrics
#            * Confusion Matrix
#            * ROC_AUC score
#         11. Choosing best model for predicting customer transaction

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
from sklearn.metrics import roc_auc_score,confusion_matrix,make_scorer,classification_report,roc_curve,auc
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids,NearMiss, RandomUnderSampler
import lightgbm as lgb
import eli5
from eli5.sklearn import PermutationImportance
from sklearn import tree
import graphviz
from pdpbox import pdp, get_dataset, info_plots
import scikitplot as skplt
from scikitplot.metrics import plot_confusion_matrix,plot_precision_recall_curve


from scipy.stats import randint as sp_randint
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

random_state=42
np.random.seed(random_state)


# **Importing the train dataset**

# In[ ]:


#importing the train dataset
train_df=pd.read_csv('../input/train.csv')
train_df.head()


# **Shape of the train dataset**

# In[ ]:


#Shape of the train dataset
train_df.shape


# Summary of the dataset

# In[ ]:


#Summary of the dataset
train_df.describe()


# **Target classes count**

# In[ ]:


get_ipython().run_cell_magic('time', '', "#target classes count\ntarget_class=train_df['target'].value_counts()\nprint('Count of target classes :\\n',target_class)\n#Percentage of target classes count\nper_target_class=train_df['target'].value_counts()/len(train_df)*100\nprint('percentage of count of target classes :\\n',per_target_class)\n\n#Countplot and violin plot for target classes\nfig,ax=plt.subplots(1,2,figsize=(20,5))\nsns.countplot(train_df.target.values,ax=ax[0],palette='husl')\nsns.violinplot(x=train_df.target.values,y=train_df.index.values,ax=ax[1],palette='husl')\nsns.stripplot(x=train_df.target.values,y=train_df.index.values,jitter=True,color='black',linewidth=0.5,size=0.5,alpha=0.5,ax=ax[1],palette='husl')\nax[0].set_xlabel('Target')\nax[1].set_xlabel('Target')\nax[1].set_ylabel('Index')")


# **Take aways:**                   
# * We have a unbalanced data,where 90% of the data is the number of customers those will not make a transaction and 10% of the data is those who will make a transaction.
# * Look at the violin plots seems that there are no relationship between the target with the index of the train dataframe.This is more dominated by the zero targets then for the ones.
# * Look at the jitter plots with violin plots. We can observed that targets looks uniformly distributed over the indexs of the dataframe.

# **Let us look distribution of train attributes**

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Distribution of train attributes\ndef plot_train_attribute_distribution(t0,t1,label1,label2,train_attributes):\n    i=0\n    sns.set_style(\'whitegrid\')\n    \n    fig=plt.figure()\n    ax=plt.subplots(10,10,figsize=(22,18))\n    \n    for attribute in train_attributes:\n        i+=1\n        plt.subplot(10,10,i)\n        sns.distplot(t0[attribute],hist=False,label=label1)\n        sns.distplot(t1[attribute],hist=False,label=label2)\n        plt.legend()\n        plt.xlabel(\'Attribute\',)\n        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})\n    plt.show()')


# Let us see first 100 train attributes will be displayed in next cell.

# In[ ]:


get_ipython().run_cell_magic('time', '', "#corresponding to negative class\nt0=train_df[train_df.target.values==0]\n#corresponding to positive class\nt1=train_df[train_df.target.values==1]\n#train attributes from 2 to 102\ntrain_attributes=train_df.columns.values[2:102]\n#plot distribution of train attributes\nplot_train_attribute_distribution(t0,t1,'0','1',train_attributes)")


# Let us see next 100 train attributes will be displayed in next cell.

# In[ ]:


get_ipython().run_cell_magic('time', '', "#train attributes from 102 to 203\ntrain_attributes=train_df.columns.values[102:203]\n#plot distribution of train attributes\nplot_train_attribute_distribution(t0,t1,'0','1',train_attributes)")


# **Take aways:**
# * We can observed that their is a considerable number of features which are significantly have different distributions for two target variables. For example like var_0,var_1,var_9,var_198 var_180 etc.
# *  We can observed that their is a considerable number of features which are significantly have same distributions for two target variables. For example like var_3,var_7,var_10,var_171,var_185 etc.
# 

# **Importing the test dataset**

# In[ ]:


#importing the test dataset
test_df=pd.read_csv('../input/test.csv')
test_df.head()


# **Shape of the test dataset**

# In[ ]:


#Shape of the test dataset
test_df.shape


# **Let us look distribution of test attributes**

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Distribution of test attributes\ndef plot_test_attribute_distribution(test_attributes):\n    i=0\n    sns.set_style(\'whitegrid\')\n    \n    fig=plt.figure()\n    ax=plt.subplots(10,10,figsize=(22,18))\n    \n    for attribute in test_attributes:\n        i+=1\n        plt.subplot(10,10,i)\n        sns.distplot(test_df[attribute],hist=False)\n        plt.xlabel(\'Attribute\',)\n        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})\n    plt.show()')


# Let us see first 100 test attributes will be displayed in next cell.

# In[ ]:


get_ipython().run_cell_magic('time', '', '#test attribiutes from 1 to 101\ntest_attributes=test_df.columns.values[1:101]\n#plot distribution of test attributes\nplot_test_attribute_distribution(test_attributes)')


# Let us see next 100 test attributes will be displayed in next cell.

# In[ ]:


get_ipython().run_cell_magic('time', '', '#test attributes from 101 to 202\ntest_attributes=test_df.columns.values[101:202]\n#plot the distribution of test attributes\nplot_test_attribute_distribution(test_attributes)')


# **Take aways:**
# * We can observed that their is a considerable number of features which are significantly have different distributions. 
#   For example like var_0,var_1,var_9,var_180 var_198 etc.
# * We can observed that their is a considerable number of features which are significantly have same distributions. 
#   For example like var_3,var_7,var_10,var_171,var_185,var_192 etc.
# 

# **Let us look distribution of mean values per rows and columns in train and test dataset**

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Distribution of mean values per column in train and test dataset\nplt.figure(figsize=(16,8))\n#train attributes\ntrain_attributes=train_df.columns.values[2:202]\n#test attributes\ntest_attributes=test_df.columns.values[1:201]\n#Distribution plot for mean values per column in train attributes\nsns.distplot(train_df[train_attributes].mean(axis=0),color='blue',kde=True,bins=150,label='train')\n#Distribution plot for mean values per column in test attributes\nsns.distplot(test_df[test_attributes].mean(axis=0),color='green',kde=True,bins=150,label='test')\nplt.title('Distribution of mean values per column in train and test dataset')\nplt.legend()\nplt.show()\n\n#Distribution of mean values per row in train and test dataset\nplt.figure(figsize=(16,8))\n#Distribution plot for mean values per row in train attributes\nsns.distplot(train_df[train_attributes].mean(axis=1),color='blue',kde=True,bins=150,label='train')\n#Distribution plot for mean values per row in test attributes\nsns.distplot(test_df[test_attributes].mean(axis=1),color='green',kde=True, bins=150, label='test')\nplt.title('Distribution of mean values per row in train and test dataset')\nplt.legend()\nplt.show()")


# **Let us look distribution of standard deviation(std) values per rows and columns in train and test dataset**

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Distribution of std values per column in train and test dataset\nplt.figure(figsize=(16,8))\n#train attributes\ntrain_attributes=train_df.columns.values[2:202]\n#test attributes\ntest_attributes=test_df.columns.values[1:201]\n#Distribution plot for std values per column in train attributes\nsns.distplot(train_df[train_attributes].std(axis=0),color='red',kde=True,bins=150,label='train')\n#Distribution plot for std values per column in test attributes\nsns.distplot(test_df[test_attributes].std(axis=0),color='blue',kde=True,bins=150,label='test')\nplt.title('Distribution of std values per column in train and test dataset')\nplt.legend()\nplt.show()\n\n#Distribution of std values per row in train and test dataset\nplt.figure(figsize=(16,8))\n#Distribution plot for std values per row in train attributes\nsns.distplot(train_df[train_attributes].std(axis=1),color='red',kde=True,bins=150,label='train')\n#Distribution plot for std values per row in test attributes\nsns.distplot(test_df[test_attributes].std(axis=1),color='blue',kde=True, bins=150, label='test')\nplt.title('Distribution of std values per row in train and test dataset')\nplt.legend()\nplt.show()")


# ****

# **Let us look distribution of skewness per rows and columns in train and test dataset**

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Distribution of skew values per column in train and test dataset\nplt.figure(figsize=(16,8))\n#train attributes\ntrain_attributes=train_df.columns.values[2:202]\n#test attributes\ntest_attributes=test_df.columns.values[1:201]\n#Distribution plot for skew values per column in train attributes\nsns.distplot(train_df[train_attributes].skew(axis=0),color='green',kde=True,bins=150,label='train')\n#Distribution plot for skew values per column in test attributes\nsns.distplot(test_df[test_attributes].skew(axis=0),color='blue',kde=True,bins=150,label='test')\nplt.title('Distribution of skewness values per column in train and test dataset')\nplt.legend()\nplt.show()\n\n#Distribution of skew values per row in train and test dataset\nplt.figure(figsize=(16,8))\n#Distribution plot for skew values per row in train attributes\nsns.distplot(train_df[train_attributes].skew(axis=1),color='green',kde=True,bins=150,label='train')\n#Distribution plot for skew values per row in test attributes\nsns.distplot(test_df[test_attributes].skew(axis=1),color='blue',kde=True, bins=150, label='test')\nplt.title('Distribution of skewness values per row in train and test dataset')\nplt.legend()\nplt.show()")


# **Let us look distribution of kurtosis values per rows and columns in train and test dataset**

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Distribution of kurtosis values per column in train and test dataset\nplt.figure(figsize=(16,8))\n#train attributes\ntrain_attributes=train_df.columns.values[2:202]\n#test attributes\ntest_attributes=test_df.columns.values[1:201]\n#Distribution plot for kurtosis values per column in train attributes\nsns.distplot(train_df[train_attributes].kurtosis(axis=0),color='blue',kde=True,bins=150,label='train')\n#Distribution plot for kurtosis values per column in test attributes\nsns.distplot(test_df[test_attributes].kurtosis(axis=0),color='red',kde=True,bins=150,label='test')\nplt.title('Distribution of kurtosis values per column in train and test dataset')\nplt.legend()\nplt.show()\n\n#Distribution of kutosis values per row in train and test dataset\nplt.figure(figsize=(16,8))\n#Distribution plot for kurtosis values per row in train attributes\nsns.distplot(train_df[train_attributes].kurtosis(axis=1),color='blue',kde=True,bins=150,label='train')\n#Distribution plot for kurtosis values per row in test attributes\nsns.distplot(test_df[test_attributes].kurtosis(axis=1),color='red',kde=True, bins=150, label='test')\nplt.title('Distribution of kurtosis values per row in train and test dataset')\nplt.legend()\nplt.show()")


# **Missing value analysis**

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Finding the missing values in train and test data\ntrain_missing=train_df.isnull().sum().sum()\ntest_missing=test_df.isnull().sum().sum()\nprint('Missing values in train data :',train_missing)\nprint('Missing values in test data :',test_missing)")


# No missing values are present in both train and test data.

# **Correlation between the attributes**

# We can observed that the correlation between the train attributes is very small.

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Correlations in train attributes\ntrain_attributes=train_df.columns.values[2:202]\ntrain_correlations=train_df[train_attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()\ntrain_correlations=train_correlations[train_correlations['level_0']!=train_correlations['level_1']]\nprint(train_correlations.head(10))\nprint(train_correlations.tail(10))")


# We can observed that the correlation between the test attributes is very small.

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Correlations in test attributes\ntest_attributes=test_df.columns.values[1:201]\ntest_correlations=test_df[test_attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()\ntest_correlations=test_correlations[test_correlations['level_0']!=test_correlations['level_1']]\nprint(test_correlations.head(10))\nprint(test_correlations.tail(10))")


# **Correlation plot for train and test data**

# We can observed from correlation distribution plot that the correlation between the train and test attributes is very very small, it means that features are independent each other.

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Correlations in train data\ntrain_correlations=train_df[train_attributes].corr()\ntrain_correlations=train_correlations.values.flatten()\ntrain_correlations=train_correlations[train_correlations!=1]\n#Correlations in test data\ntest_correlations=test_df[test_attributes].corr()\ntest_correlations=test_correlations.values.flatten()\ntest_correlations=test_correlations[test_correlations!=1]\n\nplt.figure(figsize=(20,5))\n#Distribution plot for correlations in train data\nsns.distplot(train_correlations, color="Red", label="train")\n#Distribution plot for correlations in test data\nsns.distplot(test_correlations, color="Blue", label="test")\nplt.xlabel("Correlation values found in train and test")\nplt.ylabel("Density")\nplt.title("Correlation distribution plot for train and test attributes")\nplt.legend()')


# **Feature engineering**

# Let us do some feature engineering by using
# * Permutation importance
# * Partial dependence plots

# **Permutation importance**

# Permutation variable importance measure in a random forest for classification and regression.

# In[ ]:


#training and testing data
X=train_df.drop(columns=['ID_code','target'],axis=1)
test=test_df.drop(columns=['ID_code'],axis=1)
y=train_df['target']


# Let us build simple model to find features which are more important.

# In[ ]:


#Split the training data
X_train,X_valid,y_train,y_valid=train_test_split(X,y,random_state=42)

print('Shape of X_train :',X_train.shape)
print('Shape of X_valid :',X_valid.shape)
print('Shape of y_train :',y_train.shape)
print('Shape of y_valid :',y_valid.shape)


# **Random forest classifier**

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Random forest classifier\nrf_model=RandomForestClassifier(n_estimators=10,random_state=42)\n#fitting the model\nrf_model.fit(X_train,y_train)')


# Let us calculate weights and show important features using eli5 library.

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Permutation importance\nfrom eli5.sklearn import PermutationImportance\nperm_imp=PermutationImportance(rf_model,random_state=42)\n#fitting the model\nperm_imp.fit(X_valid,y_valid)')


# Let us see important features,

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Important features\neli5.show_weights(perm_imp,feature_names=X_valid.columns.tolist(),top=200)')


# Take aways:
# * Importance of the features decreases as we move down the top of the column.
# * As we can see the features shown in green indicate that they have a positive impact on our prediction
# * As we can see the features shown in white indicate that they have no effect on our prediction
# * As we can see the features shown in red indicate that they have a negative impact on our prediction
# * The most important feature is 'Var_81'

# **Partial dependence plots**

# Partial dependence plot gives a graphical depiction of the marginal effect of a variable on the class probability or classification.While feature importance shows what variables most affect predictions, but partial dependence plots show how a feature affects predictions.

# Let us calculate partial dependence plots on random forest

# **Partial dependence plot**

# Let us see impact of the main features which are discovered in the previous section by using the pdpbox.

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Create the data we will plot \'var_53\'\nfeatures=[v for v in X_valid.columns if v not in [\'ID_code\',\'target\']]\npdp_data=pdp.pdp_isolate(rf_model,dataset=X_valid,model_features=features,feature=\'var_53\')\n#plot feature "var_53"\npdp.pdp_plot(pdp_data,\'var_53\')\nplt.show()')


# **Take aways:**
# * The y_axis does not show the predictor value instead how the value changing with  the change in given predictor variable. 
# * The blue shaded area indicates the level of confidence of 'var_53'.
# * On y-axis having a positive value means for that particular value of predictor variable it is less likely to predict the correct class and having a positive value means it has positive impact on predicting the correct class.
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Create the data we will plot \npdp_data=pdp.pdp_isolate(rf_model,dataset=X_valid,model_features=features,feature=\'var_6\')\n#plot feature "var_6"\npdp.pdp_plot(pdp_data,\'var_6\')\nplt.show()')


# **Take aways:**
# * The y_axis does not show the predictor value instead how the value changing with  the change in given predictor variable. 
# * The blue shaded area indicates the level of confidence of 'var_6'.
# * On y-axis having a positive value means for that particular value of predictor variable it is less likely to predict the correct class and having a positive value means it has positive impact on predicting the correct class.
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Create the data we will plot \npdp_data=pdp.pdp_isolate(rf_model,dataset=X_valid,model_features=features,feature=\'var_146\')\n#plot feature "var_146"\npdp.pdp_plot(pdp_data,\'var_146\')\nplt.show()')


# **Take aways:**
# * The y_axis does not show the predictor value instead how the value changing with  the change in given predictor variable. 
# * The blue shaded area indicates the level of confidence of 'var_146'.
# * On y-axis having a positive value means for that particular value of predictor variable it is less likely to predict the correct class and having a positive value means it has positive impact on predicting the correct class.
# 

# **Handling of imbalanced data**
# 
# Now we are going to explore 5 different approaches for dealing with imbalanced datasets.
# * Change the performance metric
# * Oversample minority class
# * Undersample majority class
# * Synthetic Minority Oversampling Technique(SMOTE)
# * Change the algorithm

# Now let us start with simple Logistic regression model.

# **Split the train data using StratefiedKFold cross validator**

# In[ ]:


#Training data
X=train_df.drop(['ID_code','target'],axis=1)
Y=train_df['target']
#StratifiedKFold cross validator
cv=StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
for train_index,valid_index in cv.split(X,Y):
    X_train, X_valid=X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid=Y.iloc[train_index], Y.iloc[valid_index]

print('Shape of X_train :',X_train.shape)
print('Shape of X_valid :',X_valid.shape)
print('Shape of y_train :',y_train.shape)
print('Shape of y_valid :',y_valid.shape)


# **Logistic Regression model**

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Logistic regression model\nlr_model=LogisticRegression(random_state=42)\n#fitting the lr model\nlr_model.fit(X_train,y_train)')


# **Accuracy of model**

# In[ ]:


#Accuracy of the model
lr_score=lr_model.score(X_train,y_train)
print('Accuracy of the lr_model :',lr_score)


# **Cross validation prediction of lr_model**

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Cross validation prediction\ncv_predict=cross_val_predict(lr_model,X_valid,y_valid,cv=5)\n#Cross validation score\ncv_score=cross_val_score(lr_model,X_valid,y_valid,cv=5)\nprint('cross_val_score :',np.average(cv_score))")


# Accuracy of the model is not the best metric to use when evaluating the imbalanced datasets as it may be misleading. So, we are going to change the performance metric.

# **Confusion matrix**

# In[ ]:


#Confusion matrix
cm=confusion_matrix(y_valid,cv_predict)
#Plot the confusion matrix
plot_confusion_matrix(y_valid,cv_predict,normalize=False,figsize=(15,8))


# **Reciever operating characteristics (ROC)-Area under curve(AUC) score and curve**

# In[ ]:


#ROC_AUC score
roc_score=roc_auc_score(y_valid,cv_predict)
print('ROC score :',roc_score)

#ROC_AUC curve
plt.figure()
false_positive_rate,recall,thresholds=roc_curve(y_valid,cv_predict)
roc_auc=auc(false_positive_rate,recall)
plt.title('Reciver Operating Characteristics(ROC)')
plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall(True Positive Rate)')
plt.xlabel('False Positive Rate')
plt.show()
print('AUC:',roc_auc)


# When we compare the roc_auc_score and model accuracy , model is not performing well on imbalanced data.

# **Classification report**

# In[ ]:


#Classification report
scores=classification_report(y_valid,cv_predict)
print(scores)


# We can observed that f1 score is high for number of customers those who will not make a transaction then the who will make a transaction. So, we are going to change the algorithm.

# **Model performance on test data**

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Predicting the model\nX_test=test_df.drop(['ID_code'],axis=1)\nlr_pred=lr_model.predict(X_test)\nprint(lr_pred)")


# **Oversample minority class:**
# * It can be defined as adding more copies of minority class.
# * It can be a good choice when we don't have a ton of data to work with.
# * Drawback is that we are adding information.This may leads to overfitting and poor performance on test data.
# 

# **Undersample majority class:**
# * It can be defined as removing some observations of the majority class.
# * It can be a good choice when we have a ton of data -think million of rows.
# * Drawback is that we are removing information that may be valuable.This may leads to underfitting and poor performance on test data.

# Both Oversampling and undersampling techniques have some drawbacks. So, we are not going to use this models for this problem and also we will use other best algorithms.

# **Synthetic Minority Oversampling Technique(SMOTE)**

# SMOTE uses a nearest neighbors algorithm to generate new and synthetic data to used for training the model.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from imblearn.over_sampling import SMOTE\n#Synthetic Minority Oversampling Technique\nsm = SMOTE(random_state=42, ratio=1.0)\n#Generating synthetic data points\nX_smote,y_smote=sm.fit_sample(X_train,y_train)\nX_smote_v,y_smote_v=sm.fit_sample(X_valid,y_valid)')


# Let us see how baseline logistic regression model performs on synthetic data points.

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Logistic regression model for SMOTE\nsmote=LogisticRegression(random_state=42)\n#fitting the smote model\nsmote.fit(X_smote,y_smote)')


# **Accuracy of model**

# In[ ]:


#Accuracy of the model
smote_score=smote.score(X_smote,y_smote)
print('Accuracy of the smote_model :',smote_score)


# Cross validation prediction of smoth_model

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Cross validation prediction\ncv_pred=cross_val_predict(smote,X_smote_v,y_smote_v,cv=5)\n#Cross validation score\ncv_score=cross_val_score(smote,X_smote_v,y_smote_v,cv=5)\nprint('cross_val_score :',np.average(cv_score))")


# **Confusion matrix**

# In[ ]:


#Confusion matrix
cm=confusion_matrix(y_smote_v,cv_pred)
#Plot the confusion matrix
plot_confusion_matrix(y_smote_v,cv_pred,normalize=False,figsize=(15,8))


# **Reciever operating characteristics (ROC)-Area under curve(AUC) score and curve**

# In[ ]:


#ROC_AUC score
roc_score=roc_auc_score(y_smote_v,cv_pred)
print('ROC score :',roc_score)

#ROC_AUC curve
plt.figure()
false_positive_rate,recall,thresholds=roc_curve(y_smote_v,cv_pred)
roc_auc=auc(false_positive_rate,recall)
plt.title('Reciver Operating Characteristics(ROC)')
plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall(True Positive Rate)')
plt.xlabel('False Positive Rate')
plt.show()
print('AUC:',roc_auc)


# **Classification report**

# In[ ]:


#Classification report
scores=classification_report(y_smote_v,cv_pred)
print(scores)


# **Model performance on test data**

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Predicting the model\nX_test=test_df.drop(['ID_code'],axis=1)\nsmote_pred=smote.predict(X_test)\nprint(smote_pred)")


# We can observed that smote model is performing well on imbalance data compare to logistic regression.

# **LightGBM:**
# 
# LightGBM is a gradient boosting framework that uses tree based learning algorithms. We are going to use LightGBM model.
# 

# Let us build LightGBM model

# In[ ]:


#Training the model
#training data
lgb_train=lgb.Dataset(X_train,label=y_train)
#validation data
lgb_valid=lgb.Dataset(X_valid,label=y_valid)


# **choosing of  hyperparameters**

# In[ ]:


#Selecting best hyperparameters by tuning of different parameters
params={'boosting_type': 'gbdt', 
          'max_depth' : -1, #no limit for max_depth if <0
          'objective': 'binary',
          'boost_from_average':False, 
          'nthread': 20,
          'metric':'auc',
          'num_leaves': 50,
          'learning_rate': 0.01,
          'max_bin': 100,      #default 255
          'subsample_for_bin': 100,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'bagging_fraction':0.5,
          'bagging_freq':5,
          'feature_fraction':0.08,
          'min_split_gain': 0.45, #>0
          'min_child_weight': 1,
          'min_child_samples': 5,
          'is_unbalance':True,
          }


# **Training the lgbm model**

# In[ ]:


num_rounds=10000
lgbm= lgb.train(params,lgb_train,num_rounds,valid_sets=[lgb_train,lgb_valid],verbose_eval=1000,early_stopping_rounds = 5000)
lgbm


# **lgbm model performance on test data**

# In[ ]:


X_test=test_df.drop(['ID_code'],axis=1)
#predict the model
#probability predictions
lgbm_predict_prob=lgbm.predict(X_test,random_state=42,num_iteration=lgbm.best_iteration)
#Convert to binary output 1 or 0
lgbm_predict=np.where(lgbm_predict_prob>=0.5,1,0)
print(lgbm_predict_prob)
print(lgbm_predict)


# **Let us plot the important features**

# In[ ]:


#plot the important features
lgb.plot_importance(lgbm,max_num_features=50,importance_type="split",figsize=(20,50))


# **Conclusion :**
# 
# We tried model with logistic regression,smote and lightgbm. But lightgbm model is performing well on imbalanced data compared to other models based on scores of roc_auc_score.

# In[ ]:


#final submission
sub_df=pd.DataFrame({'ID_code':test_df['ID_code'].values})
sub_df['lgbm_predict_prob']=lgbm_predict_prob
sub_df['lgbm_predict']=lgbm_predict
sub_df.to_csv('submission.csv',index=False)
sub_df.head()

