#!/usr/bin/env python
# coding: utf-8

# # Learning the Process for Machine Learning: First Kaggle Competition, Titanic
# 
# I will be learning the process of machine learning from beginning to end by preprocessing data, building machine learning models, and submitting my output/predictions to my first Kaggle competition, Titanic. In addition, this will give me the opportunity to work with new libraries and troubleshoot preprocessing issues I may encounter in the future to build parismonious machine learning models. 
# 
# Conceptually, the first thing I will need to do is to explore the data in my train set, which involves understanding the format of the variables, distributions, and interactions. Then, given that the output of the machine learning model should be 0/1, preparing my features to be inserted into a logistic regression classifier. Finally, tweaking the feature parameters as well as inclussion given bias/variance and performance.

# In[ ]:


import numpy as np
import pandas as pd 

import os
print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import missingno
import seaborn as sbrn

from sklearn.preprocessing import OneHotEncoder, label_binarize

import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, datasets, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier, Pool, cv

from IPython.display import HTML
import base64


# ## Import Datasets
# The first step is to import my datasets and save them as variables in the memory. I will use the pandas library to exectue this.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/gender_submission.csv')


# ## Exploring the Train Dataset
# Now, we want to just preview what the dataset looks like.

# In[ ]:


train.head(15)


# In[ ]:


len(train)


# The variables in the training dataset are: Passenger ID, Survived(0/1), P Class (ticket class, 1st-2nd-3rd, SES proxy), Name, Sex, Age, Sibling onboard, Parch (children on board), Ticket, Fare, Cabin and Port of Emarkment. More notes on the variables can be found on the Kaggle competition page.
# 
# Survived(0/1) is our label for each training example. And we have 891 examples to train our model on. 
# 
# We now will ask for some basic descriptions of our variables in the dataset using '.describe()'. 

# In[ ]:


train.describe()


# From this, we can see that we have full data for all of the examples except for age where roughly 177 examples do not have an age. Although it's not necessary to pinpoint exactly the percentage, but it seems that between 50% to 75% of people did not survive. Most people in the dataset were of lower SES based on their ticket class. Most people did not have a sibling onboard or children with them. And most people paid less than 31 for their fare.

# In[ ]:


train.Age.plot.hist()


# This histogram for age makes sense. Younger people were probably more excited to try out new technologies and be a part of a big event which is why we see that spike at around 18 years old. The low numbers under the age of 18 make sense since minors probably couldn't go unaccompanied and parents may have not wanted them on the trip.
# 
# **** We might want to create a feature that lets us know whether they are children, adults, or seniors. We should explore the variable a little more to decide if these groupings could be helpful.**

# In[ ]:


train.Pclass.plot.hist()


# From the histogram for ticket class, we see that roughly half were in the 3rd class, a quarter in the 2nd class, and a quarter in the 1st class. 

# In[ ]:


train.Fare.plot.hist()


# From the histogram for fare, we see that the vast majority of people paid less than 100 for their fare. 

# In[ ]:


missingno.matrix(train)


# The two variables with the most missing data are Age and Cabin. Cabin seems pretty sparse. 

# In[ ]:


train.isnull().sum()


# Age, as calculate above, has 177 missing, while Cabin has 687 missing. 

# In[ ]:


train.dtypes


# From looking at the variable types, we see which variables are continuous/numerical and which are categorical. Pclass although an integer, probably makes more sense as a categorical variable. Name and Sex are categorical variables. Age is a continuous (float) variable since children under 1 are coded with one decimal place (e.g. 0.5). SibSp and Parch are continuous but could also make more sense as a categorical variable. Ticket is categorical, but not sure what can be done with this variable yet since I assume the tickets may not follow a pattern thus being unique. Lastly, Cabin and Port of Embarkement are categorical, which makes sense in its current form.

# ## Target Feauture
# Now, we'll want to look at survivorshop of our training data set.

# In[ ]:


sbrn.countplot(y='Survived', data=train);
print(train.Survived.value_counts())


# From the quartiles that we saw above, I estimated that 50% to 75% of the training examples did not survive. Here we see 342 people did survive and 549 people did not survive. Roughly 61% of the examples in the training dataset did not survive.
# 
# ## Adding Non-missing Variables to Dataframes: PClass, Sex

# In[ ]:


#Separating our variables into categorical and continuous variables within dataframes, makes exploration easier

df_cat = pd.DataFrame()
df_con = pd.DataFrame()


# In[ ]:


#To explore the variables as related to the outcome of survival, we'll add the outcome to the empty dataframes

df_cat['Survived'] = train['Survived']
df_con['Survived'] = train['Survived']

#Adding Pclass to the dataframes since no data is missing for that variable

df_cat['Pclass'] = train['Pclass']
df_con['Pclass'] = train['Pclass']

#Adding Sex to the dataframes since no data is missing for that variable

df_cat['Sex'] = train['Sex']
#This next line of code will recode the variables into 0/1; female = 1, male = 0. Male = 0 was decided since the idea of "women and children first" would probably lead to higher survival for 'female'
df_cat['Sex'] = np.where(df_cat['Sex'] == 'male', 1, 0)
df_con['Sex'] = train['Sex']


# My assumption that women were more likely to survive than men can be confirmed by exploring Sex by survival status.

# In[ ]:


train.groupby(['Sex', 'Survived']).count()


# As you can see here, this confirms my inkling about Sex differences in survival. 74% of women survived while only 19% of men survived. This will be important to keep note of as from just the Sex variable, we should predict women to survive more often than men. 

# ## Features: SibSp 

# In[ ]:


train.SibSp.value_counts()


# Given the distribution, I might group together 2-5 siblings onboard together. Let's see how it might relate to survival.

# In[ ]:


train.groupby(['SibSp', 'Survived']).count()


# From visual inspection, it seems that those who had 2 or more siblings onboard, likely didn't survive. This cutoff of "0, 1, 2+" seems reasonabl given the differences between the groups. 

# In[ ]:


df_cat['SibSp'] = train['SibSp']
# We will rewrite them as categories where 0=0, 1=1, 2-8=2
df_cat['SibSp'] = pd.cut(train['SibSp'], [0,1,2,8], labels=['0','1','2+'], right=False)

df_con['SibSp'] = train['SibSp']


# ## Features: Parch
# We'll do a similar exploration and categorization as SibSp.

# In[ ]:


train.Parch.value_counts()


# In[ ]:


train.groupby(['Parch', 'Survived']).count()


# Parch of 2 and above have different surival dynamics from 0 and 1, so I might group them together. It seems these might be related to SibSp. If they are, we may need an interaction feature.

# In[ ]:


train.groupby(['Parch', 'SibSp']).count()


# I'm going to group Parch as 0, 1, 2, 3+ since 3-6 have similar relationships. 

# In[ ]:


df_cat['Parch'] = train['Parch']
# We will rewrite them as categories where 0=0, 1=1, 2=2, 3=3-6
df_cat['Parch'] = pd.cut(train['Parch'], [0,1,2,3,6], labels=['0','1','2','3+'], right=False)

df_con['Parch'] = train['Parch']


# ## Features: Tickets

# In[ ]:


train.Ticket.value_counts()


# This is a weird variable and I'm not sure what to do with it. I'll skip over this variable for now.

# ## Features: Fare
# 
# Given the distribution we saw, we might scale the feature. I found a cool new method to deal with very skewed, odd data. It's called the Box-Cox transformation. 

# In[ ]:


from scipy.stats import boxcox

df_cat['Fare'] = train['Fare'] + 0.1
df_cat['Fare'] = boxcox(df_cat.Fare)[0]
df_cat['Fare'] = pd.cut(df_cat['Fare'], bins=5)

df_con['Fare'] = train['Fare']


# ## Features: Cabin
# Since there are a lot of missing values for Cabin, I might create a new feature that is 0/1 to determine whether or not it's there.

# In[ ]:


train['Cabin'].head(15)


# In[ ]:


df_cat['Cabin'] = train['Cabin']
df_cat['Cabin'] = df_cat['Cabin'].replace(np.nan, '0')

df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*A+.*', '1')
df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*B+.*', '1')
df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*C+.*', '1')
df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*D+.*', '1')
df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*E+.*', '1')
df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*F+.*', '1')
df_cat['Cabin'] = df_cat['Cabin'].str.replace('.*G+.*', '1')


# Although this is not the most elegant coding, I wanted to code those with Cabin information available to be coded as 1 and those without to be coded as 0. 

# ## Features: Embarkment
# 
# From previous views, there are a couple of missing values for Embarkment. Let's see what we can do about it. 

# In[ ]:


train[train.Embarked.isna()]


# I might impute those two values if I can find a relationship with maybe the Cabin or Ticket. The two missing Embarkment examples seem to be related to each other since they have the same ticket number, fare and cabin.

# In[ ]:


train.groupby(['Embarked',train['Cabin'].str.contains("B", na=False)]).count()


# Since there are only two missing Cabin information, I'm going to impute them with C simply because it is more likely they embarked from Southhampton than Cherbourg. 

# In[ ]:


df_cat['Embarked'] = train['Embarked']
df_cat.set_value(61, 'Embarked', 'S')
df_cat.set_value(829, 'Embarked', 'S')

df_con['Embarked'] = df_cat['Embarked']


# ## Preparing Data Set for Training with Feature Encoding

# We're going to now 0/1 encode the features in the df_cat dataframe. 

# In[ ]:


df_cat.head()


# In[ ]:


# One-Hot Encoding employed
onehot_cols = df_cat.columns.tolist()
onehot_cols.remove('Survived')
df_cat_enc = pd.get_dummies(df_cat, columns=onehot_cols)

df_cat_enc.head(8)


# In[ ]:


df_con.head(15)


# We'll now encode the categorical variables in the continuous dataset, and remove the original variables. 

# In[ ]:


df_embarked_onehot = pd.get_dummies(df_con['Embarked'], 
                                     prefix='embarked')

df_sex_onehot = pd.get_dummies(df_con['Sex'], 
                                prefix='sex')

df_plcass_onehot = pd.get_dummies(df_con['Pclass'], 
                                   prefix='pclass')


# In[ ]:


# Now lets remove the original variables
df_con_enc = pd.concat([df_con, 
                        df_embarked_onehot, 
                        df_sex_onehot, 
                        df_plcass_onehot], axis=1)

# Drop the original categorical columns (because now they've been one hot encoded)
df_con_enc = df_con_enc.drop(['Pclass', 'Sex', 'Embarked'], axis=1)


# In[ ]:


df_con_enc.head(9)


# ## Testing Machine Learning Methods Over Clean and Manipulated Datasets
# 
# I suspect that a logistic regression would probably do just fine, but let's find out!

# In[ ]:


chosen_df = df_con_enc


# We're going to split the dataframe into the target feature and the independent features. 

# In[ ]:


X_train = chosen_df.drop('Survived', axis=1)
y_train = chosen_df.Survived


# This next section, I used a lot of Google searching and helping to figure out how to execute what I wanted to execute. I want to first try the methods I have picked up from my Machine Learning course with Andrew Ng: Logistic Regression, K-Nearest Neighbor, Linear Support Vector Machines, and Stochastic Gradient Descent. There are some other methods that I am not familiar with but will test them out. I'll explain the pros and cons of each method. 
# 
# I'm going to borrow some code that will streamline the information I want from testing my models. It takes as inputs the mode you wan tot use, your X_train values, your y_train values and cross validation proportion. 

# In[ ]:


def fit_ml_algo(algo, X_train, y_train, cv):
    
    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    
    # Cross Validation 
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs = -1)
    
    
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    confus_matr = metrics.confusion_matrix(y_train, train_pred)
    precision = precision_score(y_train, train_pred)
    recall = recall_score(y_train, train_pred)
    F1 = f1_score(y_train, train_pred)
    auc = roc_auc_score(y_train, train_pred)
    
    return train_pred, acc, acc_cv, confus_matr, precision, recall, F1, auc


# ## Logistic Regression
# 
# **Pros: easy to understand; when features and problem is pretty linear; robust to noise; efficient; can avoid overfitting
# 
# **Cons: doesn't do well with categorical variables; multicollinearity can be a problem

# In[ ]:


train_pred_log, acc_log, acc_cv_log, cfmt_log, prec_log, rec_log, F1_log, auc_log = fit_ml_algo(LogisticRegression(), 
                                                                                               X_train, 
                                                                                               y_train, 
                                                                                                10)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Confusion Matrix: %s" % cfmt_log)
print("Precision: ", float("{0:.3f}".format(round(prec_log, 3))))
print("Recall: ", float("{0:.3f}".format(round(rec_log, 3))))
print("F1 Score: ", float("{0:.3f}".format(round(F1_log, 3))))
print("AUC: ", float("{0:.3f}".format(round(auc_log, 3))))


# ## K-Nearest Neighbor (KNN)
# 
# **Pros: no assumption about data; simple; relative high accuracy; versatile
# 
# **Cons: computationally heavy; high memory usage; stores the training data; prediction might be slow; sensitive to unneeeded features and scale

# In[ ]:


train_pred_knn, acc_knn, acc_cv_knn, cfmt_knn, prec_knn, rec_knn, F1_knn, auc_knn = fit_ml_algo(KNeighborsClassifier(), 
                                                                                                X_train, 
                                                                                                y_train, 
                                                                                                10)
print("Accuracy: %s" % acc_knn)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print("Confusion Matrix: %s" % cfmt_knn)
print("Precision: ", float("{0:.3f}".format(round(prec_knn, 3))))
print("Recall: ", float("{0:.3f}".format(round(rec_knn, 3))))
print("F1 Score: ", float("{0:.3f}".format(round(F1_knn, 3))))
print("AUC: ", float("{0:.3f}".format(round(auc_knn, 3))))


# ## Linear Support Vector Machine (SVM)
# 
# **Pros: good with high-dimensional space; good for non-linear problems; high accuracy; flexible; multicollinearity is not an issue
# 
# **Cons: hard to interpret; inefficient to train; better for smaller problems; not meant for massive sets

# In[ ]:


train_pred_svc, acc_linear_svc, acc_cv_linear_svc, cfmt_svc, prec_svc, rec_svc, F1_svc, auc_svc = fit_ml_algo(LinearSVC(),
                                                                                                                X_train, 
                                                                                                                y_train, 
                                                                                                                10)
print("Accuracy: %s" % acc_linear_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
print("Confusion Matrix: %s" % cfmt_svc)
print("Precision: ", float("{0:.3f}".format(round(prec_svc, 3))))
print("Recall: ", float("{0:.3f}".format(round(rec_svc, 3))))
print("F1 Score: ", float("{0:.3f}".format(round(F1_svc, 3))))
print("AUC: ", float("{0:.3f}".format(round(auc_svc, 3))))


# ## Stochastic Gradient Descent
# 
# **Pros: updates parameters example by example; gets to minimum relatively quickly; not as computationally heavy; 
# 
# **Cons: variance can be an issue

# In[ ]:


train_pred_sgd, acc_sgd, acc_cv_sgd, cfmt_sgd, prec_sgd, rec_sgd, F1_sgd, auc_sgd = fit_ml_algo(SGDClassifier(), 
                                                                                                  X_train, 
                                                                                                  y_train,
                                                                                                  10)
print("Accuracy: %s" % acc_sgd)
print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)
print("Confusion Matrix: %s" % cfmt_sgd)
print("Precision: ", float("{0:.3f}".format(round(prec_sgd, 3))))
print("Recall: ", float("{0:.3f}".format(round(rec_sgd, 3))))
print("F1 Score: ", float("{0:.3f}".format(round(F1_sgd, 3))))
print("AUC: ", float("{0:.3f}".format(round(auc_sgd, 3))))


# ## Gaussian Naive Bayes
# 
# **Pros: very simple; requires less training data; converges quicker; few features
# 
# **Cons: multicollinearity can be an issue

# In[ ]:


train_pred_gaussian, acc_gaussian, acc_cv_gaussian, cfmt_gaussian, prec_gaussian, rec_gaussian, F1_gaussian, auc_gaussian = fit_ml_algo(GaussianNB(), 
                                                                                                              X_train, 
                                                                                                              y_train, 
                                                                                                               10)
print("Accuracy: %s" % acc_gaussian)
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)
print("Confusion Matrix: %s" % cfmt_gaussian)
print("Precision: ", float("{0:.3f}".format(round(prec_gaussian, 3))))
print("Recall: ", float("{0:.3f}".format(round(rec_gaussian, 3))))
print("F1 Score: ", float("{0:.3f}".format(round(F1_gaussian, 3))))
print("AUC: ", float("{0:.3f}".format(round(auc_gaussian, 3))))


# ## Decision Tree
# 
# **Pros: east to interpet; non-parametric; good for few categorical variables
# 
# **Cons: easily overfits

# In[ ]:


train_pred_dt, acc_dt, acc_cv_dt, cfmt_dt, prec_dt, rec_dt, F1_dt, auc_dt = fit_ml_algo(DecisionTreeClassifier(), 
                                                                                        X_train, 
                                                                                        y_train,
                                                                                        10)
print("Accuracy: %s" % acc_dt)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
print("Confusion Matrix: %s" % cfmt_dt)
print("Precision: ", float("{0:.3f}".format(round(prec_dt, 3))))
print("Recall: ", float("{0:.3f}".format(round(rec_dt, 3))))
print("F1 Score: ", float("{0:.3f}".format(round(F1_dt, 3))))
print("AUC: ", float("{0:.3f}".format(round(auc_dt, 3))))


# ## Gradient Boost Trees
# 
# **Pros: since it builds a tree at a time, errors correct over time; perform better than Random Forest
# 
# **Cons: prone ot overfitting; parameters are harder to 'tune'; training takes much longer

# In[ ]:


train_pred_gbt, acc_gbt, acc_cv_gbt, cfmt_gbt, prec_gbt, rec_gbt, F1_gbt, auc_gbt = fit_ml_algo(GradientBoostingClassifier(), 
                                                                                               X_train, 
                                                                                               y_train,
                                                                                               10)
print("Accuracy: %s" % acc_gbt)
print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)
print("Confusion Matrix: %s" % cfmt_gbt)
print("Precision: ", float("{0:.3f}".format(round(prec_gbt, 3))))
print("Recall: ", float("{0:.3f}".format(round(rec_gbt, 3))))
print("F1 Score: ", float("{0:.3f}".format(round(F1_gbt, 3))))
print("AUC: ", float("{0:.3f}".format(round(auc_gbt, 3))))


# ## Catboost Algorithm
# 
# This seems to be a new one that deals with categorical variables really well.
# 
# **Cons: takes a while to run

# In[ ]:


## Remove non-categorical variables
cat_features = np.where(X_train.dtypes != np.float)[0]


# ### Pool() will combine our values from the training data and categories together. 

# In[ ]:


train_pool = Pool(X_train, 
                  y_train,
                  cat_features)


# In[ ]:


# CatBoost model 
catboost_model = CatBoostClassifier(iterations=1000,
                                    custom_loss=['Accuracy'],
                                    loss_function='Logloss')

# Fit CatBoost model
catboost_model.fit(train_pool,
                   plot=False)

# CatBoost accuracy
acc_catboost = round(catboost_model.score(X_train, y_train) * 100, 2)


# In[ ]:


# Set params for cross-validation
cv_params = catboost_model.get_params()

# Run the cross-validation for 10-folds
cv_data = cv(train_pool,
             cv_params,
             fold_count=10,
             plot=False)


# CatBoost CV results save into a dataframe (cv_data), let's withdraw the maximum accuracy score
acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)


# In[ ]:


# Print out the CatBoost model metrics
print("---CatBoost Metrics---")
print("Accuracy: {}".format(acc_catboost))
print("Accuracy cross-validation 10-Fold: {}".format(acc_cv_catboost))


# ## Model Results

# In[ ]:


models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees',
              'CatBoost'],
    'Score': [
        acc_knn, 
        acc_log,  
        acc_gaussian, 
        acc_sgd, 
        acc_linear_svc, 
        acc_dt,
        acc_gbt,
        acc_catboost
    ]})
print("---Reuglar Accuracy Scores---")
models.sort_values(by='Score', ascending=False)


# In[ ]:


cv_models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees',
              'CatBoost'],
    'Score': [
        acc_cv_knn, 
        acc_cv_log,      
        acc_cv_gaussian, 
        acc_cv_sgd, 
        acc_cv_linear_svc, 
        acc_cv_dt,
        acc_cv_gbt,
        acc_cv_catboost
    ]})
print('---Cross-validation Accuracy Scores---')
cv_models.sort_values(by='Score', ascending=False)


# From these results, I'd probably pick Gradient Boosting Trees or CatBoost. 
# 
# ## Feature Importance
# 
# This will allow us to evaluate how much features are contributing to the model and whether or not we can remove some to reduce our dimensionality.

# ### Gradient Boost Tree

# In[ ]:


model_gbt = GradientBoostingClassifier()
model_gbt.fit(X_train, y_train)
# plot feature importance
print(model_gbt.feature_importances_)
plt.bar(range(len(model_gbt.feature_importances_)), model_gbt.feature_importances_)
plt.show()


# From this plot, we see that Fare, Sex, and PClass seem to be significant features.
# 
# SibSp, Parch and Embarkment can probably be removed. 

# ### CatBoost
# 
# I borrowed code to evaluated the performance of the CatBoost

# In[ ]:


def feature_importance(model, data):
    """
    Function to show which features are most important in the model.
    ::param_model:: Which model to use?
    ::param_data:: What data to use?
    """
    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns})
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
    return fea_imp


# In[ ]:


feature_importance(catboost_model, X_train)


# In[ ]:



metrics = ['Precision', 'Recall', 'F1', 'AUC']

eval_metrics = catboost_model.eval_metrics(train_pool,
                                           metrics=metrics,
                                           plot=False)

for metric in metrics:
    print(str(metric)+": {}".format(np.mean(eval_metrics[metric])))


# Given the performance of my Gradient Boosting Trees (AUC=79%) and CatBoost (AUC=88%), I'm going to go with CatBoost as my final model for predicting on the test set.

# ## Submitting Final Predictions from GBT and CatBoost
# 
# We'll need to transform the Test dataset to look like our Train dataset.

# In[ ]:


test_embarked_onehot = pd.get_dummies(test['Embarked'], 
                                     prefix='embarked')

test_sex_onehot = pd.get_dummies(test['Sex'], 
                                prefix='sex')

test_plcass_onehot = pd.get_dummies(test['Pclass'], 
                                   prefix='pclass')


# In[ ]:


# Now lets remove the original variables
test = pd.concat([test, 
                        test_embarked_onehot, 
                        test_sex_onehot, 
                        test_plcass_onehot], axis=1)


# In[ ]:


test.head(3)


# Pull out just the columns that we used for the training set.

# In[ ]:


want_test_colum = X_train.columns
want_test_colum


# ### Creating CatBoost Prediction CSV

# In[ ]:


def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# In[ ]:


predict_test_cb = catboost_model.predict(test[want_test_colum])


# In[ ]:


submission_catboost = pd.DataFrame()
submission_catboost['PassengerId'] = test['PassengerId']
submission_catboost['Survived'] = predict_test_cb
submission_catboost['Survived'] = submission_catboost['Survived'].astype(int)
submission_catboost.head()


# In[ ]:


create_download_link(submission_catboost)


# ### Creating GBT Prediction CSV

# GBT doesn't handle missing data well, so I might need to impute data for the row missing a Fare value (which is odd). 

# In[ ]:


test[test.Fare.isna()]


# Given his Pclass, I might see if I can impute from the relationship between Pclass and Fare. 

# In[ ]:


df_con.groupby(['Pclass', 'Fare']).count()


# From this, I'm going to imput the average value between 16.1000 and 39.6875, since it seems like that majority of the Pclass are between there, which is 55.7875.

# In[ ]:


test.set_value(152, 'Fare', '55.7875')
test.iloc[152]


# In[ ]:


test[want_test_colum].isna().sum()


# In[ ]:


predict_test_gbt = model_gbt.predict(test[want_test_colum])


# In[ ]:


submission_gbt = pd.DataFrame()
submission_gbt['PassengerId'] = test['PassengerId']
submission_gbt['Survived'] = predict_test_gbt
submission_gbt['Survived'] = submission_gbt['Survived'].astype(int)
submission_gbt.head()


# In[ ]:


create_download_link(submission_gbt)

