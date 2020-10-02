#!/usr/bin/env python
# coding: utf-8

# **Beginning with Kaggle**
# 
# [ Reference: https://www.kaggle.com/niklasdonges/end-to-end-project-with-python#Random-Forest]
# 
# 
# 
# 

# # 1.  Objectives
# 
# ## 1.1 Learning to code ML project 
# Entering some Kaggle competitions may be a good idea to get started. Things I am looking to learn: 
# 
# 1. More systematic use of iPython notebooks
# 2. Using pandas for python 
# 3. Data visualization
# 4. Creating and running a model using existing frameworks 
# 5. Creating and running a model from scratch (extended objective)
# 
# ## 1.2 Learning how Kaggle works
# Well, I am completely new to everything here .. except with some background in private Jupyter notebooks and python and numpy. 
# 
# 1. Learn my way around
# 2. How to write a good Kernel
# 3. How to import datasets, train and submit results
# 

# # 2. Getting Started 
# 
# ## 2.1 Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale , normalize
from sklearn.model_selection import train_test_split , StratifiedKFold
#from sklearn.feature_selection import RFECV


# In[ ]:


# Helper functions 
# Reference: https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 8 , 7 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : 0.5}, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 11 }
    )


#  

# ## 2.2 Import Data
# 
# 

# In[ ]:



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


titanic_train=pd.read_csv('../input/train.csv')
titanic_test=pd.read_csv('../input/test.csv')


# What did we just import? 

# In[ ]:


print('Size of training data:  rows:{}, cols:{}'.format(titanic_train.shape[0], titanic_train.shape[1]))
print('Size of test data:      rows:{}, cols:{}'.format(titanic_test.shape[0], titanic_test.shape[1]))
print('')


# Why do we have one less column in test?
# 
# 

# In[ ]:


titanic_train_header = list(titanic_train)
titanic_test_header = list(titanic_test)

print('Train Header:', titanic_train_header)
print('Difference between train and test headers: ', [item for item in titanic_train_header if item not in titanic_test_header ])


# Obviously, the missing field was the test result. It exists in train data because that acts as the ground truth. 
# 
# Okay, so what does the data look like?  

# In[ ]:


titanic_train.head()


# Randomly sample some data
# 

# In[ ]:


titanic_train.sample(10)


# **Do we need to know what headers mean?**
# To some extent - yes. The headers can be used to learn which features to use. For instance, it is likely that Name did not have significant influence on the survival rate? However, Sex is more likely to have had an influence. The same is true for Fare and Cabin information.  
# 
# First analysis, non statistical - this can later be re-analyzed based on correlation between survival chance. 
# 
# *ID:*  
#   * PassengerId: Not feature
# 
# *Ground Truth:*   
#   * Survived: 
# 
# *Feature: *
#   1. Pclass:   (Likely that class has an impact on survival rate)
#   2. Sex
#   3. Age
#   4. SibSp : Siblings or spouces on board
#   5. Parch: Parents/children on board
#   6. Fare
#   7. Cabin: Having a cabin or not is important 
#   8. Embarked: Port of embarkation
# 
# *Further Engineering Maybe required*
#   1. Name: could be a feature as it may represent wealth or background in some cases. We should try this if result independent of this is not satisfactory
#   2. Ticket: Ticket number may have some clues .. ?
# 

# ## 2.3 Analyzing the data  
# 
# Statistical Characteristic: 
# This is important, because you can use this information to normalize the data, and also get insight into the data

# In[ ]:


titanic_train.describe()


# Sex is not listed here, because it is a string field. Let us try to make this a binary field

# In[ ]:


# Check how many unique values exist
titanic_train['Sex'].unique()


# In[ ]:


titanic_train["SexBin"] = (titanic_train.Sex == 'female').astype(int)
titanic_train.describe()


# **Correlation between different features**
# 
# The following code draws a nice correlation plot. Ideally, this should go into a helper function. 

# In[ ]:


corr = titanic_train.corr()
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
_ = sns.heatmap(corr, 
            cmap=cmap, 
            annot=True)


# Plotting again using the helper function. 

# In[ ]:


plot_correlation_map(titanic_train)


# Plotting Distribution ( https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial) 

# In[ ]:


#plot_distribution( titanic_train , var = 'Age' , target = 'Survived' , row = 'Sex' )
facet = sns.FacetGrid(titanic_train, col="Sex", row='Survived')
#_ = facet.map(sns.kdeplot , 'Age' , shade= True )
facet.map(plt.hist, "Age", density=True)


# Survival rate of females seems to be higher as expected from the correlation plot

# **Pandas Dataframe Query** 
# There will be better ways to do this, but for now, I split the database into male and female to analyze the correlation matrix. 

# In[ ]:


df_female = titanic_train.query('SexBin == 1')
plot_correlation_map(df_female)


# In[ ]:


df_male = titanic_train.query('SexBin == 0')
plot_correlation_map(df_male)


# **Embark Location**

# In[ ]:


# Plot survival rate by Embarked
plot_categories( titanic_train , cat = 'Embarked' , target = 'Survived' )
plot_categories( df_male , cat = 'Embarked' , target = 'Survived' )
plot_categories( df_female , cat = 'Embarked' , target = 'Survived' )


# It appears that embarking location has more correlation with Male passengers than with female passengers
# 
# **Pclass**

# In[ ]:


plot_categories( titanic_train , cat = 'Pclass' , target = 'Survived' )
plot_categories( df_male , cat = 'Pclass' , target = 'Survived' )
plot_categories( df_female , cat = 'Pclass' , target = 'Survived' )


# # 2.3 Data Cleanup & Feature Definition
# 
# Lets find if there is any missing data and if it is an error. 
# Also, convert non-numeric data into numeric. 
# 
# Note that this will be done for both training and test data because we need to perform similar operations on both data sets. We did not use test data for analysis. 
# 

# In[ ]:


full_data = pd.concat([titanic_train.drop('Survived', axis=1).drop('SexBin', axis=1), titanic_test], axis = 0, sort=False) 
full_data.shape
full_data.sample(5)


# ### 2.3.1 Sex - Numeric value
# 
# New series with numerical value of "sex"

# In[ ]:


sex = (full_data.Sex == 'female').astype(int) # Male = 0, Female = 1


# ### 2.3.2 Embarked - numeric value dataframe
# New dataframe with classess assigned to embarked .. Can this instead be a single feature with 3 possible values?

# In[ ]:


# Create a new variable for every unique value of Embarked
#embarked = pd.get_dummies( full_data.Embarked , prefix='Embarked' ).idxmax(1)
embarked_list = full_data.Embarked.unique()
embarked_dict = pd.Series(range(len(embarked_list)), embarked_list)
print(embarked_dict)
embarked = pd.get_dummies( full_data.Embarked).idxmax(1)
embarked_id = embarked.map(embarked_dict).rename('Embarked').to_frame()
embarked.head()


# ### 2.3.3  Cabin - Use as category

# Many values of Cabin are undefined. This may be passengers who had no cabin. We can fill all unassigned values with 'U'

# In[ ]:


cabin_num = full_data.Cabin.fillna('U')
cabin_num.head()


# We will take the first letter from Cabin number and convert that into a feature. It turns out there are 8 different classess for cabins. 

# In[ ]:


cabin  = cabin_num.str[0]
cabin_list = cabin.unique()
cabin_dict = pd.Series(range(len(cabin_list)), cabin_list)
print(cabin_dict)
cabin_id = pd.get_dummies( cabin ).idxmax(1)
cabin_id = cabin_id.map(cabin_dict).rename('Cabin').to_frame()
cabin_id.sample(5)


# ### 2.3.4 Ticket Number - Use as category
# ---- Not for now ---- 

# ### 2.3.5 Fill missing values
# 
# Age: Mean(age)
# 

# In[ ]:


full_data.Age = full_data.Age.fillna(full_data.Age.mean())
full_data.Fare = full_data.Fare.fillna(full_data.Fare.mean())
full_data.isna().sum()


# # 3. Prepare data for training! 

# ## 3.1 Concatenate all features and normalize
# We select followign features as important: 
# 1. Class 
# 2. Sex
# 3. Age
# 4. SibSp
# 5. Parch
# 6. Cabin
# 7. Embarked
# 
# Let's drop the unnecessary columns and append the cleaned-up features (Cabin, Embarked)
# 
# normalization can be done for both training and test sets here. Alternatively, it is possible to use the same normalizaton settings for training and test sets separately

# In[ ]:


# Reference: https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c
age = full_data.Age
age_sq = (age*age).rename('Age_sq')  # rename the series
age_sex = ((sex*2-1)*age).rename('Age_sex')
#embark_sex = ((sex*2-1)*embarked_id).rename('embark_sex')
class_sex = ((sex*2-1)*full_data.Pclass).rename('Class_sex')

full_x = pd.concat([full_data[['Pclass', 'Age', 'SibSp', 'Parch']], age_sq, sex, age_sex, class_sex, cabin_id, embarked_id], axis=1)


# Min-max normalization
full_x = (full_x - full_x.min())/(full_x.max()-full_x.min())
#full_x.Sex = full_x.Sex*2 - 1

train_valid_x = full_x[0:titanic_train.shape[0]]
train_valid_y = titanic_train['Survived']

test_x = full_x[titanic_train.shape[0]:]

train_valid_x.head()


# ## 3.2 Split train, validation 
# 
# **scikit-learn**   
# scikit is an API that implements methods for various machine learning utilities. Primarily including: 
# 1. data pre-processing
# 2. statistical methods (random forest, logistic regression, svm etc)
# 
# We will first use this for pre-processing steps such as dividing the data for training and validation, and normalization. 

# In[ ]:


train_x, valid_x, train_y, valid_y = train_test_split(train_valid_x, train_valid_y, train_size=0.8)


# # 4. Model 
# 
# ## 4.1 Model Selection
# ### 4.1.1 Logistic Regression

# In[ ]:


model_lr = LogisticRegression()
model_rf = RandomForestClassifier()
model_dt = DecisionTreeClassifier()
model_svc = LinearSVC()
model_knn = KNeighborsClassifier()


# # 4.2 Training the model

# In[ ]:


model_lr.fit(train_x, train_y)
model_rf.fit(train_x, train_y)  
model_dt.fit(train_x, train_y)
model_svc.fit(train_x, train_y)
model_knn.fit(train_x, train_y)


# ## 4.3 Check prediction on validation set

# In[ ]:


print('LR Scores:  Training:{} \t Valid:{}'.format(model_lr.score(train_x, train_y), model_lr.score(valid_x, valid_y)))
print('RF Scores:  Training:{} \t Valid:{}'.format(model_rf.score(train_x, train_y), model_rf.score(valid_x, valid_y)))
print('DT Scores:  Training:{} \t Valid:{}'.format(model_dt.score(train_x, train_y), model_dt.score(valid_x, valid_y)))
print('SVC Scores:  Training:{} \t Valid:{}'.format(model_svc.score(train_x, train_y), model_svc.score(valid_x, valid_y)))
print('KNN Scores:  Training:{} \t Valid:{}'.format(model_knn.score(train_x, train_y), model_knn.score(valid_x, valid_y)))


# ## 4.4 K-Fold Cross Validation
# 
# K-Fold cross validation splits the training set into K sub-sets (folds). K-iterations of training and validation are run while K-1 sets are chosen for training and 1 for validation each time. 

# In[ ]:


from sklearn.model_selection import cross_val_score
scores_rf = cross_val_score(model_rf, train_valid_x, train_valid_y, cv=10)
scores_knn = cross_val_score(model_knn, train_valid_x, train_valid_y, cv=10)

print('RF \t Mean: {} Std:{}'.format(np.mean(scores_rf), np.std(scores_rf)))
print('KNN \t Mean: {} Std:{}'.format(np.mean(scores_knn), np.std(scores_svc)))



# # 5. Random Forest: Advanced Optimizations

# 
# We can use Randomforest classifier and perform more advanced optimizations to further improve the accuracy
# 
# ## 5.1 Feature Importance
# 
# Feature importance can be obtained in random forest which gives you an idea of how much weight each feature carries. This can be useful to eliminate less important or noisy features 

# In[ ]:


print(train_valid_x.columns.shape)
print(model_rf.feature_importances_)

importances = pd.DataFrame({'feature':train_valid_x.columns,'importance':np.round(model_rf.feature_importances_,3)})
print(importances.sort_values(by=['importance'], ascending=False))


# ## 5.2 Confusion Matrix
# 
# Confusion matrix gives an indication of which class is being misclassified more. 
# The matrix below suggests that there is some imbalance in the data

# In[ ]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(model_rf, train_valid_x, train_valid_y, cv=10)
print(' Confusion Matrix \n', confusion_matrix(train_valid_y, predictions))

print('\n Weight of training set:', np.sum(train_valid_y==1)/len(train_valid_y))


# # 5.3 Improving the model
# 
# There are a few tricks used here: 
# 
# n_estimators: complexity of random forest model 
# oob_score: this uses an OOB train and cross validation method to improve the training
# class weight: since ourput dataset has more examples of class: 0 (not survived), we can adjust for it in the training of random forest
# 
#   
# OOB score: This is a metric similar to cross validation score computed at training time

# In[ ]:


#model_rf_improved = RandomForestClassifier()
model_rf_improved = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
print('Weights:{}'.format(model_rf.class_weight))

model_rf_improved.fit(train_valid_x, train_valid_y)
print('oob_score:{}'.format(model_rf_improved.oob_score_))

scores_rf_improved = cross_val_score(model_rf_improved, train_valid_x, train_valid_y, cv=10)

min_estimators = 10
max_estimators = 200

error_rate = {}

for i in range(min_estimators, max_estimators + 1, 5):
    model_rf_improved.set_params(n_estimators=i)
    model_rf_improved.fit(train_valid_x, train_valid_y)

    oob_error = 1 - model_rf_improved.oob_score_
    error_rate[i] = oob_error
    
    #y_scores = model_rf_improved.predict_proba(train_valid_x)
    #y_scores = y_scores[:,1]

    #precision, recall, threshold = precision_recall_curve(train_valid_y, y_scores)


# In[ ]:


# Convert dictionary to a pandas series for easy plotting 
oob_series = pd.Series(error_rate)
fig, ax = plt.subplots(figsize=(10, 10))

oob_series.plot(kind='line',
                color = 'red')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')


# In[ ]:


model_rf_improved = RandomForestClassifier(n_estimators=125, oob_score=True, random_state=42, warm_start=True)
model_rf_improved.fit(train_valid_x, train_valid_y)


# # 5.4 Precision and Recall
# 

# In[ ]:


from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = model_rf_improved.predict_proba(train_valid_x)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(train_valid_y, y_scores)


# In[ ]:


def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# This suggests that the threshold for determining survival in our model should be around 0.45

# In[ ]:


def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plt.show()


# # 5.5 Submitting our prediction 
# 
# Finally, we use the model to submit our prediction. One final thing we have used here is to choose a threshold value that is different from default. Based on precision/recall curves, we choose a prediction probability of 0.3

# In[ ]:


test_data = full_data[titanic_train.shape[0]:]

threshold = 0.4

predicted = model_rf_improved.predict_proba(test_x)
#predicted [:,0] = (predicted [:,0] < threshold).astype('int')
#predicted [:,1] = (predicted [:,1] >= threshold).astype('int')

Y_prediction_default = model_rf_improved.predict(test_x)

Y_prediction = (predicted[:, 0] <= threshold).astype('int')

#print('Prediction probabilities which were changed by setting a different threshold')
#print(predicted[Y_prediction!=Y_prediction_default])


# In[ ]:



submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_prediction
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




