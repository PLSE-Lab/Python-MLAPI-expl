#!/usr/bin/env python
# coding: utf-8

# # Motivation for this Kernel

# In this Kernel I am going to create the models to make a prediction for the famous Titanic dataset. I believe that this Kernel can be helpful for people looking for the following topics:
# 
# - How to use Pipelines for data preparation
# 
# - Using K-Fold cross validation to test different models
# 
# - Evaluating results using confusion_matrix, precision_score, recall_score, f1_score, plotting precission/recall chart...
# 
# - Testing and tuning models like LogisticRegression, SVM and RandomForest
# 
# - Using GridSearchCV to test different convination of Hyperparams to tune our model
# 
# - Using RandomizedSearchCV when there are many combinations of Hyperparms to limit the time that the model takes to find the best combination
# 
# The model published from this kernel got 0.80861 in the Titanic Competition (top 11%)
#     

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# I import some extra libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('viridis')
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(42)


# # Loading and Inspecting the data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.info()
train_df.head()


# # Exploratory analysis and data preparation

# I am not going to spend too long in the Exploratory analysis and data preparation. For those interested in a good exploratory analysis, there are kernels like https://www.kaggle.com/startupsci/titanic-data-science-solutions.
# 
# For the data preparation I am going to pack every transformation that I do in a function so I can use pipelines later to call this functions.

# In[ ]:


def get_df_by_group(df, group):
    df_groupedby = df.groupby(group).agg({'PassengerId':'count', 'Survived': 'sum'}).rename(columns={'PassengerId': 'NumPassengers'})
    df_groupedby['Rate'] = df_groupedby['Survived'] / df_groupedby['NumPassengers'] 
    return df_groupedby

df = train_df.copy()


# ## Per Sex

# In[ ]:


train_groupby_sex = get_df_by_group(df, ['Sex'])
train_groupby_sex


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2)
f.set_figwidth(16)
f.set_figheight(6)
sns.barplot(x=train_groupby_sex.index, y='Survived', data=train_groupby_sex, ax=ax1)
sns.barplot(x=train_groupby_sex.index, y='NumPassengers', data=train_groupby_sex, ax=ax2)
ax1.set_title('Passengers Survived Per Sex')
ax2.set_title('Passengers Embarked Per Sex') 
ax1.plot()
ax2.plot()


# ## Per Class

# In[ ]:


train_groupby_pclass = get_df_by_group(df, ['Pclass'])
train_groupby_pclass


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2)
f.set_figwidth(16)
f.set_figheight(6)
sns.barplot(x=train_groupby_pclass.index, y='Survived', data=train_groupby_pclass, ax=ax1)
sns.barplot(x=train_groupby_pclass.index, y='NumPassengers', data=train_groupby_pclass, ax=ax2)
ax1.set_title('Passengers Survived Per Class')
ax2.set_title('Passengers Embarked Per Class') 
ax1.plot()
ax2.plot()


# ## Per Embarked

# Embarked had some nan values, we need to fill the gaps

# In[ ]:


#Embarked had a couple of null values as we saw when we called train_df.info(). Let's check the most common values and choose a good approach to fill the nan values
df['Embarked'].value_counts()


# In[ ]:


#As S is the most common value by far, we will fill the couple of Nan values with the most common one
def fill_embarked_nan(df):
    return df['Embarked'].fillna(value='S')

df['Embarked'] = fill_embarked_nan(df)
df['Embarked'].isnull().any()


# ## Per Age Range

# We also have the Nan values for age. We will use median per Pclass to fill those gaps 

# In[ ]:


class_median_age_series = df.groupby(['Pclass'])['Age'].median()
class_median_age_series


# In[ ]:


def fill_age_nan(df):
    return df[['Age', 'Pclass']].apply(lambda x: class_median_age_series.get(x['Pclass']) if(pd.isnull(x['Age'])) else x['Age'], axis=1)
df['Age'] = fill_age_nan(df)


# For Age I also want to segment the age in ranges that I will use as categories for the model

# In[ ]:


def ageclass_by_age(age):
    if age < 10:
        return '< 10'
    elif (age >= 10 and age < 20):
        return '>= 10 and < 20'
    elif (age >= 20 and age < 35):
        return '>= 20 and < 35'
    elif (age >= 35 and age < 50):
        return '>= 35 and < 50'
    elif (age >= 50 and age < 65):
        return '>= 50 and < 65'
    else:
        return '> 65'

def convert_age_to_ageclass(df):
     return df['Age'].apply(ageclass_by_age)
    
df['Age'] = convert_age_to_ageclass(df)


# In[ ]:


train_groupby_age = get_df_by_group(df, ['Age'])
train_groupby_age


# ## Per Fare

# We have all values for Fare. I will use this value as is. Maybe I could improve the model if making a category of this value like I did for Age?

# I am just going to plot the distribution of Fares per category to have an idea of the different values of it. Later I will look at the correlations

# In[ ]:


plt.figure(figsize=(10,4))
f = sns.distplot(df[df['Pclass']==1]['Fare'], bins=30)
f.plot()


# In[ ]:


plt.figure(figsize=(10,4))
f = sns.distplot(df[df['Pclass']==2]['Fare'], bins=30)
f.plot()


# In[ ]:


plt.figure(figsize=(10,4))
f = sns.distplot(df[df['Pclass']==3]['Fare'], bins=30)
f.plot()


# ## Per Title

# Title can be extracted from the name of each passenger. Below I created a function to extract the Titles.

# In[ ]:


def extract_title_from_name(df):
    return df['Name'].apply(lambda x: (x[x.index(',') + 1:x.index('.')]).strip())

df['Title'] = extract_title_from_name(df)
df['Title'].value_counts()


# I want to reduce the number of titles to a more manageable number. There are some titles that seem to be the same than I have tried to unify under the same name. Ie Lady and Miss. And there are others that have very little occurrences like Sir and Major. For those I have crated a big category called other.

# In[ ]:


def reduce_list_titles(title):
    if(title in ['Mrs', 'Miss', 'Master', 'Mr', 'Dr']):
        title = title
    elif(title in ['Ms','Mme']):
        title = 'Mrs'
    elif(title in ['Mlle', 'Lady']):
        title = 'Miss'
    elif(title in ['Don']):
        title = 'Mr'
    else:
        title = 'Other'
    return title

def convert_title_to_reduced_list(df):
    return df['Title'].apply(reduce_list_titles)

df['Title'] = convert_title_to_reduced_list(df)


# In[ ]:


train_groupby_title = get_df_by_group(df, ['Title'])
train_groupby_title.sort_values(by='Rate', ascending=False)


# ## Per Family Members

# After exploratory analysis of survival based on the number of members in a family, it seems that having a small family increases the chances of survival but having a big family (more than 3 members) reduces it. So I have created 3 categories based in the attributes Parch and SibSp.

# In[ ]:


def convert_family_size_cat(df):
    family_members=df['Parch'] + df['SibSp']
    if(family_members == 0):
        return 'NO_FAMILY'
    if(family_members <= 3 and family_members >=0):
        return 'SMALL_FAMILY'
    else:
        return 'BIG_FAMILY'
    
def convert_parch_and_sibsp_to_family_cat(df):
     return df.apply(convert_family_size_cat, axis=1)
    
df['Family'] = convert_parch_and_sibsp_to_family_cat(df)


# In[ ]:


train_groupby_family = get_df_by_group(df, ['Family'])
train_groupby_family.sort_values(by='Family', ascending=False)


# ## Per Cabin

# Cabin has a big number of nan, but the known information can still be valuable to determine if a passenger survived or not. The strategy that I choose to fill nan in this case was to fill all nan with the same value 'O' (other)

# In[ ]:


def fill_cabin_nan(df):
    return df['Cabin'].apply(lambda x: str(x)[0] if(pd.notnull(x)) else str('O'))

df['Cabin'] = fill_cabin_nan(df)


# In[ ]:


train_groupby_cabin = get_df_by_group(df, ['Cabin'])
train_groupby_cabin.sort_values(by='Rate', ascending=False)


# ## Heatmap of the Survival grouping Categories

# To visualize better the survival rates per category I am going to create a heatmap that shows the survival rate per Sex, Age, Pclass, Family and Title.

# In[ ]:


grouped_df = df.groupby(['Sex', 'Age', 'Pclass', 'Family', 'Title']).agg({'PassengerId':'count', 'Survived': 'sum'}).rename(columns={'PassengerId': 'NumPassengers'})
grouped_df['Rate'] = grouped_df['Survived'] / grouped_df['NumPassengers']
grouped_df = grouped_df.reset_index()
grouped_df.head(10)


# In[ ]:


grouped_df['Sex-AgeRange-Title'] = grouped_df['Sex'] + ' - ' + grouped_df['Age'].astype(str) + ' - ' + grouped_df['Title']
grouped_df['Class-HasFamily'] = 'Class_' + grouped_df['Pclass'].astype(str) + ' - ' + grouped_df['Family']
pivot_df = grouped_df.pivot(index='Sex-AgeRange-Title', columns='Class-HasFamily', values='Rate')
plt.figure(figsize=(16, 8))
sns.heatmap(pivot_df, annot=True)


# # Data Transformation: Scikit pipelines

# Pipelines are useful to make a sequence of transformation repeteable. 
# 
# They are also useful in case you want to parameterize your transformation functions and then use those params as hyperparams of the model using RandomizedSearchCV or GridSearchCV to find the best combination.

# ## Creating the methods to call from pipelines

# I will use this method in my pipelines to call above methods that make all the transformations of the different attributes. In this method, I start by converting the numpy array into pandas DataFrame and at the very end I return np array again as this is what the pipeline expects. 
# I would appreciate if any of the experts in kaggle can tell me if this is a good practice or not.

# In[ ]:


def convert_to_categories_and_fill_nan(X):
    df = pd.DataFrame(X, columns=cat_attributes)
    df['Title'] = extract_title_from_name(df)
    df['Title'] = convert_title_to_reduced_list(df)
    df['Embarked'] = fill_embarked_nan(df)
    df['Age'] = fill_age_nan(df)
    df['Age'] = convert_age_to_ageclass(df)
    df['Family'] = convert_parch_and_sibsp_to_family_cat(df)
    df['Cabin'] = fill_cabin_nan(df)
    df.drop(['Name', 'SibSp', 'Parch'], axis=1, inplace=True)
    return df.values


# ## Creating the pipelines to format the data

# Now, I am going to import all the classes that I will use for hte pipeline and different transformations that I will do.

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

# I print here one last time the first 5 rows to decide the attributes that I will use in the model
train_df.head()


# I name here the attributes that I will use in my model. I name some numberic attributes like 'SibSp' and 'Parch' as categoricals as I want those to go through the categorical transformations that I will do in the pipeline

# In[ ]:


cat_attributes = ['Pclass', 'Age', 'Sex', 'Embarked', 'Cabin', 'Name', 'SibSp', 'Parch']
num_attributes = ['Fare']


# Here I define the two pipelines that I will use. One for numeric attributes and one for categorical attributes:
# - num_pipeline defines 2 transformations:
#     - SimpleImputer will just replace nan with the most frequent value. In this case the num_pipeline only will be applied to Fare and it will be use the most common value to fill the gaps.
#     - StandardScaler Standardize features by removing the mean and scaling to unit variance
# - cat_pipeline: defines 2 transformations for the categorical attributes too:
#     - convert_to_categories_and_fill_nan: this is our custom function defined above
#     - OneHotEncoder: This converts categorical values into one-hot vectors.

# In[ ]:


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('convert_to_categories_and_fill_nan', FunctionTransformer(convert_to_categories_and_fill_nan, validate=False)),
    ('cat_encoder', OneHotEncoder())
])


# Finally I use ColumnTransformer Pipeline to define which attributes will be transformed in each pipeline

# In[ ]:


full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes)
])


# ## Using the pipelines to prepared the train and test data

# Now we are going to use the pipelines. First I get the train and test sets with just the attributes we are going to use in the pipelines and we also split the labels of the train set into the y_train variable.

# In[ ]:


X_train = train_df[['Fare','Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Embarked', 'Cabin', 'Name']].copy()
X_test = test_df[['Fare','Sex', 'Pclass', 'Age', 'SibSp', 'Parch','Embarked', 'Cabin', 'Name']].copy()
y_train = train_df['Survived']
X_train.info()


# We start by concating the X_train and X_test sets to process both of them at the same time by the pipeline. The reason to do this is because not all categorical values could be represented in both sets which would give different number of attributes when applying the OneHotEncoder.
# 
# After running the pipeline, we split the np array again between the train data and the test data.

# In[ ]:


X_concat = pd.concat([X_train, X_test])
X_concat_prepared = full_pipeline.fit_transform(X_concat)
X_train_prepared = X_concat_prepared[:891]
X_test_prepared = X_concat_prepared[891:]
X_train_prepared[:10]


# # Logistic Regression with K-Fold Cross Validation

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, accuracy_score


# First model I am going to try is LogisticRegression. I am going to use cross validation to check the accuracy of the model.
# 
# cross_val_score performs k-fold validation which splits the training set in n distinct subsets (specified by the param cv) and picking a different fold for evaluation every time training in the other n-1 folds.

# In[ ]:


logistic_reg = LogisticRegression(solver='lbfgs',max_iter=1000,random_state=101)
y_logistic_score = cross_val_score(logistic_reg, X_train_prepared, y_train, cv=10, scoring='accuracy')
y_logistic_score.mean()


# ## Testing the results of Logistic Regression Model

# #### Confusion Matrix

# Confusion matrix is a better way to evaluate the performance of a classifier. In a confusion matrix we are going to count the number of instances of each class that are counted as the correct class and as the incorrect class:
# - In the top row we will have the instaces that are negative and in the bottom row the instances that are positive, in this case the top row would be the not survivals and the bottom one the survivals
# - In the left column we have what the model predicted as negative and in the right one the ones predicted as positive. Thus, the left column is what the model predicted as not survivals, and the right one what the model predicted as survivals.
# 
# Therefore, the left-top and right-bottom cells are what the model got right. Top-right corner would be called "False Positives", which in this case are not survivals that the model predicted as survivals. Bottom-left corner would be False Negatives, in this case, survivals that were predicted as non-survivals by the model.

# In[ ]:


# To get the confusion matrix, we don't want to get the score but the actual predictions, so we can pass it later to the confusion_matrix function together with the actual 
# labels to get the matrix
y_logistic_pred = cross_val_predict(logistic_reg, X_train_prepared, y_train, cv=10)


# In[ ]:


confusion_matrix(y_train, y_logistic_pred)


# #### Precision Score

# Precission is a way to look at the accuracy of the positive predictions. The formula to calculate it is 
# 
# ``PRECISSION = TP / (TP + FP)``

# In[ ]:


precision_score(y_train, y_logistic_pred)


# #### Recall Score

# Also called Sensitivity or True positive Rate, this is the ratio of possitive instances that are correctly detected by the classifier. 
# 
# ``RECALL = TP / (TP + FN)``

# In[ ]:


recall_score(y_train, y_logistic_pred)


# #### F1 Score

# F1 Score, combines precision and recally into a single metric. It is useful when you want a single metric that combines both classifiers.
# 
# ``F1 = 2 / ( (1 / precision) + (1 / recall) ) = 2 * ( (precission * recall) / (precission + recall) ) = TP / (TP + ( (FN + FP) / 2) ) ``

# In[ ]:


f1_score(y_train, y_logistic_pred)


# #### Classification Report

# Classification report builds a report that contains some of the metrics that I just computed above

# In[ ]:


print(classification_report(y_train,y_logistic_pred))


# ### Using Decision Function to control threshold evaluate precision and recall

# Precission and Recall can be controlled by modifiying the threshold of the decision function that decides the predicted value. However this is a tradeoff, if we increase precission, Recall will decrease and viceversa.

# In[ ]:


y_decision_function = cross_val_predict(logistic_reg, X_train_prepared, y_train, cv=10, method='decision_function')


# First, we are going to set the threshold to a value higher than 0 and see how this modifies precision and recall

# In[ ]:


threshold = 2
print(confusion_matrix(y_train, y_decision_function > threshold))
print(precision_score(y_train, y_decision_function > threshold))
print(recall_score(y_train, y_decision_function > threshold))


# This, increased a lot the precision and there are just 4 False Positives, however, it also increased a lot the number of False Negatives reducing a lot the Recall

# Second, we are going to set the threshold to a value lower than 0 and see how precission and recall change this time

# In[ ]:


threshold = -2
print(confusion_matrix(y_train, y_decision_function > threshold))
print(precision_score(y_train, y_decision_function > threshold))
print(recall_score(y_train, y_decision_function > threshold))


# Now, the opposite happened, the number of False Negatives decreased a lot and the model is able to predict many more survivals, however this is at the expense of decreasing the precission as there are a much higher number of False Positives

# #### Precission curve per threshold vs Recall curve per threshold 

# We can plot how the precision and recall change depending on the threshold chosen

# In[ ]:


precisions, recalls, thresholds = precision_recall_curve(y_train, y_decision_function)


# In[ ]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    
plt.figure(figsize=(16, 8))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)


# #### Precision vs Recall curve

# In addition, we can also print precision vs recall curve. This is useful to choose correctly a value with a good recall just before precission falls abruptly. 

# In[ ]:


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(16, 8))
plot_precision_vs_recall(precisions, recalls)


# #### ROC Curve

# I am also printing the receiver operating characteristic curve (ROC). ROC curve plots the true positive rate vs false positive rate****

# In[ ]:


def plot_roc_curve(fpr, tpr,  label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

fpr, tpr, threshold = roc_curve(y_train, y_decision_function)
plt.figure(figsize=(16, 8))
plot_roc_curve(fpr, tpr)


# A good model, should make the curve as close as possible to the top-left corner.

# #### ROC AUC Score

# Finally, We can calculate the area under the curve. A good model would get a result as close as possible to 1, while 0.5 would be a pure random model

# In[ ]:


roc_auc_score(y_train, y_decision_function)


# 

# # Hyperparam Tunning with GridSearchCV

# Now, I am going to train again a LogisticRegression, this time, using GridSearchCV to find the best combination of Hyper Params. GridSearchCV tests all the parameters in the param_grid and does cross validation to get the combination with the best score.

# ## Logistic Regression

# In[ ]:


from sklearn.model_selection import GridSearchCV
from scipy.stats import randint

# This is the combination of Hyperparams that we are going to test
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01,0.1,0.9,0.95]
}

logistic_reg = LogisticRegression()

# As parameters for the GridSearchCV we provide the LogisticRegression instance, the param dictionary to test, 
# the number of folds that the train set should be split on for cross validation, and the scoring method.
grid_search = GridSearchCV(logistic_reg, param_grid, cv=5,scoring='accuracy')

grid_search.fit(X_train_prepared, y_train)


# We can now find the combination of hyperparms that gives the best accuracy and the score that we got with that combination

# In[ ]:


print(grid_search.best_params_)
print(grid_search.best_score_)


# And we can also print the score that we get with each of the combinations of hyperparams tested

# In[ ]:


means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# ## K Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
}

grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search_knn.fit(X_train_prepared, y_train)


# In[ ]:


print(grid_search_knn.best_params_)
print(grid_search_knn.best_score_)


# In[ ]:


means = grid_search_knn.cv_results_['mean_test_score']
stds = grid_search_knn.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search_knn.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# ## SVM Classifier

# In[ ]:


from sklearn.svm import SVC

param_grid = [
    {'kernel':['rbf'], 'C':[0.7,0.8,1], 'gamma':[0.07, 0.08, 0.09]},
    {'kernel':['poly'],'C':[0.1,1,10,100], 'gamma':['auto']}
  ]

grid_search_svc = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search_svc.fit(X_train_prepared, y_train)


# In[ ]:


print(grid_search_svc.best_params_)
print(grid_search_svc.best_score_)


# In[ ]:


means = grid_search_svc.cv_results_['mean_test_score']
stds = grid_search_svc.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search_knn.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# ## Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'max_features':[8,9,10,11,12,13],
    'max_leaf_nodes':[7,8,9,10,11],
    'min_samples_split':[3,4,5]
}

grid_search_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train_prepared, y_train)


# In[ ]:


print(grid_search_tree.best_params_)
print(grid_search_tree.best_score_)


# In[ ]:


means = grid_search_tree.cv_results_['mean_test_score']
stds = grid_search_tree.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search_tree.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# In[ ]:


feature_importance_list = grid_search_tree.best_estimator_.feature_importances_
list_categories = full_pipeline.named_transformers_['cat']['cat_encoder'].categories_
flat_list = [item for sublist in list_categories for item in sublist]
attribute_list = num_attributes + flat_list
df_attribute_importance = pd.DataFrame({'attribute_name': attribute_list, 'importance': feature_importance_list})
df_attribute_importance.sort_values('importance', ascending=False)


# ## AdaBoostClassifier

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

ada_boost_clf = AdaBoostClassifier(DecisionTreeClassifier(), random_state=42)

param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2,3],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

grid_search_ada = GridSearchCV(ada_boost_clf, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search_ada.fit(X_train_prepared, y_train)


# In[ ]:


print(grid_search_ada.best_params_)
print(grid_search_ada.best_score_)


# In[ ]:


means = grid_search_ada.cv_results_['mean_test_score']
stds = grid_search_ada.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search_ada.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# In[ ]:


feature_importance_list = grid_search_ada.best_estimator_.feature_importances_
list_categories = full_pipeline.named_transformers_['cat']['cat_encoder'].categories_
flat_list = [item for sublist in list_categories for item in sublist]
attribute_list = num_attributes + flat_list
df_attribute_importance = pd.DataFrame({'attribute_name': attribute_list, 'importance': feature_importance_list})
df_attribute_importance.sort_values('importance', ascending=False)


# ## Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gradient_boost_clf = GradientBoostingClassifier(DecisionTreeClassifier(), random_state=42)

param_grid = {'loss' : ["deviance"],
              'n_estimators' : [300, 400, 500],
              'learning_rate': [0.2, 0.15, 0.1],
              'max_depth': [3, 4, 6, 8],
              'min_samples_leaf': [60, 80, 100],
              'max_features': [0.5, 0.3, 0.1] 
              }
grid_search_gradient_boost = GridSearchCV(gradient_boost_clf, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search_gradient_boost.fit(X_train_prepared, y_train)


# In[ ]:


print(grid_search_gradient_boost.best_params_)
print(grid_search_gradient_boost.best_score_)


# In[ ]:


means = grid_search_gradient_boost.cv_results_['mean_test_score']
stds = grid_search_gradient_boost.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search_gradient_boost.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# In[ ]:


feature_importance_list = grid_search_gradient_boost.best_estimator_.feature_importances_
list_categories = full_pipeline.named_transformers_['cat']['cat_encoder'].categories_
flat_list = [item for sublist in list_categories for item in sublist]
attribute_list = num_attributes + flat_list
df_attribute_importance = pd.DataFrame({'attribute_name': attribute_list, 'importance': feature_importance_list})
df_attribute_importance.sort_values('importance', ascending=False)


# # Hyperparam Tunning using RandomizedSearchCV for Random Forest 

# Finally, I am going to train a Random Forest model and I will use RandomizedSearchCV to test a number of hyperparameters combinations. The reason to use RandomizedSearchCV instead of GridSearchCV is that we are testing a very high number of combinations, randomizedSearchCV will just test a number of random combinations instead of all possible combinations.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_grid = {
    'bootstrap': [True],
    'max_depth': [80,120,140],
    'max_features':[4,5,6,7,8],
    'min_samples_leaf':[2,3,4],
    'min_samples_split':[8,10,12,14,16],
    'n_estimators': [100,200,300,500,1000]
}

forest2_clf = RandomForestClassifier()
rnd_search = RandomizedSearchCV(forest2_clf, param_distributions=param_grid,
                                n_iter=500, cv=10, scoring='accuracy', random_state=101)
rnd_search.fit(X_train_prepared, y_train)


# We can check the best hyperparam combination tested

# In[ ]:


rnd_search.best_params_


# Also the best score that was obtained by the above hyperparam combination

# In[ ]:


rnd_search.best_score_


# And I can also print every single combination tested and the score of obtained by that combination

# In[ ]:


means = rnd_search.cv_results_['mean_test_score']
stds = rnd_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, rnd_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# I can get the feature importance of each feature, sort it, and print them in a table to analyze which features are the most important in the model

# In[ ]:


feature_importance_list = rnd_search.best_estimator_.feature_importances_


# In[ ]:


list_categories = full_pipeline.named_transformers_['cat']['cat_encoder'].categories_
flat_list = [item for sublist in list_categories for item in sublist]
attribute_list = num_attributes + flat_list
df_attribute_importance = pd.DataFrame({'attribute_name': attribute_list, 'importance': feature_importance_list})
df_attribute_importance.sort_values('importance', ascending=False)


# # Testing the model with the test set and publishing result

# Finally, I am going to use my best estimator in the test set to submit my prediction for the competition

# In[ ]:


best_estimator = rnd_search.best_estimator_


# In[ ]:


y_test_predictions = best_estimator.predict(X_test_prepared)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_test_predictions
    })
submission.head()


# In[ ]:


submission.to_csv('submission5.csv', index=False)

