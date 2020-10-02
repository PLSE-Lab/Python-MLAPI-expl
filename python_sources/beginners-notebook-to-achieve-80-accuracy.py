#!/usr/bin/env python
# coding: utf-8

# # Welcome to this beginners notebook on titanic dataset to achieve 80% accuracy.
# 
# This notebook will provide you with an overview of how you can approach a data science problem in order to achieve high prediction accuracy and to gain useful insights as well as conclusions from a given dataset.
# 
# So without further ado lets start by importing packages.
# 
# Credits:
# 
# This notebook is inspired by ideas and code implemented in the following notebooks:
# 
# https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
# 
# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy

# In[ ]:


#Machine Learning Packages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn import model_selection

#Data Processing Packages
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection

#Data Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Metrics
from sklearn import metrics

#Data Analysis Packages
import pandas as pd
import numpy as np

#Ignore Warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Loading the data sets
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")
target = train["Survived"]
data = pd.concat([train.drop("Survived", axis=1),test], axis=0).reset_index(drop=True)


# Lets have a peek at our dataset

# In[ ]:


data.head()


# ###### Things to take note here:
# 
#     1) There are 1309 data-points in train dataset.
#     2) Data types of all the columns.
#     3) Check which columns have null values.

# In[ ]:


data.info()


# ##### We can see that Age, Fare, Cabin, Embarked have null values. Since not all machine learning models can tackle Nan values we have to clean our dataset of such values.

# In[ ]:


data.isnull().sum()


# ### Dealing with missing Age values

# By analysing the barplots and the boxplots we can conclude that mean and median follow the same trends for all the feature so we can use either to fill out the missing age values(here we will use median). We can draw the following conclusions from the below visualizations:
# 
# * Age decreases with increasing Pclass. A valid argument to explain this trend is could be that old people have more money and hence can afford first class.
# 
# * Male Passengers are comparitively older than the female passengers. A valid argument to explain this trend could be that for all the couples on-board wives would be younger than husbands in general.
# 
# * Relation of Age with SibSp could be divided into two parts. The first part can be defined as SibSp in [0,1] and the second part could be defined as SibSp in [2,8]. 
#     
#     * For the first part we can see that singles are younger than couples (without children) which is an acceptable trend.
#     
#     * For the second part we can see that age decreases with the increase in number of siblings on-board which is an acceptable trend because children are like to come with their brothers and sisters along with their parents.
#     
# * Relation of Age with Parch can also be divided in two parts. The first part can be defined as Parch in [0,2] and the second part can be defined as [3,9].
# 
#     * For the first part we can see that age decreases with an increase in number of parents on-board which is an accepatble trend.
#     
#     * For the second part we can see that age increases with an increase in number of children on-board which is an acceptable trend.
#     
# * We can see that in general Age decrease with an increase in Family Size.
# 
# * Finally we can see that passengers who embarked from 'C' are older than those who embarked from 'S' who are older than those who embarked from 'Q'. 
#     
# 
# 

# In[ ]:


#Plotting the relations between Age and other features (Mean).
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,12))

#Age vs Pclass
sns.barplot(x="Pclass", y="Age", data=data, ax=ax[0,0])
#Age vs Sex
sns.barplot(x="Sex", y="Age", data=data, ax=ax[0,1])
#Age vs SibSp
sns.barplot(x="SibSp", y="Age", data=data, ax=ax[0,2])
#Age vs Parch
sns.barplot(x="Parch", y="Age", data=data, ax=ax[1,0])
#Age vs Family_size
sns.barplot(x=(data["Parch"] + data["SibSp"]), y="Age", data=data, ax=ax[1,1])
ax[1,1].set(xlabel='Family Size')
#Age vs Embarked
sns.barplot(x="Embarked", y="Age", data=data, ax=ax[1,2])


# In[ ]:


#Plotting relations between Age and other features (Median).

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,12))
#Age vs Pclass
sns.boxplot(x="Pclass", y="Age", data=data, ax=ax[0,0])
#Age vs Sex
sns.boxplot(x="Sex", y="Age", data=data, ax=ax[0,1])
#Age vs SibSp
sns.boxplot(x="SibSp", y="Age", data=data, ax=ax[0,2])
#Age vs Parch
sns.boxplot(x="Parch", y="Age", data=data, ax=ax[1,0])
#Age vs Family_size
sns.boxplot(x=(data["Parch"] + data["SibSp"]), y="Age", data=data, ax=ax[1,1])
ax[1,1].set(xlabel='Family Size')
#Age vs Embarked
sns.boxplot(x="Embarked", y="Age", data=data, ax=ax[1,2])


# The trend between Embarked and Age could be verified by the following plot. We can see that most of the passengers who embarked
# from 'C' went to first class since we know that people in first class are older we can verify that passengers who embarked from
# 'C' are older. The same conclusions can be drawn for 'S' and 'Q'.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,7))

#Relation between Pclass and Embarked
sns.countplot(x="Pclass", data=data, hue="Embarked")


# In[ ]:


# We will use Pclass, Family Size and Embarked features to fill in the missing age values

# First Lets create the feature Family_Size
data["Family_Size"] = data["SibSp"] + data["Parch"]

#Filling in the missing Age values
missing_age_value = data[data["Age"].isnull()]
for index, row in missing_age_value.iterrows():
    median = data["Age"][(data["Pclass"] == row["Pclass"]) & (data["Embarked"] == row["Embarked"]) & (data["Family_Size"] == row["Family_Size"])].median()
    if not np.isnan(median):
        data["Age"][index] = median
    else:
        data["Age"][index] = np.median(data["Age"].dropna())


# ### Dealing with missing Fare value

# In[ ]:


#Relation Between Fare and Pclass
fig, ax = plt.subplots(figsize=(7,5))
sns.boxplot(x="Pclass", y="Fare", data=data)

#Since we have only 1 missing value for Fare we can just fill it according to Pclass feature
print("Pclass of the data point with missing Fare value:", int(data[data["Fare"].isnull()]["Pclass"]))
median = data[data["Pclass"] == 3]["Fare"].median()
data["Fare"].fillna(median, inplace=True)


# ### Dealing with missing Cabin values
# 
# * Since cabin has a lot of missing values we are going to update the Cabin column as follows:
#     * For the non-missing values we are going to update Cabin with the first character of Cabin. For eg: if Cabin is C13 we are going to change it to C
#     
#     * Replace all the missing Cabin values with X

# In[ ]:


for index, rows in data.iterrows():
    if pd.isnull(rows["Cabin"]):
        data["Cabin"][index] = 'X'
    else:
        data["Cabin"][index] = str(rows["Cabin"])[0]


# ### Dealing with missing Embarked values

# In[ ]:


#Since we only have 2 missing Embarked values we will just fill the missing values with mode.
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)


# ## Feature Engineering
# 
# Now that we have cleaned our dataset it's now time to decide the important features and then to convert them into forms which our machine learning models could interpret.

# * The following conclusions can be drawn from the plots below:
#     * More people survived in first class this can be due to the fact that they were given priority on the basis of their socio-economic status
#     
#     * More female passengers survived than male passengers again this is due to the fact that women and children are given priority over males when it comes to saving lives.

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

#Survived vs Pclass
sns.barplot(x="Pclass", y=target, data=data[:891], ax=ax[0])

#Survived vs Sex
sns.barplot(x="Sex", y=target, data=data[:891], ax=ax[1])


# * The following conclusions can be drawn from the plots below:
#     
#     * Singles have a lower chance of survival. This can be verified by the fact that there are more 'male single' passengers than 'female single' passsengers
#     
#     * Family_Size of 1, 2, 3 has higher chances of survival. This is due to the fact that females and children were given priority over males 
#     
#     * Survival probability drops again for Family_Size > 4. I don't know the reason for this trend but if you have any insights do feel to post in the comment section :)
# 
# * On the basis of the above observations Family_Size could be divided into 3 groups which are:
#     
#     * Singles
#     * Family_Size -> [1,3]
#     * Family_Size >= 4

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,5))

#Survived vs Family_Size
sns.barplot(x="Family_Size", y=target, data=data[:891], ax=ax[0])

#Sex vs Single Passengers
sns.countplot(x="Sex", data=data[data["Family_Size"] == 0], ax=ax[1])

#Dividing Family_Size into 3 groups
data["Family_Size"] = data["Family_Size"].map({0:0, 1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2})


# In[ ]:


data.head()


# Clearly Embarked is an important feature to determine the target feature. Also, we can see that the passengers whose cabin values were missing have a lower chance of survival. 

# In[ ]:


plt.figure(figsize=(12,7))

#Survived vs Cabin
plt.subplot(121)
sns.barplot(x="Cabin", y=target, data=data[:891])

#Survived vs Embarked
plt.subplot(122)
sns.barplot(x="Embarked", y=target, data=data[:891])


# ### Analysing Fare Feature
# 
# We can see that passengers who survived have a higher probability for paying a high fare as compared to those who did not survived. Hence people who survived were mostly from first class (a result we saw before as well).
# 
# Again, Fare is an important feature.

# In[ ]:


plt.figure(figsize=(12,7))

#Plotting Kde for Fare
plt.subplot(121)
sns.kdeplot(data["Fare"])

#Plotting Kde for Fare with Survived as hue
plt.subplot(122)
sns.kdeplot(np.log(data[:891][target == 1]["Fare"]), color='blue', shade=True)
sns.kdeplot(np.log(data[:891][target == 0]["Fare"]), color='red', shade=True)
plt.legend(["Survived", "Not Survived"])

#Since skewness can result in false conclusions we reduce skew for fare by taking log.
data["Fare"] = np.log(data["Fare"])

#Dividing Fare into different categories
data["Fare"] = pd.qcut(data["Fare"], 5)


# ### Analysing Age feature
# 
# Since we know that children were given priority when saving passengers and that old passengers have a lower chance of survivabilty it would be a good idea to divide Age into different categories.

# In[ ]:


label = LabelEncoder()
data["Age"] = label.fit_transform(pd.cut(data["Age"].astype(int), 5))
sns.barplot(x="Age", y=target, data=data[0:891])


# ### Analysing the Name feature
# 
# Even though the name of the person plays no role in determining whether he/she survived. The title in the name can provide useful insights like socio-economic status, age, gender, etc..
# 
# Hence we are going to extract title from each name.

# In[ ]:


data["Name"] = data["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
data["Name"] = data["Name"].map({'Mr':1, 'Miss':2, 'Mrs':3, 'Ms':3, 'Mlle':3, 'Mme':3, 'Master':4, 'Dr':5, 'Rev':5, 'Col':5, "Major":5, "Dona":5, "Sir":5, "Lady":5, "Jonkheer":5, "Don":5, "the Countess":5, "Capt":5})
sns.barplot(x="Name", y=target, data=data[0:891])


# ### Analysing Ticket Feature
# 
# Again ticket as a whole may not be important but the prefix can provide some useful insights.

# In[ ]:


data["Ticket"] = data["Ticket"].apply(lambda x: x.replace(".","").replace('/',"").strip().split(' ')[0] if not x.isdigit() else 'X')
data["Ticket"] = label.fit_transform(data["Ticket"])
sns.barplot(x="Ticket", y=target, data=data[0:891])


# ### Dealing with Categorical Features
# 
# * We will use OneHot encodings for Pclass, Name, Sex, Age, Cabin, Embarked, Ticket, Family_Size (We will drop SibSp and Parch).

# In[ ]:


#OneHot encoding with pd.get_dummies
data.drop(["SibSp", "Parch"], inplace=True, axis=1)
data = pd.get_dummies(data=data, columns=["Pclass", "Name", "Sex", "Age", "Cabin", "Embarked", "Family_Size", "Ticket", "Fare"], drop_first=True)


# In[ ]:


#Splitting into train and test again
train = data[:891]
test = data[891:]


# ### Machine Learning
# 
# We will first see performance of different tuned models after that we will select the models with best performance and high diversity for ensemble modeling. 

# In[ ]:


# Modeling step Test differents algorithms 
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.7, random_state=42)

classifiers = [
    SVC(random_state=42),
    DecisionTreeClassifier(random_state=42),
    AdaBoostClassifier(DecisionTreeClassifier(random_state=42),random_state=42,learning_rate=0.1),
    RandomForestClassifier(random_state=42),
    ExtraTreesClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    MLPClassifier(random_state=42),
    KNeighborsClassifier(),
    LogisticRegression(random_state=42),
    LinearDiscriminantAnalysis()
]

cv_train_mean = []
cv_test_mean = []
cv_score_time = []
cv_fit_time = []
cv_name = []
predictions = []

for classifier in classifiers :
    cv_results = model_selection.cross_validate(classifier, train.drop(['PassengerId'], axis=1), target, cv=cv_split, return_train_score=True)
    cv_train_mean.append(cv_results['train_score'].mean())
    cv_test_mean.append(cv_results['test_score'].mean())
    cv_score_time.append(cv_results['score_time'].mean())
    cv_fit_time.append(cv_results['fit_time'].mean())
    cv_name.append(str(classifier.__class__.__name__))
    classifier.fit(train.drop(['PassengerId'], axis=1), target)
    predictions.append(classifier.predict(test.drop(['PassengerId'], axis=1)))
    

performance_df = pd.DataFrame({"Algorithm":cv_name, "Train Score":cv_train_mean, "Test Score":cv_test_mean, 'Score Time':cv_score_time, 'Fit Time':cv_fit_time})
performance_df


# In[ ]:


#Plotting the performance on test set
sns.barplot('Test Score', 'Algorithm', data=performance_df)


# ### Ensemble Modeling
# 
# * Now that we have performance of individual models we can select models for ensemble modeling on the following grounds:
#     
#     * The selected models should have high test score on cross_validation.
#     * The test set predictions of the selected models should be similar.
# 
# * Condsidering the above statements we can create three ensembles which are:
#     
#     * Ensemble 1 --> DecisionTreeClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
#     * Ensemble 2 --> LinearDiscriminantAnalysis, LogisticRegression, GradientBoostingClassifier, MLPClassifier 
#     * Ensemble 3 --> Combining both Ensemble 1 and Ensemble 2
# 

# In[ ]:


#Plotting prediction correlation of the algorithms
sns.heatmap(pd.DataFrame(predictions, index=cv_name).T.corr(), annot=True)


# ### Tuning 

# In[ ]:


tuned_clf = {
    'DecisionTreeClassifier':DecisionTreeClassifier(random_state=42),
    'AdaBoostClassifier':AdaBoostClassifier(DecisionTreeClassifier(random_state=42),random_state=42,learning_rate=0.1),
    'RandomForestClassifier':RandomForestClassifier(random_state=42),
    'ExtraTreesClassifier':ExtraTreesClassifier(random_state=42),
    
    'GradientBoostingClassifier':GradientBoostingClassifier(random_state=42),
    'MLPClassifier':MLPClassifier(random_state=42),
    'LogisticRegression':LogisticRegression(random_state=42),
    'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis()
}


# In[ ]:


#DecisionTreeClassifier
grid = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [2,4,6,8,10,None], 
        'min_samples_split': [2,5,10,.03,.05], 'min_samples_leaf': [1,5,10,.03,.05], 'max_features': [None, 'auto']}

tune_model = model_selection.GridSearchCV(tuned_clf['DecisionTreeClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)
tune_model.fit(train.drop(['PassengerId'], axis=1), target)
print("Best Parameters:")
print(tune_model.best_params_)
tuned_clf['DecisionTreeClassifier'].set_params(**tune_model.best_params_)


# In[ ]:


#AdaBoostClassifier
grid = {'n_estimators': [10, 50, 100, 300], 'learning_rate': [.01, .03, .05, .1, .25], 'algorithm': ['SAMME', 'SAMME.R'] }

tune_model = model_selection.GridSearchCV(tuned_clf['AdaBoostClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)
tune_model.fit(train.drop(['PassengerId'], axis=1), target)
print("Best Parameters:")
print(tune_model.best_params_)
tuned_clf['AdaBoostClassifier'].set_params(**tune_model.best_params_)


# In[ ]:


#RandomForestClassifier
grid = {'n_estimators': [10, 50, 100, 300], 'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10, None], 
        'oob_score': [True] }
 
tune_model = model_selection.GridSearchCV(tuned_clf['RandomForestClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)
tune_model.fit(train.drop(['PassengerId'], axis=1), target)
print("Best Parameters:")
print(tune_model.best_params_)
tuned_clf['RandomForestClassifier'].set_params(**tune_model.best_params_)


# In[ ]:


#ExtraTreesClassifier
grid = {'n_estimators': [10, 50, 100, 300], 'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10, None]}
 
tune_model = model_selection.GridSearchCV(tuned_clf['ExtraTreesClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)
tune_model.fit(train.drop(['PassengerId'], axis=1), target)
print("Best Parameters:")
print(tune_model.best_params_)
tuned_clf['ExtraTreesClassifier'].set_params(**tune_model.best_params_)


# In[ ]:


#GradientBoostingClassifier
grid = {#'loss': ['deviance', 'exponential'], 'learning_rate': [.01, .03, .05, .1, .25], 
        'n_estimators': [300],
        #'criterion': ['friedman_mse', 'mse', 'mae'], 
        'max_depth': [4] }

tune_model = model_selection.GridSearchCV(tuned_clf['GradientBoostingClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)
tune_model.fit(train.drop(['PassengerId'], axis=1), target)
print("Best Parameters:")
print(tune_model.best_params_)
tuned_clf['GradientBoostingClassifier'].set_params(**tune_model.best_params_)


# In[ ]:


#MLPClassifier
grid = {'learning_rate': ["constant", "invscaling", "adaptive"], 'alpha': 10.0 ** -np.arange(1, 7), 'activation': ["logistic", "relu", "tanh"]}

tune_model = model_selection.GridSearchCV(tuned_clf['MLPClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)
tune_model.fit(train.drop(['PassengerId'], axis=1), target)
print("Best Parameters:")
print(tune_model.best_params_)
tuned_clf['MLPClassifier'].set_params(**tune_model.best_params_)


# In[ ]:


#LogisticRegression
grid = {'fit_intercept': [True, False], 'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
 
tune_model = model_selection.GridSearchCV(tuned_clf['LogisticRegression'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)
tune_model.fit(train.drop(['PassengerId'], axis=1), target)
print("Best Parameters:")
print(tune_model.best_params_)
tuned_clf['LogisticRegression'].set_params(**tune_model.best_params_)


# In[ ]:


#LinearDiscriminantAnalysis
grid = {"solver" : ["svd"], "tol" : [0.0001,0.0002,0.0003]}
 
tune_model = model_selection.GridSearchCV(tuned_clf['LinearDiscriminantAnalysis'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)
tune_model.fit(train.drop(['PassengerId'], axis=1), target)
print("Best Parameters:")
print(tune_model.best_params_)
tuned_clf['LinearDiscriminantAnalysis'].set_params(**tune_model.best_params_)


# In[ ]:


#Evaluating the performance of our tuned models

cv_train_mean = []
cv_test_mean = []
cv_score_time = []
cv_fit_time = []
cv_name = []
predictions = []

for _, classifier in tuned_clf.items():
    cv_results = model_selection.cross_validate(classifier, train.drop(['PassengerId'], axis=1), target, cv=cv_split, return_train_score=True)
    cv_train_mean.append(cv_results['train_score'].mean())
    cv_test_mean.append(cv_results['test_score'].mean())
    cv_score_time.append(cv_results['score_time'].mean())
    cv_fit_time.append(cv_results['fit_time'].mean())
    cv_name.append(str(classifier.__class__.__name__))
    classifier.fit(train.drop(['PassengerId'], axis=1), target)
    predictions.append(classifier.predict(test.drop(['PassengerId'], axis=1)))
    

performance_df = pd.DataFrame({"Algorithm":cv_name, "Train Score":cv_train_mean, "Test Score":cv_test_mean, 'Score Time':cv_score_time, 'Fit Time':cv_fit_time})
performance_df


# In[ ]:


sns.barplot('Test Score', 'Algorithm', data=performance_df)


# ### Finally, we create a voting classifier for our final predictions!

# In[ ]:


#voting_1 = VotingClassifier(estimators=[
#    ('DecisionTreeClassifier', tuned_clf['DecisionTreeClassifier']), 
#    ('AdaBoostClassifier', tuned_clf['AdaBoostClassifier']),
#    ('RandomForestClassifier', tuned_clf['RandomForestClassifier']), 
#    ('ExtraTreesClassifier',tuned_clf['ExtraTreesClassifier'])], voting='soft', n_jobs=4)

#voting_1 = voting_1.fit(train.drop(['PassengerId'], axis=1), target)
#pred_1 = voting_1.predict(test.drop(['PassengerId'], axis=1))


# In[ ]:


#voting_2 = VotingClassifier(estimators=[
#    ('GradientBoostingClassifier',tuned_clf['GradientBoostingClassifier']), 
#    ('MLPClassifier',tuned_clf['MLPClassifier']), 
#    ('LogisticRegression', tuned_clf['LogisticRegression']), 
#    ('LinearDiscriminantAnalysis', tuned_clf['LinearDiscriminantAnalysis'])], voting='soft', n_jobs=4)

#voting_2 = voting_2.fit(train.drop(['PassengerId'], axis=1), target)
#pred_2 = voting_2.predict(test.drop(['PassengerId'], axis=1))


# In[ ]:


voting_3 = VotingClassifier(estimators=[
    ('DecisionTreeClassifier', tuned_clf['DecisionTreeClassifier']), 
    ('AdaBoostClassifier', tuned_clf['AdaBoostClassifier']),
    ('RandomForestClassifier', tuned_clf['RandomForestClassifier']), 
    ('ExtraTreesClassifier',tuned_clf['ExtraTreesClassifier']),
    ('GradientBoostingClassifier',tuned_clf['GradientBoostingClassifier']), 
    ('MLPClassifier',tuned_clf['MLPClassifier']), 
    ('LogisticRegression', tuned_clf['LogisticRegression']), 
    ('LinearDiscriminantAnalysis', tuned_clf['LinearDiscriminantAnalysis'])], voting='soft', n_jobs=4)

voting_3 = voting_3.fit(train.drop(['PassengerId'], axis=1), target)
pred_3 = voting_3.predict(test.drop(['PassengerId'], axis=1))


# In[ ]:


#sol1 = pd.DataFrame(data=pred_1, columns=['Survived'], index=test['PassengerId'])
#sol2 = pd.DataFrame(data=pred_2, columns=['Survived'], index=test['PassengerId'])
sol3 = pd.DataFrame(data=pred_3, columns=['Survived'], index=test['PassengerId'])


# In[ ]:


#sol1.to_csv("sol1.csv")
#sol2.to_csv("sol2.csv")
sol3.to_csv("sol3.csv")


# Thank you for reading this notebook. If you have any doubts/suggestions do feel free to post in the comment section!
# If you liked this notebook dont forget to upvote :)

# In[ ]:




