#!/usr/bin/env python
# coding: utf-8

# # 1. Presentation #
# The objective of the study is to analyze passengers data in order to predict which person survived.
# In the first part, I will have a first look at the data then I will process and prepare data for to training.
# 
# Right now, I am just focusing on data analysis and a single regression technique, Logistic Regression, to see the influence of converting data. Then I will look at the different techniques in order to understand the different ways to make a prediction
# 
# More details on the description [here at Kaggle][1]
# 
# First, we need to make some preparations: import python libraries and acquire data files.
# 
# 
#   [1]: https://www.kaggle.com/c/titanic#description

# In[ ]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Acquire data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # 2. Analyze data #
# 
# ## 2.1 First look at the features ##
# Data are describe [here][1] for more details. In this part, we will take a look at the dataset structure. Then, we will take a look at the features.
# 
# ### 2.1.1 Structure of the dataset ###
# Now let's take a look at the labels and the shapes.
#   [1]: https://www.kaggle.com/c/titanic/data

# In[ ]:


#Labels
print('Labels:\n','train: ',train.columns.values,'\n test: ',test.columns.values)
#Shapes
print('Shapes:\n','train: ',train.shape,'\n test: ',test.shape)


# We can see that there is 12 (resp. 11) features for train (resp. test) data. It is normal, because we want to predict which test passengers have survived, there is no *Survived* feature. **More over, because there is not so many features, I have decided to consider, in the first place, all of them relevant for the prediction.** So we will extract the feature *Survived* and mix data train and test to manipulate them only once.

# In[ ]:


labels = train['Survived']
data = pd.concat([train,test],ignore_index=True)
data = data.drop(['PassengerId','Survived'],1)


# ### 2.1.2 Structure of the features ###
# Now we can take a look at the general description:

# In[ ]:


print(data.info())


# At this point, we can extract 2 major information: the type of each variable and the number of missing values.
# Here:
# 
#  - *Age*:  float ; 263 missing values
#  - *Cabin*: object ; 1014 missing values
#  - ...
#  - *Ticket*: object ; 0 missing values
# 
# Features with missing values: *Age*, *Cabin*, *Embarked* and *Fare*.
# 
# ### 2.1.3 Summary ###
# 
#  - **Dataset structure :**  train 891 rows, test 418 rows and 11 input variables/ 1 output variable.
#  - **Categorical variables :** Embarked (2 missing values), Pclass, Sex, Survived
#  - **Continuous numerical variables :** Age (263 missing values), Fare (1 missing value)
#  - **Discrete numerical variables :** Parch, SibSp, PassengerId
#  - **Text variables :** Cabin (1014 missing values), Name, Ticket
# 
# ## 2.2 Extracting information from features ##
# Basically, this step consist in interpreting complex features as Ticket or Name.
# 
# 'Name': Contain at least 3 informations: the family name, the social/professional Title and the probable native language based on the origin of the family name.
# 
# 'Ticket': contain information about LINE employee or the class (PC/FC for Premium Class or First Class). But because one ticket can be attributed for several people, it could be interesting to add a feature representing the number of people with the same ticket (See chapter 2.4).
# 
# Because this is my first ML project, I will only focus on extracting title information.
# But to extract more information, you can use the last name and with some skills estimate the origin.
# I have no clue about how to extract more information from *Ticket* feature, so I have decided to eventually drop the feature.

# In[ ]:


data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['LastName'] = data['Name'].str.extract('([A-Za-z]+)\,', expand=False)

#First look at the Title distribution
data['Title'].value_counts()


# A simplification of this distribution seems relevant to avoid rare titles, I choose to keep only 6 Titles, according to social position and infering information about age and/or sex:
# * Mr / Master
# * Mrs / Miss
# * Hish Society 
# * Function

# In[ ]:


title_replacement_dict = {'Function': ['Dr', 'Rev', 'Col', 'Major', 'Capt'],
                          'High_society': ['Lady', 'Jonkheer', 'Dona', 'Don', 'Sir', 'Countess'],
                          'Mrs': ['Ms', 'Mme'], 'Miss': ['Mlle']}

for new_title, old_titles in title_replacement_dict.items():
    data['Title'] = data['Title'].replace(old_titles, new_title)


# *Cabin* can be interesting, I will deal with it in the next part.
# 
# ## 2.3 Missing Values: Which Strategy? ##
# We have seen that there are many missing values especially for the feature 'Cabin'. Now we should think about what is the best strategy to avoid missing values. There is a lot of methods such as:
# 
#  - Remove feature with more than X% of missing values
#  - Fill with the most common value
#  - Fill with a null element depending on the type (ex: 0 for int or Unspecified for object)
#  - ...
# 
# ### 2.3.1 *Age* ###
# For this feature, my first idea was to use the median value to fill missing values. But when I created groups of age to simplify, I realized I could use the title to refine the value.
# 
# To do so, I group the *Age* feature according to the *Title* feature allowing me to access for each title several values as the count, mean or median (etc.). Then using the *transform()* method, I get a new column with the median age associated to the title. Then I use the value of this column to fill NaN values in the *Age* feature.

# In[ ]:


data['Age'] = data['Age'].fillna(data.groupby(['Title'])['Age'].transform('median'))

#The line above do the following:
#data.loc[data['Title']=='Mr','Age'] = data.loc[data['Title']=='Mr','Age'].fillna(data[data['Title']=='Mr'].Age.median())
#data.loc[data['Title']=='Mrs','Age'] = data.loc[data['Title']=='Mrs','Age'].fillna(data[data['Title']=='Mrs'].Age.median())
#data.loc[data['Title']=='Miss','Age'] = data.loc[data['Title']=='Miss','Age'].fillna(data[data['Title']=='Miss'].Age.median())
#data.loc[data['Title']=='Master','Age'] = data.loc[data['Title']=='Master','Age'].fillna(data[data['Title']=='Master'].Age.median())
#data.loc[data['Title']=='Function','Age'] = data.loc[data['Title']=='Function','Age'].fillna(data[data['Title']=='Function'].Age.median())


# ### 2.3.2 *Embarked* ###
# We have seen there are only two missing values and because almost 70% of them are *S*, I choose to use the most frequent value.

# In[ ]:


data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])


# ### 2.3.3 *Fare* ###
# At first, because there is only 1 missing value, i thought to use the median value of all the *Fare* values but some refinement are possible.
# 
# First, what are the information on the passenger with the missing value:

# In[ ]:


print(data[data['Fare'].isnull()])


# Looking at the dataset, people with the same ticket have the same *Fare* value. The fare feature is link to the ticket and the cabin and so the occupant number. It's logical that for the same standard a bigger cabin has a bigger price. It could be interesting to add a new feature representing the Fare per person per ticket (See chapter 2.4). 
# For now, we could analyse 3 points:
# 1. Is there someone else with the same ticket? => Yes: We use the same *Fare* value / No: we skip that
# 2. What's the median fare for a single person ? Is it significantly different from people in family? => Yes
# 3. Is the *Fare* distribution different based on the *Embarked* location? => Yes

# In[ ]:


ticket_fare_missing = data.loc[data['Fare'].isnull(), 'Ticket'].values[0]
print(data[data['Ticket'] == ticket_fare_missing], '\n')
print(data.groupby(['SibSp', 'Parch'])['Fare'].median(), '\n')
print(data.groupby(['Embarked'])['Fare'].median())


# Based on these result, a solution to fill the missing value could be using the median value based on the *Fare* value of people who embarked at Southampton, alone, with a third class.
# Comparing mean and median values obtain this way, with based on the global set, there is quite a difference.

# In[ ]:


filt = ((data['Embarked'] == 'S') &
        (data['Parch'] == 0) & (data['SibSp'] == 0))
        
print('Mean:')
print(data[filt].groupby(['Pclass'])['Fare'].mean())
print(data['Fare'].mean())
print('\nMedian')
print(data[filt].groupby(['Pclass'])['Fare'].median())
print(data['Fare'].median())

data['Fare'] = data['Fare'].fillna(data[filt].groupby(['Pclass'])['Fare'].transform('median'))


# ### 2.3.4 *Cabin* ###
# This feature is the most interesting, there is almost 80% of missing values, so should I drop the feature or should I keep it? <br>
# *To avoid complexity, I have decided to drop it.*

# In[ ]:


data = data.drop(['Cabin'],1)


# ## 2.4 Creating new features ##
# With the current features, it could be interesting to engineer new ones, as:
# 1.  The number of person sharing the same ticket
# 2.  The family size
# 3.  Is alone: to indicates if the person is alone or not
# 4.  The fare per person per ticket
# 
# 2 and 3 can be combine in a single feature, *Family representation* indicating how many percent of the family the person represents

# In[ ]:


data['Num_person_per_ticket'] = data.groupby(data['Ticket'])['Sex'].transform('count')
data['Family Size'] = data['Parch'] + data['SibSp']
data['Fare_per_person_per_ticket'] = data['Fare'] / data['Num_person_per_ticket']
data['Family_representation'] = 1 / (1 + data['Parch'] + data['SibSp'])
data['Pclass_inv'] = 1 / data['Pclass']


# ## 2.4 Converting Features##
# To simplify our data, I choose to create some groups for the features *Age*, *Fare*, *SibSp* and *Parch*.
# 
#  - *Age*: I create 5 groups
#  - *Fare*: I create 6 groups
#  - *SibSp* and *Parch* : I create a new intermediate feature *Family* representing the size of the family
# 
# Then, I cut *Family* in 3 parts and I drop *SibSp* and *Parch*

# In[ ]:


#data['Age'] = pd.cut(data['Age'], (0, 11, 16, 35, 55, 120), labels = ['Child', 'Teen', 'Young', 'Adult', 'Senior'])
#data['Fare'] = pd.cut(data['Fare'], (-0.1, 0, 8, 15, 32, 80, 520), labels = ['Free', 'Low', 'MidLow', 'MidHigh', 'High', 'UltraHigh'])
#data['Family'] = data['SibSp'] + data['Parch'] + 1
#data['Family'] = pd.cut(data['Family'], (0, 1, 2, 10), labels=['Alone', 'Couple', 'Family'])
#data = data.drop(['SibSp','Parch'],1)


# Now we need drop features we do not want to use and convert remaining categorical variables into numerical variables.

# In[ ]:


data = data.drop(['Name', 'Ticket', 'LastName', 'Pclass', 'SibSp','Parch'], axis=1)
data['Sex'] = np.where(data['Sex']=='female',1,0)
print(data.info())
data = pd.get_dummies(data)
print(data.info())


# # 3 Make The First Prediction#
# 
# ## 3.1 Preparing datasets##
# 
# We separate our dataset data into the training set with labels and the test set (validation set which we try to make predictions on).
# Then we normalize our training set to optimize the training process and we split it into a training set and a testing set. The testing set is used to evaluate the classifier.

# In[ ]:


seed = 42
test_size = 0.2

trainX = data[:891]
trainY = labels
valX = data[891:]

scaler = StandardScaler()
scaler.fit(trainX)
X_train_norm = scaler.transform(trainX)
X_valid_norm = scaler.transform(valX)

X_train, X_test, y_train, y_test = train_test_split(X_train_norm, trainY, random_state=seed, test_size=test_size)


# ## 3.2 First Models##
# 
# ### 3.2.1 Logistic Regression ###

# In[ ]:


def classifier_predictions(clf, refit=True):
    if refit:
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Test Accuracy:', accuracy_score(y_test, y_pred))
    print('Test F1 score:', f1_score(y_test, y_pred))
    return y_pred

def classifier_cross_val_scores(clf):
    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
    print("Cross_val_score_mean:", scores.mean())
    return scores

logreg = LogisticRegression(random_state=seed)
y_pred_log_reg = classifier_predictions(logreg)
logreg_scores = classifier_cross_val_scores(logreg)


# Not that bad, but is it the best we can get? Should we tune its hyperparameters? We should take a look at some other classifiers first.
# 
# 
# ### 3.2.2 SVC ###

# In[ ]:


svc = SVC(random_state=seed, gamma='auto')
y_pred_svc = classifier_predictions(svc)
svc_scores = classifier_cross_val_scores(svc)


# 
# ### 3.2.3 Random Forest Classifier ###

# In[ ]:


rfc = RandomForestClassifier(random_state=seed)
y_pred_rfc = classifier_predictions(rfc)
rfc_scores = classifier_cross_val_scores(rfc)


# ### 3.2.3 Decision Tree Classifier ###

# In[ ]:


dtc = DecisionTreeClassifier(random_state=seed)
y_pred_dtc = classifier_predictions(dtc)
dtc_scores = classifier_cross_val_scores(dtc)


# ### 3.2.5 Comparison of the first models ###

# In[ ]:


def plot_box_clfs_scores(clfs_scores, clfs_labels):
    plt.figure(figsize=(12, 6))
    i = 1
    for scores in clfs_scores:
        plt.plot([i]*len(scores), scores, ".")
        i += 1
    plt.boxplot(clfs_scores, labels=clfs_labels)
    plt.ylabel("Accuracy", fontsize=14)
    plt.show()

def create_df_scores(y_dict_preds, metrics, is_clf_tuned=False):
    scores = []
    for clf, y in y_dict_preds.items():
        for m in metrics:
            scores.append({'Classifier': clf, 'Score': metrics_map[m](y_test, y), 'Type': m + " tuned"*int(is_clf_tuned)})
    df_scores = pd.DataFrame.from_records(scores)
    return df_scores

def plot_df_scores(df_scores):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Score',y='Classifier', hue='Type', data=df_scores)
    plt.xlim(0.7, df_scores.Score.max() + 0.05)

metrics_map = {'f1_score': f1_score,
               'accuracy_score': accuracy_score,
               'precision_score': precision_score,
               'recall_score': recall_score}    
    
clfs_scores = [logreg_scores, svc_scores, rfc_scores, dtc_scores]
clfs_labels = ["logreg", "SVC","Random Forest", "Decision Tree"]
plot_box_clfs_scores(clfs_scores, clfs_labels)

y_dict_preds = {'logreg': y_pred_log_reg, 'svc': y_pred_svc, 'rfc': y_pred_rfc, 'dtc': y_pred_dtc}
metrics = ['f1_score', 'accuracy_score']
df_scores_no_tuning = create_df_scores(y_dict_preds, metrics)
plot_df_scores(df_scores_no_tuning)


# # 4 Tuning hyperparameters#
# 
# GridSearchCV is one way to tune hyperparameters, it is convenient to search which are the best parameters among the user specific desired values to be tested. The inconvenient side is that it requires time to test all the possible combination
# 
# The other way is RandomizedSearchCV which allows a more efficient search among a huge number of hyperparameters combination

# In[ ]:


def tuning_hyperparameters(clf, grid, scoring='accuracy', rnd=True):
    if rnd:
        srch_clf = RandomizedSearchCV(clf, grid, random_state=seed, scoring=scoring, n_iter=20, n_jobs=-1)
        srch_type = "RandomizedSearchCV"
    else:
        srch_clf = GridSearchCV(clf, grid, scoring=scoring, n_jobs=-1)
        srch_type = "GridSearchCV"
    search = srch_clf.fit(X_train, y_train)
    print("The best parameters are:", search.best_params_)
    print("{} cross-val-mean {}:".format(srch_type, scoring), search.best_score_)
    y_best_pred = classifier_predictions(srch_clf, refit=False)
    best_scores = classifier_cross_val_scores(srch_clf)
    return srch_clf, y_best_pred, best_scores

def plot_map_2_parameters_tuning(param_1, param_2, srch_clf):
    mean_test_scores = srch_clf.cv_results_['mean_test_score']
    p1_vals = srch_clf.cv_results_['param_' + param_1]
    p2_vals = srch_clf.cv_results_['param_' + param_2]
    data = pd.DataFrame.from_dict({param_1: p1_vals, param_2: p2_vals, 'mean_test_score': mean_test_scores})
    pivot_table = data.pivot(index=param_1, columns=param_2, values='mean_test_score')
    plt.figure(figsize=(18, 4))
    sns.heatmap(pivot_table, vmin=mean_test_scores.min(), vmax=mean_test_scores.max(), annot=True, fmt=".3f", cmap="coolwarm")
    plt.show()
    
def plot_line_1_parameters_tuning(param, srch_clf):
    param_vals = srch_clf.cv_results_['param_' + param]
    mean_test_scores = srch_clf.cv_results_['mean_test_score']
    data = pd.DataFrame.from_dict({param: param_vals, 'mean_test_score': mean_test_scores})
    plt.figure(figsize=(18, 6))
    sns.lineplot(x=param, y='mean_test_score', data=data, estimator=None)
    plt.show()


# 
# ## 4.1 Quick Tuning with RandomizedSearchCV ##

# In[ ]:


log_reg_grid = {'penalty': ['l1', 'l2'],
                'C': np.random.uniform(0, 5, 20),
                'solver': ['lbfgs', 'liblinear', 'saga']
               }
best_logreg, y_pred_best_logreg, best_logreg_scores = tuning_hyperparameters(logreg, log_reg_grid)


# In[ ]:


svc_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
            'degree': [2, 3, 5],
            'C': np.random.uniform(0, 3, 10)
           }
best_svc, y_pred_best_svc, best_svc_scores = tuning_hyperparameters(svc, svc_grid)


# In[ ]:


rfc_grid = {'n_estimators': np.random.randint(10, 500, 10),
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 50, 100],
            'max_features': ['log2', 'sqrt', None]
           }
#good_rfc_1 = RandomForestClassifier(random_state=seed, n_estimators=127, max_features='log2', max_depth=10, criterion='entropy') #0.85 Test Accuracy
#good_rfc_2 = RandomForestClassifier(random_state=seed, n_estimators=275, max_features='sqrt', max_depth=10, criterion='gini') #0.87 Test Accuracy !! 0.824 CV
best_rfc, y_pred_best_rfc, best_rfc_scores = tuning_hyperparameters(rfc, rfc_grid)


# In[ ]:


dtc_grid = {'splitter': ['best', 'random'],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 50, 100],
            'max_features': ['log2', 'sqrt', None]
           }
best_dtc, y_pred_best_dtc, best_dtc_scores = tuning_hyperparameters(dtc, dtc_grid)


# In[ ]:


clfs_best_scores = [best_logreg_scores, best_svc_scores, best_rfc_scores, best_dtc_scores]
clfs_labels = ["logreg", "SVC","Random Forest", "Decision Tree"]
plot_box_clfs_scores(clfs_best_scores, clfs_labels)

y_dict_best_preds = {'logreg': y_pred_best_logreg, 'svc': y_pred_best_svc,
                     'rfc': y_pred_best_rfc, 'dtc': y_pred_best_dtc}
df_scores_with_tuning = create_df_scores(y_dict_best_preds, metrics, is_clf_tuned=True)
df_scores = pd.concat([df_scores_no_tuning, df_scores_with_tuning])
plot_df_scores(df_scores)


# Well, with this first tuning, we get the following accuracy improvement:

# In[ ]:


def get_acc_vals_from_scores(df_scores, clf):
    acc = df_scores.loc[(df_scores.Classifier == clf) & (df_scores.Type == 'accuracy_score'), 'Score'].values[0]
    acc_tuned = df_scores.loc[(df_scores.Classifier == clf) & (df_scores.Type == 'accuracy_score tuned'), 'Score'].values[0]
    return acc, acc_tuned

print("Logistic Regression:\t\t {0[0]} => {0[1]}".format(get_acc_vals_from_scores(df_scores, 'logreg')))
print("SVC:\t\t\t\t {0[0]} => {0[1]}".format(get_acc_vals_from_scores(df_scores, 'svc')))
print("Random Forest Classifier:\t {0[0]} => {0[1]}".format(get_acc_vals_from_scores(df_scores, 'rfc')))
print("Decision Tree Classifier:\t {0[0]} => {0[1]}".format(get_acc_vals_from_scores(df_scores, 'dtc')))


# It appears that we improve by +1% for SVC, +4% for RFC and + 6% for DTC, nice ! But looking at the cross-validation mean score, it seems that despite some improvement there could be some overfitting for the RFC and DTC..
# 
# ## 4.2 Tuning with GridSearchCV ##
# 
# # Logistic Regression:

# In[ ]:


log_reg_grid_gs = {'C': np.random.uniform(0.01, 0.3, 50),
                   'penalty': ['l1', 'l2']}
logreg_gs = LogisticRegression(random_state=seed, solver='saga', max_iter=500)

#best_logreg_gs, y_pred_best_logreg_gs, best_logreg_scores_gs = tuning_hyperparameters(logreg_gs, log_reg_grid_gs, rnd=False)
#plot_map_2_parameters_tuning('penalty','C', best_logreg_gs)


# In[ ]:


log_reg_grid_gs = {'C': np.random.uniform(0.04, 0.18, 100)}
logreg_gs = LogisticRegression(random_state=seed, penalty='l2', solver='saga', max_iter=500)

#best_logreg_gs, y_pred_best_logreg_gs, best_logreg_scores_gs = tuning_hyperparameters(logreg_gs, log_reg_grid_gs, rnd=False)

#plot_line_1_parameters_tuning('C', best_logreg_gs)


# During the tuning process, several combinations returned quite the same accuracy including the following one and we can compare the cross-validation mean accuracy of the three obtained logistic regressor (without tuning, with RandomizedSearchCV and with GridSearchCV): 

# In[ ]:


log_reg_gs_saga_l2 = LogisticRegression(C=0.054, random_state=seed, penalty='l2', solver='saga', max_iter=500,) #GSCV Mean Acc = 0.831449

log_reg_gs_saga_l2_pred = classifier_predictions(log_reg_gs_saga_l2)
log_reg_gs_saga_l2_scores = classifier_cross_val_scores(log_reg_gs_saga_l2)

log_reg_best_scores = [logreg_scores, best_logreg_scores, log_reg_gs_saga_l2_scores]
log_reg_labels = ["logreg", "logreg with RS", "logreg with GS saga-l2"]
plot_box_clfs_scores(log_reg_best_scores, log_reg_labels)

print('LogisticRegressor without tuning: Cross_val_mean_score -', logreg_scores.mean())
print('LogisticRegressor with RS tuning: Cross_val_mean_score -', best_logreg_scores.mean())
print('LogisticRegressor with GS tuning: Cross_val_mean_score -', log_reg_gs_saga_l2_scores.mean())


# Well, the tuning did not bring a significant improvement. May be we will be more successful with the other classifiers.
# 
# # SVC:

# In[ ]:


svc_grid_gs = {'C': np.random.uniform(4, 5, 100)}
svc_gs = SVC(random_state=seed, kernel='poly', degree=1, gamma='auto')

#best_svc_gs, y_pred_best_svc_gs, best_svc_scores_gs = tuning_hyperparameters(svc_gs, svc_grid_gs, rnd=False)
#plot_map_2_parameters_tuning('kernel','C', best_svc_gs)
#plot_line_1_parameters_tuning('C', best_svc_gs)


# In[ ]:


svc_gs_poly = SVC(C=4.4, random_state=seed, kernel='poly', degree=1, max_iter=500, gamma='auto')

svc_gs_poly_pred = classifier_predictions(svc_gs_poly)
svc_gs_poly_scores = classifier_cross_val_scores(svc_gs_poly)

svc_best_scores = [svc_scores, best_svc_scores, svc_gs_poly_scores]
svc_labels = ["SVC", "SVC with RS", "SVC with GS poly-1"]
plot_box_clfs_scores(svc_best_scores, svc_labels)

print('SVC without tuning: Cross_val_mean_score -', svc_scores.mean())
print('SVC with RS tuning: Cross_val_mean_score -', best_svc_scores.mean())
print('SVC with GS tuning: Cross_val_mean_score -', svc_gs_poly_scores.mean())


# # Random Forest Classifier

# In[ ]:


rfc_grid_gs = {'n_estimators': [345, 370, 390, 410, 420, 490]}
rfc_gs = RandomForestClassifier(max_depth=7, random_state=seed, criterion='entropy', max_features='sqrt', class_weight='balanced', n_jobs=-1)

#best_rfc_gs, y_pred_best_rfc_gs, best_rfc_scores_gs = tuning_hyperparameters(rfc_gs, rfc_grid_gs, rnd=False)
#plot_map_2_parameters_tuning('n_estimators','max_depth', best_rfc_gs)
#plot_line_1_parameters_tuning('n_estimators', best_rfc_gs)


# In[ ]:


rfc_gs = RandomForestClassifier(random_state=seed, max_features='sqrt',
                                class_weight='balanced', max_depth=7, n_estimators=345,
                                criterion='entropy') 

rfc_gs_pred = classifier_predictions(rfc_gs)
rfc_gs_scores = classifier_cross_val_scores(rfc_gs)

rfc_best_scores = [rfc_scores, best_rfc_scores, rfc_gs_scores]
rfc_labels = ["RFC", "RFC with RS", "RFC with GS"]
plot_box_clfs_scores(rfc_best_scores, rfc_labels)

print('RFC without tuning: Cross_val_mean_score -', rfc_scores.mean())
print('RFC with RS tuning: Cross_val_mean_score -', best_rfc_scores.mean())
print('RFC with GS tuning: Cross_val_mean_score -', rfc_gs_scores.mean())


# In[ ]:


dtc_grid_gs = {'max_features': np.append(np.random.uniform(0.6, 0.99,50), None)}
dtc_gs = DecisionTreeClassifier(max_depth=4, random_state=seed, splitter='best', criterion='entropy', class_weight='balanced')

#best_dtc_gs, y_pred_best_dtc_gs, best_dtc_scores_gs = tuning_hyperparameters(dtc_gs, dtc_grid_gs, rnd=False)
#plot_map_2_parameters_tuning('max_depth','max_features', best_dtc_gs)
#plot_line_1_parameters_tuning('max_features', best_dtc_gs)


# In[ ]:


dtc_gs = DecisionTreeClassifier(max_depth=4, random_state=seed, splitter='best', criterion='entropy',
                                max_features=0.65, class_weight='balanced') 

dtc_gs_pred = classifier_predictions(dtc_gs)
dtc_gs_scores = classifier_cross_val_scores(dtc_gs)

dtc_best_scores = [dtc_scores, best_dtc_scores, dtc_gs_scores]
dtc_labels = ["DTC", "DTC with RS", "DTC with GS"]
plot_box_clfs_scores(dtc_best_scores, dtc_labels)

print('DTC without tuning: Cross_val_mean_score -', dtc_scores.mean())
print('DTC with RS tuning: Cross_val_mean_score -', best_dtc_scores.mean())
print('DTC with GS tuning: Cross_val_mean_score -', dtc_gs_scores.mean())


# So, to conlude this tuning chapter, it appears that there is a slightly improvement on the training but there are still some subsets with no improvement.
# 
# 
# # 5 Ensembling / Stacking #

# In[ ]:


test_preds_dict = {'LogReg': log_reg_gs_saga_l2_pred,
                   'SVC': svc_gs_poly_pred,
                   'RandomForest': rfc_gs_pred,
                   'DecisionTree': dtc_gs_pred}
df_test_preds = pd.DataFrame().from_dict(test_preds_dict)
df_test_preds['Sum'] = df_test_preds['LogReg'] + df_test_preds['SVC'] + df_test_preds['RandomForest'] + df_test_preds['DecisionTree']
df_test_preds['Prod'] = df_test_preds['LogReg'] * df_test_preds['SVC'] * df_test_preds['RandomForest'] * df_test_preds['DecisionTree']

train_preds_dict = {'LogReg': log_reg_gs_saga_l2.predict(X_train),
                    'SVC': svc_gs_poly.predict(X_train),
                    'RandomForest': rfc_gs.predict(X_train),
                    'DecisionTree': dtc_gs.predict(X_train)}
df_train_preds = pd.DataFrame().from_dict(train_preds_dict)
df_train_preds['Sum'] = df_train_preds['LogReg'] + df_train_preds['SVC'] + df_train_preds['RandomForest'] + df_train_preds['DecisionTree']
df_train_preds['Prod'] = df_train_preds['LogReg'] * df_train_preds['SVC'] * df_train_preds['RandomForest'] * df_train_preds['DecisionTree']

X_train_2 = df_test_preds
y_train_2 = y_test
X_test_2 = df_train_preds
y_test_2 = y_train


# In[ ]:


knn = KNeighborsClassifier()

knn.fit(X_train_2, y_train_2)
y_knn_pred = knn.predict(X_test_2)
print('KNN Accuracy:', accuracy_score(y_test_2, y_knn_pred))
print('KNN F1 score:', f1_score(y_test_2, y_knn_pred))


# In[ ]:


valid_dict = {'LogReg': log_reg_gs_saga_l2.predict(X_valid_norm),
              'SVC': svc_gs_poly.predict(X_valid_norm),
              'RandomForest': rfc_gs.predict(X_valid_norm),
              'DecisionTree': dtc_gs.predict(X_valid_norm)}
df_valid = pd.DataFrame().from_dict(valid_dict)
df_valid['Sum'] = df_valid['LogReg'] + df_valid['SVC'] + df_valid['RandomForest'] + df_valid['DecisionTree']
df_valid['Prod'] = df_valid['LogReg'] * df_valid['SVC'] * df_valid['RandomForest'] * df_valid['DecisionTree']

X_valid_2 = df_valid
y_valid_pred = knn.predict(X_valid_2)


# # 5 Preparing the submission #
# Now the last step, preparing our file for subission

# In[ ]:


ids = test.PassengerId
surv_df = pd.DataFrame({ 'PassengerId' : ids, 'Survived': y_valid_pred})
surv_df.to_csv('submission_titanic.csv', index=False)

