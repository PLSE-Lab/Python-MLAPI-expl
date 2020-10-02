#!/usr/bin/env python
# coding: utf-8

# # Titanic Machine Learning compitetion

# ## Import Libraries

# In[ ]:


import re
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.feature_selection import SelectKBest,chi2


# ## Import DataSet

# Import training and testing dataset and combine them for filling missing value and for feature engineering.

# In[ ]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


df_train['train'] = 1
df_test['train']  = 0
data = df_train.append(df_test, ignore_index=True)
data.info()


# The dataset is pretty clean with Only Three columns **Age, Ebmarked, Fare'** have Some missing values and **Cabin** has lots of missing values.

# As **Cabin** as lots of missing value so we are going to leave as it now and think about it in feature engineering process. Fill missing values of Other three features.

# Let first explore them a little bit.

# In[ ]:


sns.distplot(data['Age'].dropna())
print('Min = {}, Max= {}'.format(data['Age'].min(), data['Age'].max()))


# Distribution is almost Normal Unimodel with right skewed. we can fill Age with mean.  

# In[ ]:


sns.distplot(data['Fare'].dropna())
print('Min = {}, Max= {}'.format(data['Fare'].min(), data['Fare'].max()))


# The distribution is Unimodel with right skewed and has some outlier on right side. We must fill missing value with medium because mean is skewed because of outlier.

# In[ ]:


data['Embarked'].value_counts()


# Fill missing value in Embarked with most common value **(S)**.

# ## Filling Missing value

# In[ ]:


data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Embarked'].fillna('S', inplace=True)


# In[ ]:


data.isnull().sum()


# ## Eploratory Data Analysis of Traning Data

# Now we Explore training data so we can find feature that have more impact on survival of passenger.

# In[ ]:


# Getting Training data from Full dataset
train = data[data['train']==1]


# In[ ]:


train['Survived'].value_counts()


# The outcome is not equally distributed instead it skewed class.In Our training set out of 819 only 342 peoples managed to survived.

# In[ ]:


# chi-square test to test independance of two categorical varaible.
def chi_test_categorical_feature(alpha, feature, target='Survived'):
    contigency_pclass = pd.crosstab(train[feature], train[target])
    stat, p, dof, expected = chi2_contingency(contigency_pclass)
    if p < alpha:
        print('There is relationship btw {} and target variable with p_value = {} and Chi-squared = {}'.format(feature, p, stat) )
    else:
        print('not good predictor with p_value = {} and Chi-squared = {}'.format( p, stat))


# ### Sex 

# In[ ]:


train['Sex'].value_counts()


# In[ ]:


sns.countplot(train['Sex'], hue=train['Survived'])


# In[ ]:


# for making contegancy table 
pd.crosstab(train['Sex'], train['Survived'], normalize='all')*100


# In[ ]:


chi_test_categorical_feature(0.01, 'Sex')


# There is  stong relationship btw **Sex** and **Survived**

# ### Pclass

# In[ ]:


train['Pclass'].value_counts()


# In[ ]:


sns.countplot(train['Pclass'], hue=train['Survived'])


# In[ ]:


pd.crosstab(train['Pclass'], train['Survived'], normalize='all')*100


# In[ ]:


chi_test_categorical_feature(0.01, 'Pclass')


# ### SibSp

# In[ ]:


train['SibSp'].value_counts()


# In[ ]:


sns.countplot(train['SibSp'], hue=train['Survived'])


# Here we seen that Person that has one sibling or spouses are more likly to survived. All other category have less likely to survived.

# In[ ]:


pd.crosstab(train['SibSp'], train['Survived'], normalize='all')*100


# The Person That have more then Four sibling or Spouses are very less likely to survived.

# In[ ]:


chi_test_categorical_feature(0.01, 'SibSp')


# ### Parch

# In[ ]:


train['Parch'].value_counts()


# In[ ]:


sns.countplot(train['Parch'], hue=train['Survived'])


# Here we also seen that person with one children or parent are more likely to survived.

# In[ ]:


pd.crosstab(train['Parch'], train['Survived'], normalize='all')*100


# In[ ]:


chi_test_categorical_feature(0.01, 'Parch')


# We can Create Feature Like family_size which can be helpin predicting outcome.

# ### Embarked

# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


sns.countplot(train['Embarked'], hue=train['Survived'])


# Person who get in titanic from **C** is more likely to survived then other station.

# In[ ]:


pd.crosstab(train['Embarked'], train['Survived'], normalize='all')*100


# In[ ]:


chi_test_categorical_feature(0.01, 'Embarked')


# ### Age

# As age is float varibale to so we perform different analysis on this.

# In[ ]:


sns.boxplot(y=train['Age'], x=train['Survived'])


# From above boxplot we can not seen any clear difference between distribution of survived and died.

# In[ ]:


sns.violinplot(y=train['Age'], hue=train['Survived'], x=[""]*len(train), palette="Set2")


# In[ ]:


train[['Age', 'Survived']].corr()


# As we can see very weak correlation btw Age and Survived feature.we can create bins of age feature to make it more useful.
# 

# ### Fare

# In[ ]:


sns.boxplot(y=train['Fare'], x=train['Survived'])


# In[ ]:


train[['Fare', 'Survived']].corr()


# In[ ]:


sns.violinplot(y=train['Fare'], hue=train['Survived'], x=[""]*len(train), palette="Set2")


# As fare increase mean higher Pclass so more chance of survival.

# ### Cabin

# In[ ]:


train[train['Cabin'].isnull()]['Survived'].value_counts()


# In[ ]:


train[train['Cabin'].isnull()==False]['Survived'].value_counts()


# In[ ]:


pd.crosstab([train['Cabin'].isnull()], train['Survived'], normalize='all')*100


# Person that have cabin feature is more likely to survive then person that have missing cabin.

# ## Feature Engineering

# In[ ]:


# creating featur fromcabin column if cabin exist then 1 else 0
data['Has_Cabin'] = ~data['Cabin'].isnull()
data['Has_Cabin'] = data['Has_Cabin'].astype(int)


# In[ ]:


train = data[['Has_Cabin', 'Survived']]
sns.countplot(train['Has_Cabin'], hue=train['Survived'])


# In[ ]:


# Creating feature FamilySize from SibSp and Parch
data['Family_Size'] = data['SibSp'] + data['Parch'] + 1


# In[ ]:


train = data[['Family_Size', 'Survived']]
sns.countplot(train['Family_Size'], hue=train['Survived'])


# In[ ]:


data['Is_Alone'] = data['Family_Size'].apply(lambda x: 1 if x==1 else 0)


# In[ ]:


train = data[['Is_Alone', 'Survived']]
sns.countplot(train['Is_Alone'], hue=train['Survived'])


# In[ ]:


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


# In[ ]:


data['Title'] = data['Name'].apply(get_title)


# In[ ]:


data['Title'].value_counts()


# In[ ]:


data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'OTHER')


# In[ ]:


data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')


# In[ ]:


train = data[['Title', 'Survived']]
sns.countplot(train['Title'], hue=train['Survived'])


# As we can see the Mrs, Miss, Master are more likely to survived so it is good feature for making prediction.

# ## DataTypes  &  Binning

# In[ ]:


data.info()


# In[ ]:


data['Age'].max()


# In[ ]:


bins = np.linspace(0, 80, 6)
data['Age_binned']= pd.cut(data['Age'], bins, labels=[1,2,3,4,5], include_lowest=True)
data['Age_binned'] = data['Age_binned'].astype(int)


# In[ ]:


train = data[['Age_binned', 'Survived']]
sns.countplot(train['Age_binned'], hue=train['Survived'])


# In[ ]:


bins = [-1,50,100,390, 520]
data['Fare_binned'] = pd.cut(data['Fare'], bins ,labels=[1,2,3,4], include_lowest=True)
data['Fare_binned'] = data['Fare_binned'].astype(int)


# In[ ]:


train = data[['Fare_binned', 'Survived']]
sns.countplot(train['Fare_binned'], hue=train['Survived'])


# In[ ]:


data['Sex'].replace({'male': 1, 'female': 0}, inplace=True)
data['Embarked'].replace({'S': 1, 'C': 2, 'Q': 3}, inplace=True)
data['Title'].replace({'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Master': 4, 'OTHER': 5}, inplace=True)


# ## Data Transformation

# In[ ]:


feature = ['Embarked','Pclass', 'Sex', 'SibSp','Parch', 'Has_Cabin', 'Family_Size', 'Is_Alone', 'Title', 'Age_binned', 'Fare_binned']


# In[ ]:


# converting feature to category so that we perform encoding on them.
data[feature] = data[feature].astype('category')
dummy_data = pd.get_dummies(data[feature])
# join with orginal dataset
data = pd.concat([data, dummy_data], axis=1)


# In[ ]:


# Separating training and testing data. 
training_data = data[data['train']==1]
testing_data = data[data['train']==0]


# ## Machine Learning

# All Features that we get after feature engineering.

# In[ ]:


# All features
feature_1 = ['Embarked_1', 'Embarked_2', 'Embarked_3', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_0', 'Sex_1',
             'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1',
             'Parch_2','Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Parch_9','Has_Cabin_0', 'Has_Cabin_1', 'Family_Size_1',
             'Family_Size_2', 'Family_Size_3', 'Family_Size_4', 'Family_Size_5','Family_Size_6', 'Family_Size_7',
             'Family_Size_8', 'Family_Size_11','Is_Alone_0', 'Is_Alone_1', 'Title_1','Title_2', 'Title_3', 'Title_4',
             'Title_5','Age_binned_1', 'Age_binned_2', 'Age_binned_3', 'Age_binned_4','Age_binned_5', 'Fare_binned_1',
             'Fare_binned_2', 'Fare_binned_3','Fare_binned_4'] 


# #### Selecting features By Chi-Squared Method.

# In[ ]:


feature_set = []
chi2_selector = SelectKBest(chi2, k=30)
chi2_selector.fit_transform(training_data[feature_1], y=training_data['Survived'].astype(int))
for feature, chi_result in zip(feature_1, chi2_selector.get_support()):
    if chi_result==True:
        feature_set.append(feature)


# ### Training Model

# In[ ]:


train_accuracy = pd.DataFrame(columns=['Name of Model', 'Accuracy'])


# In[ ]:


seed = 101


# In[ ]:


# 1.Decision Tree Classifier
dt = DecisionTreeClassifier(random_state = seed)

# 2.Support Vector Machines
svc = SVC(gamma = 'auto')

# 3.Random Forest Classifier

rf = RandomForestClassifier(random_state = seed, n_estimators = 100)

#4.Gaussian Naive Bayes
gnb = GaussianNB()

#5.Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state = seed)

#6.Adaboost Classifier
abc = AdaBoostClassifier(random_state = seed)

#7.ExtraTrees Classifier
etc = ExtraTreesClassifier(random_state = seed)

#10.Extreme Gradient Boosting
xgbc = XGBClassifier(random_state = seed)


# In[ ]:


clf_list = [dt, svc, rf, gnb, gbc, abc, etc, xgbc]
clf_list_name = ['dt', 'svc', 'rf', 'gnb', 'gbc', 'abc', 'etc', 'xgbc']


# In[ ]:


def train_accuracy_model(model):
    model.fit(X_train, y_train)
    accuracy = (model.score(X_train, y_train))*100
    return accuracy


# In[ ]:


# For Feature set 1 which countain all Feature.
X_train = training_data[feature_set]
y_train = training_data['Survived'].astype(int)
for clf, name in zip(clf_list, clf_list_name):
    accuracy = train_accuracy_model(clf)
    r = train_accuracy.shape[0]
    train_accuracy.loc[r] = [name, accuracy]


# In[ ]:


train_accuracy.sort_values(by='Accuracy', ascending=False)


# Looks like all the tree based models have highest train accuracy followed by SVC and GBN.But train accuracy of a model is not enough to tell if a model can be able to generalize the unseen data or not. so We perform Cross Validation. 

# ### Cross validation.

# In[ ]:


cross_val_df = pd.DataFrame(columns=['Name of Model', 'Accuracy'])


# In[ ]:


def cross_val_accuracy(model):
    score = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1).mean()
    score = np.round(score*100, 2)
    return score
    


# In[ ]:


# For Feature set 1 which countain all Feature.
X_train = training_data[feature_set]
y_train = training_data['Survived'].astype(int)
for clf, name in zip(clf_list, clf_list_name):
    accuracy = cross_val_accuracy(clf)
    r = cross_val_df.shape[0]
    cross_val_df.loc[r] = [name, accuracy]


# In[ ]:


cross_val_df.sort_values(by='Accuracy', ascending=False)


# Now Gradient-Boost Calssifier and Xgbc is doing best among all other model. So let try hyperparameter Tuning to select model.

# ### Hyperparameter Tunning

# In[ ]:


#Define dataframe for Parameter tuning.
# Accuracy here is mean value of cross validation score of model with best paramters
param_df = pd.DataFrame(columns=['Name of Model', 'Accuracy', 'Parameter'])


# In[ ]:


# For GBC, the following hyperparameters are usually tunned.
gbc_params = {'learning_rate': [0.01, 0.02, 0.05, 0.01],
              'max_depth': [4, 6, 8],
              'max_features': [1.0, 0.3, 0.1], 
              'min_samples_split': [ 2, 3, 4],
              'random_state':[seed]}

# For SVC, the following hyperparameters are usually tunned.
svc_params = {'C': [6, 7, 8, 9, 10], 
              'kernel': ['linear','rbf'],
              'gamma': [0.5, 0.2, 0.1, 0.001, 0.0001]}

# For DT, the following hyperparameters are usually tunned.
dt_params = {'max_features': ['auto', 'sqrt', 'log2'],
             'min_samples_split': [2, 3, 4, 5, 6, 7, 8], 
             'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8],
             'random_state':[seed]}

# For RF, the following hyperparameters are usually tunned.
rf_params = {'criterion':['gini','entropy'],
             'n_estimators':[10, 15, 20, 25, 30],
             'min_samples_leaf':[1, 2, 3],
             'min_samples_split':[3, 4, 5, 6, 7], 
             'max_features':['sqrt', 'auto', 'log2'],
             'random_state':[44]}


# For ABC, the following hyperparameters are usually tunned.'''
abc_params = {'n_estimators':[1, 5, 10, 50, 100, 200],
              'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3, 1.5],
              'random_state':[seed]}

# For ETC, the following hyperparameters are usually tunned.
etc_params = {'max_depth':[None],
              'max_features':[1, 3, 10],
              'min_samples_split':[2, 3, 10],
              'min_samples_leaf':[1, 3, 10],
              'bootstrap':[False],
              'n_estimators':[100, 300],
              'criterion':["gini"], 
              'random_state':[seed]}

# For XGBC, the following hyperparameters are usually tunned.
xgbc_params = {'n_estimators': (150, 250, 350,450,550,650, 700, 800, 850, 1000),
              'learning_rate': (0.01, 0.6),
              'subsample': (0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': (0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4],
              'random_state':[seed]}


# In[ ]:


clf_list = [dt, svc, rf, gbc, abc, etc, xgbc]
clf_list_name = ['dt', 'svc', 'rf', 'gbc', 'abc', 'etc', 'xgbc']
clf_param_list = [dt_params, svc_params, rf_params, gbc_params, abc_params, etc_params, xgbc_params]


# In[ ]:


# Create a function to tune hyperparameters of the selected models.'''
def tune_hyperparameters(model, params):
    from sklearn.model_selection import GridSearchCV
    # Construct grid search object with 10 fold cross validation.
    grid = GridSearchCV(model, params, verbose = 0, cv = 10, scoring = 'accuracy', n_jobs = -1)
    # Fit using grid search.
    grid.fit(X_train, y_train)
    best_params, best_score = grid.best_params_, np.round(grid.best_score_*100, 2)
    return best_params, best_score


# In[ ]:


# Tuning Parameters of all Model
X_train = training_data[feature_set]
y_train = training_data['Survived'].astype(int)
for clf, name, params in zip(clf_list, clf_list_name, clf_param_list):
    best_params, best_score = tune_hyperparameters(clf, params)
    r = param_df.shape[0]
    param_df.loc[r] = [name, best_score, best_params]


# In[ ]:


param_df.sort_values(by='Accuracy', ascending=False)


# In[ ]:


param_df.to_pickle("./dummy.pkl")


# ### Ploting Learning Curve

# In[ ]:


def ploting_learning_curve(model):
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model, 
                                                        X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))


    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--',  label="Training score")
    plt.plot(train_sizes, test_mean, label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# In[ ]:


#  RandomFrostClassifier Learning curve
model = RandomForestClassifier(criterion='entropy', max_features='log2', min_samples_leaf=1, min_samples_split=7,
                      n_estimators=20, random_state = 44)
ploting_learning_curve(model)


# In[ ]:


model = XGBClassifier(colsample_bytree= 0.9, learning_rate= 0.01, max_depth= 6, min_child_weight= 2,
                      n_estimators= 1000, random_state= 101, subsample= 0.3)
ploting_learning_curve(model)


# In[ ]:


model = GradientBoostingClassifier(learning_rate= 0.02, max_depth= 4, max_features= 0.3, min_samples_split= 4,
                                   random_state= 101)
ploting_learning_curve(model)


# As Standard deviation of XGBC is less and accuracy is also high enough so we used it for Final submission.

# ### Final submission

# In[ ]:


X_train = training_data[feature_set]
y_train = training_data['Survived'].astype(int)
model = XGBClassifier(colsample_bytree= 0.9, learning_rate= 0.01, max_depth= 6, min_child_weight= 2,
                      n_estimators= 1000, random_state= 101, subsample= 0.3)


model.fit(X_train, y_train)


# In[ ]:


X_test=testing_data[feature_set]
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)

