#!/usr/bin/env python
# coding: utf-8

# # Importing libs

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV 


# # Preprocessing

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train.info()


# On the train dataset, there are null values in features Age, Cabin and Embarked. An option could be get rid of the instances, but the dataset is too small, so the model would have less data to train on. Another option could be get rid of the features with null values instead, but for the same reason it wouldn't be good.

# # Embarked

# Instead, I chose to replace the 2 null values in Embarked for the mode (most frequent value). To keep the code clean, let's build a DataFrame selector and a pipeline to handle it.

# In[ ]:


class Selector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.feature_names].values
    
embarked_pipeline = Pipeline([
    ('selector', Selector( ['Embarked'] )),
    ('imputer', SimpleImputer( strategy='most_frequent' ))
])


# # Cabin

# Now for the Cabin feature, let's get the first letter of the Cabin and see if there's a correlation to the Survived column: 

# In[ ]:


first_letter = train['Cabin'].apply(lambda x: x[0] if not pd.isna(x) else 0)
sns.countplot(x=first_letter, hue=train['Survived'])


# It's possible to see that passengers without Cabin are more likely to die. We can create a feature that will hopefully help improve our model.

# In[ ]:


class CabinTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['First_Cabin_Letter'] = X['Cabin'].apply(lambda x: x[0] if not pd.isna(x) else 'X')
        return X

cabin_pipeline = Pipeline([
    ('cabin_tr', CabinTransformer()),
    ('selector', Selector(['First_Cabin_Letter']))
])


# # Age

# For the Age feature, let's visualize some plots of the other features.

# In[ ]:


sns.barplot(x=train['Pclass'], y=train['Age'], hue=train['Sex'])


# We can see that as the passenger class increases, the mean age of the class increases as well.

# In[ ]:


sns.barplot(x=train['Sex'], y=train['Age'])


# In[ ]:


sns.barplot(x=train['Parch'], y=train['Age'])


# In[ ]:


sns.barplot(x=train['SibSp'], y=train['Age'])


# In[ ]:


ax = sns.barplot(x=train['SibSp'] + train['Parch'] + 1, y=train['Age'])
ax.set(xlabel='Family Size')


# In[ ]:


sns.countplot(x=train['SibSp'] + train['Parch'] + 1)


# Including Parch and SibSp features when imputing Age could be dangerous due to the high variance for family size higher than 3. Later on we can create a feature on that. So I will avoid considering them to impute the missing age values. Based on that, I chose to impute the mean age based on the sex and passenger class.

# In[ ]:


male_mean_age = train[train['Sex'] == 'male'].groupby('Pclass').mean()['Age'].values
female_mean_age = train[train['Sex'] == 'female'].groupby('Pclass').mean()['Age'].values


# In[ ]:


class AgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        for index, _ in X.iterrows():
            if pd.isna(X.loc[index, 'Age']):

                if X.loc[index, 'Sex'] == 'male':
                    if X.loc[index, 'Pclass'] == 1:
                        X.loc[index, 'Age'] = male_mean_age[0]
                    elif X.loc[index, 'Pclass'] == 2:
                        X.loc[index, 'Age'] = male_mean_age[1]
                    else:
                        X.loc[index, 'Age'] = male_mean_age[2]

                else:      
                    if X.loc[index, 'Pclass'] == 1:
                        X.loc[index, 'Age'] = female_mean_age[0]
                    elif X.loc[index, 'Pclass'] == 2:
                        X.loc[index, 'Age'] = female_mean_age[1]
                    else:
                        X.loc[index, 'Age'] = female_mean_age[2]
        return X

age_pipeline = Pipeline([
    ('age_tr', AgeTransformer()),
    ('selector', Selector(['Age']))
])


# # Name

# Let's take a closer look to the Name feature.

# In[ ]:


train['Name'].head(10)


# It seems like every passenger has a last name followed by a comma and the passenger's title followed by his first name. So we can create a feature Title and see if it has a correlation with the target variable.

# In[ ]:


def getTitle(name):
    return name.split(',')[1].split('.')[0].split()[0]

train['Title'] = train['Name'].apply(getTitle)
test['Title'] = test['Name'].apply(getTitle)

train.Title.value_counts()
test.Title.value_counts()


# In[ ]:


class TitleAttributeAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        titles = X['Name'].apply(getTitle)
        less_titles_count = titles.value_counts()[4:].keys()
        titles = titles.apply(lambda x: 'residue' if x in less_titles_count else x)
        return titles.values.reshape(-1, 1)
    
name_pipeline = TitleAttributeAdder()


# I chose to place the titles with fewer counts in a single bucket to keep the features clean.

# In[ ]:


sns.countplot(name_pipeline.fit_transform(train).reshape(-1), hue=train['Survived'])


# We can see that there's a strong correlation between this feature and the Sex feature. The buckets Mr, Mrs and Miss are as expected: men with lower survival rate than women. The Master bucket, though, has a much more higher survival rate than the Mr bucket. Maybe Master was a Title for wealthier people, so they got a higher cabin in the ship (the lower classes were below) and had a better chance to survive.

# # SibSp and Parch

# Let's create a feature for handling the family size of each passenger (SibSp + Parch + 1).

# In[ ]:


train['Family Size'] = train['SibSp'] + train['Parch'] + 1
train['Family Size'].value_counts()


# In[ ]:


sns.barplot(x=train['Family Size'], y=train['Survived'])


# There is a very high variance for family size higher than 4. Let's place those in a single bucket. Also, let's join the families with size 3 and 4 and keep the 1 and 2. 

# In[ ]:


train['Family Size'] = train['Family Size'].replace(np.arange(5, 12), 'Large')
train['Family Size'] = train['Family Size'].replace([3, 4], 'Medium')
train['Family Size'] = train['Family Size'].replace(2, 'Small')
train['Family Size'] = train['Family Size'].replace(1, 'Alone')


# In[ ]:


sns.barplot(x=train['Family Size'], y=train['Survived'])


# In[ ]:


class FamilySize_Attribute_Adder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        family_size = X['SibSp'] + X['Parch'] + 1
        
        family_size = family_size.replace(np.arange(5, 12), 'Large')
        family_size = family_size.replace([3, 4], 'Medium')
        family_size = family_size.replace(2, 'Small')
        family_size = family_size.replace(1, 'Alone')
        
        return family_size.values.reshape(-1, 1)

family_pipeline = FamilySize_Attribute_Adder()


# # Pclass and Sex

# Passenger class is a strong correlated feature as the most expensive classes were in the top of the ship, having a better chance to survive. Let's see if the data matches this assumption.

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train)


# The assumption was correct, as the survival rate is higher in the first class, and the third class has the lower survival rate of all.

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train)


# In[ ]:


pcl_sex_pipeline = Selector(['Pclass', 'Sex'])


# Just like Pclass, Sex is a strong correlated feature as women and children had priority in the ship evacuation. The data corresponds to it. After analysing the remaining features, we'll one hot encode this and the other categorical features.

# # Ticket

# Let's take a closer look at the ticket feature.

# In[ ]:


# Getting 20 random instances of the train dataset
np.random.seed(42)
train.Ticket.loc[np.random.randint(0, len(train), size=20)]


# There are some tickets that are only numerical, and others have a string before the numbers. Let's extract this from data.

# In[ ]:


is_numerical = train['Ticket'].apply(lambda x: x.split()[0][0])
is_numerical = is_numerical.replace('1 2 3 4 5 6 7 8 9'.split(), 1)
is_numerical = is_numerical.apply(lambda x: 1 if type(x)==int else 0)


# In[ ]:


is_numerical.value_counts()


# In[ ]:


sns.barplot(x=is_numerical, y=train.Survived)


# Aside from the variance, if the ticket begins with numerical or non-numerical values doesn't affect the passenger survival. So I chose to drop this feature.

# # Fare

# We can see the distribution of the Fare feature in the graphic below.

# In[ ]:


sns.distplot(train.Fare)


# We can see that this distribution has a very high skewness, so let's apply a logarithm or root square to each point and take a look at the new plot.

# In[ ]:


train['Fare'] = train['Fare'].apply(np.sqrt)
sns.distplot(train.Fare)


# Much better, even though there are a few values higher than 15 that could be considered outliers. Let's check that.

# In[ ]:


first_quartile = np.quantile(train.Fare, 0.25)
third_quartile = np.quantile(train.Fare, 0.75)
interquartile_amplitude = third_quartile - first_quartile
lower_limit = first_quartile - 1.5 * interquartile_amplitude
higher_limit = third_quartile + 1.5 * interquartile_amplitude
train.loc[(train.Fare > higher_limit), 'Fare'] = higher_limit


# In[ ]:


sns.distplot(train.Fare)


# Great! Now the distribution looks more smooth.

# In[ ]:


class FareTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        fare_sqrt = X.Fare.apply(lambda x: np.log(x + 0.5))
        self.mean = np.mean(fare_sqrt)
        first_quartile = np.quantile(fare_sqrt, 0.25)
        third_quartile = np.quantile(fare_sqrt, 0.75)
        interquartile_amplitude = third_quartile - first_quartile
        lower_limit = first_quartile - 1.5 * interquartile_amplitude
        higher_limit = third_quartile + 1.5 * interquartile_amplitude
        
        self.higher_limit = higher_limit
        self.lower_limit = lower_limit
        return self
    def transform(self, X):
        fare_sqrt = X.Fare.apply(lambda x: np.log(x + 0.5))
        fare_sqrt.fillna(self.mean, inplace=True)
        fare_sqrt = fare_sqrt.where(fare_sqrt > self.lower_limit, self.lower_limit)
        fare_sqrt = fare_sqrt.where(fare_sqrt < self.higher_limit, self.higher_limit)
        return fare_sqrt.values.reshape(-1, 1)
    
fare_pipeline = FareTransformer()


# Also, the test dataset has one missing value in the Fare column. So let's take the mean and fill it.

# # Data Preparation

# To prepare the data, I will concatenate the numerical and categorical features separately. Then, standardize the numerical ones and one hot encode the categorical ones.

# In[ ]:


numerical_pipeline = FeatureUnion([ 
    ('age_pipe', age_pipeline),
    ('fare_pipe', fare_pipeline)
])

numerical_pipeline = Pipeline([
    ('num_pipe', numerical_pipeline),
    ('scaler', StandardScaler())
])

categorical_pipeline = FeatureUnion([
    ('embarked_pipe', embarked_pipeline),
    ('pcl_sex_pipe', pcl_sex_pipeline),
    ('name_pipe', name_pipeline),
    ('family_pipe', family_pipeline),
    ('cabin_pipe', cabin_pipeline)
])

categorical_pipeline = Pipeline([
    ('cat_pipe', categorical_pipeline),
    ('encoder', OneHotEncoder(drop='first'))
])


# Then we can concatenate the last two pipelines.

# In[ ]:


prepared_data_pipeline = FeatureUnion([
    ('num_pipe', numerical_pipeline),
    ('cat_pipe', categorical_pipeline)
])


# And by loading the data again, we only have to pass it through the pipeline. At this point, we end up with 22 feature columns.

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

X_train = prepared_data_pipeline.fit_transform(train)
y_train = train.Survived
X_test = prepared_data_pipeline.transform(test)


# # Model Selection

# In a first moment, I will evaluate very different kinds of model. Then I'll choose a subset of the best models and apply a Voting Classifier. Let's start by the RFC model.

# In[ ]:


scores = []
param_grid = {'n_estimators': np.arange(10, 100, 5),
              'max_depth': np.arange(3, 8)}

rf_clf_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, verbose=1)
rf_clf_grid.fit(X_train, y_train)

scores.append(('RFC', rf_clf_grid.best_score_))
rf_clf_best = rf_clf_grid.best_estimator_


# Extra Trees Classifier.

# In[ ]:


param_grid = {'n_estimators': np.arange(10, 100, 5),
              'max_depth': np.arange(2, 8)}

et_clf_grid = GridSearchCV(ExtraTreesClassifier(), param_grid, cv=3, verbose=1)
et_clf_grid.fit(X_train, y_train)

scores.append(('ETC', et_clf_grid.best_score_))
et_clf_best = et_clf_grid.best_estimator_


# Logistic Regression.

# In[ ]:


param_grid = {'C': [0.1, 1, 2, 3, 4, 5]}

log_reg_grid = GridSearchCV(LogisticRegression(), param_grid, cv=3, verbose=1)
log_reg_grid.fit(X_train, y_train)

scores.append(("Logistic Regression",log_reg_grid.best_score_))
log_reg_best = log_reg_grid.best_estimator_


# Support Vector Machine.

# In[ ]:


param_grid = {'C': np.linspace(1, 3, 11), 
              'kernel': ['rbf', 'poly','linear'], 
              'degree': [2, 3, 4]}

svc_grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, verbose=1)
svc_grid.fit(X_train, y_train)

scores.append(('SVC', svc_grid.best_score_))
svc_best = svc_grid.best_estimator_


# KNeighbors Classifier.

# In[ ]:


param_grid = {'weights': ['uniform', 'distance'], 
              'n_neighbors': np.arange(2, 20)}

kn_clf_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, verbose=1)
kn_clf_grid.fit(X_train, y_train)

scores.append(('KNC', kn_clf_grid.best_score_))
kn_clf_best = kn_clf_grid.best_estimator_


# In[ ]:


scores


# All the models got approximately the same score. As RFC and ETC are tree based models, I chose to keep RFC for the ensemble. Also, I kept the three remaining models.

# In[ ]:


voting_clf = VotingClassifier([
    ('rf_clf', rf_clf_best),
    ('log_reg', log_reg_best),
    ('svc', svc_best),
    ('kn_clf', kn_clf_best)
], voting='soft')

voting_clf.fit(X_train, y_train)
predictions = voting_clf.predict(X_test)


# # Submission

# In[ ]:


submission = pd.Series(predictions, index=test.PassengerId, name='Survived')
submission.to_csv('titanic_submission.csv', header=True)


# Thanks if you read until here. Leave a comment below if I made some mistake or tell me where can I improve. I'll be happy to answer!
