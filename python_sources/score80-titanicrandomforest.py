#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

# Standard plot using matplotlib
from matplotlib import pyplot as plt

# Enhance plotting/visulization using seaborn
# Seaborn uses matplotlib as base
import seaborn as sb
import numpy as np

# For reading csv, manipulating, analysis data in tabular format
import pandas as pd


# Preprocessing
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Model selection/evaluation
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# model import
from sklearn.ensemble import RandomForestClassifier


# Plot all the charts in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read train and test data

# In[ ]:


file_location = '/kaggle/input/titanic/'
train_data = pd.read_csv(os.path.join(file_location, 'train.csv'))
test_data = pd.read_csv(os.path.join(file_location, 'test.csv'))

# Let's join both data for preprocessing then later we can separate train and test data
total_data = pd.concat([train_data, test_data], sort=False, ignore_index=True)


# # EDA - Exploratory Data Analysis

# ### Check NaN values in data

# In[ ]:


total_data.isnull().sum()


# ### Understand more about age

# In[ ]:


# This might give idea to fillna value
plt.figure(figsize=(8,4))
sb.distplot(train_data['Age']);

# Looks like most of the passenger are in age group of 20-40


# #### Box plot will give useful view to identify outliers and also to get idea about distribution

# In[ ]:


# This will give idea whether this feature will be useful for prediction or not
plt.figure(figsize=(8,4))
sb.boxplot(x='Survived', y='Age', data=train_data);

# It is not much clear but will give get some idea, age more than 35 less survival rate


# #### Violin plot will give more idea about distribution, outliers by comparing with 3rd parameter like sex parameter below

# In[ ]:


# This will give idea whether this feature will be useful for prediction or not
plt.figure(figsize=(12,6))
sb.violinplot(x='Survived', y='Age', data=train_data, hue='Sex');

# After looking into plot, Compared to male more female passenger having age less than 40 age survived.
# Compared to female, Male passenger age between 18-40 has less survival rate


# #### Use barcharfor comparison

# In[ ]:


# This will give idea whether this feature will be useful for prediction or not
plt.figure(figsize=(8,4))
sb.countplot(x='Survived', data=train_data, hue='Sex');

# After looking into plot, it is clear female survival rate is high


# ### Understand more about Fare

# In[ ]:


# This will give idea whether this feature will be useful for prediction or not
plt.figure(figsize=(8,4))
sb.boxplot(x='Survived', y='Fare', data=train_data);

# After looking into plot, it is clear that fare contributing largely to survival rate
# (More paid having more survival rate) but with outliers
# Need to fill NaN values as well need to treat outliers
# Need to calculate fare per passenger to have better idea ***


# ### Understand more about Cabin

# In[ ]:


train_data['Cabin'].isnull().sum()
# there are more missing values so need to analyze then decide whether feature is needed or not


# In[ ]:


train_data['Cabin'].unique()
# There are more unique values too. Better let's create another feature 'HasCabin'


# # Preprocessing

# #### Preprocessing needs to be done on concatenated data

# In[ ]:


def get_title(data):
    """
    Get title of each person like Mr. Mrs, Miss etc from Name
    """
    valid_title = ['Mr', 'Mrs', 'Miss', 'Master']
    data['Title'] = data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    data.loc[data['Title'] == 'Sir', 'Title'] = 'Mr'
    data.loc[data['Title'] == 'Ms', 'Title'] = 'Miss'
    data.loc[~data['Title'].isin(valid_title), 'Title'] = 'Rare'

def calculate_fare(data):
    """
    Fare per person by grouping on ticket. Grouping done outside this function.
    """
    return data.mean()/data.count()

def treat_fare(data):
    """
    Fare 0 is not a valid value so let's calculate median based on pclass grouping. Grouping done outside this function
    """
    return data.where(data != 0.0, data.median())


# Notes:
# #### .loc of dataframe is best way for indexing column value.
# #### ***Note: Avoid chaining, like data['data'][data['data']>20**]*. Notebook will show warning if you do chaining

# In[ ]:


total_data.Embarked.value_counts()


# In[ ]:


# Fill NaN values for Embarked with S because S having more occurrence
total_data.Embarked.fillna('S', inplace=True)

# Fill 1 NaN value of Fare value using median
total_data.Fare.fillna(total_data['Fare'].median(), inplace=True)

# Fill NaN with some value, then later we can use this information to create new feature
total_data.Cabin.fillna('Z', inplace=True)

# Calculate fare per person
total_data.Fare = total_data.groupby('Ticket')['Fare'].transform(calculate_fare)

# There are fare values with 0 so let's fix it
total_data.Fare = total_data.groupby('Pclass')['Fare'].apply(treat_fare)


# ### Let's remove outliers

# In[ ]:


# This will give idea whether this feature will be useful for prediction or not
plt.figure(figsize=(8,4))
sb.boxplot(x='Survived', y='Fare', data=train_data);

# After looking into plot, it is clear that fare contributing largely to survival rate and there is outliers so let's fix it.


# In[ ]:


# # Remove otliers based on box plot analysis
total_data.loc[total_data['Fare'] > 60, 'Fare'] = 60


# ### Let's create new features.

# In[ ]:


# If there is valid cabin detail then set HasCabin as 1 otherwise 0
total_data['HasCabin'] = np.where(total_data['Cabin'] == 'Z', 0, 1)
total_data['FamilyMembers'] = total_data['SibSp'] + total_data['Parch']

# If there is valid cabin detail then set HasCabin as 1 otherwise 0
total_data['Alone'] = total_data['FamilyMembers'].map(lambda size: 0 if size > 0 else 1)

# After filling NaN value for age, let's create new feature to map passenger as senior or not
total_data['Senior'] = total_data['Age'].map(lambda size: 1 if size > 55 else 0)

# New feature after removing outliers
total_data['FareCat'] = pd.qcut(total_data['Fare'], 4, labels=range(1,5)).astype('int')

# New feature "Title"
get_title(total_data)

# Let's fill age based on title. Miss - Likely between 10-30, Master 0-10
total_data.Age = total_data.groupby('Title')['Age'].apply(lambda age: age.fillna(age.median()))

# Let's create new feature, age into categorical value (bucketing). Here 8 age groups are created
total_data['AgeCat'] = pd.qcut(total_data['Age'], 8, labels=range(1,9)).astype('int')


# ## Let's visualize

# In[ ]:


# Only train data has survived data so filter only train data from total_data for visualization
for_visualize = total_data[:891]

# Create subplot/placeholder to draw 4 plots
f, axes = plt.subplots(2, 2)
f.set_size_inches(20,10)
sb.violinplot(x='Survived',y='Age', hue='Sex', data=for_visualize, ax=axes[0][0]);
sb.boxplot(x='Survived',y='Age', data=for_visualize, ax=axes[0][1]);
sb.countplot(x='Survived', data=for_visualize, hue='AgeCat', ax=axes[1][0]);
sb.countplot(x='Survived', data=for_visualize, hue='FareCat', ax=axes[1][0]);


# ### Convert string/categories(object type) to numerial value

# In[ ]:


total_data.dtypes


# In[ ]:


# Only required features needs to be encoded
encoding_features = ['Sex','Title','Embarked']
for feature in encoding_features:
    encode = LabelEncoder()
    total_data.loc[:,feature] = encode.fit_transform(total_data.loc[:,feature])


# In[ ]:


# Features to be used for prediction
# features = ['Sex', 'AgeCat', 'Title', 'Alone', 'HasCabin', 'Parch', 'Embarked', 'FamilyMembers', 'Pclass', 'Senior', 'Fare']
features = ['Sex', 'AgeCat', 'Title', 'Alone', 'HasCabin', 'Parch', 'Embarked', 'FamilyMembers', 'FareCat', 'Senior', 'Pclass']


# ### Scale the value for better prediction
# #### Here we are going to use Randome forest classifier so scaling won't be much help.
# #### Scaling is necessary for algo like ridge, linear, lasso, knn, svm

# In[ ]:


for feature in features:
    scaling_clf = MinMaxScaler()
    total_data.loc[:, feature] = scaling_clf.fit_transform(total_data.loc[:,feature].values.reshape(-1,1))


# ### It's time to separate train and test data (EDA, pre-processing are done)

# In[ ]:


train = total_data[:891]
test = total_data[891:]
x_train_data = train.loc[:,features]
y_train_data = train.loc[:,'Survived']
x_test_data = test.loc[:,features]


# ## Check whether any redudant features are present

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sb.heatmap(x_train_data.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True);

# We have better features because features correlations are very less except Pclass and HasCabin (-0.73) but let's keep it


# # Model selection

# In[ ]:


# Instantiate RandomForestClassifier with default values
clf = RandomForestClassifier()

# Split test and train data for evaluation purpose
# train_size =0.8 (split train data 80% and test data 20%)
# random_state = 10 (For splitting data sklearn uses random seed, here let's fix the value so we can reproduce the set of values)
x_train, x_test, y_train, y_test = train_test_split(x_train_data, y_train_data, train_size=0.80, random_state=10)

# Train the model
clf.fit(x_train, y_train)


# ### Let's evaluate using different metrics and methods
# 

# In[ ]:


# # Prediction based on training
pred_y = clf.predict(x_test).astype(int)

# Zip the features, importance
feature_importance = dict(zip(x_train_data.columns, clf.feature_importances_))
print(feature_importance)

# To know the accuracy score for test data
print("Accuracy score: ", accuracy_score(y_test, pred_y))

# To know better where the misclassification happend, whether more TN or more FN
print(confusion_matrix(y_test, pred_y))

# let's cross validation instead of fixed data set and use different metrics for evalation
metrics = ('accuracy','precision','recall')
score = cross_validate(clf, x_train_data, y_train_data, cv=20, scoring=metrics, n_jobs=-1)
print('Min Accuracy:', score['test_accuracy'].min())
print('Average accuracy', score['test_accuracy'].mean())
print('Average precision', score['test_precision'].mean())
print('Average recall', score['test_recall'].mean())


# > # Tuning/Hyperparameter selection

# ## It's time to tune the hyperparameters of the model to attain more accuracy

# ### Gridsearch can be used to find best parameter

# In[ ]:


# Instantiate model
clf = RandomForestClassifier(random_state=10)

# Parameter for tuning
parameters = {
    'n_estimators': range(200,300, 20),
    'max_features': [6,7,8,],
    'min_samples_leaf': [6,7,8],
    'criterion': ['entropy', 'gini']}

# n_jobs = -1, will utilize all the CPU to do search very quickly, instead you can use 1/2/3 depending upon number of CPUs available in computer    
grid_clf = GridSearchCV(estimator=clf, param_grid=parameters, cv=30, n_jobs=-1)
grid_clf.fit(x_train_data, y_train_data)
print(grid_clf.best_params_)
print(grid_clf.best_score_)


# # Evaluate

# ## It's time to evaluate again with new best parameters found using grid search

# In[ ]:


clf = RandomForestClassifier(criterion='gini', max_features=7, min_samples_leaf=7, n_estimators=280,
                            random_state=10)
clf.fit(x_train_data, y_train_data)
print("Prediction for training data")
pred_y = clf.predict(x_train_data).astype(int)
print("Accuracy score: ", accuracy_score(y_train_data, pred_y))
# let's cross validation instead of fixed data set and use different metrics for evalation
metrics = ('accuracy','precision','recall')
score = cross_validate(clf, x_train_data, y_train_data, cv=20, scoring=metrics, n_jobs=-1)
print('Min Accuracy:', score['test_accuracy'].min())
print('Average accuracy', score['test_accuracy'].mean())
print('Average precision', score['test_precision'].mean())
print('Average recall', score['test_recall'].mean())


# # Predict

# ## It's time to predict final output

# In[ ]:


clf.fit(x_train_data, y_train_data)
pred_y = clf.predict(x_test_data).astype(int)

# Generate Submission File 
out_df = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                            'Survived': pred_y })
out_df.to_csv("output.csv", index=False)

