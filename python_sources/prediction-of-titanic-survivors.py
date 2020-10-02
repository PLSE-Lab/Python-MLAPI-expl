#!/usr/bin/env python
# coding: utf-8

# # Importing required libraries and setup some variables

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as plt # help to plot different graphs 

# Set the global default size of matplotlib figures
plt.rc('figure', figsize=(10, 5))

# Size of matplotlib figures that contain subplots
fizsize_with_subplots = (10, 10)

# Size of matplotlib histogram bins
bin_size = 10
#/kaggle/input/titanic/test.csv
#/kaggle/input/titanic/train.csv
#/kaggle/input/titanic/gender_submission.csv


# ### Reading training data with showing first 5 records

# In[ ]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_train.head()


# ### Fetching last 5 records from training data

# In[ ]:


df_train.tail()


# ### Checking datatypes of columns

# In[ ]:


df_train.dtypes


# ### Get some basic information of the DataFrame:

# In[ ]:


df_train.info()


# There are some of variables which have missing values for example Age, Cabin and Embarked. Cabin has too many missing values, whereas we might be able to infer values for Age and Embarked.

# Let's plots some basic figure to get basic idea of data.

# In[ ]:


# Set up a grid of plots
fig = plt.figure(figsize=fizsize_with_subplots)
fig_dims = (3, 2)

# Plot death and survival counts
plt.subplot2grid(fig_dims, (0, 0))
df_train['Survived'].value_counts().plot(kind='bar',
                                         title='Death and Survival Counts')

# Plot Pclass counts
plt.subplot2grid(fig_dims, (0, 1))
df_train['Pclass'].value_counts().plot(kind='bar',
                                       title='Passenger Class Counts')

# Plot Sex counts
plt.subplot2grid(fig_dims, (1, 0))
df_train['Sex'].value_counts().plot(kind='bar',
                                    title='Gender Counts')
plt.xticks(rotation=0)

# Plot Embarked counts
plt.subplot2grid(fig_dims, (1, 1))
df_train['Embarked'].value_counts().plot(kind='bar',
                                         title='Ports of Embarkation Counts')

# Plot the Age histogram
plt.subplot2grid(fig_dims, (2, 0))
df_train['Age'].hist()
plt.title('Age Histogram')


# # FEATURE: PASSENGER CLASSES
# From the above exploratory analysis we see that there are three classes as 1,2 and 3. Now We'll determine what proportion of passager survive according to different classes.
# 

# In[ ]:


pclass_xt = pd.crosstab(df_train['Pclass'], df_train['Survived'])
pclass_xt


# Ploting cross tab:

# In[ ]:


# Normalize the cross tab to sum to 1:
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0)
print(pclass_xt_pct)
pclass_xt_pct.plot(kind='bar',
                   stacked=True,
                   title='Survival Rate by Passenger Classes')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')


# From this bar gragh of passanger survival rate with passanger class it is clear that pclass also play important role to determine survival of passanger.

# # FEATURE: SEX
# 
# Sex might also be played important role to determine survival of passanger. So, as we see sex is categorical data so we change this to numerical by mapping with proper id of Male and Female. 

# In[ ]:


# mapping of gender sex variable with with an id.
sexes = sorted(df_train['Sex'].unique())
genders_mapping = dict(zip(sexes, range(0, len(sexes))))
genders_mapping


# In[ ]:


# Transfering Sex from string to number representation.
df_train['Sex_Val'] = df_train['Sex'].map(genders_mapping).astype(int)
df_train.head()


# Now, normalized a cross tab of Sex_Val with survicval rate.

# In[ ]:


sex_val_xt = pd.crosstab(df_train['Sex_Val'], df_train['Survived'])
sex_val_xt


# In[ ]:


# Normalize the cross tab of Sex to sum to 1:
sex_val_xt_pct = sex_val_xt.div(sex_val_xt.sum(1).astype(float), axis=0)
print(sex_val_xt_pct)
sex_val_xt_pct.plot(kind='bar',
                   stacked=True,
                   title='Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')


# From the above bar chart, It is clear that Survival of passanger is strongly dependent on Sex. The majority of females survived, whereas the majority of males did not.

# Now, We will see is there any relation exist between pClass and Sex.
# And count males and females in each Pclass.

# In[ ]:


# Get the unique values of Pclass:
passenger_classes = sorted(df_train['Pclass'].unique())

for p_class in passenger_classes:
    print('Male:   ', p_class, len(df_train[(df_train['Sex'] == 'male') &
                             (df_train['Pclass'] == p_class)]))
    print('Female: ', p_class, len(df_train[(df_train['Sex'] == 'female') &
                             (df_train['Pclass'] == p_class)]))


# Plot survival rate of pclass and sex.

# In[ ]:


# Plot survival rate by Sex
females_df = df_train[df_train['Sex'] == 'female']
females_xt = pd.crosstab(females_df['Pclass'], df_train['Survived'])
females_xt_pct = females_xt.div(females_xt.sum(1).astype(float), axis=0)
females_xt_pct.plot(kind='bar',
                    stacked=True,
                    title='Female Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

# Plot survival rate by Pclass
males_df = df_train[df_train['Sex'] == 'male']
males_xt = pd.crosstab(males_df['Pclass'], df_train['Survived'])
males_xt_pct = males_xt.div(males_xt.sum(1).astype(float), axis=0)
males_xt_pct.plot(kind='bar',
                  stacked=True,
                  title='Male Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')


# The vast majority of females in First and Second class survived. Males in First class had the highest chance for survival.

# # FEATURE: EMBARKED
# Embarked column might be an important variable but it has much missing values. So, we need to infer with some value or need to delete that column after seeing relation between them.

# In[ ]:


df_train[df_train['Embarked'].isnull()]


# Let's create Embarked from categorical to numerical variables. And find the relation between target varible Survivale.

# In[ ]:


df_train['Embarked'] = sorted(df_train['Embarked'].fillna('A'))
df_train['Embarked'].unique()


# In[ ]:


# Get the unique values of Embarked
embarked_locs = df_train['Embarked'].unique()

embarked_locs_mapping = dict(zip(embarked_locs,
                                 range(0, len(embarked_locs))))
embarked_locs_mapping


# Transform Embarked from a string to a number representation to prepare it for machine learning algorithms:

# In[ ]:


df_train['Embarked_Val'] = df_train['Embarked']                                .map(embarked_locs_mapping)                                .astype(int)
df_train.head()


# In[ ]:


#Ploting histograph for embarked
df_train['Embarked_Val'].hist(bins=len(embarked_locs), range=(0, 3))
plt.title('Port of Embarkation Histogram')
plt.xlabel('Port of Embarkation')
plt.ylabel('Count')
plt.show()


# In[ ]:


# As we see majority of passager is from S embarked, so we assign missing with S
df_train.replace({'Embarked_Val' :
                { embarked_locs_mapping['A'] : embarked_locs_mapping['S']
                }
             },
             inplace=True)


# In[ ]:


# Verifing replaced data
embarked_locs = sorted(df_train['Embarked_Val'].unique())
embarked_locs


# In[ ]:


# Plot cross tab for Embarked_val and Survival
embarked_val_xt = pd.crosstab(df_train['Embarked_Val'], df_train['Survived'])
embarked_val_xt_pct =     embarked_val_xt.div(embarked_val_xt.sum(1).astype(float), axis=0)
embarked_val_xt_pct.plot(kind='bar', stacked=True)
plt.title('Survival Rate by Port of Embarkation')
plt.xlabel('Port of Embarkation')
plt.ylabel('Survival Rate')


# It appears those that embarked in location 'C': 1 had the highest rate of survival. We'll dig in some more to see why this might be the case. Below we plot a graphs to determine gender and passenger class makeup for each port:
# 

# In[ ]:


# Set up a grid of plots
fig = plt.figure(figsize=fizsize_with_subplots)

rows = 2
cols = 3
col_names = ('Sex_Val', 'Pclass')

for portIdx in embarked_locs:
    for colIdx in range(0, len(col_names)):
        plt.subplot2grid((rows, cols), (colIdx, portIdx - 1))
        df_train[df_train['Embarked_Val'] == portIdx][col_names[colIdx]]             .value_counts().plot(kind='bar')


# In[ ]:


# Leaving Embarked as integers implies ordering in the values, which does not exist. Another way to represent Embarked without ordering is to create dummy variables:
df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked_Val'], prefix='Embarked_Val')], axis=1)


# # FEATURE: AGE
# The Age column seems like an important feature--unfortunately it is missing many values. We'll need to fill in the missing values like we did with Embarked.

# In[ ]:


# Filter to view missing Age values:

df_train[df_train['Age'].isnull()][['Sex', 'Pclass', 'Age']].head()


# Determine the Age typical for each passenger class by Sex_Val. We'll use the median instead of the mean because the Age histogram seems to be right skewed.
# 

# In[ ]:


# To keep Age in tact, make a copy of it called AgeFill
# that we will use to fill in the missing ages:
df_train['AgeFill'] = df_train['Age']

# Populate AgeFill
df_train['AgeFill'] = df_train['AgeFill']                         .groupby([df_train['Sex_Val'], df_train['Pclass']])                         .apply(lambda x: x.fillna(x.median()))


# In[ ]:


# Plot a normalized cross tab for AgeFill and Survived:

# Set up a grid of plots
fig, axes = plt.subplots(2, 1, figsize=fizsize_with_subplots)

# Histogram of AgeFill segmented by Survived
df1 = df_train[df_train['Survived'] == 0]['Age']
df2 = df_train[df_train['Survived'] == 1]['Age']
max_age = max(df_train['AgeFill'])
axes[0].hist([df1, df2],
             bins=int(max_age / bin_size),
             range=(1, max_age),
             stacked=True)
axes[0].legend(('Died', 'Survived'), loc='best')
axes[0].set_title('Survivors by Age Groups Histogram')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')

# Scatter plot Survived and AgeFill
axes[1].scatter(df_train['Survived'], df_train['AgeFill'])
axes[1].set_title('Survivors by Age Plot')
axes[1].set_xlabel('Survived')
axes[1].set_ylabel('Age')


# Unfortunately, the graphs above do not seem to clearly show any insights. We'll keep digging further.
# 
# Plot AgeFill density by Pclass:

# In[ ]:


for pclass in passenger_classes:
    df_train.AgeFill[df_train.Pclass == pclass].plot(kind='kde')
plt.title('Age Density Plot by Passenger Class')
plt.xlabel('Age')
plt.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')


# When looking at AgeFill density by Pclass, we see the first class passengers were generally older then second class passengers, which in turn were older than third class passengers. We've determined that first class passengers had a higher survival rate than second class passengers, which in turn had a higher survival rate than third class passengers.

# In[ ]:


# Set up a grid of plots
fig = plt.figure(figsize=fizsize_with_subplots)
fig_dims = (3, 1)

# Plot the AgeFill histogram for Survivors
plt.subplot2grid(fig_dims, (0, 0))
survived_df = df_train[df_train['Survived'] == 1]
survived_df['AgeFill'].hist(bins=int(max_age / bin_size), range=(1, max_age))

# Plot the AgeFill histogram for Females
plt.subplot2grid(fig_dims, (1, 0))
females_df = df_train[(df_train['Sex_Val'] == 0) & (df_train['Survived'] == 1)]
females_df['AgeFill'].hist(bins=int(max_age / bin_size), range=(1, max_age))

# Plot the AgeFill histogram for first class passengers
plt.subplot2grid(fig_dims, (2, 0))
class1_df = df_train[(df_train['Pclass'] == 1) & (df_train['Survived'] == 1)]
class1_df['AgeFill'].hist(bins=int(max_age / bin_size), range=(1, max_age))


# In the first graph, we see that most survivors come from the 20's to 30's age ranges and might be explained by the following two graphs. The second graph shows most females are within their 20's. The third graph shows most first class passengers are within their 30's.

# # FEATURE: FAMILY SIZE
# Feature enginering involves creating new features or modifying existing features which might be advantageous to a machine learning algorithm.
# 
# Define a new feature FamilySize that is the sum of Parch (number of parents or children on board) and SibSp (number of siblings or spouses):

# In[ ]:


df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_train.head()


# In[ ]:


# Plot a histogram of FamilySize:
df_train['FamilySize'].hist()
plt.title('Family Size Histogram')


# Plot a histogram of AgeFill segmented by Survived:

# In[ ]:


# Get the unique values of Embarked and its maximum
family_sizes = sorted(df_train['FamilySize'].unique())
family_size_max = max(family_sizes)

df1 = df_train[df_train['Survived'] == 0]['FamilySize']
df2 = df_train[df_train['Survived'] == 1]['FamilySize']
plt.hist([df1, df2],
         bins=family_size_max + 1,
         range=(0, family_size_max),
         stacked=True)
plt.legend(('Died', 'Survived'), loc='best')
plt.title('Survivors by Family Size')


# Based on the histograms, it is not immediately obvious what impact FamilySize has on survival. The machine learning algorithms might benefit from this feature.
# 
# Additional features we might want to engineer might be related to the Name column, for example honorrary or pedestrian titles might give clues and better predictive power for a male's survival.

# # FINAL DATA PREPARATION FOR MACHINE LEARNING
# Many machine learning algorithms do not work on strings and they usually require the data to be in an array, not a DataFrame.
# 
# Show only the columns of type 'object' (strings):

# In[ ]:


df_train.dtypes[df_train.dtypes.map(lambda x: x == 'object')]


# In[ ]:


# Drop the columns we won't use:

df_train = df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],
                         axis=1)
df_train = df_train.drop(['Age', 'SibSp', 'Parch', 'PassengerId', 'Embarked_Val'], axis=1)
df_train.dtypes


# Convert the DataFrame to a numpy array:

# In[ ]:


train_data = df_train.values
train_data


# # DATA WRANGLING SUMMARY
# 
# Below is a summary of the data wrangling we performed on our training data set. We encapsulate this in a function since we'll need to do the same operations to our test set later.

# In[ ]:


def clean_data(df, drop_passenger_id):

    # Get the unique values of Sex
    sexes = sorted(df['Sex'].unique())

    # Generate a mapping of Sex from a string to a number representation
    genders_mapping = dict(zip(sexes, range(0, len(sexes))))

    # Transform Sex from a string to a number representation
    df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)

    # Get the unique values of Embarked
    df['Embarked'] = sorted(df['Embarked'].fillna('A'))
    embarked_locs = sorted(df['Embarked'].unique())

    # Generate a mapping of Embarked from a string to a number representation
    embarked_locs_mapping = dict(zip(embarked_locs,
                                     range(0, len(embarked_locs))))

    # Transform Embarked from a string to dummy variables
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked_Val')], axis=1)

    # Fill in missing values of Embarked
    # Since the vast majority of passengers embarked in 'S': 3,
    # we assign the missing values in Embarked to 'S':
    if (len(df[df['Embarked']=='A']) > 0):
        df.replace({'Embarked_Val' :
                        { embarked_locs_mapping['A'] : embarked_locs_mapping['S']
                        }
                    },
                    inplace=True)

    # Fill in missing values of Fare with the average Fare
    if (len(df[df['Fare'].isnull()]) > 0):
        avg_fare = df['Fare'].mean()
        df.replace({ None: avg_fare }, inplace=True)

    # To keep Age in tact, make a copy of it called AgeFill
    # that we will use to fill in the missing ages:
    df['AgeFill'] = df['Age']

    # Determine the Age typical for each passenger class by Sex_Val.
    # We'll use the median instead of the mean because the Age
    # histogram seems to be right skewed.
    df['AgeFill'] = df['AgeFill']                         .groupby([df['Sex_Val'], df['Pclass']])                         .apply(lambda x: x.fillna(x.median()))

    # Define a new feature FamilySize that is the sum of
    # Parch (number of parents or children on board) and
    # SibSp (number of siblings or spouses):
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # Drop the columns we won't use:
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # Drop the Age column since we will be using the AgeFill column instead.
    # Drop the SibSp and Parch columns since we will be using FamilySize.
    # Drop the PassengerId column since it won't be used as a feature.
    df = df.drop(['Age', 'SibSp', 'Parch'], axis=1)

    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)

    return df


# # RANDOM FOREST: TRAINING

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
# Training data features, skip the first column 'Survived'
train_features = train_data[:, 1:]

# 'Survived' column values
train_target = train_data[:, 0]

# Fit the model to our training data
clf = clf.fit(train_features, train_target)
score = clf.score(train_features, train_target)
print("Accuracy: %.2f%%" % (score * 100.0))


# # XGBoost Model: TRAINING
# 

# In[ ]:


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# fit model no training data
model = XGBClassifier()
model.fit(train_features, train_target)
# make predictions for test data
y_pred = model.predict(train_features)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(train_target, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# From RF and XGBOOST, RF work better for this prediction. So, we will predict actual survival through Random Forest.

# # RANDOM FOREST: PREDICTING

# In[ ]:


df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
# Data wrangle the test set and convert it to a numpy array
df_test = clean_data(df_test, drop_passenger_id=False)
test_data = df_test.values


# Take the decision trees and run it on the test data:
# 
# 

# In[ ]:


# Get the test data features, skipping the first column 'PassengerId'
test_x = test_data[:, 1:]

# Predict the Survival values for the test data
test_y = clf.predict(test_x)


# Take the XG boost and run it on the test data:
# 

# In[ ]:


test_y = model.predict(test_x)


# # XGBOOST ALGO: PREPARE FOR KAGGLE SUBMISSION
# Create a DataFrame by combining the index from the test data with the output of predictions, then write the results to the output:
# 

# In[ ]:


df_test['Survived'] = test_y
df_test['Survived'] = df_test['Survived'].astype(int)
df_test[['PassengerId', 'Survived']]     .to_csv('/kaggle/working/submission.csv', index=False)


# In[ ]:


# Link to download file for submission
from IPython.display import FileLink
FileLink('/kaggle/working/submission.csv')


# # EVALUATE MODEL ACCURACY
# Submitting to Kaggle will give you an accuracy score. It would be helpful to get an idea of accuracy without submitting to Kaggle.
# 
# We'll split our training data, 80% will go to "train" and 20% will go to "test":
# 

# In[ ]:


from sklearn import metrics
from sklearn.model_selection import train_test_split


# Split 80-20 train vs test data
train_x, test_x, train_y, test_y = train_test_split(train_features,
                                                    train_target,
                                                    test_size=0.20,
                                                    random_state=0)
print (train_features.shape, train_target.shape)
print (train_x.shape, train_y.shape)
print (test_x.shape, test_y.shape)


# In[ ]:


# Use the new training data to fit the model, predict, and get the accuracy score:

clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)

from sklearn.metrics import accuracy_score
print ("Accuracy = %.2f" % (accuracy_score(test_y, predict_y)))


# In[ ]:


# Get the model score and confusion matrix:

model_score = clf.score(test_x, test_y)
print ("Model Score %.2f \n" % (model_score))

confusion_matrix = metrics.confusion_matrix(test_y, predict_y)
print ("Confusion Matrix ", confusion_matrix)

print ("          Predicted")
print ("         |  0  |  1  |")
print ("         |-----|-----|")
print ("       0 | %3d | %3d |" % (confusion_matrix[0, 0],
                                   confusion_matrix[0, 1]))
print ("Actual   |-----|-----|")
print ("       1 | %3d | %3d |" % (confusion_matrix[1, 0],
                                   confusion_matrix[1, 1]))
print ("         |-----|-----|")


# In[ ]:


#Display the classification report:

from sklearn.metrics import classification_report
print(classification_report(test_y,
                            predict_y,
                            target_names=['Not Survived', 'Survived']))

