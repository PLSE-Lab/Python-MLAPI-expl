'''
    Using titanic_test_train_combo.py as a base, then working to improve model accuracy through
    better feature engineering.

    Base model is achieving 0.78468 on validation / test set from Kaggle.

    Improvements to make:
    - Use pd.cut instead of pd.qcut - qcut produces evenly sized quantiles (same number of records in each bin).
        Cut doesn't do that, so in our case is more representative of the data.
    - Ensure any bins / buckets have a minimum of 10 samples
    - Use LabelEncoder for ordinal data, OneHotEncoder for non-ordinal categorical
    - Transform age bins, fare bins and title bins into label categories?
    - A 90/10 training to test split is probably not enough. Try closer to 75/25. Possibly overfitting at the moment.

    Goal: Get to >= 0.85 accuracy on public LD
'''

import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing:
from sklearn.preprocessing import StandardScaler

# Scikit Learn Models
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import train_test_split

# Performance assessment
import shap
from sklearn import metrics

''' Exploration '''
raw_x_train = pd.read_csv('../input/train.csv')
raw_x_test = pd.read_csv('../input/test.csv')

#Get passengerId of Test
test_index = raw_x_test.PassengerId.head(1).item()

combined_df = raw_x_train.append(raw_x_test, sort=False).reset_index(drop=True)

combined_df.loc[combined_df.Ticket == combined_df.sample(1).Ticket.item()]

# Data Cleanup
# Determine title of each group
combined_df['Title'] = combined_df['Name'].apply(lambda x: x.split('.')[0].split(',')[1].strip())

# Re map names, and use names to determine average ages of missing data
# Could further improve this by taking class and ticket value into account to better categorize 'upper class' vs 'commoners'
# 0: Male Commoner 1: Female Commoner - Adults
# 2: Male Commoner 3: Female Commoner - Children / young adults
# 4: Male Upper 5: Female Upper - Adults
updated_titles = {  'Mr': 0,
                    'Mrs': 1,
                    'Miss': 3,
                    'Master': 2,
                    'Don': 4,
                    'Dona': 5,
                    'Rev': 4,
                    'Dr': 4,
                    'Mme': 5,
                    'Ms': 5,
                    'Major': 4,
                    'Lady': 5,
                    'Sir': 4,
                    'Mlle': 5,
                    'Col': 4,
                    'Capt': 4,
                    'the Countess': 5,
                    'Jonkheer': 4}


# Find all the NaN age values for each title type
parch_range = combined_df.Parch.unique()
for title in updated_titles.keys():
    for fam_size in parch_range:
        parch_age = combined_df.loc[(combined_df.Title == title) & (combined_df.Parch == fam_size)].Age.median()

        # Gendered title median age. parch_age can be NaN in extreme cases (fam_size == 9)
        if np.isnan(parch_age):
            # Just use median title age
            parch_age = combined_df.loc[(combined_df.Title == title)].Age.median()

        combined_df.loc[(combined_df.Age.isnull() == True) & (combined_df.Title == title) & (combined_df.Parch == fam_size), 'Age'] = parch_age

# Arrange into buckets. Using pd.cut instead of pd.qcut, as qcut removes information on age distribution
combined_df['Age_buckets'] = pd.cut(combined_df.Age, 5, labels=[1,2,3,4,5])
combined_df['Age_buckets'] = combined_df['Age_buckets'].astype(int)
combined_df = combined_df.join(pd.get_dummies(combined_df['Age_buckets'], prefix='age_bucket'))

# Clean up missing Fare values based on PClass
classes = [1, 2, 3]
for each in classes:
    class_median = combined_df.loc[combined_df['Pclass'] == each, 'Fare'].median()
    combined_df.loc[(combined_df['Fare'] == 0) & (combined_df['Pclass'] == each), 'Fare'] = class_median
    combined_df.loc[(combined_df['Fare'].isnull() == True) & (combined_df['Pclass'] == each), 'Fare'] = class_median

# Determine family size. Unsure why some people add +1 to this.
combined_df['fam_size'] = combined_df['SibSp'] + combined_df['Parch']

# Determine group size based on shared ticket numbers:
combined_df['group_size'] = 1
for ticket in combined_df.Ticket.unique():
    group_size = len(combined_df.loc[combined_df.Ticket == ticket])
    combined_df.loc[combined_df.Ticket == ticket, 'group_size'] = group_size

# Convert categorical into categorical ints
combined_df['Sex'].value_counts()
combined_df['Sex'] = combined_df['Sex'].apply(lambda x: 0 if x == 'male' else 1)

# Fill NA with most common value
combined_df['Embarked'] = combined_df['Embarked'].map({'S': 0, 'C':1, 'Q':2})
combined_df['Embarked'].fillna(combined_df['Embarked'].mode().item(), inplace=True)

# Finally remap title to categorical dummy variable
combined_df['Title'] = combined_df['Title'].map(updated_titles)
combined_df = combined_df.join(pd.get_dummies(combined_df['Title'], prefix='title'), how='left')

''' ---- Visualisations & Stats ---- '''
'''
viz_train = combined_df.loc[combined_df.PassengerId < test_index]

# Names of columns to plot
bar_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'Age_buckets', 'fam_size', 'group_size']
scatter_cols = ['Age', 'Fare']

for col in bar_cols:
    # Sex vs survival
    sns.catplot(x=col, y='Survived', hue='Sex', data=viz_train, kind='bar', height=10)

for col in scatter_cols:
    # Pclass vs survival
    sns.swarmplot(x=col, hue='Survived', data=viz_train)

sns.catplot(x='Age_buckets', y='Survived', hue='Title', data=viz_train, kind='bar', height=10)

sns.catplot(x='Sex', y='Age_buckets', hue='Survived', data=viz_train, kind='bar', height=10)
'''

''' ---- Final Check ---- '''
for col in combined_df.columns:
    print('{}: \t {:>10}'.format(col, combined_df[col].isnull().sum()))

combined_df.Survived.value_counts()

combined_df.columns

''' ---- Model ---- '''

keep_labels = set(['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Age_buckets', 'group_size'])

all_labels = set(combined_df.columns)

drop_labels = all_labels - keep_labels

# Setup results dictionary for Random Forest

training_set = combined_df.loc[combined_df.Survived.isnull() != True].drop(labels=drop_labels, axis=1)

validation_set = combined_df.loc[combined_df.Survived.isnull() == True].drop(labels=drop_labels, axis=1)

clf_df = training_set.copy()

# Create train and test subsets to try provider a better estimate of classifier performance
split = 0.75

x_train, x_test, y_train, y_test = train_test_split(clf_df.drop(labels='Survived', axis=1), clf_df.loc[:, 'Survived'], test_size=0.25)

# Fit & transform training data
#scaler = StandardScaler(with_mean=False, with_std=False)
#scaler.fit(x_train)
#x_train[x_train.columns] = scaler.transform(x_train[x_train.columns])
#x_test[x_test.columns] = scaler.transform(x_test[x_test.columns])

clf_model = RFC(n_estimators=500,
                criterion='gini',
                max_depth=None,
                min_samples_split=10,
                min_samples_leaf=10,
                min_weight_fraction_leaf=0.,
                max_features='sqrt',
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                bootstrap=False,
                oob_score=False,
                n_jobs=4,
                random_state=None,
                verbose=0,
                warm_start=False,
                class_weight={0: 0.66, 1:0.33})


# Train and predict

clf_model.fit(x_train, y_train)

clf_model.score(x_train, y_train)

# Predict on test set

clf_model.score(x_test, y_test)

label_array = ['dead', 'alive']
label_array_ints = [0, 1]
print(metrics.classification_report(y_test, clf_model.predict(x_test), label_array_ints))

# Feature Importance log for random forests
importances_df = pd.DataFrame({})
importances = zip(x_train.columns, clf_model.feature_importances_)
importances_dict = dict((key,value) for key, value in importances)
importances_df = importances_df.append(importances_dict, ignore_index=True)
importances_df.median().sort_values(ascending=False).plot(figsize=(18,18), kind='bar')


''' ---- Generate Predictions ---- '''
submission_df = pd.DataFrame(raw_x_test.PassengerId)
submission_df = submission_df.reset_index(drop=True)

#x_test[x_test.columns] = scaler.transform(x_test[x_test.columns])

submission_df['Survived'] = clf_model.predict(validation_set.drop(labels='Survived', axis=1))
submission_df['Survived'] = submission_df['Survived'].astype(int)
submission_df.to_csv("../working/submit.csv", index=False)