import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import data sets
titanic_train = pd.read_csv('../train.csv')

titanic_test = pd.read_csv(('../test.csv')

# Combine training and testing sets for data manipulation
titanic_train.set_index('PassengerId', inplace=True)
titanic_test.set_index('PassengerId', inplace=True)
titanic_all = pd.concat([titanic_train, titanic_test], keys=['train', 'test'])

# Split the full names into last names, prefix and rest of names.
# Extract only the first character of all cabin labels
# Drop Ticket numbers and Names
df_names = titanic_all[:]['Name'].str.replace('.', ',').str.split(',', expand=True).\
               iloc[:, 0:2].apply(lambda x: x.str.strip())
titanic_all = titanic_all[:].drop(['Name', 'Cabin', 'Ticket'], axis=1)
titanic_all['Last_Name'], titanic_all['Prefix'] = df_names.loc[:, 0], df_names.loc[:, 1]

# Generate mean ages for different prefix and fill the missing data
titanic_all['Age'] = titanic_all.groupby('Prefix').\
    apply(lambda x: x.loc[:, ['Age']].fillna(x['Age'].mean()))

# Create family size
titanic_all['Family_Size'] = titanic_all['Parch'] + titanic_all['SibSp'] + 1

# Create fare per person
titanic_all['Fare_Per_Person'] = titanic_all['Fare'] / titanic_all['Family_Size']

# Generate mean fare per person for different Pclass, apply it to fares which are zeros or missing
titanic_all['Fare_Per_Person'] = titanic_all.groupby('Pclass').\
    apply(lambda x: x.loc[:, ['Fare_Per_Person']].replace(to_replace=0, value=x['Fare_Per_Person'].mean()))

titanic_all['Fare_Per_Person'] = titanic_all.groupby('Pclass').\
    apply(lambda x: x.loc[:, ['Fare_Per_Person']].fillna(x['Fare_Per_Person'].mean()))

# Apply fare per person to compute true fare per family
titanic_all.loc[:, 'Fare'].loc[:, titanic_all['Fare'].isnull()] = 0
titanic_all['Fare'] = titanic_all.apply(
    lambda x: x['Fare_Per_Person'] * x['Family_Size']
    if x['Fare'] == 0 else x['Fare'], axis=1)

# Deal with missing embarked data
titanic_all.loc[:, 'Embarked'].loc[:, titanic_all['Embarked'].isnull()] = 'S'

# Feature engineering on Prefix
titanic_all.loc[:, 'Prefix'].loc[:, titanic_all['Prefix'].isin(['Mme'])] = 'Mrs'
titanic_all.loc[:, 'Prefix'].loc[:, titanic_all['Prefix'].isin(['Mlle', 'Ms'])] = 'Miss'
titanic_all.loc[:, 'Prefix'].loc[:, titanic_all['Prefix'].isin(['Capt', 'Don', 'Col', 'Major', 'Sir', 'Jonkheer'])] = 'Sir'
titanic_all.loc[:, 'Prefix'].loc[:, titanic_all['Prefix'].isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'

# Create family id. Thanks to Trevor Stephens's tutorial http://trevorstephens.com/kaggle-titanic-tutorial/
titanic_all['Family_id'] = titanic_all['Family_Size'].astype(str) + titanic_all['Last_Name']
titanic_all.loc[:, 'Family_id'].loc[:, titanic_all['Family_Size'] <= 2] = 'Small_Family'
knockout_list = \
    titanic_all.groupby('Family_id')['Last_Name'].count()[titanic_all.groupby('Family_id')['Last_Name'].count() <= 2]
titanic_all.loc[:, 'Family_id'].loc[:, titanic_all['Family_id'].isin(knockout_list.index)] = 'Small_Family'
titanic_all = titanic_all[:].drop(['Last_Name'], axis=1)


# Label and one hot encode categorical features
titanic_encoded = titanic_all.copy()
for elem in ['Embarked', 'Sex', 'Prefix', 'Pclass', 'Family_id']:
    titanic_encoded = titanic_encoded.merge\
        (pd.get_dummies(titanic_encoded[elem]), left_index=True, right_index=True)
titanic_encoded = titanic_encoded.drop(['Embarked', 'Sex', 'Prefix', 'Pclass', 'Family_id'], axis=1)

# Data preparation
train_data = titanic_encoded.ix['train']
test_data = titanic_encoded.ix['test']
predictors = [x for x in titanic_encoded.columns if x not in ['Survived']]

# Split into training and validation set
training_set, validation_set = train_test_split(train_data, test_size=0.3, random_state=42)


# Performance Report
def show_performance(data_set, model):
    model.fit(training_set[predictors], training_set['Survived'])
    predictions = model.predict(data_set[predictors])
    probabilities = model.predict_proba(data_set[predictors])[:, 1]
    print 'Performance Report'
    print 'Accuracy : {:.2%}'.format(metrics.accuracy_score(data_set['Survived'].values, predictions))
    print 'AUC Score (Test): {0}'.format(metrics.roc_auc_score(data_set['Survived'], probabilities))


# Output prediction file based on testing set
def output_prediction(model):
    model.fit(train_data[predictors], train_data['Survived'])
    test_predictions = model.predict(test_data[predictors])
    d = {'PassengerId': test_data.index.values, 'Survived': test_predictions.astype(int)}
    df_output = pd.DataFrame(data=d)

    df_output.to_csv('../submission.csv', index=False)

# XGBoost
gbm = xgb.XGBClassifier\
    (max_depth=3, n_estimators=200, gamma=5, colsample_bytree=0.8, learning_rate=0.05, seed=42)

# Random Forest
# drf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)

show_performance(training_set, gbm)
show_performance(validation_set, gbm)

output_prediction(gbm)