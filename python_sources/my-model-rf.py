# data analysis and wrangling
import pandas as pd
import numpy as np

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV

def cleanTicket(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = list(map(lambda t : t.strip() , ticket))
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'XXX'

def get_dummy_data():
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    ids = test['PassengerId']

    # Combine the two data sets just to make sure each end with same set of features
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)

    # Data Clean Up
    # --------------------------------------------------------------------------------------------------

    # Columns that do not seem important: PassengerID, Name, Ticket

    # Convert sex to 0/1
    combined['Sex'] = combined['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # Get Cabin letter from Cabin and convert to number
    combined['Cabin'] = combined.Cabin.str.extract('([A-Za-z])', expand=False)
    combined['Cabin'] = combined['Cabin'].fillna("NA")
    combined = pd.get_dummies(combined, columns = ['Cabin'])

    # Clean up NAs from fare
    combined['Fare'].fillna(combined['Fare'].dropna().mean(), inplace=True)

    # Clean up NAs from age
    guess_ages = np.zeros((2,3))
    guess_ages_test = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = combined[(combined['Sex'] == i) & (combined['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            combined.loc[(combined.Age.isnull()) & (combined.Sex == i) & (combined.Pclass == j+1), 'Age'] = guess_ages[i,j]

    combined['Age'] = combined['Age'].astype(int)

    # Convert Age into ranges
    combined.loc[ combined['Age'] <= 6, 'Age'] = 0
    combined.loc[(combined['Age'] > 6) & (combined['Age'] <= 21), 'Age'] = 1
    combined.loc[(combined['Age'] > 21) & (combined['Age'] <= 26), 'Age'] = 2
    combined.loc[(combined['Age'] > 26) & (combined['Age'] <= 36), 'Age'] = 3
    combined.loc[(combined['Age'] > 36), 'Age'] = 4

    combined = pd.get_dummies(combined, columns = ['Age'])

    # Get titles from name
    combined['Title'] = combined.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    combined['Title'] = combined['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    combined['Title'] = combined['Title'].replace('Mlle', 'Miss')
    combined['Title'] = combined['Title'].replace('Ms', 'Miss')
    combined['Title'] = combined['Title'].replace('Mme', 'Mrs')

    # Convert titles to numbers
    combined['Title'] = combined['Title'].fillna("Unk")

    combined = pd.get_dummies(combined, columns = ['Title'])

    # Convert Embarked to numbers
    combined['Embarked'] = combined['Embarked'].fillna("S")
    combined = pd.get_dummies(combined, columns = ['Embarked'])

    # Find number of family members on board
    combined['FamilyMems'] = combined['Parch'] + combined['SibSp']

    # Get fare ranges
    combined.loc[ combined['Fare'] <= 7.91, 'Fare'] = 0
    combined.loc[(combined['Fare'] > 7.91) & (combined['Fare'] <= 14.454), 'Fare'] = 1
    combined.loc[(combined['Fare'] > 14.454) & (combined['Fare'] <= 31), 'Fare']   = 2
    combined.loc[ combined['Fare'] > 31, 'Fare'] = 3
    combined['Fare'] = combined['Fare'].astype(int)

    # Variables from ahmedbesbes's solution
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilyMems'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilyMems'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilyMems'].map(lambda s : 1 if 5<=s else 0)


    combined = pd.get_dummies(combined, columns = ['FamilyMems'])
    combined = pd.get_dummies(combined, columns = ['Fare'])
    combined = pd.get_dummies(combined, columns = ['Pclass'])
    combined = pd.get_dummies(combined, columns = ['Ticket'])

    combined = combined.drop(["PassengerId", "Name", "SibSp", "Parch"], axis=1)

    train = combined.ix[0:890]
    test = combined.ix[891:]

    return (train, test.drop(["Survived"], axis=1), ids)
    
    
def k_folds(model, X_train, Y_train):
    num_test = 0.20

    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        Xf_train, Xf_test = X_train.values[train_index], X_train.values[test_index]
        Yf_train, Yf_test = Y_train.values[train_index], Y_train.values[test_index]
        model.fit(Xf_train, Yf_train)
        predictions = model.predict(Xf_test)
        accuracy = accuracy_score(Yf_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))

    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))
    
def grid_search(model, parameters, X_train, Y_train):
    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    # Run the grid search
    grid_obj = GridSearchCV(model, parameters, scoring=acc_scorer)
    grid_obj = grid_obj.fit(X_train, Y_train)

    print(grid_obj.best_params_)

    # Set the clf to the best combination of parameters
    return grid_obj.best_estimator_
    
(train, test, ids) = get_dummy_data()

num_test = 0.20
X_all = train.drop("Survived", axis=1)
Y_all = train["Survived"]
X_test  = test

# Pick a subset of parameters to use
clf = RandomForestClassifier()
clf = clf.fit(X_all, Y_all)

features = pd.DataFrame()
features['feature'] = X_all.columns
features['importance'] = clf.feature_importances_
features.sort_values(['importance'],ascending=False)

X_all_new = X_all[["Title_Mr", "Sex", "Title_Mrs", "Pclass_3", "Title_Miss", "Cabin_NA", "Fare_0", "Age_3", "Age_2", "Embarked_C"]]
X_test_new = X_test[["Title_Mr", "Sex", "Title_Mrs", "Pclass_3", "Title_Miss", "Cabin_NA", "Fare_0", "Age_3", "Age_2", "Embarked_C"]]

# model = SelectFromModel(clf, prefit=True)
# X_all_new = model.transform(X_all)
X_all_new.shape
# X_test_new = model.transform(X_test)
X_test_new.shape

# Choose some parameter combinations to try for grid search
parameters = {'n_estimators': [10, 100, 1000],
              'max_features': ['log2', 'sqrt'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [3, 30, 100, 300, 1000],
              'min_samples_split': [1.0, 3, 10, 30],
              'min_samples_leaf': [1, 3, 10, 30]
             }

model = RandomForestClassifier(n_estimators=10, min_samples_leaf=3)
# model = mlh.grid_search(model, parameters, X_all_new, Y_all)

k_folds(model, X_all_new, Y_all)

X_train, X_dev, Y_train, Y_dev = train_test_split(X_all_new, Y_all, test_size=num_test, random_state=23)

#Find score of model on training set and dev set
model.fit(X_train, Y_train)
acc_random_forest = round(model.score(X_train, Y_train) * 100, 2)
acc_random_forest
acc_random_forest = round(model.score(X_dev, Y_dev) * 100, 2)
acc_random_forest
#train model on all the data
model.fit(X_all_new, Y_all)
Y_pred = model.predict(X_test_new)

