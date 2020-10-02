import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.Age.fillna(train.Age.mean(), inplace=True)  # replace missing age in train data by average age
train.Embarked.fillna(("S"), inplace=True)  # replace missing embarked value in train data by 'S'

# compare survived and dead passengers by gender
survived_sex = train[train['Survived'] == 1]['Sex'].value_counts()
dead_sex = train[train['Survived'] == 0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex, dead_sex])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(9, 8), color=['r', 'b'], title='Survival Rates By Gender')
plt.show()  # there were more survived female passengers than dead, while male passengers had higher dead rates
# than survived

# compare survived and dead passengers by age
age_figure = plt.figure(figsize=(9, 8))
plt.hist([train[train['Survived'] == 1]['Age'], train[train['Survived'] == 0]['Age']], stacked=True, color=['g', 'r'],
         bins=30, label=['Survived', 'Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()

# survival rates of children vs adult passengers
fig, axs = plt.subplots(1, 2)
train[train["Age"] < 18].Survived.value_counts().plot(kind='bar', ax=axs[0], title='Children Survived')
train[train["Age"] >= 18].Survived.value_counts().plot(kind='bar', ax=axs[1], title='Adult Survived')
plt.show()  # the graph shows that children have higher survived rates

# compare survived and dead passengers by fare
fare_figure = plt.figure(figsize=(9, 8))
plt.hist([train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']], stacked=True, color=['g', 'r'],
         bins=30, label=['Survived', 'Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()  # the histogram shows that number of passenger with low fare has higher dead rates

# survival rates by passenger class
survived_class = train[train['Survived'] == 1]['Pclass'].value_counts()
dead_class = train[train['Survived'] == 0]['Pclass'].value_counts()
df = pd.DataFrame([survived_class, dead_class])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(9, 8))
plt.legend(['1st class', '2nd class', '3rd class'], loc='upper left')
plt.show()

# survival rates by embarked
survived_embark = train[train['Survived'] == 1]['Embarked'].value_counts()
dead_embark = train[train['Survived'] == 0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark, dead_embark])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(9, 8))
plt.legend(['Southampton', 'Cherbourg', 'Queenstown'], loc='upper left')
plt.show()


def get_combined_data():
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    train.pop("Survived")

    combined_data = train.append(test)
    combined_data.reset_index(inplace=True)
    combined_data.drop('index', inplace=True, axis=1)

    return combined_data

combined_data = get_combined_data()


def get_titles():
    # get passengers titles and match them to their categories.
    global combined_data

    combined_data['Title'] = combined_data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    Title_Dict = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }

    combined_data['Title'] = combined_data.Title.map(Title_Dict)

get_titles()


def fill_age():
    # replace unknown age by median age in each group based by gender, class and title.
    global combined_data

    combined_data["Age"] = combined_data.groupby(['Sex', 'Pclass', 'Title'])['Age'].transform(
        lambda x: x.fillna(x.median()))


fill_age()


def drop_names():
    # remove Name column as we will be using Title instead
    global combined_data
    combined_data.drop('Name', axis=1, inplace=True)

    titles_dummies = pd.get_dummies(combined_data['Title'], prefix='Title')
    combined_data = pd.concat([combined_data, titles_dummies], axis=1)

    combined_data.drop('Title', axis=1, inplace=True)


drop_names()


def fill_fares():
    # replace unknown fare by average fare.
    global combined_data

    combined_data.Fare.fillna(combined_data.Fare.mean(), inplace=True)

fill_fares()


def fill_embarked():
    # replace unknown embarked by 'S'.
    global combined_data
    combined_data.Embarked.fillna('S', inplace=True)

    embarked_dummies = pd.get_dummies(combined_data['Embarked'], prefix='Embarked')
    combined_data = pd.concat([combined_data, embarked_dummies], axis=1)
    combined_data.drop('Embarked', axis=1, inplace=True)


fill_embarked()


def fill_cabin():
    global combined_data

    # replacing missing cabins with U (for Unknown)
    combined_data.Cabin.fillna('U', inplace=True)

    # mapping each Cabin value with the cabin letter
    combined_data['Cabin'] = combined_data['Cabin'].map(lambda c: c[0])
    cabin_dummies = pd.get_dummies(combined_data['Cabin'], prefix='Cabin')
    combined_data = pd.concat([combined_data, cabin_dummies], axis=1)
    combined_data.drop('Cabin', axis=1, inplace=True)


fill_cabin()


def process_sex():
    # replace value in Sex column with 1 for male and 0 for female
    global combined_data
    combined_data['Sex'] = combined_data['Sex'].map({'male': 1, 'female': 0})


process_sex()


def process_pclass():
    global combined_data
    pclass_dummies = pd.get_dummies(combined_data['Pclass'], prefix="Pclass")
    combined_data = pd.concat([combined_data, pclass_dummies], axis=1)
    combined_data.drop('Pclass', axis=1, inplace=True)

process_pclass()


def get_ticket_prefix():
    global combined_data

    # extract each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = list(filter(lambda t: not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'

    combined_data['Ticket'] = combined_data['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined_data['Ticket'], prefix='Ticket')
    combined_data = pd.concat([combined_data, tickets_dummies], axis=1)
    combined_data.drop('Ticket', inplace=True, axis=1)

get_ticket_prefix()


def get_family():
    global combined_data
    # create new feature which combine members of a family together
    combined_data['FamilySize'] = combined_data['Parch'] + combined_data['SibSp'] + 1

    # another feature tells whether the passenger is single or has small/large family
    combined_data['Singleton'] = combined_data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined_data['SmallFamily'] = combined_data['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined_data['LargeFamily'] = combined_data['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

get_family()


def features_list():
    # create a list of features that will be used
    global combined_data

    features = list(combined_data.columns)
    features.remove('PassengerId')
    combined_data[features] = combined_data[features].apply(lambda x: x / x.max(), axis=0)

features_list()


def recover_train_test_target():
    # reassign train, test and target variable
    global combined_data

    train1 = pd.read_csv('../input/train.csv')

    target = train1.Survived
    train = combined_data.ix[0:890]
    test = combined_data.ix[891:]

    return train, test, target


train, test, target = recover_train_test_target()

clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, target)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_

print(features.sort_values(['importance'], ascending=False))

# select important features
model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
print(train_new.shape)

test_new = model.transform(test)
print(test_new.shape)

# train data with RandomForestClassifier and choose the best parameter.
forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {
    'max_depth': [4, 5, 6, 7, 8],
    'n_estimators': [200, 210, 240, 250],
    'criterion': ['gini', 'entropy']
}

cross_validation = StratifiedKFold(target, n_folds=10)

# find the best parameters
grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train_new, target)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# write prediction output in .csv file
output = grid_search.predict(test_new).astype(int)
df_output = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": output
})
df_output[['PassengerId', 'Survived']].to_csv('output.csv', index=False)
