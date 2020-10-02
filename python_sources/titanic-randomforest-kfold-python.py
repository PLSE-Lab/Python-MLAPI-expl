import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
import warnings
import seaborn as sns
import csv as csv
from sklearn.utils import check_array
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

le = preprocessing.LabelEncoder()

def process_Cabin(df):
    df['Cabin'].fillna('N')
    # cleaning up data
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if isinstance(x, str) else '$')
    le = preprocessing.LabelEncoder()
    df['Cabin'] = le.fit_transform(df['Cabin'])
    return df

def process_Ticket(df):
    df['Ticket'].fillna('N')
    # cleaning up data
    df['Ticket'] = df['Ticket'].apply(lambda x: x[0:1] if isinstance(x, str) else '$')
    le = preprocessing.LabelEncoder()
    df['Ticket'] = le.fit_transform(df['Ticket'])
    return df


def process_ports(df):
    if len(df.Age[df.Embarked.isnull()]) > 0:
        df.loc[(df.Embarked.isnull()), 'Embarked'] = '$'
    le = preprocessing.LabelEncoder()
    df['Embarked'] = le.fit_transform(df['Embarked'])
    return df


def process_gender(df):
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    return df


def process_age(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories

    df['Age'] = le.fit_transform(df['Age'])
    return df



def process_names(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['Lname'] = le.fit_transform(df['Lname'])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    #print(df['NamePrefix'].unique())
    df['NamePrefix'] = le.fit_transform(df['NamePrefix'])
    return df




def run_kfold(clf,X_all,y_all,folds):
    kf = KFold(X_all.count()[0], n_folds=folds)
    outcomes = []
    fold = 0

    X_all = X_all.copy()

    median_age = X_all['Age'].dropna().median()
    if len(X_all.Age[X_all.Age.isnull()]) > 0:
        X_all.loc[(X_all.Age.isnull()), 'Age'] = median_age

    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))




desired_width = 320
pd.set_option('display.width', desired_width)


data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv('../input/test.csv')

data_train = process_Cabin(data_train)
data_train = process_ports(data_train)
data_train = process_gender(data_train)
data_train = process_Ticket(data_train)
data_train = process_age(data_train)
data_train = process_names(data_train)
# All the ages with no input -> make the median of all Ages
num = data_train.count()
#print(num)

#print(median_age)
#print(data_train[0:100])
#print(data_train.Embarked.unique())
#print(data_train.Ticket.unique())
#print(data_train[0:100])


X_all = data_train.drop(['Survived', 'PassengerId','Name'], axis=1)
y_all = data_train['Survived']



num_test = 0.50
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

clf = RandomForestClassifier(n_estimators=10)

run_kfold(clf,X_train,y_train,5)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Total accuracy: {0}".format( accuracy))


#print("score is {0}".format(clf.score(X,Y)))


with open("submission.csv", "w") as predictions_file:
    predictions_file_object = predictions_file
    predictions_file_object.write('PassengerId,Survived\r\n')

    test_data = pd.read_csv("../input/test.csv")
    test_passengerId = test_data['PassengerId'].copy()

    test_data = process_gender(test_data)
    test_data = process_Cabin(test_data)
    test_data = process_ports(test_data)
    test_data = process_Ticket(test_data)
    test_data = process_age(test_data)
    test_data = process_names(test_data)

    #TODO: add cabin and see if it makes difference.
    test_data = test_data.drop(['Name', 'PassengerId'], axis=1)

    test_inputs = test_data

    for ix in range(len(test_data.index)):
        inputVector = test_inputs.irow(ix)
        inputVector = np.nan_to_num(inputVector)
        inputVector = check_array(inputVector, accept_sparse='csr', dtype=np.float64, order="C")
        estimatedLabel = output = clf.predict(inputVector)
        if estimatedLabel[0] == -1 :
            estimatedLabel[0] = 0
        predictions_file_object.write("{0},{1}\r\n".format(test_passengerId[ix],estimatedLabel[0]))

# Close out the files
    predictions_file.close()