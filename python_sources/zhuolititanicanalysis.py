from __future__ import division

# https://www.kaggle.com/l3nnys/titanic/dense-highway-neural-network-for-titanic
import pandas as pd
import numpy as np
import re as re


# Clean data, convert non numerical data to numerical data
# Use the Regular Expression to get the title from the name field.
pattern = re.compile(r'.*?,(.*?)\.')


def getTitle(x):
    result = pattern.search(x)
    if result:
        return result.group(1).strip()
    else:
        return ''
        

def cleanData(train, test):
    train['Title'] = train['Name'].map(getTitle)
    test['Title'] = test['Name'].map(getTitle)

    # Set the missing Age of Title 'Master'
    master_age_mean = train['Age'][(train['Title'] == 'Master') & (train['Age'] > 0)].mean()
    train.loc[train[(train['Title'] == 'Master') & (train['Age'].isnull())].index, 'Age'] = master_age_mean
    test.loc[test[(test['Title'] == 'Master') & (test['Age'].isnull())].index, 'Age'] = master_age_mean

    # Set the missing Age of Title 'Mr'
    mr_age_mean = train['Age'][(train['Title'] == 'Mr') & (train['Age'] > 0)].mean()
    train.loc[train[(train['Title'] == 'Mr') & (train['Age'].isnull())].index, 'Age'] = mr_age_mean
    test.loc[test[(test['Title'] == 'Mr') & (test['Age'].isnull())].index, 'Age'] = mr_age_mean

    # Set the missing Age of Title 'Miss' or 'Ms'
    miss_age_mean = train['Age'][(train['Title'] == 'Miss') & (train['Age'] > 0)].mean()
    train.loc[train[(train['Title'] == 'Miss') & (train['Age'].isnull())].index, 'Age'] = miss_age_mean
    test.loc[
        test[((test['Title'] == 'Miss') | (test['Title'] == 'Ms')) & (test['Age'].isnull())].index, 'Age'] = miss_age_mean

    # Set the missing Age of Title 'Mrs'
    mrs_age_mean = train['Age'][(train['Title'] == 'Mrs') & (train['Age'] > 0)].mean()
    train.loc[train[(train['Title'] == 'Mrs') & (train['Age'].isnull())].index, 'Age'] = mrs_age_mean
    test.loc[test[(test['Title'] == 'Mrs') & (test['Age'].isnull())].index, 'Age'] = mrs_age_mean

    # Set the missing Age of Title 'Dr'
    dr_age_mean = train['Age'][(train['Title'] == 'Dr') & (train['Age'] > 0)].mean()
    train.loc[train[(train['Title'] == 'Dr') & (train['Age'].isnull())].index, 'Age'] = dr_age_mean
    test.loc[test[(test['Title'] == 'Mrs') & (test['Age'].isnull())].index, 'Age'] = dr_age_mean

    sex_to_int = {'male': 1, 'female': 0}
    train['SexInt'] = train['Sex'].map(sex_to_int)
    embark_to_int = {'S': 0, 'C': 1, 'Q': 2}
    train['EmbarkedInt'] = train['Embarked'].map(embark_to_int)
    train['EmbarkedInt'] = train['EmbarkedInt'].fillna(0)
    test['SexInt'] = test['Sex'].map(sex_to_int)
    test['EmbarkedInt'] = test['Embarked'].map(embark_to_int)
    test['EmbarkedInt'] = test['EmbarkedInt'].fillna(0)
    test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
    train['FamilySize'] = train['SibSp'] + train['Parch']
    test['FamilySize'] = test['SibSp'] + test['Parch']

    ticket = train[train['Parch'] == 0]
    ticket = ticket.loc[ticket.Ticket.duplicated(False)]
    grouped = ticket.groupby(['Ticket'])
    # The Friends field indicate if the passenger has frineds/SibSp in the boat.
    train['Friends'] = 0
    # The below fields statistic how many are survived or not survived by sex.
    train['Male_Friends_Survived'] = 0
    train['Male_Friends_NotSurvived'] = 0
    train['Female_Friends_Survived'] = 0
    train['Female_Friends_NotSurvived'] = 0
    for (k, v) in grouped.groups.items():
        for i in range(0, len(v)):
            train.loc[v[i], 'Friends'] = 1
            train.loc[v[i], 'Male_Friends_Survived'] = train[
                (train.Ticket == k) & (train.index != v[i]) & (train.Sex == 'male') & (
                train.Survived == 1)].Survived.count()
            train.loc[v[i], 'Male_Friends_NotSurvived'] = train[
                (train.Ticket == k) & (train.index != v[i]) & (train.Sex == 'male') & (
                train.Survived == 0)].Survived.count()
            train.loc[v[i], 'Female_Friends_Survived'] = train[
                (train.Ticket == k) & (train.index != v[i]) & (train.Sex == 'female') & (
                train.Survived == 1)].Survived.count()
            train.loc[v[i], 'Female_Friends_NotSurvived'] = train[
                (train.Ticket == k) & (train.index != v[i]) & (train.Sex == 'female') & (
                train.Survived == 0)].Survived.count()

    test_ticket = test[test['Parch'] == 0]
    test['Friends'] = 0
    test['Male_Friends_Survived'] = 0
    test['Male_Friends_NotSurvived'] = 0
    test['Female_Friends_Survived'] = 0
    test['Female_Friends_NotSurvived'] = 0

    grouped = test_ticket.groupby(['Ticket'])
    for (k, v) in grouped.groups.items():
        temp_df = train[train.Ticket == k]
        length = temp_df.shape[0]
        if temp_df.shape[0] > 0:
            for i in range(0, len(v)):
                test.loc[v[i], 'Friends'] = 1
                test.loc[v[i], 'Male_Friends_Survived'] = temp_df[(temp_df.Sex == 'male') & (temp_df.Survived == 1)].shape[
                    0]
                test.loc[v[i], 'Male_Friends_NotSurvived'] = \
                temp_df[(temp_df.Sex == 'male') & (temp_df.Survived == 0)].shape[0]
                test.loc[v[i], 'Female_Friends_Survived'] = \
                temp_df[(temp_df.Sex == 'female') & (temp_df.Survived == 1)].shape[0]
                test.loc[v[i], 'Female_Friends_NotSurvived'] = \
                temp_df[(temp_df.Sex == 'female') & (temp_df.Survived == 0)].shape[0]

    train['FatherOnBoard'] = 0
    train['FatherSurvived'] = 0
    train['MotherOnBoard'] = 0
    train['MotherSurvived'] = 0
    train['ChildOnBoard'] = 0
    train['ChildSurvived'] = 0
    train['ChildNotSurvived'] = 0
    grouped = train[train.Parch > 0].groupby('Ticket')
    for (k, v) in grouped.groups.items():
        for i in range(0, len(v)):
            if train.loc[v[i], 'Age'] < 19:
                temp = train[(train.Ticket == k) & (train.Age > 18)]
                if temp[temp.SexInt == 1].shape[0] == 1:
                    train.loc[v[i], 'FatherOnBoard'] = 1
                    train.loc[v[i], 'FatherSurvived'] = temp[temp.SexInt == 1].Survived.sum()
                if temp[temp.SexInt == 0].shape[0] == 1:
                    train.loc[v[i], 'MotherOnBoard'] = 1
                    train.loc[v[i], 'MotherSurvived'] = temp[temp.SexInt == 0].Survived.sum()
            else:
                temp = train[(train.Ticket == k) & (train.Age < 19)]
                length = temp.shape[0]
                if length > 0:
                    train.loc[v[i], 'ChildOnBoard'] = 1
                    train.loc[v[i], 'ChildSurvived'] = temp[temp.Survived == 1].shape[0]
                    train.loc[v[i], 'ChildNotSurvived'] = temp[temp.Survived == 0].shape[0]

    test['FatherOnBoard'] = 0
    test['FatherSurvived'] = 0
    test['MotherOnBoard'] = 0
    test['MotherSurvived'] = 0
    test['ChildOnBoard'] = 0
    test['ChildSurvived'] = 0
    test['ChildNotSurvived'] = 0
    grouped = test[test.Parch > 0].groupby('Ticket')
    for (k, v) in grouped.groups.items():
        temp = train[train.Ticket == k]
        length = temp.shape[0]
        if length > 0:
            for i in range(0, len(v)):
                if test.loc[v[i], 'Age'] < 19:
                    if temp[(temp.SexInt == 1) & (temp.Age > 18)].shape[0] == 1:
                        test.loc[v[i], 'FatherOnBoard'] = 1
                        test.loc[v[i], 'FatherSurvived'] = temp[(temp.SexInt == 1) & (temp.Age > 18)].Survived.sum()
                    if temp[(temp.SexInt == 0) & (temp.Age > 18)].shape[0] == 1:
                        test.loc[v[i], 'MotherOnBoard'] = 1
                        test.loc[v[i], 'MotherSurvived'] = temp[(temp.SexInt == 0) & (temp.Age > 18)].Survived.sum()
                else:
                    length = temp[temp.Age < 19].shape[0]
                    if length > 0:
                        test.loc[v[i], 'ChildOnBoard'] = 1
                        test.loc[v[i], 'ChildSurvived'] = temp[(temp.Age < 19) & (temp.Survived == 1)].shape[0]
                        test.loc[v[i], 'ChildNotSurvived'] = temp[(temp.Age < 19) & (temp.Survived == 0)].shape[0]

    title_to_int = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 1, 'Dr': 4, 'Rev': 4, 'Mlle': 2, 'Major': 4, 'Col': 4,
                    'Ms': 3, 'Lady': 3, 'the Countess': 4, 'Sir': 4, 'Mme': 3, 'Capt': 4, 'Jonkheer': 4, 'Don': 1,
                    'Dona': 3}
    train['TitleInt'] = train['Title'].map(title_to_int)
    test['TitleInt'] = test['Title'].map(title_to_int)
    train.loc[train[train['Age'] < 13].index, 'TitleInt'] = 5
    test.loc[test[test['Age'] < 13].index, 'TitleInt'] = 5

    train['FareCat'] = pd.cut(train['Fare'], [-0.1, 50, 100, 150, 200, 300, 1000], right=True,
                              labels=[0, 1, 2, 3, 4, 5])
    test['FareCat'] = pd.cut(test['Fare'], [-0.1, 50, 100, 150, 200, 300, 1000], right=True,
                             labels=[0, 1, 2, 3, 4, 5])
    train['AgeCat'] = pd.cut(train['Age'], [-0.1, 12.1, 20, 30, 35, 40, 45, 50, 55, 65, 100], right=True,
                             labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    test['AgeCat'] = pd.cut(test['Age'], [-0.1, 12.1, 20, 30, 35, 40, 45, 50, 55, 65, 100], right=True,
                            labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    return train, test

# Defining and scaling train and test data
# Utility split method
def split_data(x, y, split_value, indices=None):
    # Keeping the indices is usefull sometimes
    if indices is None:
        indices = np.arange(x.shape[0]) # x.shape is 2d tuple, 0 index is the row size
        # shuffling
        np.random.shuffle(indices)
    data = x[indices]
    labels = y[indices]
    nb_test_samples = int(split_value * data.shape[0])
    # Splitting
    x_ = data[:-nb_test_samples] # from zero index until the last nb_test_samples index
    y_ = labels[:-nb_test_samples]
    _x = data[-nb_test_samples:] # from the last nb_test_samples index to end
    _y = labels[-nb_test_samples:]
    return x_, y_, _x, _y, indices

def standardScalerData(x_train, x_test, test, columns):
    from sklearn.preprocessing import StandardScaler

    # data scaling
    scaler = StandardScaler()
    scaled_x_train = scaler.fit_transform(x_train)
    scaled_x_test = scaler.transform(x_test)
    scaled_test = scaler.transform(test[columns])
    return scaled_x_train, scaled_x_test, scaled_test

import pandas as pd
import numpy as np
from keras.utils import np_utils

# Reading data
# Train data
traindata = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})

# Test data
testdata  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})

#Print to standard output, and see the results in the "log" section below after running your script
# print("\n\nTop of the training data:")
# print(traindata.head())
# print(testdata.shape)

traindata, testdata = cleanData(traindata, testdata)

#Print to standard output, and see the results in the "log" section below after running your script
# print("\n\nTop of the training data:")
# print(traindata.head())
# print(testdata.shape)


# Defining columns to use in the model
columns = ['Pclass', 'SexInt', 'EmbarkedInt', 'Age', 'TitleInt', 'Fare',
           'Friends', 'Male_Friends_Survived', 'Male_Friends_NotSurvived', 'Female_Friends_Survived',
           'Female_Friends_NotSurvived',
           'MotherOnBoard', 'MotherSurvived', 'ChildOnBoard', 'ChildSurvived', 'ChildNotSurvived']

# train/test split
x_train, y_train, x_test, y_test, _ = split_data(traindata[columns].values, traindata['Survived'].values, 0.2)

## Transform labels to one-hot encoding
## i.e., from '7' to [0,0,0,0,0,0,0,1,0,0]
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)

scaled_x_train, scaled_x_test, scaled_test = standardScalerData(x_train=x_train, x_test=x_test, test=testdata, columns=columns)

# After data scalling
print("x train shape:   " + str(scaled_x_train.shape))
print("x test shape:    " + str(scaled_x_test.shape))
print("test shape:  " + str(scaled_test.shape))
print(y_train[:5])

from keras.models import Sequential
model = Sequential()
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#
# First hidden layer with 'first_layer_width' neurons.
# Also need to specify input dimension.
# 'Dense' means fully-connected.
dropout_rate = 0
second_layer_width=128
learning_rate = 0.1
model.add(Dense(128, input_dim=scaled_x_train.shape[1], W_regularizer=None))
model.add(Activation("relu"))
if dropout_rate > 0:
    model.add(Dropout(0.5))

## Second hidden layer.
model.add(Dense(second_layer_width))
model.add(Activation("relu"))
if dropout_rate > 0:
    model.add(Dropout(0.5))

model.add(Dense(second_layer_width))
model.add(Activation("relu"))
if dropout_rate > 0:
    model.add(Dropout(0.5))

model.add(Dense(y_train.shape[1]))
## For classification, the activation is softmax
model.add(Activation('softmax'))
## Define optimizer. In this tutorial/codelab, we select SGD.
## You can also use other methods, e.g., opt = RMSprop()
opt = SGD(lr=learning_rate, clipnorm=5.)
## Define loss function = 'categorical_crossentropy' or 'mean_squared_error'
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit(scaled_x_train, y_train, nb_epoch=20, batch_size=32)

# Final test predict
test_proba = model.predict(scaled_test)
print(test_proba.shape)
print(test_proba[:5])
test_classes = np_utils.probas_to_classes(test_proba)
print(test_classes.shape)
print(test_classes[:5])

# Save to csv
# Competition submission
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_classes
    })
submission.to_csv("titanic_predict_highwaynn.csv", index=False)
