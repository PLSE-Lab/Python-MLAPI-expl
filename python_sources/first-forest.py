import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier


def normalize_data(file):
    """
    function reads a file and normalizes the data. Such as replacing null values
    with averages and enumerated strings with integer representations.
    :param file:
    :return: panda data frame
    """
    
    # load data into a data frame
    df = pd.read_csv(file, header=0)

    # convert strings to integer classifiers FEMALE = 0, MALE = 1
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # replace missing data with most common
    if len(df.Embarked[df.Embarked.isnull()]) > 0:
        df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
    Ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
    Ports_dict = {name: i for i, name in Ports}              # set up a dictionary in the form  Ports : index
    df.Embarked = df.Embarked.map(lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    # replace missing age data with the median age
    median_age = df['Age'].dropna().median()
    if len(df.Age[df.Age.isnull()]) > 0:
        df.loc[(df.Age.isnull()), 'Age'] = median_age

    # All the missing Fares -> assume median of their respective class
    if len(df.Fare[df.Fare.isnull()]) > 0:
        median_fare = np.zeros(3)
        for f in range(0, 3):  # loop 0 to 2
            median_fare[f] = df[df.Pclass == f + 1][
                'Fare'].dropna().median()
        for f in range(0, 3):  # loop 0 to 2
            df.loc[(df.Fare.isnull()) & (df.Pclass == f + 1), 'Fare'] = median_fare[f]

    # Collect the data's PassengerIds before dropping it
    ids = df['PassengerId'].values

    df['Name_Title'] = df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    Name_Titles = list(enumerate(np.unique(df['Name_Title'])))    # determine all values of Embarked,
    Name_Title_dict = {name: i for i, name in Name_Titles}
    df.Name_Title = df.Name_Title.map(lambda x: Name_Title_dict[x]).astype(int)

    # df['Cabin_Letter'] = df['Cabin'].apply(lambda x: str(x)[0])

    # Remove the Name column, Cabin, Ticket, and Sex
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch'], axis=1)

    return df, ids

train_df, train_ids = normalize_data('../input/train.csv')
test_df, test_ids = normalize_data('../input/test.csv')

# The data is now ready to go. So lets fit to the train,
# then predict to the test!

# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

print('Train')
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=700,
                                min_samples_split=10,
                                min_samples_leaf=1,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
print("%.4f" % forest.oob_score_)

print('Predict')
# train_data = train_df.drop(['Survived'], axis=1).values
output = forest.predict(test_data).astype(int)

predictions_file = open("titanic_output.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(test_ids, output))
predictions_file.close()
print('Complete.')
