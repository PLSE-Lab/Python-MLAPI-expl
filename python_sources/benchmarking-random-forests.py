import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def df_cleaner(df):
    """
    Clean up a few variables in the training/test sets.
    """
    
    # Clean up ages.
    for passenger in df[(df['Age'].isnull())].index:
        df.loc[passenger, 'Age'] = np.average(df[(df['Age'].notnull())]['Age'])

    # Clean up fares.
    for passenger in df[(df['Fare'].isnull())].index:
        df.loc[passenger, 'Fare'] = np.average(df[(df['Fare'].notnull())]['Fare'])

    # Manually convert values to numeric columns for clarity.
    # Change the sex to a binary column.
    df['Sex'][(df['Sex'] == 'male')] = 0
    df['Sex'][(df['Sex'] == 'female')] = 1
    df['Sex'][(df['Sex'].isnull())] = 2

    # Transform to categorical data.
    df['Embarked'][(df['Embarked'] == 'S')] = 0
    df['Embarked'][(df['Embarked'] == 'C')] = 1
    df['Embarked'][(df['Embarked'] == 'Q')] = 2
    df['Embarked'][(df['Embarked'].isnull())] = 3

    return df


def main():
    """
    Visualization of random forest accuracy as function of
    the number of tress available in the ensemble.
    """

    # Read in the training data.
    train = pd.read_csv('../input/train.csv')

    # Set sampling.
    sampling = .75

    # Clean it up.
    # Remove unused columns, clean age, and convert gender to binary column.
    train.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
    train = df_cleaner(train)

    # Split it into coordinates.
    x_train = train[:int(len(train) * sampling)][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y_train = train[:int(len(train) * sampling)][['Survived']]
    x_test = train[int(len(train) * sampling):][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y_test = train[int(len(train) * sampling):]['Survived']

    # See how it scores as a dependent on the number of trees.
    scores = []
    values = range(1, 200)
    for trees in values:
        model = RandomForestClassifier(n_estimators=trees)
        model.fit(x_train, np.ravel(y_train))
        scores.append(model.score(x_test, y_test))

    # Plot the score as a function of the trees.
    plt.plot(values, scores, '-r')
    plt.xlabel('Trees')
    plt.ylabel('Score')
    plt.title('Correct Predictions by Number of Trees')
    plt.show()    

    # Save the image so it shows up in Kaggle.
    plt.savefig("rf-benchmark.png")

if __name__ == '__main__':
    main()
