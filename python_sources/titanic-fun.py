# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.
import keras
from sklearn import preprocessing as prep
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.linear_model import LinearRegression

np.random.seed(10)

TRAINING_DATA_PATH = Path('../input/train.csv')
TEST_DATA_PATH = Path('../input/test.csv')
OUTPUT_PATH = Path('surviving_passengers.csv')
RF_OUTPUT_PATH = Path('rf_surviving_passengers.csv')


def prepare_data(data_frame, scaler=prep.StandardScaler()):
    data_frame['Title'] = data_frame.Name.map(lambda x: _get_title(x))
    data_frame['Title'] = data_frame.apply(replace_titles, axis=1)
    # data_frame['Age'] = data_frame.apply(_deduce_missing_ages, axis=1)
    data_frame['Embarked'] = data_frame.apply(fill_embarked, axis=1)
    data_frame['FamilySize'] = data_frame['SibSp'] + data_frame['Parch']
    data_frame['Deck'] = data_frame.Cabin.str[0]
    data_frame['Deck'].fillna('Z', inplace=True)
    data_frame['Fare'] = _fill_missing_fare(data_frame)
    # data_frame['Child'] = data_frame.Age.map(lambda x: 1 if float(x) < 18.0 else 0)

    binarified_gender = pd.get_dummies(data_frame['Sex'])
    binarified_titles = pd.get_dummies(data_frame['Title'])
    binarified_embarkation = pd.get_dummies(data_frame['Embarked'])
    binarified_social_class = pd.get_dummies(
        data_frame['Pclass'],
        prefix='class'
    )
    binarified_deck = pd.get_dummies(data_frame['Deck'], prefix='deck')
    if 'deck_T' not in binarified_deck.columns:
        binarified_deck['deck_T'] = 0
    print(binarified_deck.columns)
    transformed_data_frame = pd.concat(
        [
            binarified_gender,
            binarified_social_class,
            binarified_titles,
            binarified_embarkation,
            binarified_deck,
            data_frame['Age'],
            data_frame['Fare'],
            data_frame['FamilySize'],
        ],
        axis=1
    )
    ages = _deduce_missing_ages(transformed_data_frame)
    transformed_data_frame['Age'] = ages
    transformed_data_frame.fillna(0, inplace=True)
    try:
        scaler.scale_
    except AttributeError:
        scaler.fit(transformed_data_frame.values)

    scaled_data = scaler.transform(transformed_data_frame.values)

    return pd.DataFrame(scaled_data)


def _deduce_missing_ages(x):
    test_set = x.where(x['Age'].isnull()).dropna(how='all').drop(['Age'], axis=1)
    train_set = x.where(x['Age'].notnull()).dropna(how='all')

    linear_model = LinearRegression()
    learning_features = train_set.drop(['Age'], axis=1).fillna(0)
    learning_targets = train_set['Age']
    linear_model.fit(X=learning_features.values, y=learning_targets.values)
    predictions = linear_model.predict(X=test_set.fillna(0).values)
    missing_ages = pd.DataFrame({'Age': predictions})

    return pd.concat([learning_targets, missing_ages['Age']], ignore_index=True)


def _fill_missing_fare(df):
    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()
    return df["Fare"].fillna(median_fare)


def _extract_titles(data_frame):
    return [_get_title(name) for name in data_frame['Name']]


def _get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'


def fill_embarked(x):
    embarked = x['Embarked']
    fare = float(x['Fare'])
    if embarked:
        return embarked
    else:
        if fare >= 50:
            return 'C'
        elif fare < 50 and fare > 15:
            return 'S'
        else:
            return 'Q'


def replace_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Lord']:
        return 'Mr'
    elif title in ['the Countess', 'Mme', 'Lady', 'Dona']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title


def main():
    csv_train_data = pd.read_csv(TRAINING_DATA_PATH)
    csv_test_data = pd.read_csv(TEST_DATA_PATH)

    targets = csv_train_data['Survived']

    prepared_data = prepare_data(csv_train_data)

    training_data = prepared_data[:700]
    training_targets = targets[:700]
    validation_data = prepared_data[701:]
    validation_targets = targets[701:]
    r_forest_model = RandomForestClassifier(n_jobs=16)

    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            10,
            input_dim=24,
            kernel_initializer='random_normal',
            activation='relu'
        )
    )
    model.add(keras.layers.Dense(5, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    sgd_optimizer = keras.optimizers.SGD(lr=0.3, momentum=0.5)
    model.compile(
        loss='binary_crossentropy',
        optimizer=sgd_optimizer,
        metrics=['binary_accuracy']
    )
    model.fit(
        training_data.values,
        training_targets.values,
        epochs=5,
        batch_size=128,
        shuffle=True
    )

    r_forest_model.fit(training_data.values, training_targets.values)

    prepared_test_data = prepare_data(csv_test_data)

    metrics = model.evaluate(
        validation_data.values,
        validation_targets.values,
        verbose=1
    )

    print('\n', 'Validation Loss + Accuracy:', metrics)
    print('\n', 'R Forest Score:', r_forest_model.score(validation_data.values, validation_targets.values))

    classes = model.predict(prepared_test_data.values, batch_size=128)

    discritized_classes = pd.DataFrame(
        {'Survived': [1 if x[0] >= 0.5 else 0 for x in classes]}
    )
    final_output = pd.concat(
        [csv_test_data['PassengerId'], discritized_classes],
        axis=1
    )

    r_forest_out = pd.DataFrame({'Survived': r_forest_model.predict(prepared_test_data.values)})
    final_r_forest_output = pd.concat(
        [csv_test_data['PassengerId'], r_forest_out],
        axis=1
    )
    final_output.to_csv(str(OUTPUT_PATH), index=False)
    final_r_forest_output.to_csv(str(RF_OUTPUT_PATH), index=False)


if __name__ == '__main__':
    main()
