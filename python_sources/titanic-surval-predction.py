# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

train_original = pd.read_csv('../input/train.csv')
test_original = pd.read_csv('../input/test.csv')

total = [train_original, test_original]

train_data_num = train_original[['Age', 'Fare']]
test_data_num = test_original[['Age', 'Fare']]


class CateImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        If the Series is of dtype Object, then impute with the most frequent object.
        If the Series is not of dtype Object, then impute with the mean.

        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def process_cat_data(raw_data):
    data_cat = raw_data[['Pclass', 'Sex', 'Embarked', 'Name']]

    # train_data_cat['FamilySize'] = train_original['SibSp'] + train_original['Parch'] + 1
    data_cat['FamilySize'] = raw_data.apply(lambda row: row['SibSp'] + row['Parch'] + 1, axis=1)
    data_cat['IsAlone'] = data_cat['FamilySize'].apply(lambda x: 'false' if x > 1 else 'true')
    data_cat = data_cat.drop(['FamilySize'], axis=1)

    data_cat['Salutation'] = data_cat.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    print(pd.crosstab(data_cat['Salutation'], data_cat['Sex']))
    data_cat['Salutation'] = data_cat['Salutation'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data_cat['Salutation'] = data_cat['Salutation'].replace('Mlle', 'Miss')
    data_cat['Salutation'] = data_cat['Salutation'].replace('Ms', 'Miss')
    data_cat['Salutation'] = data_cat['Salutation'].replace('Mme', 'Mrs')
    # train_data_cat['Salutation'] = pd.factorize(train_data_cat['Salutation'])[0]
    data_cat = data_cat.drop(['Name'], axis=1)

    mf_imputer = CateImputer()
    mf_imputer.fit(data_cat)
    data_cat_tr = pd.DataFrame(mf_imputer.transform(data_cat), columns=data_cat.columns)
    data_cat_tr['Pclass'] = data_cat_tr['Pclass'].astype(np.str)
    return pd.get_dummies(data_cat_tr).join(train_data_num_tr)

# mean_imputer = Imputer(strategy='median')
# mean_imputer.fit(data_num)
# data_num_tr = pd.DataFrame(mean_imputer.transform(data_num), columns=data_num.columns)
num_pipeline = Pipeline([
    ('imputer', Imputer(strategy='median')),
    ('scaler', StandardScaler())
])


train_data_num_tr = pd.DataFrame(num_pipeline.fit_transform(train_data_num), columns=['Age', 'Fare'])
print(train_data_num.isnull().any())
test_data_num_tr = pd.DataFrame(num_pipeline.fit_transform(test_data_num), columns=['Age', 'Fare'])
print(test_data_num.isnull().any())

train_data_cat = train_original[['Pclass', 'Sex', 'Embarked', 'Name']]

# train_data_cat['FamilySize'] = train_original['SibSp'] + train_original['Parch'] + 1
train_data_cat['FamilySize'] = train_original.apply(lambda row: row['SibSp'] + row['Parch'] + 1, axis=1)
train_data_cat['IsAlone'] = train_data_cat['FamilySize'].apply(lambda x: 'false' if x > 1 else 'true')
train_data_cat = train_data_cat.drop(['FamilySize'], axis=1)

train_data_cat['Salutation'] = train_data_cat.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
print(pd.crosstab(train_data_cat['Salutation'], train_data_cat['Sex']))
train_data_cat['Salutation'] = train_data_cat['Salutation'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data_cat['Salutation'] = train_data_cat['Salutation'].replace('Mlle', 'Miss')
train_data_cat['Salutation'] = train_data_cat['Salutation'].replace('Ms', 'Miss')
train_data_cat['Salutation'] = train_data_cat['Salutation'].replace('Mme', 'Mrs')
# train_data_cat['Salutation'] = pd.factorize(train_data_cat['Salutation'])[0]
train_data_cat = train_data_cat.drop(['Name'], axis=1)

mf_imputer = CateImputer()
mf_imputer.fit(train_data_cat)
train_data_cat_tr = pd.DataFrame(mf_imputer.transform(train_data_cat), columns=train_data_cat.columns)
train_data_cat_tr['Pclass'] = train_data_cat_tr['Pclass'].astype(np.str)
train_data_cat_tr_1hot = pd.get_dummies(train_data_cat_tr)
# X_input = train_data_cat_tr_1hot.join(train_data_num_tr)
X_input = process_cat_data(train_original)
y_input = train_original['Survived']
X_output = process_cat_data(test_original)
print(X_input.info())
# X_train = train_data_cat_tr_1hot.join(train_data_num_tr).values
# y_train = pd.get_dummies(train_original['Survived']).values
X_train, X_test, y_train, y_test = train_test_split(X_input.values, y_input.values, random_state=42)


def main():
    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=17)]

    # Build 3 layer DNN with 512, 256, 128 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[512, 1024, 512],
                                                n_classes=2,
                                                optimizer=tf.train.ProximalAdagradOptimizer(
                                                    learning_rate=0.15,
                                                    l1_regularization_strength=0.001
                                                ))

    # Define the training inputs
    def get_train_inputs():
        x = tf.constant(X_train)
        y = tf.constant(y_train)
        return x, y

    # Fit model.
    classifier.fit(input_fn=get_train_inputs, steps=1200)

    # Define the test inputs
    def get_test_inputs():
        x = tf.constant(X_test)
        y = tf.constant(y_test)

        return x, y

    # Evaluate accuracy.
    # print(classifier.evaluate(input_fn=get_test_inputs, steps=1))
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
    graph_location = '/tmp/tensorflow/car-evaluation'
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    print("Test Accuracy: {0:f}".format(accuracy_score))

    # Classify two new flower samples.
    # med,med,5more,more,med,high,vgood
    # med,med,4,2,small,high,unacc
    # def new_samples():
    #     return np.array([[2, 2, 3, 2, 1, 2], [2, 2, 2, 0, 0, 2]], dtype=np.float32)
    #
    def output_samples():
        return X_output.values

    predictions = list(classifier.predict(input_fn=output_samples))
    passenger_id = test_original['PassengerId'].values
    predict = np.column_stack((passenger_id, np.array(predictions)))
    np.savetxt("predict.csv", predict, fmt='%i', delimiter=",", header='PassengerId,Survived', comments='')
    print("New Samples, Class Predictions: {}".format(predictions))


if __name__ == "__main__":
    main()
