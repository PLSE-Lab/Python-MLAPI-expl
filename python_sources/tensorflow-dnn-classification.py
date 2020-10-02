# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import argparse
import tensorflow as tf
import pandas as pd

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"

CSV_COLUMNS_TRAIN = ['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Survived']
CSV_COLUMNS_TEST = ['PassengerId', 'Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']

def handle_data(data):
    data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    data["Age"] = data["Age"].fillna(data["Age"].mean())
    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())

def load_data():
    train = pd.read_csv(TRAIN_PATH)[CSV_COLUMNS_TRAIN]
    handle_data(train)
    train_x, train_y = train, train.pop('Survived')

    test = pd.read_csv(TEST_PATH)[CSV_COLUMNS_TEST]
    handle_data(test)
    test_x, passenger_id = test, test.pop('PassengerId')

    return (train_x, train_y), (passenger_id, test_x)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=20000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (passenger_id, test_x) = load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Three hidden layers of 10 nodes.
        hidden_units=[10, 10],
        # The model must choose between 2 classes.
        n_classes=2)

    # Train the Model.
    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)
    print('train complete')

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(train_x, train_y, args.batch_size))
    print('\nTest set accuracy: {accuracy:0.4f}\n'.format(**eval_result))

    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(test_x, labels=None, batch_size=args.batch_size))

    prediction_ids = [prediction['class_ids'][0] for prediction in predictions]

    submission = pd.DataFrame({
      "PassengerId": passenger_id,
      "Survived": prediction_ids
      })
    submission.to_csv("nn_submission.csv", index=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)