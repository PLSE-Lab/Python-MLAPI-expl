import pandas as pd
import tensorflow as tf
import tempfile

def input_fn(data_file, num_epochs, shuffle):
    data_file = pd.read_csv(tf.gfile.Open(data_file), names=[
        'PassengerId',
        'Survived',
        'Pclass',
        'Name',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Ticket',
        'Fare',
        'Cabin',
        'Embarked'
    ], skiprows=1, engine='python').drop(["Name", "Ticket", "Fare", "Cabin", "Embarked"], axis=1).dropna(how='any', axis=0)
    data_file['Pclass'] = data_file['Pclass'].astype(str)
    data_file['Sex'] = pd.get_dummies(data_file['Sex'])
    data_file['Parch'] = pd.get_dummies(data_file['Parch'])
    return tf.estimator.inputs.pandas_input_fn(
        x=data_file,
        y=data_file['Survived'].apply(lambda x: bool(x)),
        batch_size=100,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=5
    )


sex = tf.feature_column.categorical_column_with_vocabulary_list('Sex', [0, 1])
pclass = tf.feature_column.categorical_column_with_vocabulary_list('Pclass', ['1', '2', '3'])
sibsp = tf.feature_column.numeric_column('SibSp')
parch = tf.feature_column.numeric_column('Parch')
age = tf.feature_column.numeric_column('Age')
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[1, 16, 30, 60, 100])

model = tf.estimator.LinearClassifier(model_dir='tmp/tf_titanic_log_reg', feature_columns=[sex, pclass, sibsp, parch, age_buckets])

model.train(input_fn=input_fn('../input/train.csv', None, True), steps=500)

results = model.evaluate(input_fn=input_fn('../input/test.csv', 1, False), steps=None)

for key in sorted(results):
    print('%s: %s' % (key, results[key]))