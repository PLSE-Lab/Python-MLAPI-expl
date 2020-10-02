# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv('../input/HR_comma_sep.csv')
columns = list(data.columns.values)

### Categorical columns

# employee dept is actually listed under the 'sales' column, probably in error
departments = list(pd.unique(data['sales']))
department = tf.feature_column.categorical_column_with_vocabulary_list(
        'sales', departments)

salaries = list(pd.unique(data['salary']))
salary = tf.feature_column.categorical_column_with_vocabulary_list('salary', salaries)

project_numbers = list(pd.unique(data['number_project']))
project_number = tf.feature_column.categorical_column_with_vocabulary_list('number_project', project_numbers)

# binary columns are also categorical
work_accident = tf.feature_column.categorical_column_with_vocabulary_list('Work_accident', [0, 1])
promotion_last_5years = tf.feature_column.categorical_column_with_vocabulary_list('promotion_last_5years', [0, 1])

### Continuous columns
satisfaction_level = tf.feature_column.numeric_column('satisfaction_level')
last_evaluation = tf.feature_column.numeric_column('last_evaluation')
average_monthly_hours = tf.feature_column.numeric_column('average_montly_hours') # typo intentional, from data
time_spend_company = tf.feature_column.numeric_column('time_spend_company')

### Base columns
base_columns = [department, salary, project_number, satisfaction_level, last_evaluation,
                average_monthly_hours, time_spend_company]

### Crossed columns
crossed_columns = [
    tf.feature_column.crossed_column(['sales', 'number_project'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(['sales', 'salary'], hash_bucket_size=1000)
]

### Deep columns
deep_columns = [
    tf.feature_column.indicator_column(department),
    tf.feature_column.indicator_column(salary),
    tf.feature_column.indicator_column(project_number),
    tf.feature_column.indicator_column(work_accident),
    tf.feature_column.indicator_column(promotion_last_5years),
    satisfaction_level,
    last_evaluation,
    average_monthly_hours,
    time_spend_company
]

### Build the estimator (a DNNLinearCombinedClassifier)
def build_estimator(model_dir):
    return tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=crossed_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])

### Split train/test sets
clean_data = data.dropna(how='any', axis=0)
msk = np.random.rand(len(clean_data)) < 0.8
y = clean_data['left']
x = clean_data.drop('left', axis=1)
train_x = x[msk]
train_y = y[msk]
test_x = x[~msk]
test_y = y[~msk]

### The input functions, as per the Tensorflow estimator's expectation
def input_fn(num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
            x=train_x,
            y=train_y,
            batch_size=100,
            num_epochs=num_epochs,
            shuffle=shuffle,
            num_threads=5)
    
### The input function, as per the Tensorflow estimator's expectation
def test_input_fn(num_epochs=1, shuffle=False):
    return tf.estimator.inputs.pandas_input_fn(
            x=test_x,
            y=test_y,
            batch_size=100,
            num_epochs=num_epochs,
            shuffle=shuffle,
            num_threads=5)
    
### Train and evaluate the model    
def train(model_dir, train_steps):
    m = build_estimator(model_dir)
    return m.fit(input_fn=input_fn(), steps=train_steps)
        
    
### Run it
model_dir = './models'
train_steps = 10000 
trained_model = train(model_dir, train_steps)
results = trained_model.evaluate(input_fn=test_input_fn(), steps=None)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
