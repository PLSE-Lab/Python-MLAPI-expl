'''
We have used Tensorflow version 1.x
'''

#Importing
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

#Creating train dataframe
df = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')
df.dropna(inplace=True)
Y = df['y']
X = df.drop('y', axis=1)

#Train Test Split 30% ratio
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
'''
The x_test, y_test created from the training data will be used as cross-validation set
'''

#MinMaxScaling
mms = MinMaxScaler()
mms.fit(x_train)
x_train = pd.DataFrame(mms.transform(x_train), columns=x_train.columns, index=x_train.index)
x_test = pd.DataFrame(mms.transform(x_test), columns=x_test.columns, index=x_test.index)

#Creating feature column for tf and init DNNR
feat_col = [ tf.feature_column.numeric_column('x') ]
estimator = tf.estimator.DNNRegressor(hidden_units=[10, 10, 10], feature_columns=feat_col)

#Creating estimator input function
X_train = tf.estimator.inputs.pandas_input_fn(x_train, y_train, batch_size=10, num_epochs=1000, shuffle=True)

#Train
estimator.train(X_train, steps=10000)

#Test
X_test = tf.estimator.inputs.pandas_input_fn(x_test, num_epochs=1, shuffle=False)

#MAE
predictions = [ x['predictions'][0] for x in list(estimator.predict(X_test)) ]
print('Cross-Validations MAE score: ' + str(mean_absolute_error(predictions, y_test)))

#Actual Test file predictions
main_test_df = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')
main_test_X = main_test_df.drop('y', axis=1)
main_test_Y = main_test_df['y']
main_test_X = pd.DataFrame(mms.transform(main_test_X), columns=main_test_X.columns, index=main_test_X.index)
main_test_X_input = tf.estimator.inputs.pandas_input_fn(main_test_X, num_epochs=1, shuffle=False)
main_test_predictions = [ x['predictions'][0] for x in list(estimator.predict(main_test_X_input)) ]
print('Testset MAE score: ' + str(mean_absolute_error(main_test_predictions, main_test_Y)))

#TestSet plotting
plt.scatter(main_test_X, main_test_Y)
plt.plot(main_test_X, main_test_predictions, 'r')
plt.show()