import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

train_path = '/kaggle/input/cdal-competition-2019-fall/train.csv'
valid_path = '/kaggle/input/cdal-competition-2019-fall/valid.csv'
test_path = '/kaggle/input/cdal-competition-2019-fall/test.csv'
# train_path = 'train.csv'
# valid_path = 'valid.csv'
# test_path = 'test.csv'
train = pd.read_csv(train_path)
valid = pd.read_csv(valid_path)
test = pd.read_csv(test_path)

submission = pd.DataFrame(columns=['id', 'rating'])
submission['id'] = test['id']

select_columns = ['effectiveness', 'sideEffects', 'rating']
select_columns_test = ['effectiveness', 'sideEffects']
train = train[select_columns]
valid = valid[select_columns]
test = test[select_columns_test]

train = pd.get_dummies(train, columns=['effectiveness',  'sideEffects'])
valid = pd.get_dummies(valid, columns=['effectiveness', 'sideEffects'])
test = pd.get_dummies(test, columns=['effectiveness', 'sideEffects'])

print('train shape: ', train.shape)

x_train = train.loc[:, train.columns != 'rating']
x_valid = valid.loc[:, valid.columns != 'rating']
y_train = train.loc[:, train.columns == 'rating']
y_valid = valid.loc[:, valid.columns == 'rating']

model = LinearRegression()
model.fit(x_train, y_train)
y_predict_lr = model.predict(x_valid)

regression_model_mse = mean_squared_error(y_predict_lr, y_valid)

print('lr valid rmse', math.sqrt(regression_model_mse))
print('lr valid score', model.score(x_valid, y_valid))

y_predict_lr_test = model.predict(test)

y_predict_lr_test = pd.DataFrame(y_predict_lr_test)
y_predict_lr_test = y_predict_lr_test.apply(lambda x: 1 if x.item() < 1 else x.item(), axis=1)

submission['rating'] = y_predict_lr_test
submission.to_csv('submission.csv', index=False)


