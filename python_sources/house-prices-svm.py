import numpy as np
import pandas as pd
from sklearn import svm


## Read in data.
from os import listdir
data = {filename.split('.')[0]: pd.read_csv(f'../input/{filename}') for filename in listdir("../input") if '.csv' in filename}


## Create train and test set.
DATA_COLUMNS = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
                'FullBath', 'GarageArea', 'YrSold']
TARGET_COLUMNS = ['SalePrice']

train_data = data['train'][DATA_COLUMNS]
train_target = data['train'][TARGET_COLUMNS]

test_data = data['test'][DATA_COLUMNS]

print(test_data.columns)

## Clean up data.
for column in DATA_COLUMNS:
    average = 0 #np.mean(train_data[column])

    train_data[column].fillna(average, inplace=True)
    test_data[column].fillna(average, inplace=True)


## Fit model.
model = svm.SVC()
model.fit(train_data, train_target)


## Generate submission.
test_guesses = model.predict(test_data)

output = pd.DataFrame()
output['Id'] = data['test']['Id']
output['SalePrice'] = test_guesses

output.to_csv('output.csv', index=False)
