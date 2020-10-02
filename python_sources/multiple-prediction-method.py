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
from sklearn import ensemble
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from scipy.stats.mstats import mode

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train['train_or_test'] = 'train'
test['train_or_test'] = 'test'
df = pd.concat([train,test], sort=False) # Combine the train and test data
del train, test

# Create new columns
df['HF1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
df['HF2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
df['HR1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
df['HR2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
df['FR1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
df['FR2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])
df['Adjusted_Elevation'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
df['Slope_Hydrology'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5
df.Slope_Hydrology = df.Slope_Hydrology.map(lambda x: 0 if np.isinf(x) else x)
df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points * df.Horizontal_Distance_To_Hydrology * df.Horizontal_Distance_To_Roadways) ** (1/3)
df['Mean_Fire_Hyd'] = (df.Horizontal_Distance_To_Fire_Points * df.Horizontal_Distance_To_Hydrology) ** (1/2)

# Separate back into train/test data
train = df[(df['train_or_test'] == 'train')]
test = df[(df['train_or_test'] == 'test')]
train.drop(['train_or_test'], axis=1, inplace=True)
test.drop(['train_or_test'], axis=1, inplace=True)

# Choose only the features we want for our algorithm
feature = [col for col in train.columns if col not in ['Cover_Type','Id']]
train_X = train[feature]
train_y = train['Cover_Type']
test_X = test[feature]
predictions = pd.DataFrame()

model_1 = ensemble.AdaBoostClassifier(ensemble.ExtraTreesClassifier(n_estimators=500), n_estimators=250, learning_rate=0.01, algorithm='SAMME')
model_1.fit(train_X, train_y)
predictions["model_1"] = model_1.predict(test_X)
print("Prediction 1 finished")

model_2 = ensemble.ExtraTreesClassifier(n_estimators=550)
model_2.fit(train_X, train['Cover_Type'])
predictions["model_2"] = model_2.predict(test_X)
print("Prediction 2 finished")

model_3 = XGBClassifier(max_depth=20, n_estimators=1000)
model_3.fit(train_X, train['Cover_Type'])
predictions["model_3"] = model_3.predict(test_X)
print("Prediction 3 finished")

model_4 = LGBMClassifier(n_estimators=2000, max_depth=15, verbose=-1)
model_4.fit(train_X, train['Cover_Type'])
predictions["model_4"] = model_4.predict(test_X)
print("Prediction 4 finished")

model_5 = ensemble.AdaBoostClassifier(ensemble.GradientBoostingClassifier(n_estimators=1000, max_depth=10), n_estimators=1000, learning_rate=0.01, algorithm="SAMME")
model_5.fit(train_X, train['Cover_Type'])
predictions["model_5"] = model_5.predict(test_X)
print("Prediction 5 finished")

model_6 = SGDClassifier(loss='hinge')
model_6.fit(train_X, train['Cover_Type'])
predictions["model_6"] = model_6.predict(test_X)
print("Prediction 6 finished")

model_7 = RandomForestClassifier(criterion = 'gini', max_depth= 60, max_features= 0.5, n_estimators= 200)
model_7.fit(train_X, train['Cover_Type'])
predictions["model_7"] = model_7.predict(test_X)
print("Prediction 7 finished")

print("HEAD:",predictions.head())
prediction = predictions.mode(axis=1)

sub = pd.DataFrame({"Id": test['Id'], "Cover_Type": prediction[0].astype('int').values})
print("Sending...")
sub.to_csv("submission.csv", index=False)
print("Done")