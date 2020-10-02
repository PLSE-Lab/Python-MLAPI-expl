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
data=pd.read_csv('../input/train.csv')

rich_features = pd.concat([data[['Fare', 'Pclass', 'Age']],
                           pd.get_dummies(data['Sex'], prefix='Sex'),
                           pd.get_dummies(data['Embarked'], prefix='Embarked')],
                          axis=1)

rich_features_no_male = rich_features.drop('Sex_male', 1)
rich_features_final = rich_features_no_male.fillna(rich_features_no_male.dropna().median())
survived_column = data['Survived']
target = survived_column.values
data_test=pd.read_csv('../input/test.csv')

rich_features_test = pd.concat([data_test[['Fare', 'Pclass', 'Age']],
                           pd.get_dummies(data_test['Sex'], prefix='Sex'),
                           pd.get_dummies(data_test['Embarked'], prefix='Embarked')],
                          axis=1)

rich_features_no_male_test = rich_features_test.drop('Sex_male', 1)
rich_features_final_test = rich_features_no_male_test.fillna(rich_features_no_male_test.dropna().median())
#Non Linear Model Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,subsample=.8, max_features=.5)

#scores = cross_val_score(gb, rich_features_final, target, cv=5, n_jobs=4,scoring='accuracy')
#print("Gradient Boosted Trees CV scores:")
#print("min: {:.3f}, mean: {:.3f}, max: {:.3f}".format(scores.min(), scores.mean(), scores.max()))

gb.fit(rich_features_final, survived_column)


predsGB = gb.predict(rich_features_final_test)
my_submissionGB = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': predsGB})
my_submissionGB.to_csv(f'submissionGB.csv', index=False)