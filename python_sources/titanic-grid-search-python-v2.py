
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore', DeprecationWarning)




data = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")




survived_column = data['Survived']

target = survived_column.values

rich_features = pd.concat([data.get(['Fare', 'Pclass', 'Age']),
                           pd.get_dummies(data.Sex, prefix='Sex'),
                           pd.get_dummies(data.Embarked, prefix='Embarked')],
                          axis=1)

rich_features_no_male = rich_features.drop('Sex_male', 1)

rich_features_final = rich_features_no_male.fillna(rich_features_no_male.dropna().median())


rich_features_test = pd.concat([data_test.get(['Fare', 'Pclass', 'Age']),
                           pd.get_dummies(data_test.Sex, prefix='Sex'),
                           pd.get_dummies(data_test.Embarked, prefix='Embarked')],
                          axis=1)

rich_features_no_male_test = rich_features_test.drop('Sex_male', 1)

rich_features_final_test = rich_features_no_male_test.fillna(rich_features_no_male_test.dropna().median())



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.grid_search import GridSearchCV

gb = GradientBoostingClassifier(n_estimators=100, subsample=.8)

params = {
    'learning_rate': [0.05, 0.1, 0.5],
    'max_features': [0.5, 1],
    'max_depth': [3, 4, 5],
}
gs = GridSearchCV(gb, params, cv=5, scoring='roc_auc', n_jobs=4)
gs.fit(rich_features_final, target)


    
rich_gs_predictions = gs.predict(rich_features_final_test)
rich_gs_predictions = pd.DataFrame(rich_gs_predictions, columns=['Survived'])
test = pd.read_csv(os.path.join('../input', 'test.csv'))
rich_gs_predictions = pd.concat((test.iloc[:, 0], rich_gs_predictions), axis = 1)
rich_gs_predictions.to_csv('submission.csv', sep=",", index = False)









