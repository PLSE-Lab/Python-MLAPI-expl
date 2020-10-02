# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

features = ['Pclass', 'Age', 'Fare']
features_str = ['Sex'] # will be converted


plt.figure(figsize=(7,4))
fig, axes = plt.subplots(nrows=1, ncols=3)
# plots a bar graph of those who surived vs those who did not.               
#plt.title("Sobreviventes sem acompanhantes")    
train_df.Survived[train_df.Parch==0].value_counts().plot(kind='bar', ax=axes[0], title="sem acomp.")

#plt.title("Sobreviventes com acompanhantes")    
train_df.Survived[train_df.Parch>0].value_counts().plot(kind='bar' , ax=axes[1],  title="com acomp.")
train_df.withpart = train_df.Parch>0
train_df.withpart[train_df.Survived==1].value_counts().plot(kind='bar' , ax=axes[2],  title="sobre. com/sem parc.")




def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if str(big_string).find(substring) != -1:
            return substring
    #print (big_string)
    return np.nan

title_list= ["Mr", "Ms", "Miss", "Master", "Rev", "Dr", "Mlle", "Col", "Countess", "Capt.", "Don", "Major", "Jonkheer", "Mme"]

train_df['Title'] = train_df['Name'].map(lambda x: substrings_in_string(x, title_list))
test_df['Title'] = test_df['Name'].map(lambda x: substrings_in_string(x, title_list))

features_str.append('Title')

#Turning cabin number into Deck
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
train_df['Deck']=train_df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
test_df['Deck']=test_df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

#features_str.append('Deck')

# check if there are still null
print("---->", train_df.Title.value_counts(dropna=False),  train_df.Name[train_df.Title.isnull()], train_df.info())





features.extend(features_str)

joined_df = train_df[features].append(test_df[features])
joined_df.dropna()
test_df.dropna()
print(joined_df.tail())
# convert str features to Number 
le = LabelEncoder()
for f in features_str:
    
    #print(f,"--->", joined_df[f], le.fit_transform(joined_df[f]))
    joined_df[f] = le.fit_transform(joined_df[f].fillna(0))
    
#joined_df['Sex'] = joined_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


train_X = joined_df[0:train_df.shape[0]].as_matrix()
test_X = joined_df[train_df.shape[0]::].as_matrix()

train_y = train_df['Survived']

cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}

gbm = xgb.XGBClassifier(**ind_params).fit(train_X, train_y)


optimized_GBM = GridSearchCV(gbm, cv_params, scoring = 'accuracy', cv = 5, n_jobs = -1) 

optimized_GBM.fit(train_X, train_y)


predictions_opt = optimized_GBM.predict(test_X)
subfile = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions_opt })
subfile.to_csv("predictions_opt.csv", index=False)