import os
import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier



def extract_extra_features(data_set):

   # data inferred from passenger names
   names = data_set["Name"]
   data_set["Surname"] = names.str.extract('^(\w+)')

   unique_surnames = data_set["Surname"].value_counts()
   data_set["FamilySize"] = data_set["Surname"].map(lambda x: unique_surnames[x])

   data_set["NameTitle"] = names.apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

   data_set["NameLen"] = names.apply(lambda x: len(x))

   # data inferred from info about tickets
   data_set["TicketLen"] = data_set['Ticket'].apply(lambda x: len(x))

   data_set["DeckId"] = data_set['Ticket'].apply(lambda x: str(x)[0])
   data_set['DeckId'] = np.where((data_set['DeckId']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), 
                                 data_set['DeckId'],
                                 np.where((data_set['DeckId']).isin(['W', '4', '7', '6', 'L', '5', '8']), 
                                          'Low_ticket', 
                                          'Other_ticket')
                                 )

   # data inferred from cabin numbers
   data_set["CabinLetter"] = data_set["Cabin"].apply(lambda x: str(x)[0])

   # passenger age imputation
   age_grouping = data_set.groupby(["NameTitle", "Pclass"])["Age"]
   data_set["Age"] = age_grouping.transform(lambda x: x.fillna(x.mean()))
   data_set["Age"].fillna(data_set["Age"].mean(), inplace=True)

   # imputing the missing embarkments
   data_set["Embarked"] = data_set["Embarked"].fillna('S')

   # imputing the missing fares
   data_set['Fare'].fillna(data_set['Fare'].mean(), inplace = True)

   # drop obsolete columns
   del data_set["PassengerId"]
   del data_set["Name"]
   del data_set["Surname"]
   del data_set["SibSp"]
   del data_set["Parch"]
   del data_set["Ticket"]
   del data_set["Cabin"]

def convert_columns_to_OHV(data_set, columns):

   for column in columns:

      data_set[column] = data_set[column].apply(lambda x: str(x))

      good_cols = [column+'_'+i for i in data_set[column].unique() if i in data_set[column].unique()]
      data_set = pd.concat((data_set, pd.get_dummies(data_set[column], prefix = column)[good_cols]), axis = 1)

      del data_set[column]

   return data_set

def split_dataset(data_set, train_set_length):

   train = data_set.iloc[:train_set_length, :]
   test = data_set.iloc[train_set_length:, :]

   return train, test

#
# main
#
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

all_data = pd.concat([train, test])
extract_extra_features(all_data)
all_data = convert_columns_to_OHV(all_data, columns = ['Pclass', 'Sex', 'Embarked', 'DeckId', 'CabinLetter', 'NameTitle', 'FamilySize']) 

train, test = split_dataset(all_data, len(train))

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

y = train["Survived"]
x = train[train.columns.difference(["Survived"])]

rf.fit(x, y)


# test the model

test_x = test[test.columns.difference(["Survived"])]

survival_predictions = rf.predict(test_x).astype(int)
survival_predictions_df = pd.DataFrame(survival_predictions, columns=['Survived'])

original_test_data = pd.read_csv("../input/test.csv")
predictions = pd.concat((original_test_data["PassengerId"], survival_predictions_df), axis = 1)
predictions.to_csv('submission.csv', sep=",", index = False)
