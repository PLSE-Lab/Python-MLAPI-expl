import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64, "Pclass": np.int64} )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64, "Pclass": np.int64} )

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

# My code starts from here:
def sex_str_to_num(sex_entry):
    if sex_entry == 'male':
        return -1
    else:
        return 1

# Numeric
train['Sex_num'] = train['Sex']
test['Sex_num'] = test['Sex']
train['Sex_num'] = train['Sex_num'].apply(lambda x: sex_str_to_num(x))
test['Sex_num'] = test['Sex_num'].apply(lambda x: sex_str_to_num(x))

# cleaning data
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Fare'] = train['Fare'].fillna(train['Fare'].mean())

test['Age'] = test['Age'].fillna(train['Age'].mean())
test['Fare'] = test['Fare'].fillna(train['Fare'].mean())

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

# Feature Normalization
#train['Age_norm'] = (train['Age'] - (train.Age.mean()))/train.Age.std()
train['Fare_norm'] = (train['Fare'] - train.Fare.mean())/train.Fare.std()
test['Fare_norm'] = (test['Fare'] - train.Fare.mean())/train.Fare.std()

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
predictors = ['Pclass', 'Sex_num', 'Fare_norm']
alg = LogisticRegression()
model = alg.fit(train[predictors],train['Survived'])
print(model)

# Make predictions
results = model.predict(test[predictors])
df_results = pd.DataFrame({'PassengerId': test['PassengerId'],\
                            'Survived': results})
df_results.to_csv('submit_001.csv', index=False)




