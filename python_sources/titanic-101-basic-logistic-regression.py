
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


keep_cols = ["Pclass","Age", "SibSp","Parch","Fare"]
train_small = train[keep_cols+ ['Survived']]
train_small= train_small.dropna()

reg = LogisticRegression().fit(train_small[keep_cols], train_small[['Survived']])

test_small = test[keep_cols+['PassengerId']]
test_small= test_small.dropna()
test_small['Survived'] = reg.predict(test_small[keep_cols])

result = pd.merge(test[['PassengerId']], test_small[['PassengerId', "Survived"]], how="left")

result.loc[result['Survived'].isna(),'Survived'] = 0

result.to_csv("submission.csv",index=False)