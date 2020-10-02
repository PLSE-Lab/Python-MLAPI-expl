import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

_Outcome = "Outcome"
_ID = "ID"

original = pd.read_csv("../input/diabetes.csv")
y = original[[_Outcome]]
x = original.drop([_Outcome], axis=1)

train_X, test_X, train_Y, test_Y = train_test_split(x, y, random_state=0)

train = train_X.join(train_Y)
validate = test_X.join(test_Y)
test = validate.drop([_Outcome], axis=1)

train.to_csv("train.csv", index_label=_ID)
test.to_csv("test.csv", index_label=_ID)
validate.to_csv("validate.csv", index_label=_ID)
