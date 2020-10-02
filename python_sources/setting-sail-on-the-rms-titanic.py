import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", index_col='PassengerId', dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", index_col='PassengerId', dtype={"Age": np.float64}, )


print("\n\nTop of the training data:")
print(train.head())


#y = train['Survived']
#del train['Survived']