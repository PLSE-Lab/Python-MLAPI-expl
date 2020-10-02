import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


print(train.head())

feature_col=['Pclass','Parch']

# X = train.loc[:,feature_col]
 
# y=train[["survived"]]