# considering 709 peolple survived in the
# Age wise priority
import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

train_data=train
train = train[train['Survived']==1]
train.sort_index(by="Age",inplace=True)
survivedindex=[i for i in train.index]
for i in train_data.index:
    if i in survivedindex[:708]:
        train_data['Survived'][i]=1
    else:
        train_data['Survived'][i]=0
#Any files you save will be available in the output tab below
train_data[['PassengerId','PassengerId']].to_csv('copy_of_the_training_data.csv', index=False)