# Tutorial 1: Gender Based Model (0.76555) - by Myles O'Neill
import numpy as np
import pandas as pd
import pylab as plt

# (1) Import the Data into the Script
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# (2) Create the submission file with passengerIDs from the test file
submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pd.Series(dtype='int32')})

# (3) Fill the Data for the survived column, all females live (1) all males die (0)
for i in range(len(test)):
    if test.Sex[i] == 'female':
        if test.Age[i] < 25.0: 
            submission.Survived[i] = 1
        else:
            submission.Survived[i] = 0
    else:
        submission.Survived[i] = 0

# submission.Survived = [1 if x == 'female' else 0 for x in test['Sex']]

# (4) Create final submission file
submission.to_csv("submission.csv", index=False)

