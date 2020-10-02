import numpy as np
import pandas as pd

#(1) Import the Data into the Script

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# (2) Create the submission file with passengerIDs from the test file
submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pd.Series(dtype='int32')})

# (3) Create the Data for the survived column, all females live all males die

for i, x in enumerate(test['Sex']):
    if x == "female":
        submission.Survived.set_value(i, 1)
    else:
        submission.Survived.set_value(i, 0)

# (4) Add the survived Column to the submission file


print(submission)

submission.to_csv("submission.csv", index=False)
