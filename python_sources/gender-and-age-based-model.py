import numpy as np
import pandas as pd
import pylab as plt


train = pd.read_csv("../input/train.csv", dtype={"Class": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Class": np.float64}, )

submission_naive = pd.DataFrame({"PassengerId": 
    test['PassengerId'], "Survived": pd.Series(dtype='int32')})

submission_naive.Survived = 0

submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pd.Series(dtype='int32')})

# (5) Fill the Data for the survived column, all females live (1) all males die (0)
for i in range(len(test)):
    if test.Sex[i]=='femal':
        if test.Age[i]<40.0:
             submission.Survived[i]=1
        else:
            submission.Survived[i]=0
    else:
        submission.Survived[i]=0

submission_naive.to_csv("submission_naive.csv", index=False)
submission.to_csv("submission_gender.csv", index=False)