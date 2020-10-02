import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

predictions = {}

for passenger_index, passenger in test.iterrows():
    passenger_id = passenger['PassengerId']

    if passenger['Sex'] == 'female':
        predictions[passenger_id] = 1
    elif passenger['Age'] < 15 and passenger['SibSp'] < 3:
        predictions[passenger_id] = 1
    else:
        predictions[passenger_id] = 0

submission = pd.DataFrame({"PassengerId": list(predictions.keys()), "Survived": list(predictions.values())})
submission.to_csv("submission.csv", index=False)