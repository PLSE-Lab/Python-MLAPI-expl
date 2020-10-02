import pandas as pd

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# only keep column PassengerId and Gender
submission_data = test_data.copy()[['PassengerId']]

# if Sex == female, set Survived = int(True)
submission_data['Survived'] = (test_data['Sex'] == 'female').astype(int)

# write submission
submission_data.to_csv('submission.csv', index=False)