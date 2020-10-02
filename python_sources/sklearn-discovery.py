import pandas as pd
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

train_values = train.values
test_values = test.values
x_training,x_test,y_training,y_test = train_test_split(train_values[0::,1::],train_values[0::,0],test_size=0.2,random_state=0)

rdForestClass = RandomForestClassifier(n_estimators=500)
rdForestClass = rdForestClass.fit(x_training,y_training)
result = rdForestClass.predict(x_test)
print(accuracy_score(y_test, result))
print(result)

submission = rdForestClass.predict(test_values)

prediction_file = open("rdForestClass.csv", "w")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["ImageId", "Label"])
indexes = range(len(submission))
indexes = map(lambda x:x+1,indexes)
rows = zip(indexes,submission)
prediction_file_object.writerows(rows)
prediction_file.close()

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs