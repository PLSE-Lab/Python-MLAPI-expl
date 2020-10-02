import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

# Train Random Forest Models
label = train["label"]
training = train.filter(regex=("pixel.*"))
rf = RandomForestClassifier()
rf.fit(training, label)
predict = rf.predict(test)

# Print to file
print("Prediction set has {0[0]} items".format(predict.shape))
submit = pd.DataFrame(predict.T, columns=["Label"], index=range(1, len(predict)+1))
submit.to_csv("submission.csv", index_label="ImageId")