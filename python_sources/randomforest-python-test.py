import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
labeled = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
# print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
# print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

features = labeled.columns[1:]

labeled['is_train'] = np.random.uniform(0, 1, len(labeled)) <= .75

train = labeled[labeled['is_train'] == True]
cv = labeled[labeled['is_train'] == False]

y = train[train.columns[0]]
y_cv = cv[cv.columns[0]]

best_model = None
best_model_acc = -1

for i in range(20):
    clf = RandomForestClassifier()
    clf.fit(train[features], y)
    cv_preds = clf.predict(cv[features])
    accuracy = sum(cv_preds == y_cv) * 1.0 / len(y_cv)
    print(accuracy)
    if accuracy > best_model_acc:
        best_model_acc = accuracy
        best_model = clf

preds = best_model.predict(test)
output = pd.DataFrame(index=range(len(preds)+1)[1:])
output['Label'] = preds
output.to_csv('output.csv', index_label='ImageId')

