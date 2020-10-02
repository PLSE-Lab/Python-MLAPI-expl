import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Write to the log:

target = train["label"].values
features_one =(train.transpose())[1:].transpose().values

print(target)
print(features_one)

my_tree_one = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_tree_one = my_tree_one.fit(features_one, target)

test_features_one = test.values

my_prediction = my_tree_one.predict(test_features_one)
print(my_prediction)

my_solution = pd.DataFrame(my_prediction, columns = ["Label"])
my_solution.index += 1
my_solution.to_csv("my_solution_one.csv", index_label = ["ImageId"])
# Any files you write to the current directory get shown as outputs