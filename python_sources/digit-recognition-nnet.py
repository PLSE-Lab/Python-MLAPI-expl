import pandas as pd
from sklearn.neural_network import MLPClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

target = train["label"].values
features = train.copy()
features_test = test.copy()
features = features.drop("label",1).values
features_test = features_test.values

neurnet = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50), random_state=1)
my_net = neurnet.fit(features,target)
my_prediction = my_net.predict(features_test)

print(my_net.score(features,target))

digit_prediction = pd.DataFrame(my_prediction,columns=["label"])
digit_prediction.index += 1

digit_prediction.to_csv("digit_prediction_nnet.csv",index_label="imageId")