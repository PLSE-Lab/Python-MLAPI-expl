import pandas as pd
from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

features = train.columns[1:]
y = train[train.columns[0]]
train = train[features]

rbm = BernoulliRBM()
logistic = linear_model.LogisticRegression()

rbm.learning_rate = 0.06
rbm.n_components = 100
rbm.n_iter = 20

logistic.C = 6000.0

clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
clf.fit(train, y)

preds = clf.predict(test)

output = pd.DataFrame(index=range(len(preds)+1)[1:])
output['Label'] = preds
output.to_csv('output.csv', index_label='ImageId')
