import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

train_x = train.ix[:,1:]
train_y = train.label

# import sklearn.linear_model

# classifier = sklearn.linear_model.LogisticRegression(penalty='l2')
# classifier.fit(train_x, train_y)
# prediction = classifier.predict(test)

with open('prediction.csv', 'w') as f:
    f.write('ImageId,Label\n')
    for index in range(len(test)):
        f.write('%d,%d\n' % (index + 1, 0))