import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# The competition datafiles are in the directory ../input
# Read competition data files:
train_data = pd.read_csv("../input/train.csv")
test_data  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train_data.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test_data.shape))
# Any files you write to the current directory get shown as outputs


print("Partitioning training data into cross-validation and training...")

cross_validation_data = train_data.sample(frac = 0.2)
train_data_x_cv = train_data.drop(cross_validation_data.index)

print("Length of original training set "+str(len(train_data)))
print("Length of cross-validation set "+str(len(cross_validation_data)))
print("Length of final training set "+str(len(train_data_x_cv)))

x_variables = list(train_data.columns)
x_variables.remove('label')
y_variable = 'label'

Y = train_data_x_cv[y_variable]
X = train_data_x_cv[x_variables]

print("Training a logistic regression model ... ")
logreg = linear_model.LogisticRegression(penalty = 'l2',C = 1,fit_intercept = True,solver = 'sag',max_iter = 300)
logreg.fit(X,Y)
predicted = logreg.predict(cross_validation_data[x_variables])

print(np.corrcoef(cross_validation_data[y_variable],predicted))

predicted_test_labels = logreg.predict(test_data[x_variables])
data_dict = {'ImageId':range(1,len(predicted_test_labels)+1),'Label':predicted_test_labels}
test_labels = pd.DataFrame(data_dict)
test_labels.to_csv(r'Output.csv',index = False)


