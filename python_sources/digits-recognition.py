import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

num_train_samples = 10000

Y_train = train.values[0:num_train_samples, 0]
X_train = train.values[0:num_train_samples, 1:].astype(float)

X_test = test.values[:].astype(float)

# Step 1: Import the model
from sklearn.linear_model import LogisticRegression

# Step 2: Instantiate the model
logreg = LogisticRegression(random_state=5)

# Step 3: Fit the model
logreg.fit(X_train, Y_train)

# # Step 4: Predict
y_pred = logreg.predict(X_test)

print(y_pred)