import pandas
import sklearn
from sklearn import ensemble
from sklearn import cross_validation



# Importing the data
input_data = pandas.read_csv("../input/train.csv")
test_data = pandas.read_csv("../input/test.csv")


# Splitting the training data to validate model
train_data, validation_data = cross_validation.train_test_split(input_data,
                                                                       train_size=0.80,
                                                                       random_state=20161204)


# Fitting a Random Forest model to the training data
RF = ensemble.RandomForestClassifier()
X = train_data.iloc[:, range(1, 785)]
y = train_data.iloc[:, 0]
RF_model = RF.fit(X=X, y=y)

## Validation accuracy
X = validation_data.iloc[:, range(1, 785)]
y = validation_data.iloc[:, 0]
print("Model score on train data ====> {0:f}%".format(RF_model.score(X, y)*100))

# Predictions on the Random Forest
X = test_data.iloc[:, range(0, 784)]
pred_test_data = RF_model.predict(X)
print(pred_test_data)
