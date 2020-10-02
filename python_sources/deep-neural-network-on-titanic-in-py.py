import numpy as np
import tensorflow as tf
import pandas as pd

# Data sets
TRAINING = "../input/train.csv"
TEST = "../input/test.csv"


# Load datasets.
training_set = pd.read_csv(TRAINING, sep=',',header=0)
prediction_set = pd.read_csv(TEST, sep=',',header=0)

# Clean up datasets
training_set["Sex"] = training_set["Sex"].astype('category').cat.codes
training_set["Embarked"] = training_set["Embarked"].astype('category').cat.codes
prediction_set["Sex"] = prediction_set["Sex"].astype('category').cat.codes
prediction_set["Embarked"] = prediction_set["Embarked"].astype('category').cat.codes
 
training_set=training_set.interpolate()
prediction_set=prediction_set.interpolate()
 
# Define the training inputs
x = training_set[["Pclass","Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
y = training_set.Survived.values

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=7)]
# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 10, 10],
                                            n_classes=10,
                                            model_dir="tmp DNN")
# Define the training inputs
def get_train_inputs():
  x1 = tf.constant(x)
  y1 = tf.constant(y)
  return x1, y1

# Fit model.
classifier.fit(input_fn=get_train_inputs, steps=2000)

# Classify new samples.
new_samples = prediction_set[["Pclass","Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
pred = np.array(list(classifier.predict(new_samples))).reshape((418,1))
ident = prediction_set.PassengerId.values.reshape((1,418))
identPred = np.concatenate((ident.T, pred), axis=1)
identPredDf = pd.DataFrame(identPred, columns=['PassengerId', 'Survived'])
identPredDf.to_csv('PassengerId Survived Py 3.csv', index=False)
print("Predictions in ident label: \n{}".format(identPredDf))

