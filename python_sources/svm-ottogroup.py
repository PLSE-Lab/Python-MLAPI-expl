import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sampleSubmission.csv")
training_labels = LabelEncoder().fit_transform(train['target'])

print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

train_features = train.drop('target', axis=1)
train_features[train_features > 4] = 4

print(sample_submission.columns[4])

#model = LinearSVC().fit(train_features, training_labels)

#prediction = model.predict(test)
#print(prediction)
# create submission file

