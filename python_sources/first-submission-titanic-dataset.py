import numpy as np
import pandas as pd
import matplotlib as plot
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

#print("\n\nSummary statistics of training data")
print(train.describe())

train_labels = train['Survived']
train_features = train.drop('Survived', axis = 1)
print(test.describe())
#test_labels = test['Survived']
#test_features = test.drop('Survived', 1)

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

clf = SVC()
#clf.fit(train_features, train_labels)
#clf.predict(test)
