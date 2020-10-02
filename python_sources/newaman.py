# Any files you write to the current directory get shown as outputs
import re
import sys
# trainpath = (sys.argv[1])
# testpath = (sys.argv[2])
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
svm = SVC()


train_data = pd.read_csv("../input/train.csv")
test_data  = pd.read_csv("../input/test.csv")

# train_data = pd.read_csv("train.csv")
# test_data  = pd.read_csv("test.csv")


given_labels=train_data["label"]
train_data=train_data.drop("label",axis=1)
# print("aman")
principal_component_analysis = PCA(n_components=60, whiten=True)
# print("ayush")
principal_component_analysis.fit(train_data)
# print("seema")
train_data = principal_component_analysis.transform(train_data)
test_data = principal_component_analysis.transform(test_data)
svm.fit(train_data,given_labels)
Prediction = svm.predict(test_data)

Final_answer = pd.DataFrame(Prediction, columns=['Label'])
Final_answer.index.name = 'ImageId'
Final_answer.to_csv("output.csv")

