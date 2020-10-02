from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

x=train.values[:, 1:].astype(float)
y=train.values[:, 0]

clf = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(1000,), random_state=1)
clf.fit(x,y)

MLPClassifier(learning_rate_init=0.015, max_iter=50, momentum=0.99)

result=clf.predict(test)

out_file = open("predictions.csv", "w")
out_file.write("ImageId,Label\n")
for i in range(len(result)):
    out_file.write(str(i+1) + "," + str(int(result[i])) + "\n")
out_file.close()