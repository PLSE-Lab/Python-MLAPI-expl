import numpy as np
import pandas as pd
from sklearn import tree
from math import isnan

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
def prepare(data):
    out = []
    for sample in data:
        out.append([])
        for i in range(len(sample)):
            if i in [2, 9, 10]:
                continue
            elif i == 3:
                out[-1].append({"male": 0, "female": 1}[sample[i]])
            elif i == 7:
                out[-1].append(ord(sample[i][0]))
            else:
                if isnan(sample[i]): sample[i] = -1
                out[-1].append(sample[i])
    return np.array(out)
data = list(train.values)
Y = np.array([sample[1] for sample in data])
for i in range(len(data)):
    data[i] = list(data[i])
    del data[i][1]
X = prepare(data)
T = prepare(list(test.values))

model = tree.DecisionTreeClassifier()
model.fit(X, Y)
O = model.predict(T)
with open("submission.csv", "w") as file:
    for i in range(len(O)):
        file.write("%d,%d\n" % (T[i][0], O[i]))
with open("submission.csv") as file:
    print(file.read())
quit()