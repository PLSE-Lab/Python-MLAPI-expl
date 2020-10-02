import numpy as np
import pandas as pd

def data(arr):
	x = []
	for str in arr:
		if str != str:
			str = 0.
		x.append(str)
	return x

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#sol = pd.read_csv("../input/gender_submission.csv")

x_train = np.array(data(train['Age'])).reshape(-1, 1)
y_train = np.array(train['Survived'])
x_test = np.array(data(test['Age'])).reshape(-1, 1)
#y_test = np.array(sol['Survived'])

from sklearn.cluster import KMeans
clf = KMeans(n_clusters=1).fit(x_train, y_train)
pred = clf.predict(x_test)

df1 = pd.DataFrame(test['PassengerId'])
df2 = pd.DataFrame(pred)
my_df = pd.concat([df1, df2], axis=1)
my_df.to_csv('solution.csv', index=False, header=["PassengerId", "Survived"])
