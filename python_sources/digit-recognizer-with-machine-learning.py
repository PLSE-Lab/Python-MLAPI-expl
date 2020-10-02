# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def load_data(location):
	if(location == 1): # Desktop
		path_train = "D:/DataScience/Kaggle/HousePrice/Data/train.csv"
		path_test = "D:/DataScience/Kaggle/HousePrice/Data/test.csv"
	elif(location == 2): # Laptop
		path_train = "C:/Users/User/Documents/GitHub/Digit/Data/train.csv"
		path_test = "C:/Users/User/Documents/GitHub/Digit/Data/test.csv"
	elif(location == 3): # Kaggle Kernel
		path_train = "../input/train.csv"
		path_test =  "../input/test.csv"
	else:
		print("No location found")
	return pd.read_csv(path_train), pd.read_csv(path_test)
	
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
size = 1000
#size = 30000
# Load data set	
data, data_test = load_data(3)

X, y = data.drop("label", axis=1), data['label']

X_train, X_test, y_train, y_test = X.loc[:size], X.loc[size:size*2], y.loc[:size], y.loc[size:size*2]
#X_train, X_test, y_train, y_test = X.loc[:size], X.loc[size:], y.loc[:size], y.loc[size:]

some_digit = X.loc[5]


some_digit_image = some_digit.values.reshape(28,28)
y.loc[5]


#forest_clf = RandomForestClassifier()
#forest_clf.fit(X_train, y_train)
#forest_clf.predict([some_digit])
#forest_clf.predict_proba([some_digit])
#print(forest_clf.predict_proba([some_digit]))

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_clf.predict(X.loc[3].values.reshape(1,-1))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(knn_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
print(cross_val_score(knn_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

row_size = data_test.shape[0]
ids = [i for i in range(1,row_size+1)]
#ids = df_test['Id']		
predictions = knn_clf.predict(data_test) # Lasso performed better and I modify here.

output = pd.DataFrame({ 'ImageId' : ids, 'Label': predictions })
output.to_csv('digit_knn_youngseok.csv', index = False)
output.head()

