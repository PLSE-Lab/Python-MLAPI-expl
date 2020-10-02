import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
y = train_data[["label"]]
X_train = train_data.drop(["label"], axis=1)

X_train_stan = StandardScaler().fit_transform(X_train)
X_test_stan = StandardScaler().fit_transform(test_data)
test_size = len(test_data.count(axis = 1)); 
	
model = SVC(C=1, kernel='linear')
model.fit(X_train_stan, y.iloc[:,0])    
predictions = model.predict(X_test_stan)
counter = np.arange(1,28001)
c1 = pd.DataFrame({'ImageId': counter})
c2 = pd.DataFrame({'Label':predictions})
res = pd.concat([c1, c2], axis = 1)
res.to_csv('output.csv', index = False)
