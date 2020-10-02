import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv").values

train_x = train.iloc[:,1:].values.astype(np.float)
train_x = train_x / 255
train_y = train.iloc[:,0].values

type_count = np.unique(train_y).shape[0]
flat_train_y = np.zeros((train_y.shape[0], type_count)).astype(np.uint8)
flat_train_y.flat[train_y+np.arange(train_y.shape[0])*type_count] = 1
         
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
train_x_pca = pca.fit_transform(train_x)
                  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x_pca, train_y)

print('Waiting...')
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
clf = xgb.XGBClassifier(max_depth=7, learning_rate=0.2, n_estimators=200)
clf.fit(X_train, y_train)
print(clf)
print(accuracy_score(y_test, clf.predict(X_test)))

print('Waiting...')
test_x_pca = pca.fit_transform(test)
pred = clf.predict(test_x_pca)

result = pd.DataFrame(columns=['ImageId', 'Label'])
result.ImageId = np.arange(test.shape[0]) + 1
result.Label = pred
result.to_csv('result.csv', index=False)