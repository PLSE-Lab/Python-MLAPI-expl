import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv").values

train_x = train.iloc[:,1:].values.astype(np.float)
train_x = train_x / 255
train_y = train.iloc[:,0].values

from skimage.transform import rotate
ARTIFICIAL_DATA_SIZE = 2500
n = len(train_y)
rnd_indexes = np.random.randint(0, n, size=ARTIFICIAL_DATA_SIZE)
train_x = np.vstack((train_x, np.zeros((ARTIFICIAL_DATA_SIZE, train_x.shape[1]))))
train_y1 = np.zeros(ARTIFICIAL_DATA_SIZE+n)
train_y1[:n] = train_y
train_y = train_y1.astype(np.int)
k = n
for ind in rnd_indexes:
    img = train_x[ind, :].copy().reshape(28,28)
    angle = np.random.randint(-20,20)
    img_rot = rotate(img, angle)
    train_x[k, :] = img_rot.reshape(1,-1)
    train_y[k] = train_y[ind]
    k = k + 1

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
train_x = scale(train_x)
pca = PCA(n_components=0.8, whiten=False)
pca.fit(train_x)
train_x = pca.transform(train_x)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1, random_state=77)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
def get_model(estimator, parameters, X_train, y_train, scoring):
    model = GridSearchCV(estimator, param_grid=parameters, scoring=scoring)
    model.fit(X_train, y_train)
    return model.best_estimator_
clf = SVC()
scorer = make_scorer(accuracy_score, greater_is_better=True)
parameters = {
              'C': [2.0]
              }
clf_best = get_model(clf, parameters, X_train, y_train, scorer)
print(clf_best)
print(clf_best.score(X_test, y_test))

test = scale(test)
test_x = pca.transform(test)
test_pred = clf_best.predict(test_x)
result = pd.DataFrame(columns=['ImageId', 'Label'])
result.ImageId = np.arange(1,test.shape[0]+1)
result.Label = test_pred
result.to_csv('result.csv', index=False)