# Import libraries.
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
# Load data.
print('Loading data')
t1 = time.time()
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
t2 = time.time()
print('Done loading data ({0:.3f}sec)'.format(t2-t1))
# Let's have a look at the data.
train.head(3)
train['label'].value_counts()
train_x = train.values[:, 1:].astype(float)
train_y = train.values[:, 0]

print('Optimizing the number of trees')
n_trees = [25, 50 ,100]
cvscore_mn = []
cvscore_std = []
for n_tree in n_trees:
    print("the number of trees : {0}".format(n_tree))
    t1 = time.time()
    model = RandomForestClassifier(n_tree)
    score = cross_val_score(model, train_x, train_y)
    cvscore_mn.append(np.mean(score))
    cvscore_std.append(np.std(score))
    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(n_tree, t2-t1))
sc = np.array(cvscore_mn)
std = np.array(cvscore_std)
print('Cross validation Score mean : ', sc)
print('Standard deviation          : ', std)
# Let's plot the result of cross-validation.
plt.figure(figsize=(4,4))
plt.plot(n_trees, sc)
plt.plot(n_trees, sc + std, 'b--')
plt.plot(n_trees, sc - std, 'b--')
plt.ylabel('CV score')
plt.xlabel('number of trees')
plt.show()
model = RandomForestClassifier(100)
model.fit(train_x, train_y)
model.predict(test)