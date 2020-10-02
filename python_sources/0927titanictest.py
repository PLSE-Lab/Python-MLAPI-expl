import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import cross_validation

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

train['Sex'] = train['Sex'].replace(['male','female'],[1,0])
test['Sex'] = test['Sex'].replace(['male','female'],[1,0])

print (len(train))
print (len(train.dropna()))
'''
#Construct X and y
X = np.asarray(train.drop(['PassengerId','Survived','Name','Ticket','Cabin','Embarked'], 1))
y = np.asarray(train['Survived'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

for min_samples_split in [1, 2, 3]:
    for min_samples_leaf in [1, 2, 3]:
        for max_leaf_nodes in [10, 25, 80]:
            clf = ensemble.RandomForestClassifier(min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf,
                                                max_leaf_nodes=max_leaf_nodes, n_jobs=-1)
            clf = clf.fit(X_train,y_train)

            score = clf.score(X_test, y_test)
            print ('-------------------')
            print ('min_samples_split: ', min_samples_split)
            print ('min_samples_leaf: ', min_samples_leaf)
            print ('max_leaf_nodes: ', max_leaf_nodes)
            print (score)

'''