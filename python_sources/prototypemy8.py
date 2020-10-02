from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# create the training & test sets, skipping the header row with [1:]
data_file = pd.read_csv("../input/train.csv")
labels = data_file[[0]].values.ravel()
pixels = data_file.iloc[:,1:].values
test = pd.read_csv("../input/test.csv").values
print('labels')
print(labels)
print(labels[0])
print('labels length')
print(len(labels))
print('pixels')
print(pixels)
print('pixel length')
print(len(pixels))

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
##rf = RandomForestClassifier(n_estimators=100)
##rf.fit(train, target)
##pred = rf.predict(test)

##np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
