from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import csv as csv
data = pd.read_csv('../input/train.csv', header = 0)
train = data.values
train_long = train[0::, 1::]
lab = train[0::,0]
test = pd.read_csv('../input/test.csv', header = 0)
test_arr = test.values
svc = SVC()

pca = PCA(n_components = 50, whiten = True)
pca.fit(train_long)
feach = pca.transform(train_long)
svc.fit(feach, lab)
test_data = pca.transform(test_arr)
pred = svc.predict(test_data)

print('ImageId,', 'Label')
i = 0
for val in pred:
    i = i + 1
    print(i,',', val)