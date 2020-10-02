import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd

print('Read training data...')
train_data = pd.read_csv('../input/train.csv')
train_label = np.array(train_data['label'])
train_data = np.array(train_data.iloc[:,1:])
print('Loaded ' + str(len(train_label)))

print('Reduction...')
COMPONENT_NUM = 35
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train_data)
train_data = pca.transform(train_data)

print('Train SVM...')
svc = SVC()
svc.fit(train_data, train_label)

print('Read testing data...')
test_data = pd.read_csv('../input/test.csv')
test_data = np.array(test_data)
print('Loaded ' + str(len(test_data)))

print('Predicting...')
test_data = pca.transform(test_data)
predict = svc.predict(test_data)

print('Saving...')
with open('predict.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')

