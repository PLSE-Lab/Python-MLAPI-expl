import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
im_train = np.loadtxt('../input/train.csv', delimiter=',', dtype=np.uint8, skiprows=1)
im_test = np.loadtxt('../input/test.csv', delimiter=',', dtype=np.uint8, skiprows=1)
train_labels = im_train[:, 0]
im_train = im_train[:, 1:]
classifier = KNeighborsClassifier(n_neighbors=5).fit(im_train, train_labels)
pd.DataFrame({"ImageId": np.arange(1, len(im_test)+1), "Label": classifier.predict(im_test)}).to_csv('submission.csv', index=False)