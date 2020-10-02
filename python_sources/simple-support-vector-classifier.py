# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


train = pd.read_csv("../input/train.csv")
test = (pd.read_csv("../input/test.csv").values).astype('float32')
train_label = train['label']

train_image = (train.ix[:,1:].values).astype('float32')

train_image = train_image / 255
test = test / 255

from sklearn import decomposition
pca = decomposition.PCA(n_components = 40)
pca.fit(train_image)
train_pca = np.array(pca.transform(train_image))
test_pca = np.array(pca.transform(test))

from sklearn.svm import LinearSVC, SVC
clf = SVC(C=10,
          kernel='rbf',
          class_weight='balanced',
          random_state=50
         )
clf.fit(train_pca, train_label)

predictions = clf.predict(test_pca)

result = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
result.to_csv("output.csv", index=False, header=True)