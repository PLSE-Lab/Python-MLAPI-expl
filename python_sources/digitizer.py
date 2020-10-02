# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier as KNC
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

data = data.ix[:10000]

subject = data.drop('label', axis=1)
label = data['label']

knn = KNC(n_neighbors=4, weights='distance')
knn.fit(subject, label)
# Any results you write to the current directory are saved as output.
preds = knn.predict(test)

result = pd.DataFrame()
result['ImageId'] = range(1, len(preds)+1)
result['Label'] = preds

print(result)
result.to_csv('results.csv', index=False)