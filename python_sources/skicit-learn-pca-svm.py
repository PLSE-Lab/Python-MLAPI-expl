# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import svm
#%matplotlib inline
from sklearn.decomposition import PCA

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
COMPONENT_NUM =40
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:,1:]
labels = labeled_images.iloc[0:,:1]
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(images)
clf = svm.SVC()
images=pca.transform(images)
clf.fit(images,  labels.values.ravel())
test_labeled_images = pd.read_csv('../input/test.csv')
test_labeled_images =pca.transform(test_labeled_images )
results=clf.predict(test_labeled_images )
#print (results);
pd.DataFrame({"ImageId": range(1,len(results)+1), "Label":results }).to_csv('results.csv', index=False, header=True)

# Any results you write to the current directory are saved as output.
