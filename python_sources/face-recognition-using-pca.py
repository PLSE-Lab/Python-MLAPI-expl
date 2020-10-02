# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC

###########
#function for displaying Original images
def show_original_images(pixels):
    fig, axes = plt.subplots(6,10, figsize=(11,7),
                            subplot_kw={'xticks':[],'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(64,64), cmap='gray')
    plt.show()
    
#function to display Eigenfaces
def show_eigenfaces(pca):
    fig, axes = plt.subplots(3,8,figsize=(9,4),
                            subplot_kw={'xticks':[],'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(64,64),cmap='gray')
        ax.set_title("PC "+ str(i+1))
    plt.show()

######

df = pd.read_csv("/kaggle/input/face_data.csv")

print(df.head())

labels = df["target"]
pixels = df.drop(["target"],axis=1)

show_original_images(pixels)

#split data
x_train,x_test, y_train, y_test = train_test_split(pixels,labels)


#Perform PCA
pca = PCA(n_components=135).fit(x_train)


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()


show_eigenfaces(pca)

#project training data to PCA
x_train_pca = pca.transform(x_train)

#Initialize classifier and fit training data

clf = SVC(kernel='rbf', C=1000, gamma=0.01)
clf = clf.fit(x_train_pca,y_train)


#testing and get classification report
x_test_pca = pca.transform(x_test)

y_pred = clf.predict(x_test_pca)


print(classification_report(y_test,y_pred))