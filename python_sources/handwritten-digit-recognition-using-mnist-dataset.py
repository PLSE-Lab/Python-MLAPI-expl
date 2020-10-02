# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

data= pd.read_csv('../input/train.csv')
labels = pd.read_csv('../input/test.csv')
# Any results you write to the current directory are saved as output.
x=data.iloc[:, 1:].values
y=data.iloc[:,0].values
labels = labels.iloc[0:784, 0].values 


print(x.dtype, y.dtype, labels.dtype)
print(x.shape, y.shape, labels.shape)

min(5, len(data))
#viewing the digits
import matplotlib.pyplot as plt
def plot_images(mnist, labels):
    n_cols= min(5, len(mnist))
    n_rows=len(mnist)//n_cols
    fig=plt.figure(figsize=(8,8))
    
    
    for i in range(n_rows * n_cols):
        sp=fig.add_subplot(n_rows, n_cols, i+1)
        plt.axis('off')
        plt.imshow(mnist[i],  cmap=plt.cm.gray)
        sp.set_title(labels[i])
plt.show()


#Plotting random 20 images
p=np.random.permutation(len(x))
p=p[:20]
plot_images(x[p].reshape(-1, 28, 28), y[p])
plt.show()


#splitting the dataset into train asnd test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, )


# fitting the train sets into the algorithm
from sklearn.naive_bayes import MultinomialNB
cls = MultinomialNB(alpha=3,  )
cls.fit(x_train, y_train)

cls.partial_fit(x,y, classes=None, sample_weight=None)

cls.score(x_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix
prediction = cls.predict(x_test)
print(classification_report(y_test, prediction))



#Plotting random 20 images
p=np.random.permutation(len(x_test))
p=p[:20]
plot_images(x_test[p].reshape(-1, 28, 28), prediction[p])


