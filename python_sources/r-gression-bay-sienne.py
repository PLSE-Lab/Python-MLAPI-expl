# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from keras.models import Sequential
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

train_path = '/kaggle/input/digit-recognizer/train.csv'
test_path = '/kaggle/input/digit-recognizer/test.csv'
    
file=open(train_path,"r")
data_train = pd.read_csv(train_path)
data_test = pd.read_csv(test_path)

data_test = (data_test.iloc[:,0:].values).astype('float32')

Train_X = (data_train.iloc[:,1:].values).astype('float32') # all pixel values
Train_Y = data_train.iloc[:,0].values.astype('int32')
print(data_test, data_test.shape)
print(Train_X, Train_X.shape)


print(Train_X.shape)
X_train, X_test, y_train, y_test = train_test_split(Train_X, Train_Y, test_size=0.2, random_state=0)

#Train_X = Train_X.reshape(Train_X.shape[0], 28, 28)


'''plt.figure(figsize=(6,8))
# plot first few images
for i in range(1, 10):
    # define subplot
    plt.subplot(3, 3, i)
    # plot raw pixel data
    plt.imshow(Train_X[i], cmap=plt.get_cmap('gray'))
    plt.title(Train_Y[i])
# show the figure
plt.show()'''

#On instancie les classifieurs
clf_nb = MultinomialNB()
clf_dt = BernoulliNB()

scores_nb = cross_val_score(clf_nb, X_train, y_train, cv=10)
scores_dt = cross_val_score(clf_dt, X_train, y_train, cv=10)

#Now let's fit the model and evaluate the model on the test dataset
clf_nb.fit(X_train, y_train)
clf_dt.fit(X_train, y_train)

print("Précision test Multibinomial Naive Bayes: ", clf_nb.score(X_test, y_test))
print("précision test Bernoulli Naive Bayes: ", clf_dt.score(X_test, y_test))

predictions = clf_dt.predict(data_test)

print(predictions.shape)
    
test_images = data_test.reshape(data_test.shape[0], 28, 28)

plt.figure(figsize=(6,8))
for i in range(1, 10):
    # define subplot
    plt.subplot(3, 3, i)
    # plot raw pixel data
    plt.imshow(test_images[i], cmap=plt.get_cmap('gray'))
    plt.title(predictions[i])
# show the figure
plt.show()

c = csv.writer(open("/kaggle/working/output.csv", "w"))
c.writerow(["ImageId","Label"])
for i in range(len(predictions)):
    c.writerow([i+1,predictions[i]])
    
t = pd.read_csv("/kaggle/working/output.csv")
print(t.shape)

    
            
    
    
       

    


    
    
    
    
    
    
    
    
    
    
# Any results you write to the current directory are saved as output.