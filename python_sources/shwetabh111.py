# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 21:34:24 2017

@author: DELL
"""

#importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#importing the dataset

dataset = pd.read_csv('../input/criminal_train.csv')
print(dataset.shape)

x=dataset.iloc[: , 0:71].values
y = dataset.iloc[: , 71].values

#splitting the dataset into Traning set and Test set
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x , y , test_size =0.25, random_state=0)

#features scalling 
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Applying PCA(dimensionality reduction technique)(principle component analysis)
"""from sklearn.decomposition import PCA
pca = PCA(n_components = 2 )
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)
explained_variance = pca.explained_variance_ratio_"""


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)




#Fitting Logestic Regression to the traning set

"""from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train , y_train)"""

#predicting the test set result
y_predict = classifier.predict(x_test)

#making the confusion matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test , y_predict)

#Applying K-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator =classifier ,X =x_train, y = y_train , cv =10)
accuracies.mean()
accuracies.std()

#Applying Grid Search and to  find the best model and the best parameter
from sklearn.model_selection import GridSearchCV 
parameters = [{'C' : [1 ,10 , 100 , 1000], 'kernel' : ['linear'] },
             {'C' : [1 , 10 , 100 , 1000] , 'kernel' : ['rbf'] , 'gamma' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grid_search =  GridSearchCV(estimator =classifier ,
                            param_grid=parameters,
                            scoring ='accuracy',
                            cv =10,
                            n_jobs =-1) 
grid_search = grid_search.fit(x_train , y_train)
best_accuracy =  grid_search.best_score_ 
best_parameter =grid_search.best_params_     


"""#Visualising the training set result
plt.figure(1)

from matplotlib.colors import ListedColormap
x_set , y_set = x_train , y_train
x1 ,x2 = np.meshgrid(np.arange(start = x_set[: , 0].min() -1 , stop = x_set[: , 0].max() + 1 , step =0.01),
                     np.arange(start = x_set[:  ,1].min() -1 , stop =x_set[: ,1].max() + 1 , step =0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel() ,x2.ravel()]).T).reshape(x1.shape),
                                               alpha = 0.75 , cmap = ListedColormap(('red','green')))
plt.xlim(x1.min() ,x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set ==j , 0], x_set[y_set == j , 1],
                c=ListedColormap(('red', 'green' ))(i) ,label = j)
    
plt.title('Logistic Regression(Traning set)')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend()
plt.show()
plt.savefig('foo1.png')


#Visualising the Test Set result
plt.figure(2)

from matplotlib.colors import ListedColormap
x_set , y_set = x_test , y_test
x1 ,x2 = np.meshgrid(np.arange(start = x_set[: , 0].min() -1  , stop = x_set[: , 0].max() +1  , step =0.01),
                     np.arange(start = x_set[:  ,1].min() -1 , stop =x_set[: ,1].max() +1  , step =0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel() ,x2.ravel()]).T).reshape(x1.shape),
                                               alpha = 0.75 , cmap = ListedColormap(('red','green','blue')))
plt.xlim(x1.min() ,x1.max())
plt.ylim(x2.min(), x2.max())

#using this loop I have printed all the data points as red and green
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set ==j , 0], x_set[y_set == j , 1],
                c=ListedColormap(('red', 'green' ,'blue'))(i) ,label = j)
    
plt.title('Logistic Regression(Test set)')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend()
plt.show()
plt.savefig('foo2.png')"""