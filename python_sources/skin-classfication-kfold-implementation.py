#!/usr/bin/env python
# coding: utf-8

# **Skin Classification using RGB data with Kfold (short and long implementation)**
# * Import data
# * K-fold implementation with visualization 
# * K-fold two lines implementation

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Data is RGB value of images for skin or non-skin classification. Three attibutes with red, green, and blue intensity values of images. Use these feature to predict image is skin or not.

# In[ ]:


data = pd.read_csv('../input/skin.csv')
data = np.random.permutation(data)

X = data[:10000,:-1]
Y = data[:10000,-1]


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=33)
logreg = LogisticRegression(solver='newton-cg')


# This is my implementaion with sklearn and you can see this a rather long. But I added visualization so that the whole process is clear. I think this makes the whole process easy to understand and I like to see the decision boundary if it is possible to plot(3D or less). 

# In[ ]:


from mpl_toolkits import mplot3d
fold_num = 0

accuracy_all = []
for train_index, test_index in skf.split(X, Y):
    fold_num +=1
    print("This is fold number:",fold_num)
    #split data
    X_train,X_test = X[train_index,:],X[test_index,:]
    Y_train,Y_test = Y[train_index],Y[test_index]
    
    #train data with training set
    logreg.fit(X_train,Y_train)
    print('The weight vectors are:',logreg.intercept_,logreg.coef_)
    fig = plt.figure()
    ax = plt.axes(projection = '3d' )
    ax.scatter3D(X_train[:,0], X_train[:,1], X_train[:,2], c=Y_train, cmap='winter',label='training data');
    x1p,x2p = np.meshgrid(np.linspace(0,300,30),np.linspace(0,300,30))
    x3p = -logreg.intercept_/logreg.coef_[0,2]- (logreg.coef_[0,0]/logreg.coef_[0,2])*x1p - (logreg.coef_[0,1]/logreg.coef_[0,2])*x2p
    ax.plot_surface(x1p,x2p,x3p,label='decision boundary')
    plt.xlabel('Intensity of Blue'); plt.ylabel('Intesity of Green'); ax.set_zlabel('Intesity of Red');
    #plt.legend()
    plt.title('Training model with train data')
    plt.show()
    
    
    #predict on test data
    Y_predict= logreg.predict(X_test)
    correct_predict = sum(Y_predict==Y_test)
    accuracy = 100*correct_predict/len(Y_test)
    fig = plt.figure()
    ax = plt.axes(projection = '3d' )
    ax.scatter3D(X_test[:,0], X_test[:,1], X_test[:,2], c=Y_test, cmap='winter',label='test data');
    ax.plot_surface(x1p,x2p,x3p,label='decision boundary')
    plt.xlabel('Intensity of Blue'); plt.ylabel('Intesity of Green'); ax.set_zlabel('Intesity of Red');
    #plt.legend()
    plt.title('Testing model with test data with accuracy:'+ str(accuracy))
    plt.show()
    print("Fold ",fold_num,"accuracy is:",accuracy)
    accuracy_all.append(accuracy)


# Averge k-fold CV from long implementation

# In[ ]:


print("Average model accuracy for",skf.get_n_splits(),"fold CV is:",np.mean(accuracy_all))


# **Sklearn implementaion: What? just two lines...**
# * Note: If you compare both implementation, the results are exactly the same!
# * I also did k-fold implementaion without using any sklearn functions, if you are interested check out my other kernels.

# In[ ]:


from sklearn.model_selection import cross_val_score
acc = cross_val_score(logreg,X,Y,cv=5,scoring='accuracy')
print("Individual fold accuracy:",acc)
print("Average accuracy:",np.mean(acc))


# **Let me know if you are intersted more implementation like this!**
