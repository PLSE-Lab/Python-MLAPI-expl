#!/usr/bin/env python
# coding: utf-8

# In this script I will explore the basic and famous classification algorithm of SVM, in comparison to the logistic regression. including:
# 
#  1. SVM and logistic regression decision line and accuracy
#  2. SVM effect of regularization
#  3. SVM and non-linear Kernel
#  4. SVM and class imbalance 
# 
# This script is the first in a series of scripts going over Andrew Ng's canonical [ML course][1] on Coursera. In the future you can find other scripts exploring the algorithms in my Kaggle profile.
# 
#  I have used some of the code which is available in [scikit-learn documentation][2].
# 
#  
# 
# 
#   [1]: https://www.coursera.org/learn/machine-learning/home/welcome
#   [2]: http://scikit-learn.org/stable/modules/svm.html#svm

# First, let's load all the relevant libraries and read the data: 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from subprocess import check_output
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.style.use('fivethirtyeight')
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/data.csv')


# Now, let's explore some correlations, similarly to what other Kagglers have done with this data set:

# In[ ]:


g = sns.PairGrid(df[[df.columns[1],df.columns[2],df.columns[3],df.columns[4], df.columns[5],df.columns[6]]],hue='diagnosis')
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter, s = 3)


# For a more visual exploration, it would be best to work in a 2-dimensional world. However, a lot of feature pairs divide nicely the data to a similar extent. therefore, it makes sense to use one of the dimensionality reduction methods to try to use as many features as possible and maintian as much information as possible when working with only 2 dimensions. I will use PCA. based on [Anistropic][1]'s great script it also seems that standardizing the features is crucial.
# 
# 
#   [1]: https://www.kaggle.com/arthurtok/d/uciml/breast-cancer-wisconsin-data/tsne-pca-quick-and-dirty-visuals

# In[ ]:



df_std = StandardScaler().fit_transform(df.drop(['id','diagnosis','Unnamed: 32'], axis = 1))
pca = PCA(n_components=2)
pca.fit(df_std)
TwoD_Data = pca.transform(df_std)
PCA_df = pd.DataFrame()
PCA_df['PCA_1'] = TwoD_Data[:,0]
PCA_df['PCA_2'] = TwoD_Data[:,1]


plt.plot(PCA_df['PCA_1'][df.diagnosis == 'M'],PCA_df['PCA_2'][df.diagnosis == 'M'],'o', alpha = 0.7, color = 'r')
plt.plot(PCA_df['PCA_1'][df.diagnosis == 'B'],PCA_df['PCA_2'][df.diagnosis == 'B'],'o', alpha = 0.7, color = 'b')
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.legend(['Malignant','Benign'])


# ## Linear Logistic Regrssion
# 
# Let's now use a logistic regression classifier. first, we will assess the sensitivity of the results to the regularization coefficient:

# In[ ]:


def model(x):
    return 1 / (1 + np.exp(-x))

PCA_df['target'] = 0
PCA_df['target'][df.diagnosis == 'M'] = 1

traindf, testdf = train_test_split(PCA_df, test_size = 0.3)

X = traindf[['PCA_1','PCA_2']]
y = traindf['target']
Reg = np.linspace(0.1,10,100)
accuracy = []
for C in Reg:
    clf = LogisticRegression(penalty='l2',C=C)
    clf.fit(X,y)
    prediction = clf.predict(testdf[['PCA_1','PCA_2']])
    loss = prediction - testdf['target']
    accuracy.append(1 - np.true_divide(sum(np.abs(loss)),len(loss)))
#loss = model(clf.coef_*X + clf.intercept_)


plt.plot(Reg,accuracy,'o')
plt.xlabel('Regularization')
plt.ylabel('Validation Score')



# The model is not very sensitive to the regularization coefficient.
# 
# Now let's train the model with the default regularization and predict on the test set. let us also plot the boundary line determined by the classifier:

# In[ ]:


clf = LogisticRegression(penalty='l2',C=0.5)
clf.fit(X,y)
print('Training Accuracy.....',clf.score(X,y))
prediction = clf.predict(testdf[['PCA_1','PCA_2']])
print('Validation Accuracy....',clf.score(testdf[['PCA_1','PCA_2']],testdf['target']))
loss = prediction - testdf['target']
accuracy = 1 - np.true_divide(sum(np.abs(loss)),len(loss))

radius = np.linspace(min(X.PCA_1), max(X.PCA_2), 100)
line = (-clf.coef_[0][0]/clf.coef_[0][1])*radius + np.ones(len(radius))*(-clf.intercept_/clf.coef_[0][1])
plt.plot(radius,line)
plt.plot(PCA_df['PCA_1'][df.diagnosis == 'M'],PCA_df['PCA_2'][df.diagnosis == 'M'],'o', alpha = 0.7)
plt.plot(PCA_df['PCA_1'][df.diagnosis == 'B'],PCA_df['PCA_2'][df.diagnosis == 'B'],'o', color = 'b', alpha = 0.7)
plt.legend(['Decision Line','Malignant','Benign'])
plt.title('Logistic Regression. Accuracy:' + str(accuracy)[0:4])
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')


# ## SVM with a linear kernel
# 
# Now let's train a linear Support Vector Classifier. a SVM classifier finds the hyper-plane that maximizes the margin between the 2 groups if they are linearly separable. If the data is not linearly separable, as in our case, there is a trade-off between the margin size and the number of points that are on the wrong side of the decision boundary. this is determined by the regularization coefficient. We would see that in our case the effect would be very small.

# In[ ]:


C = 1
clf2 = SVC(kernel = 'linear',C =C)
clf2.fit(X, y)
print('training accuracy...',clf2.score(X, y, sample_weight=None))
print('validation accuracy...',clf2.score(testdf[['PCA_1','PCA_2']],testdf['target']))

w = clf2.coef_[0]
a = -w[0] / w[1]
xx =  np.linspace(min(X.PCA_1), max(X.PCA_2), 100)
yy = a * xx - (clf2.intercept_[0]) / w[1]
plt.scatter(PCA_df.PCA_1[PCA_df.target == 1],PCA_df.PCA_2[PCA_df.target == 1], alpha = 0.8, color = 'r')
plt.scatter(PCA_df.PCA_1[PCA_df.target == 0],PCA_df.PCA_2[PCA_df.target == 0], alpha = 0.8, color = 'b')
plt.scatter(clf2.support_vectors_[:, 0], clf2.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, color = 'g')
plt.plot(xx, yy)
plt.title('SVM.' + ' Reg =' + str(C) + 'Accuracy:' + str(clf2.score(testdf[['PCA_1','PCA_2']],testdf['target']))[0:4], fontsize = 10)


mu_vec1 = np.array([0,0])
cov_mat1 = np.array([[2,0],[0,2]])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
mu_vec1 = mu_vec1.reshape(1,2).T # to 1-col vector

mu_vec2 = np.array([1,2])
cov_mat2 = np.array([[1,0],[0,1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, 100)
mu_vec2 = mu_vec2.reshape(1,2).T
print('number of supporting points...',clf2.n_support_ )

plt.legend(['Decision Line','Malignant','Benign'])


# The samples denoted with a green circle are the supporting points. In a linearly separable case, these would determine the margin. 

# In[ ]:


plt.plot(xx, yy,'g')
plt.plot(radius,line,'m')
plt.title('Comparison of Decision Boundaries')
plt.legend(['SVM','Logistic Regression'])
plt.ylim([-10,15])
plt.xlim([-5,15])
plt.scatter(PCA_df.PCA_1[PCA_df.target == 1],PCA_df.PCA_2[PCA_df.target == 1], alpha = 0.8, color = 'r')
plt.scatter(PCA_df.PCA_1[PCA_df.target == 0],PCA_df.PCA_2[PCA_df.target == 0], alpha = 0.8, color = 'b')


# We can see in the pot above that the decision boundaries of the 2 classifiers are almost parallel. The accuracy level is therefore similar too.

# ## SVM with different regularization
# 
# the parameter C denotes the regularization strength (though inversely). If it is very large, the classifier will act like the hard margin classifier for the linearly separable case. Higher C values would yield a smaller number of support vectors.
# 

# In[ ]:


plt.subplot(1,2,1)
C = 1
clf2 = SVC(kernel = 'linear',C =C)
clf2.fit(X, y)
print('training accuracy...',clf2.score(X, y, sample_weight=None))
print('validation accuracy...',clf2.score(testdf[['PCA_1','PCA_2']],testdf['target']))

w = clf2.coef_[0]
a = -w[0] / w[1]
xx =  np.linspace(min(X.PCA_1), max(X.PCA_2), 100)
yy = a * xx - (clf2.intercept_[0]) / w[1]
plt.scatter(PCA_df.PCA_1[PCA_df.target == 1],PCA_df.PCA_2[PCA_df.target == 1], alpha = 0.8, color = 'r')
plt.scatter(PCA_df.PCA_1[PCA_df.target == 0],PCA_df.PCA_2[PCA_df.target == 0], alpha = 0.8, color = 'b')
plt.scatter(clf2.support_vectors_[:, 0], clf2.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, color = 'g')
plt.plot(xx, yy)
plt.title('SVM.' + ' Reg =' + str(C) + 'Accuracy:' + str(clf2.score(testdf[['PCA_1','PCA_2']],testdf['target']))[0:4], fontsize = 10)


mu_vec1 = np.array([0,0])
cov_mat1 = np.array([[2,0],[0,2]])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
mu_vec1 = mu_vec1.reshape(1,2).T # to 1-col vector

mu_vec2 = np.array([1,2])
cov_mat2 = np.array([[1,0],[0,1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, 100)
mu_vec2 = mu_vec2.reshape(1,2).T
print(clf2.n_support_ )

plt.subplot(1,2,2)
C = 300
clf2 = SVC(kernel = 'linear',C =C)
clf2.fit(X, y)
print('training accuracy...',clf2.score(X, y, sample_weight=None))
print('validation accuracy...',clf2.score(testdf[['PCA_1','PCA_2']],testdf['target']))

w = clf2.coef_[0]
a = -w[0] / w[1]
xx =  np.linspace(min(X.PCA_1), max(X.PCA_2), 100)
yy = a * xx - (clf2.intercept_[0]) / w[1]
plt.scatter(PCA_df.PCA_1[PCA_df.target == 1],PCA_df.PCA_2[PCA_df.target == 1], alpha = 0.8, color = 'r')
plt.scatter(PCA_df.PCA_1[PCA_df.target == 0],PCA_df.PCA_2[PCA_df.target == 0], alpha = 0.8, color = 'b')
plt.scatter(clf2.support_vectors_[:, 0], clf2.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, color = 'g')
plt.plot(xx, yy)
plt.title('SVM.' + ' Reg =' + str(C) + 'Accuracy:' + str(clf2.score(testdf[['PCA_1','PCA_2']],testdf['target']))[0:4], fontsize = 10)

mu_vec1 = np.array([0,0])
cov_mat1 = np.array([[2,0],[0,2]])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
mu_vec1 = mu_vec1.reshape(1,2).T # to 1-col vector

mu_vec2 = np.array([1,2])
cov_mat2 = np.array([[1,0],[0,1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, 100)
mu_vec2 = mu_vec2.reshape(1,2).T
print(clf2.n_support_ )


# Such as in the case of Logistic Regression, the model sensitivity to regularization is very low. a large change in C results in just a small change in the number of support vectors and almost no change in accuracy

# ## Non-linear kernels
# 
# SVM can use the "kernel trick" in order to produce a non-linear decision boundary. 

# In[ ]:


clf3 = SVC(kernel = 'poly',degree = 3)
clf3.fit(X, y)
print('Polynomial kernel - training accuracy...',clf3.score(X, y, sample_weight=None))
print('Polynomial kernel - validation accuracy...',clf3.score(testdf[['PCA_1','PCA_2']],testdf['target']))
print('Polynomial kernel - number of supporting points...',clf3.n_support_ )

clf4 = SVC(kernel = 'rbf',gamma=0.1)
clf4.fit(X, y)
print('Gaussian kernel - training accuracy...',clf4.score(X, y, sample_weight=None))
print('Gaussian kernel - validation accuracy...',clf4.score(testdf[['PCA_1','PCA_2']],testdf['target']))
print('Gaussian kernel - number of supporting points...',clf4.n_support_ )



# ## Decision boundaries of different classifiers
# 
# Let's see the decision boundaries produced by the linear, Gaussian and polynomial classifiers. 

# In[ ]:


plt.figure(figsize = (10,10))
plt.subplot(3,1,1)
x_min = X.PCA_1.min()
x_max = X.PCA_1.max()
y_min = X.PCA_2.min()
y_max = X.PCA_2.max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf2.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = clf2.decision_function(np.c_[XX.ravel(), YY.ravel()])

Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

plt.scatter(PCA_df.PCA_1[PCA_df.target == 1],PCA_df.PCA_2[PCA_df.target == 1], alpha = 0.8, color = 'r')
plt.scatter(PCA_df.PCA_1[PCA_df.target == 0],PCA_df.PCA_2[PCA_df.target == 0], alpha = 0.8, color = 'b')

plt.title('Linear Kernel')
plt.legend(['Malignant', ' Benign'])


plt.subplot(3,1,2)
x_min = X.PCA_1.min()
x_max = X.PCA_1.max()
y_min = X.PCA_2.min()
y_max = X.PCA_2.max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf3.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = clf3.decision_function(np.c_[XX.ravel(), YY.ravel()])

Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

plt.scatter(PCA_df.PCA_1[PCA_df.target == 1],PCA_df.PCA_2[PCA_df.target == 1], alpha = 0.8, color = 'r')
plt.scatter(PCA_df.PCA_1[PCA_df.target == 0],PCA_df.PCA_2[PCA_df.target == 0], alpha = 0.8, color = 'b')

plt.ylabel('PCA_2')
plt.title('Polynomial Kernel')

plt.subplot(3,1,3)

x_min = X.PCA_1.min()
x_max = X.PCA_1.max()
y_min = X.PCA_2.min()
y_max = X.PCA_2.max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf4.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = clf4.decision_function(np.c_[XX.ravel(), YY.ravel()])

Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

plt.scatter(PCA_df.PCA_1[PCA_df.target == 1],PCA_df.PCA_2[PCA_df.target == 1], alpha = 0.8, color = 'r')
plt.scatter(PCA_df.PCA_1[PCA_df.target == 0],PCA_df.PCA_2[PCA_df.target == 0], alpha = 0.8, color = 'b')

plt.xlabel('PCA_1')
plt.title('Gaussian Kernel')




# ## SVM and class imbalance
# 
# Let's quantify our class imbalance

# In[ ]:


print('Malignant samples...',len(df[df.diagnosis == 'M']))
print('Benign samples...',len(df[df.diagnosis == 'B']))


# So we have a mild class imbalance problem. However, since we deal with cancer, we have a class importance imbalance - false negative is worse than false positive. 
# 
# In SVM, we can introduce a weight to take this into account.
# 
# Let's first see how the non-weighted classifier performs in terms of precision and recall, in addition to accuracy. Let's only deal with the linear classier as it's simplest, and performs as well as the more complex kernels.
# 
# 

# In[ ]:


from sklearn.metrics import precision_recall_fscore_support

y_true = testdf['target']
y_pred = clf2.predict(testdf[['PCA_1','PCA_2']])
[precision,recall,fscore,support] = precision_recall_fscore_support(y_true, y_pred,pos_label=1)
print('Precision:', precision,'Recall:', recall, 'fscore:',fscore)


# Now let's introduced a weighted classifier and put a larger weight on the malignant class:

# In[ ]:


wclf = SVC(kernel='linear', class_weight={1:10})
wclf.fit(X,y)

y_true = testdf['target']
y_pred = wclf.predict(testdf[['PCA_1','PCA_2']])
[precision,recall,fscore,support] = precision_recall_fscore_support(y_true, y_pred,pos_label=1)
print(' Weighted Precision:', precision,' Weighted Recall:', recall, ' Weighted fscore:',fscore)


# We see that we have higher recall values for the positive class. this means lower chance to miss a true positive. let's verify that the false negative is indeed 0:

# In[ ]:


diff = y_pred - y_true
print('Number of false negative:',len(diff[diff == -1]))


# Let's now plot the weighted and non-weighted decision boundaries:

# In[ ]:


plt.figure(figsize = (10,10))
plt.subplot(2,1,1)
x_min = X.PCA_1.min()
x_max = X.PCA_1.max()
y_min = X.PCA_2.min()
y_max = X.PCA_2.max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf2.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = clf2.decision_function(np.c_[XX.ravel(), YY.ravel()])

Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

plt.scatter(PCA_df.PCA_1[PCA_df.target == 1],PCA_df.PCA_2[PCA_df.target == 1], alpha = 0.8, color = 'r')
plt.scatter(PCA_df.PCA_1[PCA_df.target == 0],PCA_df.PCA_2[PCA_df.target == 0], alpha = 0.8, color = 'b')

plt.title('Linear Kernel')
plt.legend(['Malignant', ' Benign'])


plt.subplot(2,1,2)
x_min = X.PCA_1.min()
x_max = X.PCA_1.max()
y_min = X.PCA_2.min()
y_max = X.PCA_2.max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = wclf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = wclf.decision_function(np.c_[XX.ravel(), YY.ravel()])

Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

plt.scatter(PCA_df.PCA_1[PCA_df.target == 1],PCA_df.PCA_2[PCA_df.target == 1], alpha = 0.8, color = 'r')
plt.scatter(PCA_df.PCA_1[PCA_df.target == 0],PCA_df.PCA_2[PCA_df.target == 0], alpha = 0.8, color = 'b')

plt.title('Weighted Linear Kernel')
plt.legend(['Malignant', ' Benign'])


# We can see how the decision boundary moved to the right to make sure to include all malignant cases, with the price of more false positives.
