#!/usr/bin/env python
# coding: utf-8

# # What are Support Vector Machines?
# Support Vector Machines (SVM's) are versatile models that can perform both classification and regression. They can also be used to detect outliers. SVM's are very popular and you should definitely get to know how to use them. They are particularly suited for complex but small classification datasets, because they are strong but slow.
# 
# ## Linear SVM Classification 
# A Linear SVM Classification model tries to create a decision 'line' to classify it's data, these lines are called Support Vectors. Any data on the one 'side' of the line is classified as one thing, any data on the other is classified as something else. SVM's try to fit this line as evenly as possible, so it'll try to stay as far away as possible from the data points of different classifications that are closest to each other. This creates a large margin for new data that is not in the training dataset. That is why this is called _large margin classification_.
# 
# ![An example of large margin classification](https://www.saedsayad.com/images/SVM_2.png)
# 
# If we don't want any data points on the Support Vectors it's called _hard margin classification_. There are a few issues with using _hard margin classifcation_. It only works with data that is linearly separable and is very sensitive to outliers. If a red data point outlier would be among the green data points, it wouldn't be possible to create any Support Vectors.
# 
# To avoid these issues you can use a more flexible model that tries to find a good balance between keeping the distance between Support Vectors as big as possible and limiting _margin violations_ (i.e. limiting outliers that are between or over the wrong side of the Support Vectors). This method of classification is called _soft margin classification_.
# 
# In Scikit-Learn you can control the balance by tweaking the C hyperparameter. A smaller C value creates more distance between Support Vectors and thus more _margin violations_. In general, you can reduce overfitting by reducing the C value.
# 
# Now, let's test the Scikit-Learn model on a dataset. We'll use the iris dataset from the datasets module.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
print(iris['DESCR'])


# In[ ]:


X = iris["data"][:, (2,3)]
y = (iris["target"] == 2).astype(np.float64)


# In[ ]:


plt.scatter(x=X[:,0], y=X[:,1], c=y, cmap=plt.cm.Paired)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")


# As you can see, we have some datapoints that overlap into the other classification, this will make it harder for the SVM to classify them. Let's see how it will deal with this.

# In[ ]:


# Code from https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html
def plotSVM(clf, X, y, size=(5,5), xlab=None, ylab=None, title=None):
    # figure number
    fignum = 1


    clf.fit(X, y)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-3, 8)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=size)
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
            facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
            edgecolors='k')

    plt.axis('tight')
    x_min = 0
    x_max = 8
    y_min = 0
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:800j, y_min:y_max:800j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=size)
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if title is not None:
        plt.title(title)
    
    plt.show()


# In[ ]:


for c in [0.01, 0.1, 1, 10, 1000]:
    clf = SVC(kernel='linear', C=c)
    plotSVM(clf, X, y, (5,5), "Petal Length", "Petal Width", "C = "+str(c))


# Shown in the plots above, you can clearly see that a larger C creates closer Support Vectors. These have less _margin violations_ than the smaller C's.
# 
# ## Nonlinear SVM Classification
# Most datasets aren't linearly seperable. This means we'll have to use some extra steps to classify them. One approach is to add more features (like polynomial features) to create a linearly separable dataset. To do this with Support Vector Machines we'll use the Polynomial Kernel
# 
# ## Polynomial Kernel
# To add polynomial features to a SVM we can select the _poly_ Kernel in SVC. You can set the polynomial degree to control the amount of features added. It can do these things by using a _kernel trick_. This is quite an advanced mathematical techinique so I won't go into details here, but you can find a simple explanation [here](https://www.youtube.com/watch?v=3Xw6FKYP7e4) and the mathematical explanation [here](https://www.youtube.com/watch?v=3Xw6FKYP7e4).
# 
# Let's create some data that isn't seperable by a Linear SVM. We'll use the Scikit-learn _make moons_ dataset for this. It'll create 2 half moons that are opposite and inside each other.

# In[ ]:


from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.13, random_state=42)

# Code From https://github.com/ageron/handson-ml/blob/master/05_support_vector_machines.ipynb
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


# In[ ]:


plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])


# Now that we have some data, we can plot how the different degrees affect the SVM. Remember, a 1 degree poly SVM is just a normal linear SVM.

# In[ ]:


for d in [1, 2, 3, 10]:
    clf = SVC(kernel="poly", degree=d, C=5, coef0=10, gamma='auto')
    clf.fit(X,y)
    plt.title('Degree = '+str(d), fontsize=20)
    plot_predictions(clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.show()


# By adding more degrees to the Kernel, we can get a more 'flexible' _Decision Function_. This can be very useful but also very dangerous. As you can see above, the 10 degree poly SVM is very close to overfitting. If we added more degrees to this SVM we would certainly overfit it.
# 
# ## Similarity Features
# Another option to classify non linear data is to add features computed using a _similarity function_. This is a function that measures how much each instance resembles a particular _landmark_. These _landmarks_ are often the location of each data point in the dataset. The downside of this is that if you have a training set with $m$ data points, it will also get $m$ features. This can create some really big datasets when you're using large $m$ datasets.
# 
# ## Gaussian RBF Kernel
# Just like with Polynomial Kernels, the SVM has a kernel for _similarity features_. This is the "rbf" kernel. This again uses the _kernel trick_ to do its magic. You can tweak the _gamma_ to adjust the _range of influence_ that a _land mark_ has. If you increase the _gamma_ then the _range of influence_ will decrease.

# In[ ]:


for g in [1, 3, 5, 10, 30]:
    clf = SVC(kernel="rbf", C=5, coef0=10, gamma=g)
    clf.fit(X,y)
    plt.title('Gamma = '+str(g), fontsize=20)
    plot_predictions(clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.show()


# Increasing the gamma creates an "island" of the _decision boundary_ around the blue data points. The last plot is definitely overfitted and shows a very limited area for the blue data points.
# 
# ## SVM Regression
# We've already seen how we can classify data using SVM's but now we'll take a look at how we can use SVM's for regression problems. Instead of creating a 'street' that is as large as possible between the 2 different types of data, the SVM will now try to fit as many data points as possible onto the 'street'. The width of the street is controlled by $\epsilon$ (epsilon).

# In[ ]:


# Code from https://github.com/ageron/handson-ml/blob/master/05_support_vector_machines.ipynb
np.random.seed(42)
m = 50
X = 2 * np.random.rand(m, 1)
y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)


# In[ ]:


from sklearn.svm import SVR
for e in [0.1, 1, 2]:
    reg = SVR(epsilon=e, gamma='auto', kernel='linear')
    reg.fit(X,y)
    plot_svm_regression(reg, X, y, [0, 2, 3, 11])
    plt.title("Epsilon = " + str(e))
    plt.show()


# Using these plots we can see that a bigger $\epsilon$ equals a bigger street.
# 
# When using SVM's for regression we can use the same _kernel tricks_ as we used with classification. So we could create a 'poly' or 'rbf' kernel instead of just a linear one.

# In[ ]:


for k in ['linear', 'poly', 'rbf']:
    reg = SVR(epsilon=1, gamma='auto', kernel=k)
    reg.fit(X,y)
    plot_svm_regression(reg, X, y, [0, 2, 3, 11])
    plt.title("Kernel = " + str(k))
    plt.show()


# ## Conclusion
# So now we know a bit about how Support Vector Machines work. There is also a lot of math involved in using them but we'll skip that for now. Maybe we'll work that out at a later date. For now it's enough to know how the different types of SVM's work and how we can use them.
# 
# ### Previous Kernel
# [What Is Polynomial Regression?](https://www.kaggle.com/veleon/what-is-polynomial-regression)
# ### Next Kernel
# [How do Decision Trees Work?](https://www.kaggle.com/veleon/how-do-decision-trees-work)
