#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


# In[ ]:


data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")


# In[ ]:


data.columns


# In[ ]:


data.describe()


# In[ ]:


data['species'].value_counts()


# In[ ]:


data.head()


# In[ ]:


sepal_length = data['sepal_length']
petal_length = data['petal_length']
sepal_width=data['sepal_width']
petal_width=data['petal_width']


# In[ ]:


#density distribution of petal length, petal width, sepal length, sepal width of Iris-setosa

iris_setosa=data[data['species'].str.contains('Iris-setosa')]
iris_versicolor=data[data['species'].str.contains('Iris-versicolor')]
iris_virginica=data[data['species'].str.contains('Iris-virginica')]


# #  Density Plot for Sepal Length and Sepal Width

# In[ ]:


sns.kdeplot(iris_setosa['sepal_length'],iris_setosa['sepal_width'], 
            color='r', shade=True, Label='Iris_Setosa', 
            cmap="Reds", shade_lowest=False).set_title('Density distribution of Sepal Length and Sepal Width of Iris Setosa')


# # Density Plot for Petal Length and Petal Width

# In[ ]:


ax = sns.kdeplot(iris_setosa['petal_length'],iris_setosa['petal_width'], 
            color='g', shade=True, Label='Iris_Setosa', 
            cmap="Greens", shade_lowest=False)
ax = sns.kdeplot(iris_versicolor['petal_length'],iris_versicolor['petal_width'], 
            color='b', shade=True, Label='Iris_Versicolor', 
            cmap="Reds", shade_lowest=False).set_title('Density distribution of Petal length and Petal Width of Iris-Versicolor and Iris-Setosa')


# # Density distribution of Sepal Length

# In[ ]:


ax=sns.kdeplot(data['sepal_length'],shade=True,color="g").set_title('Density distribution of Sepal Length')


# # Density distribution of Iris Setosa

# In[ ]:


iris_setosa.plot.density(title = 'Density distribution plot of Iris Setosa')


# # Density distribution of all flowers with all 4 attributes

# In[ ]:


data.plot.density(title='Density distribution of all the flowers',grid='true')


# # Histogram for Petal Length

# In[ ]:


sns.distplot(data['petal_length'],kde = False).set_title('Histogram for petal length')


# # Petal length of all flowers

# In[ ]:


ax = sns.countplot(x="petal_length",data=data).set_title('Petal length distribution of all flowers')
rcParams['figure.figsize'] = 20,8.27


# # Sepal Length of 3 species

# In[ ]:


sns.barplot(x="species",y="sepal_length",data=data).set_title("Sepal Length of three species")
rcParams['figure.figsize'] = 10,8.27


# # Sepal Length and width, Petal length and width of 3 flowers

# In[ ]:


sns.boxplot(data=data).set_title("Distribution of Sepal_length, Sepal_width, petal_length and petal_width of 3 flowers")


# # Sepal width of 3 flowers

# In[ ]:


sns.boxplot(data=data,x="species",y="sepal_width").set_title("sepal_width distribution of three flowers")


# # Sepal Length and width

# In[ ]:


sns.scatterplot(x=data.sepal_length,y=data.sepal_width,hue=data.species).set_title("Sepal length and Sepal width distribution of three flowers")


# # Petal length and width

# In[ ]:


cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.scatterplot(x="petal_length", y="petal_width",hue="species",size="species",sizes=(20,200),legend="full",data=data)


# # Sepal width of 3 species

# In[ ]:


sns.violinplot(x="species", y = "sepal_width",data=data, palette="muted").set_title("sepal width of 3 species")


# # Petal width of 3 species

# In[ ]:


sns.violinplot(x="species", y = "petal_width",data=data, palette="muted").set_title("petal width of 3 species")


# # Petal length and width of 3 species

# In[ ]:


sns.lineplot(x="petal_length", y="petal_width", hue="species",
                  data=data).set_title("Distribution of petal length and petal width of the 3 species")


# # Pairplot

# In[ ]:


g = sns.pairplot(data, hue="species", palette="husl")


# # Pairplot - Sepal Length and width

# In[ ]:


sns.pairplot(data, vars=["sepal_width", "sepal_length"],diag_kind="kde")


# In[ ]:


sns.pairplot(data,x_vars=["sepal_width", "sepal_length"],y_vars=["petal_width", "petal_length"])


# # Corelation Matrix of all attributes

# In[ ]:


data.corr()
sns.heatmap(data.corr(),center=0).set_title("Corelation of attributes (petal length,width and sepal length,width) among Iris species")


# # Corelation of attributes

# In[ ]:


sns.heatmap(data.corr(), annot=True, fmt="f").set_title("Corelation of attributes (petal length,width and sepal length,width) among Iris species")


# In[ ]:


sns.heatmap(data.corr(), cmap="YlGnBu").set_title("Corelation of attributes (petal length,width and sepal length,width) among Iris species")


# In[ ]:


#corr = np.corrcoef(np.random.randn(10, 200))
mask = np.zeros_like(data.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(data.corr(), mask=mask, vmax=.3, square=True).set_title("Corelation of attributes (petal length,width and sepal length,width) among Iris species")


# In[ ]:


X=data.iloc[:,0:4].values
y=data.iloc[:,4].values


# In[ ]:


#Train and Test split
from sklearn.model_selection import KFold,train_test_split,cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
y_test.shape


# # Feature Scaling

# In[ ]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# # ML Algorithms - Classification

# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
lg_class=LogisticRegression(random_state=0)
lg_class.fit(X_train,y_train)


# In[ ]:


y_pred_logit=lg_class.predict(X_test)


# In[ ]:


ldf=pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred_logit.flatten()})


# In[ ]:


plt.plot(y_test.flatten(),y_pred_logit.flatten())
plt.show()


# # Plot decision boundary for Logistic Regression classifier

# In[ ]:


from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

logreg = LogisticRegression(C=1e5)

# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()


# Confusion Matrix - Metrics evaluation

# In[ ]:


#confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
ca=confusion_matrix(y_test,y_pred_logit)
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp=plot_confusion_matrix(lg_class,X_train, y_train,display_labels=class_names,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)


# Accuracy score - Metrics evaluation

# In[ ]:


#accuracy_score
from sklearn.metrics import accuracy_score
acc_logistic=accuracy_score(y_test, y_pred_logit)
acc_logistic


# Classification Report - Metrics Evaluation

# In[ ]:


#classification_report
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_test, y_pred_logit, target_names=target_names))


# # KNN Classification

# In[ ]:


#KNN Classification
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
a_index = list(range(1,11))
a = pd.Series()
x = [1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    k_class=KNeighborsClassifier(n_neighbors=5) 
    k_class.fit(X_train,y_train)
    y_pred_neigh=k_class.predict(X_test)
    a=a.append(pd.Series(metrics.accuracy_score(y_pred_neigh,y_test)))
plt.plot(a_index, a)
plt.title("KNN Prediction")
plt.xticks(x)
    


# Decision Boundary of Neighbour classifier

# In[ ]:


#KNN 
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()


# In[ ]:


kdf=pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred_neigh.flatten()})


# Confusion Matrix

# In[ ]:


#confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
ca=confusion_matrix(y_test,y_pred_neigh)
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp=plot_confusion_matrix(k_class,X_train, y_train,display_labels=class_names,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)


# Accuracy Score

# In[ ]:


#accuracy_score
from sklearn.metrics import accuracy_score
acc_knn=metrics.accuracy_score(y_test, y_pred_neigh)
acc_knn


# Classification Report

# In[ ]:


#classification_report
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_test, y_pred_neigh, target_names=target_names))


# # SVM Classification

# In[ ]:


from sklearn.svm import SVC
svm_class=SVC(kernel='linear',random_state=0)
svm_class.fit(X_train,y_train)


# In[ ]:


y_pred_svc=svm_class.predict(X_test)


# In[ ]:


svdf=pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred_svc.flatten()})


# Confusion Matrix

# In[ ]:


#confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
ca=confusion_matrix(y_test,y_pred_svc)
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp=plot_confusion_matrix(svm_class,X_train, y_train,display_labels=class_names,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)


# Accuracy Score - metrics evaluation

# In[ ]:


#accuracy_score
from sklearn.metrics import accuracy_score
acc_svm=metrics.accuracy_score(y_test, y_pred_svc)


# Classification Report metrics evaluation

# In[ ]:


#classification_report
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_test, y_pred_svc, target_names=target_names))


# Decision boundary plot of SVM Classification

# In[ ]:


from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()


# # Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
n_class=GaussianNB()
n_class.fit(X_train,y_train)


# In[ ]:


y_pred_bayes=n_class.predict(X_test)


# In[ ]:


bdf=pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred_bayes.flatten()})


# In[ ]:


plt.plot(y_test.flatten(),y_pred_bayes.flatten())
plt.show()


# Confusion Matrix - metrics evaluation

# In[ ]:


#confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
ca=confusion_matrix(y_test,y_pred_bayes)
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp=plot_confusion_matrix(n_class,X_train, y_train,display_labels=class_names,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)


# Accuracy score - metrics evaluation

# In[ ]:


#accuracy_score
from sklearn.metrics import accuracy_score
acc_bayes=metrics.accuracy_score(y_test, y_pred_bayes)
acc_bayes


# Classification Report - Metrics Evaluation

# In[ ]:


#classification_report
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_test, y_pred_bayes, target_names=target_names))


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
d_class=DecisionTreeClassifier(criterion='entropy')
model_all_params = d_class.fit(X_train,y_train)
plt.figure(figsize = (20,10))
plot_tree(model_all_params,filled=True)
plt.show()


# In[ ]:


y_pred_tree=d_class.predict(X_test)


# Plot decision surface boundary for decision tree in Iris dataset

# In[ ]:


import matplotlib.pyplot as plt
# import the needed dataset.
from sklearn.datasets import load_iris
# Import the model and an additional visualization tool.
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Define a variable to establish three classes/species.
class_count = 3
# Define standard RGB color scheme for visualizing ternary classification in order to match the color map used later.
plot_colors = 'brg'
# Define marker options for plotting class assignments of training data.
markers = 'ovs'
# We also need to establish a resolution for plotting.  I favor clean powers of ten, but this is not by any means a hard and fast rule.
plot_res = 0.01

# Load the iris dataset from scikit-learn (note the use of from [library] import [function] above)
iris = load_iris()

# Set the size of the figure used to contain the subplots to be generated.
plt.figure(figsize=(20,10))

# Create an empty list of models to store the results of each pairwise model fit.
models = []

# Use enumerate() to define the possible pairs of features available and iterate over each pair.
for pair_index, pair in enumerate([[0, 1], [0, 2], [0, 3], 
                                           [1, 2], [1, 3], 
                                                   [2, 3] ]):

    # We only take the two features corresponding to the pair in question...
    X, y = iris.data[:, pair] , iris.target
    
    # ... to fit the decision tree classifier model.
    model = DecisionTreeClassifier().fit(X, y)
    
    # Append the results to the models list
    models.append(model)
    
    # Establish a two row by three column subplot array for plotting.
    plt.subplot(2, 3, pair_index + 1)
    
    # Define appropriate x and y ranges for each plot...
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # ... and use each range to define a meshgrid to use as the plotting area.
    xx, yy = np.meshgrid(np.arange(x_min, 
                                   x_max, 
                                   plot_res),
                         np.arange(y_min, 
                                   y_max, 
                                   plot_res) )
    # Use plt.tight_layout() to establish spacing of the subplots.
    plt.tight_layout(h_pad = 0.5, 
                     w_pad = 0.5, 
                       pad = 4.0 )
    
    # Predict the classification of each point in the meshgrid based on the calculated model above.
    # The numpy methods .c_() and .ravel() reshape our meshgrid values into a format compatible with our model.predict() method,
    Z = model.predict(np.c_[xx.ravel(), 
                            yy.ravel() ])
    # Reshape the predictions to match xx...
    Z = Z.reshape(xx.shape)
    # ... and prepare a contour plot that reflects the predictions .
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.brg)
    
    # Define the subplot axis labels after title casing while preserving case on the unit of measure 
    plt.xlabel(iris.feature_names[pair[0]].title()[0:-4] + iris.feature_names[pair[0]][-4:])
    plt.ylabel(iris.feature_names[pair[1]].title()[0:-4] + iris.feature_names[pair[1]][-4:])
    
    # Plot the training points for each species in turn
    for i, color, marker in zip(range(class_count), plot_colors, markers):
        # Subset the data to the class in question with the np.where() method
        index = np.where(y == i)
        # Plot the class in question on the subplot
        plt.scatter(X[index, 0], 
                    X[index, 1], 
                    c = color,
                    marker = marker,
                    label = iris.target_names[i],
                    cmap = plt.cm.brg, 
                    edgecolor = 'black', 
                    s = 15                       )

# Define a title for the overall collection of subplots after each subplot is fully defined
plt.suptitle('Decision Surface of a Decision Tree Using Paired Features',
             size = 24                                                   )

# Define the legend for the subplot collection
plt.legend(loc = 'lower right',
           fontsize = 16,
           borderpad = 0.1, 
           handletextpad = 0.1 )

# Set limits just large enough to show everything cleanly
plt.axis("tight")


# In[ ]:


decdf=pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred_tree.flatten()})


# Confusion Matrix - metrics evaluation

# In[ ]:


#confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
ca=confusion_matrix(y_test,y_pred_tree)
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp=plot_confusion_matrix(d_class,X_train, y_train,display_labels=class_names,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)


# Accuracy score - metrics evaluation

# In[ ]:


#accuracy_score
from sklearn.metrics import accuracy_score
acc_dec=accuracy_score(y_test, y_pred_tree)
acc_dec


# Classification Report - Metrics evaluation

# In[ ]:


#classification_report
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_test, y_pred_tree, target_names=target_names))


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
ran_class=RandomForestClassifier(n_estimators=10,criterion='entropy')
ran_class.fit(X_train,y_train)


# In[ ]:


y_pred_forest=ran_class.predict(X_test)


# Confusion Matrix

# In[ ]:


#confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
ca=confusion_matrix(y_test,y_pred_forest)
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp=plot_confusion_matrix(ran_class,X_train, y_train,display_labels=class_names,cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)


# Accuracy score metrics

# In[ ]:


#accuracy_score
from sklearn.metrics import accuracy_score
acc_ran=metrics.accuracy_score(y_test, y_pred_forest)
acc_ran


# Classification Report metrics

# In[ ]:


#classification_report
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_test, y_pred_forest, target_names=target_names))


# In[ ]:


randf=pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred_forest.flatten()})


# Plot decision surface boundary for Random Forest

# In[ ]:


from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
n_estimators = 30
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

# Load data
iris = load_iris()

plot_idx = 1

models = [DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(n_estimators=n_estimators),
          ExtraTreesClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             n_estimators=n_estimators)]

for pair in ([0, 1], [0, 2], [2, 3]):
    for model in models:
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        model.fit(X, y)

        scores = model.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(
            ".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(
                len(model.estimators_))
        print(model_details + " with features", pair,
              "has a score of", scores)

        plt.subplot(3, 4, plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title, fontsize=9)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                         yy_coarser.ravel()]
                                         ).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                c=Z_points_coarser, cmap=cmap,
                                edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(X[:, 0], X[:, 1], c=y,
                    cmap=ListedColormap(['r', 'y', 'b']),
                    edgecolor='k', s=20)
        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of the Iris dataset", fontsize=12)
plt.axis("tight")
plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
plt.show()


# # Accuracy score output

# In[ ]:


dict={'Logistic_Regression' : [acc_logistic],
     'KNN' : [acc_knn],
     'SVM' : [acc_svm],
     'Naive_Bayes' : [acc_bayes],
     'Decision_Tree' : [acc_dec],
     'Random_Forest' : [acc_ran]
     }
models = pd.DataFrame.from_dict(dict,orient='index')
models.transpose()



# In[ ]:


models.to_csv('mycsvfile.csv',index=False)



# # Classification report of all the ML Models

# In[ ]:


print("Classification_Report of Logistic Regression : \n",classification_report(y_test, y_pred_forest, target_names=target_names))
print("Classification_Report of SVM : \n",classification_report(y_test, y_pred_svc, target_names=target_names))
print("Classification_Report of KNN : \n",classification_report(y_test, y_pred_neigh, target_names=target_names))
print("Classification_Report of Naive Bayes : \n",classification_report(y_test, y_pred_bayes, target_names=target_names))
print("Classification_Report of Decision Tree : \n",classification_report(y_test, y_pred_tree, target_names=target_names))
print("Classification_Report of Random Forest : \n",classification_report(y_test, y_pred_forest, target_names=target_names))


# # Cross_val_score of ML Models

# In[ ]:



print("Logistic Regression: ")
print(cross_val_score(lg_class, X_train, y_train, scoring='accuracy', cv = 10))
accuracy = cross_val_score(lg_class, X_train, y_train, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of Logistic regression train set is: " , accuracy)
print(cross_val_score(lg_class, X_test, y_test, scoring='accuracy', cv = 10))
accuracy_test = cross_val_score(lg_class, X_test, y_test, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of Logistic regression test set is: " , accuracy_test)


# In[ ]:



print("KNN: ")
print(cross_val_score(k_class, X_train, y_train, scoring='accuracy', cv = 10))
accuracy = cross_val_score(k_class, X_train, y_train, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of KNN train set is: " , accuracy)
print(cross_val_score(k_class, X_test, y_test, scoring='accuracy', cv = 10))
accuracy_test = cross_val_score(k_class, X_test, y_test, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of Logistic regression test set is: " , accuracy_test)


# In[ ]:



print("SVM: ")
print(cross_val_score(svm_class, X_train, y_train, scoring='accuracy', cv = 10))
accuracy = cross_val_score(svm_class, X_train, y_train, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of SVM train set is: " , accuracy)
print(cross_val_score(svm_class, X_test, y_test, scoring='accuracy', cv = 10))
accuracy_test = cross_val_score(svm_class, X_test, y_test, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of SVM test set is: " , accuracy_test)


# In[ ]:



print("Naive Bayes: ")
print(cross_val_score(n_class, X_train, y_train, scoring='accuracy', cv = 10))
accuracy = cross_val_score(n_class, X_train, y_train, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of Naive Bayes train set is: " , accuracy)
print(cross_val_score(n_class, X_test, y_test, scoring='accuracy', cv = 10))
accuracy_test = cross_val_score(n_class, X_test, y_test, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of Naive Bayes test set is: " , accuracy_test)


# In[ ]:



print("Decision Tree: ")
print(cross_val_score(d_class, X_train, y_train, scoring='accuracy', cv = 10))
accuracy = cross_val_score(d_class, X_train, y_train, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of Decision Tree train set is: " , accuracy)
print(cross_val_score(d_class, X_test, y_test, scoring='accuracy', cv = 10))
accuracy_test = cross_val_score(d_class, X_test, y_test, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of Decision Tree test set is: " , accuracy_test)


# In[ ]:



print("Random Forest Classification: ")
print(cross_val_score(ran_class, X_train, y_train, scoring='accuracy', cv = 10))
accuracy = cross_val_score(ran_class, X_train, y_train, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of Logistic regression train set is: " , accuracy)
print(cross_val_score(ran_class, X_test, y_test, scoring='accuracy', cv = 10))
accuracy_test = cross_val_score(ran_class, X_test, y_test, scoring='accuracy', cv = 10).mean() * 100
print("Accuracy of Logistic regression test set is: " , accuracy_test)


# # Model Selection - KFold for SVM

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.svm import SVR
scores = []
best_svr = SVR(kernel='rbf')
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_svr.fit(X_train, y_train)
    scores.append(best_svr.score(X_test, y_test))
    


# In[ ]:


print(np.mean(scores))


# # Clustering algorithms

# In[ ]:


d = data.iloc[:,0:4].values
d
#X=data.iloc[:,0:4].values
#y=data.iloc[:,4].values


# # K-Means Clustering

# In[ ]:


from sklearn.cluster import KMeans
wcss={}
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=1000).fit(d)
    wcss[i]=kmeans.inertia_
plt.figure()
plt.plot(list(wcss.keys()),list(wcss.values()))
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=3).fit(d)
centroids = kmeans.cluster_centers_
y_kmeans=kmeans.fit_predict(d)
print(centroids)
plt.scatter(d[y_kmeans==0,0], d[y_kmeans==0,1], c= 'red',s=50, label='Cluster1')
plt.scatter(d[y_kmeans==1,0], d[y_kmeans==1,1], c= 'blue',s=50, label='Cluster2')
plt.scatter(d[y_kmeans==2,0], d[y_kmeans==2,1], c= 'green',s=50, label='Cluster3')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=50,label='centroids')
plt.title('Cluster of flowers')
plt.legend()
plt.show()


# # Hierarchical Clustering

# In[ ]:



#from scipy.cluster.hierarchy import dendrogram, linkage

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(30, 17))  
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(d, method='ward'))


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
y_hc=cluster.fit_predict(d)
plt.scatter(d[y_hc==0,0], d[y_hc==0,1], c= 'red',s=100, label='Cluster1')
plt.scatter(d[y_hc==1,0], d[y_hc==1,1], c= 'blue',s=100, label='Cluster2')
plt.scatter(d[y_hc==2,0], d[y_hc==2,1], c= 'green',s=100, label='Cluster3')
plt.title('Cluster of flowers')
plt.legend()
plt.show()

