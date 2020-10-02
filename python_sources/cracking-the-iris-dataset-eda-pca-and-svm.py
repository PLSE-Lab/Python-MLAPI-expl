#!/usr/bin/env python
# coding: utf-8

# <h1>Classifying the Iris dataset using **Support Vector Machines**  (SVMs)</h1>
# 
# In this tutorial I are going to show how to explore and classify the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) and also how to analyse the results of classification using SVMs. In this tutorial i will be using Support vector machines with dimentianility reduction techniques like PCA and Scalers to classify the dataset efficiently.
# 1. Data Exploration and Cleaning.
# 2. Dimentality Reduction
# 3. Algorithm Choice

# <h2>Data exploration and cleaning</h2>
# Data exploration is very important befoe trying to classify or fit the data as it will help you choose the best algorithm to use. Data exploration will also till you about the initial connfiguration of some hyper parameters like the degree of polynomial or the type of the kernel.
# Now let's import the libraries we are going to use in this stage and then load the data from the dataset. The X represents the **features matrix** and the y represents the **target vector**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import pandas as pd

# mapping the 3 target classes names to numeric 
# values to be able to use them later in our model
#since our model only accepts numeric numbers
X = pd.read_csv("../input/Iris.csv")

# To map the strings you need to use map(dict) function 
# from the dataframe, This function accepts a dictionary 
# with the values to be replaced as Keys and the values 
# that would replace them as the values for those keys 
z = {'Iris-setosa' : 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3 }

X['Species'] = X['Species'].map(z)
print ("Number of data points ::", X.shape[0])
print("Number of features ::", X.shape[1])


# In[ ]:


X.head()


# Now let's discover the seperability of data and the relation between features using plots, histograms and heatmaps

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
# X.species holds the classes that we have after mapping which are {1, 2, 3}
classes = np.array(list(X.Species.values))

# Now we will use matplotlib to plot the classes with the axis x and y representing two of the features that we have
# We do that to see if some features contribute to the seperablity of the dataset more than the others
def plotRelation(first_feature, sec_feature):
    
    plt.scatter(first_feature, sec_feature, c = classes, s=10)
    plt.xlabel(first_feature.name)
    plt.ylabel(sec_feature.name)
    
f = plt.figure(figsize=(25,20))
f.add_subplot(331)
plotRelation(X.SepalLengthCm, X.SepalWidthCm)
f.add_subplot(332)
plotRelation(X.PetalLengthCm, X.PetalWidthCm)
f.add_subplot(333)
plotRelation(X.SepalLengthCm, X.PetalLengthCm)
f.add_subplot(334)
plotRelation(X.SepalLengthCm, X.PetalWidthCm)
f.add_subplot(335)
plotRelation(X.SepalWidthCm, X.PetalLengthCm)
f.add_subplot(336)
plotRelation(X.SepalWidthCm, X.PetalWidthCm)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
# X.species holds the classes that we have after mapping which are {1, 2, 3}
classes = np.array(list(X.Species.values))

# Now we will use matplotlib to plot the classes with the axis x and y representing two of the features that we have
# We do that to see if some features contribute to the seperablity of the dataset more than the others
import seaborn as sns

Exploration_columns = X.drop('Id' ,   axis = 1)
sns.pairplot(Exploration_columns, hue = "Species")


# It is time to see the correlation between features and the label vector as well.

# In[ ]:


import seaborn as sns

# Here we use the seaborn library to visualize the correlation matrix
# The correlation matrix shows how much are the features and the target correlated
# This gives us some hints about the feature importance
import matplotlib.pyplot as plt


corr = X.corr()
f, ax = plt.subplots(figsize=(15, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5)


# From the previous Heatmap of correlation it seems that all features have either strong positive or negative correlation with the Target(label) vector species that we want to classify.

# Let's now check the histograms of the data features to know more about the data stastically

# In[ ]:


f = plt.figure(figsize=(25,10))
f.add_subplot(221)
X.SepalWidthCm.hist()
f.add_subplot(222)
X.SepalLengthCm.hist()
f.add_subplot(223)
X.PetalLengthCm.hist()
f.add_subplot(224)
X.PetalWidthCm.hist()


# Now we will explore outliers in the data using boxplot and zscores methods. We will consider any point with score greater than 2.5 std as an outlier. you can know more about this [here](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba).

# In[ ]:


f = plt.figure(figsize=(25,10))
f.add_subplot(221)
sns.boxplot(x=X['PetalWidthCm'])
f.add_subplot(222)
sns.boxplot(x=X['PetalLengthCm'])
f.add_subplot(223)
sns.boxplot(x=X['SepalLengthCm'])
f.add_subplot(224)
sns.boxplot(x=X['SepalWidthCm'])


sns.boxplot(x=X['PetalWidthCm'])


# It seems we have some outliers in the petalWidthCm so we will make further outlier analysis using the zscore method

# In[ ]:


from scipy import stats
import numpy as np

z = np.abs(stats.zscore(X))

zee = (np.where(z > 2.5))[1]

print("number of data examples greater than 3 standard deviations = %i " % len(zee))


# In[ ]:


data_delete = X[(z >= 2.5)]
data_delete.drop_duplicates(keep='first', inplace=True)
unique, count= np.unique(data_delete["Species"], return_counts=True)
print("The number of occurances of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )


# In[ ]:


X = X[(z <= 2.5)]


# Many machine learning algorithms require the data to be in normal distribution that is the Mean equals 0 and with unit vairance. This will help the features to represent the data better since a feature with high variance will not dominate the others, to know more about that refere to the Sklearn doumentation [here](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) .
# Here we will use the **Standard Scaler** to transform the data.

# In[ ]:


# Removing the label from the training data
y = X['Species']
X = X.drop(["Species"], axis = 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# <h2>Dimentionality Reduction</h2>
# Dimentionality reduction is a really important concept in Machine Learning since it reduces the number of features in a dataset and hence reduces the computations needed to fit the model. PCA is one of the well known efficient dimentaniolaty reduction techniques. in this tutorial we will use  **PCA** which compreses the data by projecting it to a new subspace that can help in reducing the effect of the **curse of dimentionality**.
# Our dataset consists of 4 dimentions(4 features) so we will project it to a 3 dimentions space and Plot in in a 3d graph.

# In[ ]:


fig = plt.figure(1, figsize=(16, 9))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X_scaled)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
print("The number of features in the new subspace is " ,X_reduced.shape[1])


# Below is an interactive 3d visualization after performing PCA with three components

# In[ ]:


import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2], mode='markers', marker=dict( size=4, color=y, colorscale= "Portland", opacity=0.))])

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# Now les't perform the train test split on the transformed data. We will use the convention that splits the data in 80% training/validation and 20% test set. We set the random state to a constant value in order to get consistent results when we rerun the code.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                        X_reduced, y, test_size=0.2, random_state=42)


# ## Algorithm choice
# Since we have only 150 training examples then I will try **SVMs**. If you don't know which estimator or algorithm to use you can check the Scikit Learn Cheat sheet below.
# ![](http://scikit-learn.org/stable/_static/ml_map.png)

# Let's now try fitting a Linear SVM that is imported from sklearn.

# In[ ]:


from sklearn.svm import LinearSVC

clf = LinearSVC(penalty='l2', loss='squared_hinge',
                dual=True, tol=0.0001, C=100, multi_class='ovr',
                fit_intercept=True, intercept_scaling=1, class_weight=None,verbose=0
                , random_state=0, max_iter=1000)
clf.fit(X_train,y_train)

print('Accuracy of linear SVC on training set: {:.2f}'.format(clf.score(X_train, y_train)))

print('Accuracy of linear SVC on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# It's time to tune the C parameter and see if we can reach better results. we will perform the tuning automatically using grid search algorithm that is imported from scikit learn. we will use exponential range for c parameters with base = 1.02.

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
    
c = np.logspace(start = -15, stop = 1000, base = 1.02)
param_grid = {'C': c}


grid = GridSearchCV(clf, param_grid =param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid.fit(X_train, y_train)
  
print("The best parameters are %s with a score of %0.0f" % (grid.best_params_, grid.best_score_ * 100 ))
print( "Best estimator accuracy on test set {:.2f} ".format(grid.best_estimator_.score(X_test, y_test) * 100 ) )


# The training results are not improved after tuning the c parameter with grid search and the test score is still 100 which is a good indication that the model is not overfitting nor underfitting. but we can still try improve the training accuracy to ensure that it will catch more test data if new data were intrerduced as the dataset is really small and that may be a bit tricky.

# Let's not try non linear SVC and see the results

# In[ ]:


from sklearn.svm import SVC

clf_SVC = SVC(C=100.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, 
          probability=False, tol=0.001, cache_size=200, class_weight=None, 
          verbose=0, max_iter=-1, decision_function_shape="ovr", random_state = 0)
clf_SVC.fit(X_train,y_train)

print('Accuracy of SVC on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))

print('Accuracy of SVC on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))


# It seems the training score is a little higher than linear SVC but let's try to tune the parameters using grid search to optmize more

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
    
c_SVC = np.logspace(start = 0, stop = 10, num = 100, base = 2 , dtype = 'float64')
print( 'the generated array of c values')
print ( c_SVC )
param_grid_S = {'C': c_SVC}



print("\n Array of means \n")
clf = GridSearchCV(clf_SVC, param_grid =param_grid_S, cv=20 , scoring='accuracy')
clf.fit(X_train, y_train)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
print(means)

y_true, y_pred = y_test, clf.predict(X_test)
print( '\nClassification report\n' )
print(classification_report(y_true, y_pred))


# It seems the model will not improve further using SVC after tuning the c parameter.
