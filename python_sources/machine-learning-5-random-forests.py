#!/usr/bin/env python
# coding: utf-8

# **Table of Content**
# * Decision Trees
#     * Decision tree classification
#         * Introduction
#         * Gini impurity
#         * Training algorithm
#         * Overfitting and regularization
#         * Decision tree on the Red Wine Quality dataset
#     * Decision tree regression
#         * Introduction
#         * Example
# * Random Forests
#     * What is a random forest?
#     * Example - random forest classification on the Red Wine Quality dataset
#     * Example - random forest regression on the wine recognition dataset
# * Appendix: CART algorithm
#     * Learning the weights given a fixed structure
#     * Learning the structure
#     * Pruning|

# In[ ]:


import numpy as np
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import os
print(os.listdir("../input"))


# In[ ]:


plt.rc('axes', lw = 1.5)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 5, width = 3)
plt.rc('ytick.major', size = 5, width = 3)


# # Decision Trees
# ## Decision tree classification
# ### Introdution
# Assume that we have an input dataset $X$ with $n$ features (feature 1,2,...,n), and each instance of the dataset has a classification (say, class A, B, and C). A decision tree classifier trained on this dataset with depth = 2 will *approximately* look like this:
# <img src="https://imgur.com/aJLCV2J.png" width="500px"/>
# To make a prediction, the decision tree first compares an instance's feature $k_1$ with threshold $t_{k1}$. If $k_1 \leq t_{k1}$, then the instance is classified as "Class A". If $k_1 > t_{k1}$, then the decision tree checks if this instance's feature $k_2$ is less than or equal to threshold $t_{k2}$. If yes, then the instance is classified as "Class B", otherwise it is classified as "Class C". The node on the very top is called *root node* (depth = 0), and the nodes that do not have any children are called *leaf node*. <br/>
# Let's apply a decision tree classifier to the wine recognition dataset that comes with the sklearn module:

# In[ ]:


from sklearn.datasets import load_wine
wine = load_wine()
X = wine.data
y = wine.target


# We will use sklearn's **DecisionTreeClassifier** to train the decision tree:

# In[ ]:


tree = DecisionTreeClassifier(max_depth = 2, random_state = 0)
tree.fit(X,y)


# To visualize the tree, you will need the **graphviz** module in python. I have already imported this module through 
# > import graphviz
# 
# Run the following code to show the tree graph:

# In[ ]:


dot_data = export_graphviz(tree,
                out_file = None,
                feature_names = wine.feature_names,
                class_names=wine.target_names,
                rounded = True,
                filled = True)

graph = graphviz.Source(dot_data)
graph.render() 
graph


# As you can see, the root node (depth=0) chose "proline" as the splitting feature, and 755.0 as the threshold value. The two depth 1 nodes chose "od280/od315_of_diluted_wines" and "flavanoids" as the their splitting feature, respectively. The four leaf nodes (depth=2) each predicts a class - two of them predict class_2, one predicts class_1 and one predicts class_0. You can also see other node attributes such as "gini"and "value" shown in the graph. These attributes are explained below:
# 
# * In each node, **"samples"** represents how many training instances fall in that node. For example, there are 67 samples that has proline > 755.0. 
# * The attribute **"value"** gives how many instances fall in each class for each node. For example, in the leftmost leaf node, there are 0 instances in class_0, 6 instances in class_1, and 40 instances in class_2. 
# * Each node is also assigned a **"class"** based on which class has the majority of the instances in that node. Therefore, even though the root node has comparable number of instances in each of the three classes, the root node's class label is "class_1" since 71 > 59 > 48. 
# * **To make a prediction, you keep going along the tree until you reach a leaf node. The probability of each class is the fraction of training samples of each class in a leaf node.** For example, the bottom left leaf node predicts class_2 with a probaility of $\frac{40}{46}\approx$87.0%. This attribute can be obtained through DecisionTreeClassifier's *predict_proba()* function. **Note that all samples that fall in the same leaf node share the same prediction probability**.

# ### Gini impurity
# There is another node attribute in the above figure: **Gini impurity**. Gini impurity decribes how "pure" the sample composition is in a node. For example, if a node has 0 samples in class_0 and class_1, but 20 samples in class_2, Gini impurity will be 0. The mathematical definition of gini impurity is:  
# <center>
# $G=1-\sum_{k=1}^{n}p_k^2$
# </center>
# where $p_k$ is the ratio of class k instances in the node. For the bottom left node in the above figure, we have
# <center>
# $G = 1 - (\frac{0}{46})^2 - (\frac{6}{46})^2 - (\frac{40}{46})^2 \approx 0.227.$    
# </center>
# **Gini impurity serves as a criterion for splitting the nodes in decision trees**, which we will talk more about below. 

# ### Training algorithm
# The Classification and Regression Trees (CART) algorithm works by **splitting a node into two children nodes at a time**. The splitting criterion consists of a feature $k$ and a threshold for this feature $t_k$, which are selected by minimizing an impurity cost function:
# <center>
#     $J(k, t_k) = G_{\mathrm {left}}\times{m_{\mathrm {left}}}/{m}+ G_{\mathrm {right}}\times{m_{\mathrm {right}}}/{m}$
# </center>
# This is a **weighted sum of the gini impurity of the left and right child node**. Here $G_{\mathrm {left}}$ ($G_{\mathrm {right}}$) is the gini impurity of the left(right) child node, $m_{\mathrm {left}}$ ($m_{\mathrm {right}}$) is the number of samples in the left (right) child node, and $m$ is the number of samples in the parent node $m = m_{\mathrm {left}} + m_{\mathrm {right}}$. The algorithm will first split the root node into two, then split each of the two children nodes into two, and so on. The process usually stops if the tree has reached its user-defined maximum depth, or if the after-split impurity $G_{\mathrm {left}}\times{m_{\mathrm {left}}}/{m}+ G_{\mathrm {right}}\times{m_{\mathrm {right}}}/{m}$ is bigger than the partent node's impurity $G_{\mathrm{parent}}$, but other stopping criteria exist, too.
# 
# There are several things I'd like to point out about this algorithm and sklearn's DecisionTreeClassifier:
# * The CART algorithm is a greedy algorithm that searches for **locally optimal decisions**. Optimal split is searched at each node, *without* considering whether it leads to a globally smallest impurity at the bottom of the tree. This is also mentioned in DecisionTreeClassifier's [documentation](https://scikit-learn.org/stable/modules/tree.html#tree).
# * Given the algorithm, which DecisionTreeClassifier follows if using default settings, a deterministic result is expected given a certain training set (i.e. no randomness). However, **DecisionTreeClassifier does display randomness even at default settings**. This is because the classifier will randomly re-order all features and then test each of them. If two splits are tied, the one that happened first will be selected. More discussion can be found on this [github page](https://github.com/scikit-learn/scikit-learn/issues/8443).
# * To accelerate to training, you may set max_features of DecisionTreeClassifier to be smaller than the total number of features. In this case, the algorithm will search the best split from a randomly sampled subset of features.
# 
# 

# ### Overfitting and regularization
# Without regulation, **decision tree classifier will easily overfit**. The classifier will try hard to get classification 100% correct, which often results in leaf nodes with very few samples. In this section, we will see what decision tree overfitting looks like and how to fix it. To help with visualization, we will train the classifier on just two features: flavanoids and proline levels.

# In[ ]:


wine = load_wine()
X = wine.data[:,[6,12]] # flavanoids and proline
y = wine.target

# random_state is set to guarantee consistent result. You should remove it when running your own code.
tree1 = DecisionTreeClassifier(random_state=5) 
tree1.fit(X,y)


# We can plot the decision boundaries as follows:

# In[ ]:


# preparing to plot the decision boundaries
x0min, x0max = X[:,0].min()-1, X[:,0].max()+1
x1min, x1max = X[:,1].min()-10, X[:,1].max()+10
xx0, xx1 = np.meshgrid(np.arange(x0min,x0max,0.02),np.arange(x1min, x1max,0.2))
Z = tree1.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z = Z.reshape(xx0.shape)


# In[ ]:


plt.subplots(figsize=(12,10))
plt.contourf(xx0, xx1, Z, cmap=plt.cm.RdYlBu)
plot_colors = "ryb"
n_classes = 3
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)
plt.legend(fontsize=18)
plt.xlabel('flavanoids', fontsize = 18)
plt.ylabel('proline', fontsize = 18)
plt.show()


# As you can see, the decision tree classifier is heavily overfitting. Though it fits to the training set perfectly, it will not generalize well. If you plot the tree, you can see that the tree is very deep (depth = 8). There are several parameters of DecisionTreeClassifier that you can adjust to regularize the classifier:  
# * max_depth: decrease tree depth.
# * min_samples_split: increase the minimum samples a node needs to have to be split.
# * min_samples_leaf: increase the minumum samples a leaf node must have.
# * max_leaf_nodes: decrease the maximum number of leaf nodes.
# * max_features: decrease the maximum number of features to search through at each split (default is to search through all features).
# * ...
# 
# Here I will show results from regularizing **max_depth** and **max_leaf_nodes**, respectively:

# In[ ]:


# limit maximum tree depth
tree1 = DecisionTreeClassifier(max_depth=3,random_state=5) 
tree1.fit(X,y)

# limit maximum number of leaf nodes
tree2 = DecisionTreeClassifier(max_leaf_nodes=4,random_state=5) 
tree2.fit(X,y)

x0min, x0max = X[:,0].min()-1, X[:,0].max()+1
x1min, x1max = X[:,1].min()-10, X[:,1].max()+10
xx0, xx1 = np.meshgrid(np.arange(x0min,x0max,0.02),np.arange(x1min, x1max,0.2))

Z1 = tree1.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z1 = Z1.reshape(xx0.shape)
Z2 = tree2.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z2 = Z2.reshape(xx0.shape)

fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
ax[0].contourf(xx0, xx1, Z1, cmap=plt.cm.RdYlBu)
ax[1].contourf(xx0, xx1, Z2, cmap=plt.cm.RdYlBu)
plot_colors = "ryb"
n_classes = 3
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    ax[0].scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)
    ax[1].scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)
ax[0].legend(fontsize=14)
ax[0].set_xlabel('flavanoids', fontsize = 18)
ax[0].set_ylabel('proline', fontsize = 18)
ax[0].set_ylim(260,1690)
ax[0].set_title('max_depth = 3', fontsize = 14)
ax[1].legend(fontsize=14)
ax[1].set_xlabel('flavanoids', fontsize = 18)
ax[1].set_ylabel('proline', fontsize = 18)
ax[1].set_ylim(260,1690)
ax[1].set_title('max_leaf_nodes = 4', fontsize = 14)
plt.show()


# The classifier *tree1* (limit maximum tree depth) and *tree2* (limit maximum number of leaf nodes) both have a more regularized behaviour compared to the classifier without any regulation. Or as people like to say, the regularized classifiers have **lower variance (less overfitting) and higher bias (more prediction error)** than the default classifier. **You should always regularize your decision tree classifier to make it generalize well to test sets**. However, which parameter to regularize and to what extent depend on the specific goal of the analysis.  
# Before we proceed, I would like to note that **there is usually no need to standardize the data before training a decision tree classifier**, as the algorithm focuses on one feature at a time.

# ### Decision tree classifier on the Red Wine Quality dataset
# I will now train decision tree classifiers on the Red Wine Quality dataset which we previously used in the [logistic regression tutorial](https://www.kaggle.com/fengdanye/machine-learning-3-logistic-and-softmax-regression) (note that this is NOT the dataset we used in above illustration, even though both datasets are related to wine). Let's read the data first:

# In[ ]:


wineData = pd.read_csv('../input/winequality-red.csv')
wineData.head()


# Just like what we did in the logistic regression tutorial, **we define wine as 'good'(1) if its quality is larger than or equal to 7, and 'not good'(0) otherwise**:

# In[ ]:


wineData['category'] = wineData['quality'] >= 7
wineData.head()


# In[ ]:


X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)


# We then carry out a train-test split of the dataset, with 30% samples in the test set and 70% in the training set. The 'random_state' parameter is fixed for repeatable results, and you may remove it when running the code yourself.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)


# As mentioned earlier, there are multiple parameters that can regulate the classifier. Here I focus on three of them:
# * max_features: maximum number of features to consider at each node. A float number means the proportion of the features being considered. If max_features < 1.0, the algorithm searches for the best feature among a random subset of features at each node.
# * max_depth: maximum tree depth. According to the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), by default, max_depth=None, meaning "nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples". But here we want to limit it to a specific number.
# * min_samples_leaf: minimum number of samples in each leaf node. By default, min_samples_leaf=1.
# 
# To compare classifiers with different combinations of values of the above parameters, we use sklearn's **GridSearchCV** function, which was introduced in the last section of [my previous tutorial](https://www.kaggle.com/fengdanye/machine-learning-4-support-vector-machine). Here I will repeat what I typed in that tutorial:  
# > We will use Scikit-Learn's GridSearchCV function to conduct model selection. This function iterates through all possible combinations of hyperparameters and runs cross-validation on each combination (i.e. model). The best model is the one that produces the best score during cross validation. Note that during cross-validation, GridSearchCV will split (X_train,y_train) further into train and test set. Once the best model is found, GridSearchCV train the best model on the whole train data (X_train,y_train), and it is this trained best model that will be used for further prediction.
# 
# Here, we will choose AUC (area under the ROC curve) as the score of interest. You can go to the[ logistic regression](https://www.kaggle.com/fengdanye/machine-learning-3-logistic-and-softmax-regression) tutorial to learn about ROC curve and AUC.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


tuned_parameters = {'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 'max_depth': [2,3,4,5,6,7],'min_samples_leaf':[1,10,100],'random_state':[14]} 
# random_state is only to ensure repeatable results. You can remove it when running your own code.

clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)


# You can use print(clf.cv_results_) to produce a detailed report of each classifier's performance. I am not showing the report here since we are testing $6\times6\times3=108$ classifiers, making the report rather long. Let's print the best model and its performance on the training set:

# In[ ]:


print('The best model is: ', clf.best_params_)
print('This model produces a mean cross-validated score (auc) of', clf.best_score_)


# Now let's look at the classifier's performance on the test set (X_test, y_test):

# In[ ]:


from sklearn.metrics import precision_score, accuracy_score
y_true, y_pred = y_test, clf.predict(X_test)
print('precision on the evaluation set: ', precision_score(y_true, y_pred))
print('accuracy on the evaluation set: ', accuracy_score(y_true, y_pred))


# * Among all wine predicted to be good, 55.8% are acutually good.
# * 87.3% of the wine have their quality predicted correctly.  
# 
# Now let's plot the ROC curve and calculate AUC on the test set:

# In[ ]:


from sklearn.metrics import auc
from sklearn.metrics import roc_curve
phat = clf.predict_proba(X_test)[:,1]


# In[ ]:


plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()


# In[ ]:


print('AUC is: ', auc(fpr,tpr))


# ## Decision tree regression
# ### Introduction
# Decision trees can also perform regression tasks. The algorithm still works by dividing a node into two children nodes at a time with a feature $k$ and its threshold $t_k$, but with a different cost function:
# <center>
#     $J(k, t_k) = \mathrm {MSE_{left}}\times{m_{\mathrm {left}}}/{m}+ \mathrm {MSE_{right}}\times{m_{\mathrm {right}}}/{m}$
# </center>
# where
# <center>
#     $\mathrm {MSE_{left (right)}} =\frac{1}{m_{left(right)}} \sum_{i\in \mathrm{left (right)}}(\hat{y}_{\mathrm {left (right)}}-y^{(i)})^2$    
# </center>
# <center>
#     $\hat{y}_{\mathrm{left(right)}} = \frac{1}{m_{\mathrm{left(right)}}} \sum_{i \in \mathrm{left(right)}} y^{(i)}$
# </center>
# The training process is explained step by step below to help understand the above equations:
# * The algorithm chooses a feature $k$ and a threshold $t_k$ and splits samples in the parent node into a left node ($k$ values <= $t_k$) and right node ($k$ values > $t_k$).
# * The predicted $y$ value for the left node $\hat{y}_{\mathrm{left}}$ is the mean of $y$ values of all samples in the left node. Similarly, the predicted $y$ value for the right node $\hat{y}_{\mathrm{right}}$ is the mean of $y$ values of all samples in the right node.
# * The algorithm calculates the mean squared error within each node: $\mathrm{MSE_{left}}$ and $\mathrm{MSE_{right}}$.
# * The algorithm searches all possible ($k, t_k$) to find the split that minimizes the cost function $J(k, t_k)$.
# * The algorithm proceeds to split the children nodes.
# * ...
# 
# Basically, instead of penalizing impurity, decision tree regression algorithm penalizes mean squared errors (MSE). Let's look at an simple example:

# ### Example
# Let's reuse the wine recognition dataset that comes with sklearn. Here, we will take the flavanoid values as input "$x$", and proline values as the output "$y$".

# In[ ]:


wine = load_wine()
x = wine.data[:,6] # flavanoids
y = wine.data[:,12] # proline


# In[ ]:


plt.scatter(x,y)
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()


# To run decision tree regression, we can use sklearn's **DecisionTreeRegressor** function:

# In[ ]:


x = x.reshape(-1,1)
tree = DecisionTreeRegressor(max_depth = 2, random_state = 5) # max tree depth is limited to 2
tree.fit(x,y)


# In[ ]:


dot_data = export_graphviz(tree,
                out_file = None,
                feature_names = ['flavanoids'],
                rounded = True,
                filled = True)

graph = graphviz.Source(dot_data)
graph.render() 
graph


# Let's plot the predictions of this trained regressor:

# In[ ]:


xx = np.arange(0,5.3, step = 0.01).reshape(-1,1)
yy = tree.predict(xx)

plt.scatter(x,y)
plt.plot(xx,yy,color='r')
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()


# This regressor is only coarsely fitting the data. What if we try DecisionTreeRegressor with no constrain on tree depth?

# In[ ]:


x = x.reshape(-1,1)
tree = DecisionTreeRegressor(random_state = 5)
tree.fit(x,y)

xx = np.arange(0,5.3, step = 0.01).reshape(-1,1)
yy = tree.predict(xx)

plt.scatter(x,y)
plt.plot(xx,yy,color='r')
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()


# As expected, the regressor is heavily overfitting. It is also very sensitive to outliers as it tries to predict every value perfectly. Let's try to limit the maximum tree depth to 3:

# In[ ]:


x = x.reshape(-1,1)
tree = DecisionTreeRegressor(max_depth = 3, random_state = 5)
tree.fit(x,y)

xx = np.arange(0,5.3, step = 0.01).reshape(-1,1)
yy = tree.predict(xx)

plt.scatter(x,y)
plt.plot(xx,yy,color='r')
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()


# This is certainly better than the overfitting regressor, but would you say that it is a good fit? **One of the major characteristics of a decision tree is that it does not make any assumptions about your data**. This can be a problem sometimes. **A decision tree can be sensitive to outliers**, for example, as you can see in the above figure where the prediction line drops down just to fit the rightmost data point. If the data is noisy, **decision tree might not be able to distinguish between noise and signal**, since there is no underlying model (e.g. linear model, polynomial model) to guide the regression. 
# 
# Regularizing the tree can help with the above issues, but in some cases decision tree regessors just don't work well. In practice, it is a good idea to just try several possible regressors (decision tree, polynomial...) and choose the one that performs the best. An ensemble of decision trees (random forests) also tend to work better than individual trees, which we will talk about next.
# 
# **For more on the CART algorithm for tree training: see the newly added Appendix.**

# # Random Forests
# ## What is a random forest?
# **A random forest is an ensemble of decision trees. In a random forest, each decision tree is trained on a random subset of the training set, usually sampled with replacement, with sample size equal to training set size.  In case of classifications, the class with the highest mean probability across all trees is selected. In case of regressions, the predicted $y$ value is the mean of predicted $y$ values across all trees. **
# 
# In Scikit-learn, [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) is used for random forest classification and [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) is used for random forest regression. As the [documentation](https://scikit-learn.org/stable/modules/ensemble.html#forest) mentioned, "(because of the random subsampling in training) the bias of the forest usually slightly increases (with respect to the bias of a single non-random tree) but, due to averaging, its variance also decreases, usually more than compensating for the increase in bias, hence yielding an overall better model." **Therefore, random forest is generally preferred over a single tree**. **The documentation also noted that unlike DecisionTreeClassifier, at default, RandomForectClassifier does not search all features for the best split, but only a random subset of them.** According to [RandomForestClassifier's documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), max_features is default to sqrt(n_features). 
# 
# Let's try a random forest classifier on the Red Wine Quality dataset:

# ## Example - random forest classification on the Red Wine Quality dataset
# Read data and define binarized wine quality like we did in the earlier section:

# In[ ]:


wineData = pd.read_csv('../input/winequality-red.csv')

wineData['category'] = wineData['quality'] >= 7

X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)


# Then split the dataset into a training set and a test set, with the same random_state as the one used in the "Decision tree classifier on the Red Wine Quality dataset" section:

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)


# We then use scikit-learn's **RandomForestClassfier** to train random forest classifiers. RandomForestClassfier has all the paramters from DecisionTreeClassifier. Just like what we did in the "Decision tree classifier on the Red Wine Quality dataset" section, we will use **GridSearchCV** to test different combinations of hyperparameters such as max_features and max_depth. 
# 
# **RandomForestClassfier** also has its unique paramters such as n_estimators and n_jobs. The n_estimators is the number of trees in the forest, and n_jobs is the number of jobs to run in parallel. Here we set n_estimators to 500, and n_jobs to -1 (meaning to use all possible processors).

# In[ ]:


tuned_parameters = {'n_estimators':[500],'n_jobs':[-1], 'max_features': [0.5,0.7,0.9], 'max_depth': [3,5,7],'min_samples_leaf':[1,10],'random_state':[14]} 
# random_state is only to ensure repeatable results. You can remove it when running your own code.

clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)


# The training might take some time. After it's done, print out the best model and it's mean cross-validated AUC:

# In[ ]:


print('The best model is: ', clf.best_params_)
print('This model produces a mean cross-validated score (auc) of', clf.best_score_)


# Now let's see how this model perfroms on the test set:

# In[ ]:


from sklearn.metrics import precision_score, accuracy_score
y_true, y_pred = y_test, clf.predict(X_test)
print('precision on the evaluation set: ', precision_score(y_true, y_pred))
print('accuracy on the evaluation set: ', accuracy_score(y_true, y_pred))


# In[ ]:


from sklearn.metrics import auc
from sklearn.metrics import roc_curve
phat = clf.predict_proba(X_test)[:,1]


# In[ ]:


plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()


# In[ ]:


print('AUC is: ', auc(fpr,tpr))


# The result is much better than a single decision tree.

# ## Example - random forest regression on the wine recognition dataset
# Rememer earlier we tested decision tree regressor on the following dataset:

# In[ ]:


wine = load_wine()
x = wine.data[:,6] # flavanoids
y = wine.data[:,12] # proline

plt.scatter(x,y)
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()


# Now let's try a random forest regressor on this same dataset. We will again use 500 trees. Each tree has a maximum depth of 2:

# In[ ]:


reg = RandomForestRegressor(n_estimators=500, n_jobs=-1, max_depth=2, random_state = 5)
x = x.reshape(-1,1)
reg.fit(x,y)


# In[ ]:


xx = np.arange(0,5,0.02).reshape(-1,1)
yhat = reg.predict(xx)
plt.plot(xx,yhat,color='red')
plt.scatter(x,y)
plt.xlabel('flavanoids',fontsize=16)
plt.ylabel('proline',fontsize=16)
plt.show()


# The prediction is much smoothier, but it is still affected by the outliers.

# ----------------------------------------------------
# # Appendix: CART algorithm
# Earlier, we introduced the CART algorithm for training decision tree classifiers (using gini impurity) and regressors (using MSE). Here, we will introduce the CART algorithm in a more general sense. <font color='blue'>The understanding of this algorithm is important for underdanding the boosting algorithm that we will introduce in tutorial 7</font>. The content of this Appendix is based on [this Master thesis](https://brage.bibsys.no/xmlui/handle/11250/2433761) by Didrik Nielsen.
# 
# **The CART algorithm trains the tree by splitting a node into two children nodes at a time. Among all the possible splits at each step (node to split, feature to split on, threshold of feature), the one that minimizes the training error is chosen**. 
# 
# A tree can be expressed as follows: 
# <center>
# $f(x) = \sum_{j=1}^{T} w_j {\mathrm I}(x\in R_j)$
# </center>
# Where $w_j$ is called the **weight** of *j*th leaf node, and $\{R_j\} ( j=1,2,...,T)$ is called the **structure** of the tree. The $T$ is the number of leaf nodes of this tree. The ${\mathrm I}(x\in R_j)$ equals one if sample $x$ belongs to area $R_j$, and zero otherwise. Therefore, $f(x)$ will predict $w_j$ if the sample $x$ falls in area $j$. In the figure below, we have $T=4$. The sample space is divided into four disjoint areas: $R_1=\{x: x\leq 1.21\}$, $R_2=\{x: 1.21 < x\leq 2.31\}$, $R_3=\{x: 2.31 < x \leq 3.185 \}$, and $R_4=\{x: x>3.185 \}$. Here $x$ is the flavanoids value. The weights for the four areas are: $w_1=647.889$, $w_2=525.357$, $w_3=882.193$, and $w_4=1204.35$.
# <img src="https://imgur.com/YDuPS3X.png" width="500px"/>

# Now, assuming we have $n$ samples in the training set, our goal is to train a tree to minimize the cost function:
# <center>
# $J(f)=\sum_{i=1}^{n}L(y_i, f(x_i))=\sum_{i=1}^{n}L(y_i, \sum_{j=1}^{T}w_j {\mathrm I}(x\in R_j))$
# </center>
# Where $y_i$ is the true response of sample $x_i$. **Generally speaking, the training consists of three stages: (1) learning the weights given a fixed structure; (2) learning the structure; (3) pruning**. We will go through them one by one.  

# ## Learning the weights given a fixed structure
# Given the disjoint nature of the areas $R_j$, and the fact that the union of all areas gives the complete sample space, the cost function above can be further written as:
# <center>
# $J(f)=\sum_{j=1}^{T}\sum_{x_i \in R_j}L(y_i, w_j)$
# </center>
# Given a fixed tree structure (i.e. all $R_j$s are defined and fixed, and $T$ is fixed), minimizing $J(f)$ is equivalent to minimizing $\sum_{x_i \in R_j}L(y_i, w_j)$ for each $j$. Therefore, given a fixed structure, we have
# <center>
# $w_j^* = \underset{w}{\operatorname{argmin}}\sum_{x_i \in R_j}L(y_i, w)$
# </center>
# For a <font color='blue'>squared loss function</font>, we have $w_j^* = \underset{w}{\operatorname{argmin}}\sum_{x_i \in R_j}(y_i-w)^2$. Setting $\frac{\partial \sum_{x_i \in R_j}(y_i-w)^2}{\partial w}$ to $0$ gives:
# <center>
# $w_j^* = \frac{\sum_{x_i \in R_j}y_i}{n_j}$
# </center>
# where $n_j$ is the number of samples that fall in area $R_j$. **That is, with a squared loss function, the estimated weight $w_j$ is simply the average of repsonses in the region $R_j$. This is exactly what we described in the Decision Tree Regression section**. If the loss function $L$ is defined differently, $w_j^*$ will have different expressions. For example, if we use absolute loss. where $L(y_i,w)=|y_i-w|$, $w_j^*$ should be the median of the responses in area $R_j$.
# 

# ## Learning the structure
# Now that we know what the weights should be given a structure, the cost function can be written as:
# <center>
# $J(f)=\sum_{j=1}^{T}\sum_{x_i \in R_j}L(y_i, w_j^*)  \equiv \sum_{j=1}^{T} L_j^*$
# </center>
# where $L_j^*$ is called the aggregated loss at leaf node $j$. **Note that the estimated weight $w_j^*$ is used**. Imagine that we are training a tree and we are considering the next split on leaf node $k$. Before this spit takes place, the overall cost is:
# <center>
# $J_{\mathrm {before}}= \sum_{j\neq k} L_j^* + L_k^*$
# </center>
# After the split on leaf node $k$, we obtain a left child node ("L") and a right child node ("R"). The new cost is:
# <center>
# $J_{\mathrm {after}}= \sum_{j\neq k} L_j^* + L_L^*+L_R^*$
# </center>
# The <font color='blue'>gain</font> of the considered split is defined as:
# <center>
# ${\mathrm {Gain}}=J_{\mathrm {before}} - J_{\mathrm {after}} = L_k^* - (L_L^*+L_R^*)$ 
# </center>
# The larger the gain, the bigger the decrease in the cost function $J$. **At every step of tree training, the gain for every possible split (node to split, feature to split on, threshold of feature) is calculated, and the split that maximizes the gain is chosen**. Since $L_k^*$ is a constant when considering different splits, we just need to minimize $(L_L^*+L_R^*)$. In the case of <font color='blue'>squared loss</font>, we have:
# <center>
# $L_L^*+L_R^* = \sum_{x_i \in \mathrm{left}}(y_i-\bar{y}_{\mathrm{left}})^2 + \sum_{x_i \in \mathrm{right}}(y_i-\bar{y}_{\mathrm{right}})^2 = n_{\mathrm{left}}\mathrm{MSE}_{\mathrm{left}} +  n_{\mathrm{right}}\mathrm{MSE}_{\mathrm{right}}$ 
# </center>
# where $n_{\mathrm{left(right)}}$ is the number of training samples in the left(right) node. The $\bar{y}_{\mathrm{left(right)}}$ is just the estimated weight for the left(right) node under squared loss, as we have shown in the previous step. **Note that this quantity to minimize is proportional to the cost function $J(k,t_k)$ we defined in the Decision Tree Regression section. That is, everything we defined in that section is just a specific case where the general CART algorithm is applied with a squared loss.**

# ## Pruning
# During tree training, we can give up on splitting if all possible gain is negative. However, this might be too local of a decision, since further splits after the current split can have positive gain and overall decreases the cost. To address this issue, trees are typically grown until certain stopping criteria is met (e.g. maximum number of leaf nodes is reached). During the growing period, an optimized split with maximized gain is carried out at each step, even if the gain is negative. **After tree growing, nodes with negative gain are removed in a bottom-up fashion, thus further reduces the overall cost**. This technique of cutting back the tree is called **pruning**.

# In tutorial 7, we will talk about tree boosting algorithms (gradient boosting machine and XGBoost). Understanding how the above algorithm works is important for understanding tree boosting. 

# ----------------------------------------------------------
# Update Logs:  
# 
# **2019-01-18**
# * I noticed a " timeout by a memory leak" warning while running GridSearchCV using RandomForestClassifier(). This problem did not happen in any of the previous commits. Somehow the computing capability of Kaggle platform changed in the past two weeks? Anyways, I have fixed the problem by reducing the searched parameters from "'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 'max_depth': [2,3,4,5,6,7],'min_samples_leaf':[1,10,100]" to "'max_features': [0.5,0.7,0.9], 'max_depth': [3,5,7],'min_samples_leaf':[1,10]". This does not affect the training outcome.
# * Added an appendix to explain the CART algorithm for decision tree training. This is also important for understanding of the boosting algorithm introduced in Tutorial #7.

# ---------------------------------------------
# Please upvote the notebook if you enjoyed this tutorial :) If you have any questions or comments, let me know!
# 
# For my previous tutorials, please see:
# * [Machine Learning 1 - Regression, Gradient Descent](https://www.kaggle.com/fengdanye/machine-learning-1-regression-gradient-descent)
# * [Machine Learning 2 Regularized LM, Early Stopping](https://www.kaggle.com/fengdanye/machine-learning-2-regularized-lm-early-stopping)
# * [Machine Learning 3 Logistic and Softmax Regression](https://www.kaggle.com/fengdanye/machine-learning-3-logistic-and-softmax-regression)
# * [Machine Learning 4 Support Vector Machine](https://www.kaggle.com/fengdanye/machine-learning-4-support-vector-machine)
