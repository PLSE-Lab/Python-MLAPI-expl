#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# This notebook aims to be an introduction to basic concepts about how a tree-based algorithm works and how to tune it in order to improve the efficiency of your ML models.
# 
# The choice of focusing on tree-based algorithms is driven by how easy is to understand the basics. Moreover, the hope is that most of the concepts can be transferred to other types of algorithms.
# 
# The goal is, by any means, get to a better performance but rather explore some pitfalls I have encountered in my limited experience. In particular, the most important part of a machine learning project (definition of the problem, data exploration and preparation, which I poorly performed in [this kernel](https://www.kaggle.com/lucabasa/credit-card-default-a-very-pedagogical-notebook/)) is not explored here. In this sense, the chosen dataset is perfect due to the number of entries and of features.
# 
# This notebook is organized as follow:
# 
# * Preprocessing: eliminating features, creating new ones, preparation of training and testing sets
# * Decision Tree: basic functioning and how the key hyperparameters affect the result
# * Random Forest: how is it better than a tree and how to tune it efficiently
# * XGBoost: how it gets so much better and tuning strategies
# * Overfitting: Early stopping and learning curves.
# 
# ## Preprocessing
# 
# We will use the credit card default dataset since it has enough entries and it is clean enough. The goal is to predict if a client is going to default next month or not. The two target classes are imbalanced. Life is not perfect.
# 
# We will not consider features on purpose to speed up the learning of our trees. Just as a test, we will also produce all the possible interactions between the features to observe if the performance changes. We go from 17 to 153 features (and 1 target, the default).

# In[1]:


# standard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#To create more features
from itertools import combinations, product 

#Needed for the tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

#Needed for the forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

#Needed for XGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve

import warnings
warnings.filterwarnings('ignore') #because we are animals


# In[2]:


df = pd.read_csv('../input/UCI_Credit_Card.csv')
df.columns


# In[3]:


df['default.payment.next.month'].value_counts() #77.88%


# This means that our baseline model, which would simply predict 0 every time, has an accuracy of 77.88%. This value is crucial to evaluate our models, it gives us a baseline to confront our results with.

# In[4]:


# This is just to speed up the training
del df['ID']
del df['BILL_AMT5']
del df['BILL_AMT6']
del df['PAY_AMT5']
del df['PAY_AMT6']
del df['PAY_5']
del df['PAY_6']
df.info()


# In[5]:


# Create a bunch of new features
target = df['default.payment.next.month'].copy()
cols = df.columns[:-1]
data = df[cols].copy()

cc = list(combinations(data.columns,2))
datacomb = pd.concat([data[c[0]] * data[c[1]] for c in cc], axis=1, keys=cc)
datacomb.columns = datacomb.columns.map(''.join)

df_large = pd.concat([data, datacomb, target], axis = 1)
df_large.head()


# In[6]:


y = df['default.payment.next.month'].copy()
X = df.copy().drop('default.payment.next.month', axis = 1)
XL = df_large.copy().drop('default.payment.next.month', axis = 1)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=895)
XL_train, XL_test, yL_train, yL_test = train_test_split(XL, y, test_size=0.33, random_state=895)


# ## How a tree works <a class="anchor" id="tree"></a>
# 
# We will focus on DecisionTreeClassifiers, which are the simplest thing to imagine. 
# 
# The core idea is to recursively split a sample of the data with the best possible choice until some conditions are met.
# 
# The most common way of evaluating the splits is by using the Gini Index. It measures how mixed the sample is. 
# 
# The core idea is:
# 
# * You put all the data in a node, you test every possible split by splitting the dataset based on every possible value of every possible feature.
# * You pick the best split (the one reducing the impurity the most) and divide your data in two.
# * Repeat until a condition is reached (maximum number of splits, minimum number of samples per node, minimum increase in purity of the sample, no more features to split on, etc.)
# 
# This means that a DecisionTree is a **greedy and exhaustive algorithm**. Thus it does take the best decision every time, but not necessarily the best  for the final result.
# 
# The advantages are:
# 
# * Easy to explain, being a white box algorithm.
# * It can learn non-linear relationships.
# * It is robust to outliers.
# 
# However, it is easy to fall into the overfitting trap. This is why ensemble methods are used.
# 
# In sklearn it is very easy to build a tree and it gives us the opportunity to start learning about the key parameters.

# In[118]:


tree = DecisionTreeClassifier()
tree


# According to the sklearn user guide, the good practices are:
# 
# * With a large number of features, trees tend to overfit. The key is finding the right ratio features to samples.
# * Use **max_depth** = 3 and then monitor how the fitting goes by increasing the depth.
# * **min_sample_split** and **min_sample_leaf** control the number of samples at a leaf node. Small numbers will usually lead to overfitting, large numbers will prevent learning. Try min_sample_leaf = 5 and min_sample_split = 0.5-1% of the total values as initial value. A lower value is usually helpful if we have imbalanced class problems because the leaves where the minority class can be in majority are very small.
# * Balance the dataset before training, either but sampling or by balancing the weight. If the samples are weighted, using **min_weight_fraction_leaf** will ensure that leaf nodes contain at least a fraction of the overall sum.
# * Use **max_features** to control the number of features to consider while searching for the best split. A rule of thumb is to use the square root of the total number.
# 
# Let's see some examples.

# In[119]:


tree.fit(X_train, y_train)
prediction = tree.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))


# With more features we get

# In[120]:


tree.fit(XL_train, yL_train)
prediction = tree.predict(XL_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))


# These are very bad results, considering that a baseline model would get a 77.88% accuracy.
# 
# Both trees are learning with no limits to their depths, let's see how this parameter is influencing the performance. 
# 
# **Exercise**: before peeking at the next cell, what do you expect is it going to happen?

# In[121]:


dp_list = np.arange(3, 30)
train = []
test = []
trainL = []
testL = []

for depth in dp_list:
    tree = DecisionTreeClassifier(max_depth=depth)
    tree.fit(X_train, y_train)
    prediction = tree.predict(X_test)
    trainpred = tree.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    tree.fit(XL_train, yL_train)
    prediction = tree.predict(XL_test)
    trainpred = tree.predict(XL_train)
    train_acc = accuracy_score(yL_train, trainpred)
    test_acc = accuracy_score(yL_test, prediction)
    trainL.append(train_acc)
    testL.append(test_acc)
    
performance = pd.DataFrame({'max_depth':dp_list,'Train_acc':train,'Test_acc':test})
performanceL= pd.DataFrame({'max_depth':dp_list,'Train_acc':trainL,'Test_acc':testL})

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))
x_axis = dp_list
ax1.plot(x_axis, performance['Train_acc'], label='Train')
ax1.plot(x_axis, performance['Test_acc'], label='Test')
ax1.legend()
plt.ylabel('accuracy')
plt.title('Tree accuracy vs depth')
ax2.plot(x_axis, performanceL['Train_acc'], label='Train')
ax2.plot(x_axis, performanceL['Test_acc'], label='Test')
ax2.legend()
plt.ylabel('accuracy')
plt.title('Tree accuracy vs depth')
plt.show()


# As it is expected, the deeper the tree the better it learns the training data and the worse its predictive power gets.
# 
# Now, what about min_sample_leaf and min_sample_split? 

# In[122]:


sam_list = np.arange(1,30)
train = []
test = []
trainL = []
testL = []

for sam in sam_list:
    tree = DecisionTreeClassifier(min_samples_leaf=sam)
    tree.fit(X_train, y_train)
    prediction = tree.predict(X_test)
    trainpred = tree.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    tree.fit(XL_train, yL_train)
    prediction = tree.predict(XL_test)
    trainpred = tree.predict(XL_train)
    train_acc = accuracy_score(yL_train, trainpred)
    test_acc = accuracy_score(yL_test, prediction)
    trainL.append(train_acc)
    testL.append(test_acc)
    
performance = pd.DataFrame({'min_samples_leaf':sam_list,'Train_acc':train,'Test_acc':test})
performanceL= pd.DataFrame({'min_samples_leaf':sam_list,'Train_acc':trainL,'Test_acc':testL})

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))
x_axis = sam_list
ax1.plot(x_axis, performance['Train_acc'], label='Train')
ax1.plot(x_axis, performance['Test_acc'], label='Test')
ax1.legend()
plt.ylabel('Accuracy')
plt.title('Tree accuracy vs min_sample_leaf')
ax2.plot(x_axis, performanceL['Train_acc'], label='Train')
ax2.plot(x_axis, performanceL['Test_acc'], label='Test')
ax2.legend()
plt.ylabel('Accuracy')
plt.title('Tree accuracy vs min_sample_leaf')
plt.show()


# In[123]:


sam_list = np.arange(2,40, 2)
train = []
test = []
trainL = []
testL = []

for sam in sam_list:
    tree = DecisionTreeClassifier(min_samples_split=sam)
    tree.fit(X_train, y_train)
    prediction = tree.predict(X_test)
    trainpred = tree.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    tree.fit(XL_train, yL_train)
    prediction = tree.predict(XL_test)
    trainpred = tree.predict(XL_train)
    train_acc = accuracy_score(yL_train, trainpred)
    test_acc = accuracy_score(yL_test, prediction)
    trainL.append(train_acc)
    testL.append(test_acc)
    
performance = pd.DataFrame({'min_samples_split':sam_list,'Train_acc':train,'Test_acc':test})
performanceL= pd.DataFrame({'min_samples_split':sam_list,'Train_acc':trainL,'Test_acc':testL})

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))
x_axis = sam_list
ax1.plot(x_axis, performance['Train_acc'], label='Train')
ax1.plot(x_axis, performance['Test_acc'], label='Test')
ax1.legend()
plt.ylabel('Accuracy')
plt.title('Tree accuracy vs min_sample_split')
ax2.plot(x_axis, performanceL['Train_acc'], label='Train')
ax2.plot(x_axis, performanceL['Test_acc'], label='Test')
ax2.legend()
plt.ylabel('Accuracy')
plt.title('Tree accuracy vs min_sample_split')
plt.show()


# We thus see how these two parameters function as regulators for our tree.
# 
# Before start tuning our tree, it is better to prepare the folds for our cross-validation.

# Stratified K Fold makes sure that every class is well represented in every fold.

# In[8]:


skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=268)


# One way of tuning the hyperparameters is to perform a grid search, a very exhaustive way of testing many configuration and picking the one that performs better with cross-validation. In the following case, we will grow about 14000 trees.

# In[125]:


#Don't run during class, the result is in the next cell.

param_grid = {'max_depth': np.arange(2,10),
              'min_samples_split' : np.arange(2,20,2),
              'min_samples_leaf' : np.arange(1,21,2),
             'random_state': [42]}

#create a grid
grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'accuracy', n_jobs=-1, cv=skf)

#training
get_ipython().run_line_magic('time', 'grid_tree.fit(X_train, y_train)')

#let's see the best estimator
best_tree = grid_tree.best_estimator_
print(best_tree)
print("_"*40)
#with its score
print("Cross-validated best score {}%".format(round(grid_tree.best_score_ * 100,3)))
#score on test
predictions = best_tree.predict(X_test)
print("Test score: {}%".format(round(accuracy_score(y_true = y_test, y_pred = predictions) * 100,3)))


# In[9]:


best_tree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=13, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')


# A good thing to do is to is to analyze how the model really performed. We can use a confusion matrix or the classification report. 
# 
# Or both.

# In[127]:


cm = confusion_matrix(y_test, predictions)
cmap = plt.cm.Blues
classes = [0,1]
thresh = cm.max() / 2.
fmt = 'd'

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()


# In[128]:


target_names = ['Not Default', 'Default']
print(classification_report(y_test, prediction, target_names=target_names))


# Now, a good data scientist would probably tell you that a better metric to train, or balancing your classes, would give you a better score. But the purpose is having an idea about how tuning the model can influence the result. We will move on and see how building a lot of trees and picking the best is not the only strategy.
# 
# ## Save the planet, grow a forest
# 
# The idea is to train a large number of simple learners and combine them to produce a more complex one with more accuracy and stability.
# 
# One way of doing so is with a **bagging algorithm** (bootstrap aggregating). Here, a number of strong learners (meaning, unconstrained, such as a very deep tree) are trained in parallel and then combined to produce a model less prone to overfitting.
# 
# The Random Forest is a type of bagging algorithm. Here, a set of classifiers is created by introducing randomness in their construction. The randomness comes from the fact that each tree is built from a sample drawn with replacement (this is why it is a bagging algorithm) from the training set. Moreover, the split is not chosen by picking the best possible split, but rather the best one among a random subset of the features. This will augment the bias but drastically reduce the variance.
# 
# The original idea was letting each classifier vote for a single class. The sklearn implementation combines classifiers by averaging their probabilistic predictions.
# 
# One step further would be the **extremely randomized trees**, which step up the randomness by picking also the threshold for the splits from a randomly generated set. This again reduces the variance, at the bias expenses. But we will not cover that here.

# In[129]:


forest = RandomForestClassifier(n_jobs = -1, random_state=42)
forest


# In my experience, which is limited, the most important parameters to tune are:
# 
# * **Deeper trees** help since it is in the spirit of the algorithm to produce a bunch of overfitting trees to get to a non-overfitting model.
# * **max_features**, controlling the number of features to consider when looking for the best split. The higher the number the better, at the expenses of the time of execution.
# * **n_estimators**, controlling the number of trees to grow. Again, the higher the better, giving more stability to your model. Again, the time of execution will increase.
# * **min_sample_leaf** will capture more noise if it gets smaller and smaller.

# In[130]:


forest.fit(X_train, y_train)
prediction = forest.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))


# In[131]:


def get_feature_importance(clsf, ftrs):
    imp = clsf.feature_importances_.tolist()
    feat = ftrs
    result = pd.DataFrame({'feat':feat,'score':imp})
    result = result.sort_values(by=['score'],ascending=False)
    return result


# In[132]:


get_feature_importance(forest, X.columns).head(10)


# And with more features...

# In[133]:


forest.fit(XL_train, yL_train)
prediction = forest.predict(XL_test)
print("Accuracy: {}%".format(round(accuracy_score(yL_test, prediction) * 100,3)))


# Which is not significantly worse because RandomForest implicitly selects the features.

# In[134]:


get_feature_importance(forest, XL.columns).head(10)


# In[135]:


get_feature_importance(forest, XL.columns).tail(10)


# From now onwards, we will not care about the large dataset, just to speed up the exploration.
# 
# **Exercise**: what would you expect is the effect of the max_depth parameter?

# In[136]:


dp_list = np.arange(3, 30)
train = []
test = []

for depth in dp_list:
    forest = RandomForestClassifier(max_depth=depth, n_jobs = -1, random_state=42)
    forest.fit(X_train, y_train)
    prediction = forest.predict(X_test)
    trainpred = forest.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'n_estimators':dp_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = dp_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('Forest accuracy vs depth')
plt.show()


# If for a single tree increasing the depth was compromising the performance on the test set a lot, a forest doesn't really care about that. There is a decrease in accuracy, but it is also a very small forest.
# 
# **Exercise**: did I just lie to your face? Check it. Or don't, I am not your mother.
# 
# We can see the effect of growing more trees by changing the n_estimators parameter.

# In[137]:


tree_list = np.arange(3, 80)
train = []
test = []

for tree in tree_list:
    forest = RandomForestClassifier(n_estimators=tree, n_jobs = -1, random_state=42)
    forest.fit(X_train, y_train)
    prediction = forest.predict(X_test)
    trainpred = forest.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'n_estimators':tree_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = tree_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('Forest accuracy vs n_estimators')
plt.show()


# We can see how with more trees the performance becomes more stable.

# In[138]:


leaf_list = np.arange(1, 100)
train = []
test = []

for leaf in leaf_list:
    forest = RandomForestClassifier(min_samples_leaf=leaf, n_jobs = -1, random_state=42)
    forest.fit(X_train, y_train)
    prediction = forest.predict(X_test)
    trainpred = forest.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'min_samples_leaf':leaf_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = leaf_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('Forest accuracy vs min_samples_leaf')
plt.show()


# In[139]:


leaf_list = np.arange(2, 80, 2)
train = []
test = []

for leaf in leaf_list:
    forest = RandomForestClassifier(min_samples_split=leaf, n_jobs = -1, random_state=42)
    forest.fit(X_train, y_train)
    prediction = forest.predict(X_test)
    trainpred = forest.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'min_samples_split':leaf_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = leaf_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('Forest accuracy vs min_samples_split')
plt.show()


# Now, if we wanted to GridSearch properly, this is the parameter space we want to search

# In[140]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 30, num = 15)]
max_depth.append(None)
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(2, 50, num = 25)]
# Minimum number of samples to split an internal node
min_samples_split = [int(x) for x in np.linspace(2, 100, num = 50)]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf,
               'min_samples_split' : min_samples_split,
               'n_jobs' : [-1],
               'random_state' : [42]}


# That means we want to grow 1000000 forests, about 100 million trees... Not really doable with our laptops. 
# 
# A random search picks a certain amount of points (picked randomly) in this parameter space and finds the best estimator among them (only 50 for us due to speed up the search). Depending on the problem, the results of a random search are not significantly worse than the one of a grid search, it really depends on what you want to achieve.

# In[141]:


# Don't run during class, the result is in the next cell.

grid_forest = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, 
                               n_iter = 50, cv = skf, random_state=42, n_jobs = -1,
                                scoring = 'accuracy')

#training
get_ipython().run_line_magic('time', 'grid_forest.fit(X_train, y_train)')

#let's see the best estimator
best_forest = grid_forest.best_estimator_
print(best_forest)
print("_"*40)
#with its score
print("Cross-validated best score {}%".format(round(grid_forest.best_score_ * 100,3)))
#score on test
predictions = best_forest.predict(X_test)
print("Test score: {}%".format(round(accuracy_score(y_true = y_test, y_pred = predictions) * 100,3)))


# If this method still takes too much time for you, you can also use the validation curves above to guess your best estimator. We can let all the individual trees overfit, train at least 100 of them, and play a bit with the size of the leaves. Maybe the performance is lower, but it saves a lot of time.

# In[142]:


best_forest2 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=15, min_samples_split=30,
            min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=-1,
            oob_score=False, random_state=42, verbose=0, warm_start=False)

get_ipython().run_line_magic('time', 'best_forest2.fit(X_train, y_train)')

predictions = best_forest2.predict(X_test)
print("Test score: {}%".format(round(accuracy_score(y_true = y_test, y_pred = predictions) * 100,3)))


# In[143]:


cm = confusion_matrix(y_test, predictions)
cmap = plt.cm.Blues
classes = [0,1]
thresh = cm.max() / 2.
fmt = 'd'

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()


# In[31]:


target_names = ['Not Default', 'Default']
print(classification_report(y_test, prediction, target_names=target_names))


# The model is much more precise in predicting the Default, with a precision improved by 23%. Looking at the confusion matrix, it is essentially predicting 1 more often, reducing the number of false negatives by about 90 cases and increasing the number of false positives by about 100 cases. It is up to you and your product owner to decide if it is better or not.
# 
# Other things you can do:
# 
# * Control the maximum number of features
# * Use max_leaf_nodes to not let the trees get too wide
# * Use min_impurity_decrease to avoid pointless splits
# * Use warm_start to do many experiments and save some time
# * Balance your classes
# * Use oob_score to estimate the generalization accuracy
# 
# Or, as we will do for the next algorithm, study the learning curve and take decisions from that.

# ## Get popular in every bar with your boosting algorithms
# 
# This is another class of algorithms that trains several learners to then combine them. In this case, however, the learners are not trained in parallel but rather in sequence. Moreover, each learner is trained to correct the errors of the previous one (samples that produced the bigger error get a higher weight in the next learning process).
# 
# The basic learners are weak (so very constrained) and each one of them individually is slightly better than a random guess. However, by progressively correcting its own mistakes, the final learner is much more accurate.
# 
# Here, we use the very popular XGBoost algorithm, which is a normal boosted trees algorithm with some advantages that are listed better in these posts: [A Gentle Introduction to XGBoost for Applied Machine Learning](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/), [Complete Guide to Parameter Tuning in XGBoost](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/) (a bit outdated on the syntax, but solid). In brief, it can be used in substitution of a boosting algorithm and generally is faster, more accurate, and less heavy on your hardware.

# In[145]:


xgb = XGBClassifier(n_jobs = -1)
xgb


# There are some common practices to tune XGBoost. The key hyperparameters are **learning_rate** and **n_estimators**. Depending on the problem, you might want to introduce some regularization with reg_alpha and reg_lambda. **base_score** is not very influential if you go enough trees, but we can do some experiments.
# 
# Some common practices are:
# 
# * Not so deep trees, usually between 2 and 8
# * Learning rate at 0.1 or lower, the lower the more trees you will need.
# * Feel the learning first, we will see later how to do that, just keep in mind that what is coming next is very limited.
# * subsample can help make the model more robust by reducing variance, a popular value is around 0.8
# 
# A popular strategy is to choose a fixed number of trees (in the order of the 1000 trees) and play around with the learning rate. 
# 
# Another effective strategy is to pick a high learning rate, find the optimum number of trees (keep it short or you waste a lot of time), tune the individual trees, and then lower the learning rate while increasing the trees proportionally.

# In[146]:


xgb.fit(X_train, y_train)
prediction = xgb.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))


# In[147]:


tree_list = np.arange(10, 1000, 10) # 500500 trees...
train = []
test = []

for tree in tree_list:
    xgb = XGBClassifier(n_estimators=tree, n_jobs = -1, random_state=42)
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    trainpred = xgb.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'n_estimators':tree_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = tree_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('XGB accuracy vs n_estimators')
plt.show()


# In[148]:


learn_list = np.arange(0.01, 0.99, 0.01) # About 20000 trees
train = []
test = []

for tree in learn_list:
    xgb = XGBClassifier(n_estimators=200, learning_rate=tree, n_jobs = -1, random_state=42)
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    trainpred = xgb.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'learning_rate':learn_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = learn_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('XGB accuracy vs learning_rate')
plt.show()


# In[149]:


tree_list = np.arange(2, 10) 
train = []
test = []

for tree in tree_list:
    xgb = XGBClassifier(max_depth=tree, n_jobs = -1, random_state=42)
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    trainpred = xgb.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    
performance = pd.DataFrame({'max_depth':tree_list,'Train_acc':train,'Test_acc':test})

fig, ax = plt.subplots()
x_axis = tree_list
ax.plot(x_axis, performance['Train_acc'], label='Train')
ax.plot(x_axis, performance['Test_acc'], label='Test')
ax.legend()
plt.ylabel('accuracy')
plt.title('XGB accuracy vs max_depth')
plt.show()


# The next cell is just to see how less relevant the base score is if you have enough trees.

# In[150]:


tree_list = np.arange(0.1, 0.9, 0.1)
train = []
test = []
train2 = []
test2 = []

for tree in tree_list:
    xgb = XGBClassifier(base_score=tree, n_jobs = -1, random_state=42)
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    trainpred = xgb.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train.append(train_acc)
    test.append(test_acc)
    xgb = XGBClassifier(base_score=tree, n_estimators=500, n_jobs = -1, random_state=42)
    xgb.fit(X_train, y_train)
    prediction = xgb.predict(X_test)
    trainpred = xgb.predict(X_train)
    train_acc = accuracy_score(y_train, trainpred)
    test_acc = accuracy_score(y_test, prediction)
    train2.append(train_acc)
    test2.append(test_acc)
    
performance = pd.DataFrame({'base_score':tree_list,'Train_acc':train,'Test_acc':test})
performance2 = pd.DataFrame({'base_score':tree_list,'Train_acc':train2,'Test_acc':test2})

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 5))

x_axis = tree_list
ax1.plot(x_axis, performance['Train_acc'], label='Train')
ax1.plot(x_axis, performance['Test_acc'], label='Test')
ax1.legend()
plt.ylabel('accuracy')
plt.title('Tree accuracy vs base_score')
ax2.plot(x_axis, performance2['Train_acc'], label='Train')
ax2.plot(x_axis, performance2['Test_acc'], label='Test')
ax2.legend()
plt.ylabel('accuracy')
plt.title('Tree accuracy vs base_score')
plt.show()


# From these plots, I can guess a learner as the following, saving myself some time
# 
# ** Exercise**: pick a better combination of parameters given the previous plots.

# In[152]:


best_XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=-1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)


# In[153]:


best_XGB.fit(X_train, y_train)
prediction = best_XGB.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))


# In[154]:


cm = confusion_matrix(y_test, predictions)
cmap = plt.cm.Blues
classes = [0,1]
thresh = cm.max() / 2.
fmt = 'd'

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()


# In[155]:


target_names = ['Not Default', 'Default']
print(classification_report(y_test, prediction, target_names=target_names))


# ## Overfitting is like Ted Cruz
# 
# Nobody likes Ted Cruz.
# 
# XGBoost has a built-in method to observe if you are training your model too much: it helps keeping track of the performance on training and test set.

# In[156]:


eval_set = [(X_train, y_train), (X_test, y_test)]

best_XGB.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=False)


# In[157]:


# retrieve performance metrics
results = best_XGB.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot rmse
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('error')
plt.title('XGBoost accuracy')
plt.show()


# To get the sweet spot of this curve, we can use the early_stopping_rounds parameter.

# In[158]:


eval_set = [(X_test, y_test)]
best_XGB.fit(X_train, y_train, early_stopping_rounds=200, 
             eval_metric="error", eval_set=eval_set, verbose=True)


# ** Exercise **: how do you expect the early_stopping_rounds parameter to influence the outcome? Do some experiments.

# In[159]:


best_XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=381,
       n_jobs=-1, nthread=None, objective='binary:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=1)


# In[160]:


best_XGB.fit(X_train, y_train)
prediction = best_XGB.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))


# In[161]:


cm = confusion_matrix(y_test, predictions)
cmap = plt.cm.Blues
classes = [0,1]
thresh = cm.max() / 2.
fmt = 'd'

plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.show()


# In[162]:


target_names = ['Not Default', 'Default']
print(classification_report(y_test, prediction, target_names=target_names))


# ### Learning Curves, a good friend you forgot about
# 
# We get a significant improvement in the performance, but we still have little idea about how our model is learning. To this purpose, we want to study the **learning curve**: how the performance on training and validation evolve if we vary the size of the training set.
# 
# This is done through cross-validation and it is a very useful indication on how good is your model. We are doing it at the very end, but this procedure can be done at any point in your workflow, giving you useful insights on where are you heading to.

# In[10]:


train_sizes = [1, 500, 1000, 3000, 5000, 7000, 9000, 11000, 13000, 15599] #there are 19500 entries in the train set

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = best_tree, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70)
print('\nValidation scores:\n\n', validation_scores)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Decision tree', fontsize = 18, y = 1.03)
plt.legend()


# In[12]:


train_sizes = [1, 500, 1000, 3000, 5000, 7000, 9000, 11000, 14399]

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = best_forest2, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70)
print('\nValidation scores:\n\n', validation_scores)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Random Forest', fontsize = 18, y = 1.03)
plt.legend()


# In[14]:


train_sizes = [1, 500, 1000, 3000, 5000, 7000, 9000, 11000, 14399]

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = best_XGB, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70)
print('\nValidation scores:\n\n', validation_scores)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('XGBoost', fontsize = 18, y = 1.03)
plt.legend()


# Here is what we can see on the curve:
# 
# * with a training set of only 1 data point, we learn it perfectly, but our prediction is in the mid 60's.
# * From roughly 500/1000 data points the accuracy on the validation doesn't really increase. If we wanted a better accuracy, adding more data points would lead somewhere but one should evaluate how costly this process can be in relation to the improvement in performance. The random forest *stops learning* at 1000 data points, the single tree needs 3000 data points, and XGBoost keeps improving (slowly).
# 
# Once the two curves converge, adding more data points is very unlikely to help. In our case, we still have margins of improvement for XGBoost, but we don't have really a way of doing it (to be fair, we could simply reduce the size of the test, but let's assume that set is really untouchable).
# 
# If we want to check if our model is **biased**, we can look at the accuracy on the validation set: 
# * DecisionTree: around 82%.
# * RandomForest: around 81%.
# * XGBoost: around 82%.
# 
# (I admit this is not the best example)
# 
# How good is that? It depends, domain knowledge and error characterization will definitely help you in deciding if the model suffers from a bias problem. The main indicator is a low accuracy on the validation.
# 
# To decide if it is a low or high bias problem, we can look at the training error. A model with high bias would have a low accuracy on the training data. In that case, we can conclude that the model is not learning well enough.
# 
# If we want to analyze if there is a **variance** problem, we can look at the gap between the curves (bigger gap for bigger variance) and examine the value and the evolution of the training error. A high variance means a low generalization capability.
# 
# Low performances on the training are a quick way to detect low variance since this means that the model has troubles in fitting the training data as the training sets change (this is called *underfit*).
# 
# If we have a problem with **high bias and low variance**, we need to increase the complexity of the model. We can do the following:
# 
# * Training on more features.
# * Decrease the regularization.
# 
# If we are in the opposite situation (**low bias and high variance**) we would see large gaps and high training accuracy (thus our old friend: *overfitting*). In this case, one would have the following options:
# 
# * Add more training entries.
# * Increase the regularization.
# * Reduce the number of features (this would definitely increase the bias).
# 
# **Exercise**: look at the learning curves of the three models what can you conclude?
# 
# **Exercise**: Using the model you prefer, make it underfit and overfit at the best of your patience and ability. Can you tell the difference by looking at the classification report?

# In[25]:


best_tree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=13, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')

#Change the next two

over_tree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=13, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')

under_tree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=13, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')


# In[26]:


tree = best_tree #change here

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = tree, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Decision tree', fontsize = 18, y = 1.03)
plt.legend()

tree.fit(X_train, y_train)
prediction = tree.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))

print(classification_report(y_test, prediction, target_names=target_names))


# In[48]:


best_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=15, min_samples_split=30,
            min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=-1,
            oob_score=False, random_state=42, verbose=0, warm_start=False)

# Change the next two

over_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=15, min_samples_split=30,
            min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=-1,
            oob_score=False, random_state=42, verbose=0, warm_start=False)

under_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=15, min_samples_split=30,
            min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=-1,
            oob_score=False, random_state=42, verbose=0, warm_start=False)


# In[49]:


forest = best_forest #change here

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = forest, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Random Forest', fontsize = 18, y = 1.03)
plt.legend()

forest.fit(X_train, y_train)
prediction = forest.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))

print(classification_report(y_test, prediction, target_names=target_names))


# In[66]:


best_XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=381,
       n_jobs=-1, nthread=None, objective='binary:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=1)

#change the next two

over_XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=381,
       n_jobs=-1, nthread=None, objective='binary:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=1)

under_XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=381,
       n_jobs=-1, nthread=None, objective='binary:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=1)


# In[67]:


XGB = best_XGB #change here

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = XGB, X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = skf,
                                                   scoring = 'accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('XGBoost', fontsize = 18, y = 1.03)
plt.legend()

XGB.fit(X_train, y_train)
prediction = XGB.predict(X_test)
print("Accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100,3)))

print(classification_report(y_test, prediction, target_names=target_names))


# ## Conclusions
# 
# This notebook had hopefully sparked some ideas on what one can do to achieve the best result for a specific problem. I want to stress again that your model will always be as good as the data you put into it, thus make that your absolute priority if you want to achieve good results.
# 
# A couple of exercises I found useful to better understand these learning machines:
# 
# ** Exercise **: research what is a more appropriate metric to evaluate a classification model and re-run this notebook by using that one instead of accuracy.
# 
# ** Exercise**: create a custom metric for this problem and amaze your friends
# 
# Any feedback will be very welcome, what is your strategy to tune a model?
