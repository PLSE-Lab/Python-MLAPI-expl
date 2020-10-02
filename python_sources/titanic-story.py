#!/usr/bin/env python
# coding: utf-8

# ## Part 1: Exploring the Titanic
# 
# This notebook is adapted from my homework for UC Berkeley's Applied Machine Learning class (2017). To get started, read about the prediction problem on [Kaggle](https://www.kaggle.com/c/titanic). Then, download the data from [here](https://www.kaggle.com/c/titanic/data). 
# 
# Then follow the notebook to unveil a beautiful story about men helping women and children survive sacrificing their own lives, about a group of outliers from the lowest part of the boat (3rd class) who managed to survive against all odds, and a story about several ML classifiers that failed to detect those outliers and sinked amidst the NaNs and human inconsistency.
# 
# 
# ### 1.1 Exploratory data analysis
# 
# Let's create 2-3 figures and tables to get a feel for the data. We'll check the data type of each variable, understand which variables have missing observations, and understand the distribution of each variable (to determine whether the variables should be normalized or not). Are any variables collinear or highly correlated?

# In[ ]:


import IPython
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import graphviz 
import statsmodels.formula.api as smf
import copy

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_set_full = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')


# Now, let's see how likely it is to survive depending on your passenger class, gender, age.

# In[ ]:


train_set_full[['Pclass', 'Survived']].groupby(['Pclass'], 
                                               as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_set_full[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


g = sns.FacetGrid(train_set_full, col='Survived')
g.map(plt.hist, 'Age', bins=20)
g.title = 'Survival rate depending on age'


# Seems like it is more likely to suvive if you are a woman or an infant, or traveling first class. As we can see from the diagram, there were many young people of 20-30 who did not survive, and who, probably, travelled 3rd class.

# Further exploration of titles in the passengers' names I borrowed from https://www.kaggle.com/startupsci/titanic-data-science-solutions.
# 
# Using regex, we'll extract everything before the "." in the name which will be our title, and create 'Title' column in both train and test sets. We then combine rare titles into "rare" category and group the set by a title to see the survival rate among different titles. Let's explore this on a full train set first.

# In[ ]:


combined_set = [train_set_full, test_set]


# In[ ]:


for dataset in combined_set:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


pd.crosstab(train_set_full['Title'], train_set_full['Sex'])


# In[ ]:


for dataset in combined_set:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Dr'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace(['Mme', 'Countess', 'Dona'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir',], 'Mr')
    
train_set_full[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# If we filter out rare titles, turnes out, married women (and mothers) had the highest chances to survive. Then unmarried women and girls follow. Boys under 18 (Masters) had significantly lower chances to survive, and Misters survived the least.
# 
# Extracting titles is also usefull if we want to guess the age of some people for whom this data is NaN in the "Age" column. Let's take a look at some observations with missing age. How many Masters and Misters are there?

# In[ ]:


train_set_full[np.isnan(train_set_full['Age'])].groupby(['Title'], as_index=False).PassengerId.count()


# #### Can we guess their age? 
# 
# The title - Master, Miss, etc. - gives us a vague clue about the person's age.
# 
# So, according to early 20th century eticket:
# 
# Title | Age
# --|--
# Master | under 18
# Miss | 0-90+
# Mr | 18-90+
# Mrs | 18-90+
# Rare | 0-90+
# 
# 
# 
# Most of the passengers with missing data were Misters, of the age 18 to 90+. The same range will likely apply to Mrs. The age range of Misses is even wider - from 0 to 90+. And only Masters, who are boys under 18, give us a smaller range. That's the finest granularity we can get. So it's pretty hard to deduce a heuristic here to put in the missing numbers, let alone run a ML model to predict the missing age (we don't have enough data).
# 
# This is why, for now we'll just drop the rows with missing age (all 177 of them) and create a new train set. We'll also reset index for it.

# But before, let's convert titles and gender to numerical values for future use. The bigger chance to survive - the higher value we assign.

# In[ ]:


for dataset in combined_set:
    dataset['Title'] = dataset['Title'].replace('Master', 2)
    dataset['Title'] = dataset['Title'].replace('Miss', 3)
    dataset['Title'] = dataset['Title'].replace('Mr', 0)
    dataset['Title'] = dataset['Title'].replace('Mrs', 4)
    dataset['Title'] = dataset['Title'].replace('Rare', 1)
    dataset['Sex'] = dataset['Sex'].replace('male', 0)
    dataset['Sex'] = dataset['Sex'].replace('female', 1)


# In[ ]:


train_set = train_set_full[np.isfinite(train_set_full['Age'])].copy(deep=True)
train_set = train_set.reset_index(drop=True)
train_set.head()


# Now, for this new working train set, we'll create a new variable called 'Agegroup' and assign the following values to each observation: 1 if the age < 5, 2 if the age < 16, 3 if the age < 50, 4 for over 50.

# In[ ]:


def age2cat(age):
    if age < 5:
        return 1 #toddler
    elif age < 16:
        return 2 #child
    elif age < 50:
        return 3 #adult
    else:
        return 4 #elderly

train_set['Agegroup'] = [age2cat(x) for x in train_set.Age]   


# In[ ]:


train_set['Agegroup'].unique()


# #### Let's add some more features! 
# 
# 1. Summing up Parent-child vector with Sibling-spouse should give us the family size. How likely were families to survive?
# 
# 2. Fare is given for a ticket, and since families traveled on one ticket, it's fair to divide the fare by family size - just to get a fair fare for each passenger.

# In[ ]:


train_set['Family_Size']=train_set['SibSp']+train_set['Parch']
test_set['Family_Size']=test_set['SibSp']+test_set['Parch']


train_set['Fare_p_Person']=train_set['Fare']/(train_set['Family_Size']+1)
test_set['Fare_p_Person']=test_set['Fare']/(test_set['Family_Size']+1)


# ### 1.2 Correlates of survival
# 
# Let's figure out what factors seem to determine whether or not a person would survive the sinking of the Titanic.

# In[ ]:


corr_table = train_set.corr(method='pearson')
corr_table.style.background_gradient(cmap='RdYlGn', axis=1)


# #### Conclusions:
# 
# The confusion matrix shows a relatively high positive correlation of survival rate with title and sex, and a significant negative correlation with passenger class (meaning, 1 class had better chances, 3 class had the worst chances). Other features, especially passenger ID, are not very useful. But we can try combine some features, as it seems to be a productive idea. For example, if you look at Fare_p_Person and Fare, you'll see that Fare correlates stronger. This might be because Fare is in fact a combined variable of fare per person and number of people in a family travelling by this fare since this is what a price of a ticket tells us about. That's why Fare takes some prediction power from family size while Fare_p_Person doesn't.

# ## Part 2: Decision Trees

# ### 2.1 Decision Tree
# Using the basic [Decision Tree](http://scikit-learn.org/stable/modules/tree.html#tree) library in sklearn and a 10-fold cross-validation, we'll fit a model to predict titanic survival. What would be the average training and testing accuracy across our 10 folds? Also, we'll plot a diagram of the tree just to understand it better.

# In[ ]:


from sklearn import tree
# Enter your code here
from sklearn.model_selection import cross_val_score
dt_classifier = tree.DecisionTreeClassifier(criterion='gini',  # or 'entropy' for information gain
                       splitter='best',  # or 'random' for random best split
                       max_depth=None,  # how deep tree nodes can go
                       min_samples_split=2,  # samples needed to split node
                       min_samples_leaf=1,  # samples needed for a leaf
                       min_weight_fraction_leaf=0.0,  # weight of samples needed for a node
                       max_features=None,  # number of features to look for when splitting
                       max_leaf_nodes=None,  # max nodes
                       min_impurity_decrease=1e-07, #early stopping
                       random_state = 10) #random seed


# In[ ]:


X = pd.DataFrame(train_set, columns=['Pclass', 'Title', 'Sex', 'Fare'])
Y = train_set.Survived

cross_val_score(dt_classifier, X, Y, cv=10).mean()


# In[ ]:


dt_model = dt_classifier.fit(X, Y)

dt_model.feature_importances_


# What Feature importance tells us is that, of these four parameters - passenger class, title, gender and fare, - **title** and **fare** seem to have a higher impact on the result.

# In[ ]:


##full tree can be found in titanic_tree.pdf in the working dir
# plot_tree = tree.export_graphviz(dt_classifier, out_file=None) 
# graph = graphviz.Source(plot_tree) 
# graph.render("titanic_tree") 


# In[ ]:


dot_data = tree.export_graphviz(dt_model, out_file=None,
                                max_depth=2, 
                         feature_names=[x for x in X.columns],  
                         class_names=Y.name,  
                         filled=True, rounded=True,
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


# Our starting accuracy = 81.25% using 10-fold cross validation.
# 
# Seems like 'Title and 'Fare' are indeed good features. What's interesting about them is that they both are "combined" feautures: title implicitly contains information about gender and age while fare - about family size and passenger class.

# ### 2.2 Tree tuning
# 
# Now, we'll use all of the data to re-fit a single decision tree with max_depth = 4 (i.e., no cross-validation). We'll show the tree diagram and also plot the feature importances.

# In[ ]:


# new decision tree classifier
dt_classifier_new = tree.DecisionTreeClassifier(criterion='gini',  # or 'entropy' for information gain
                       splitter='best',  # or 'random' for random best split
                       max_depth=4,  # how deep tree nodes can go
                       min_samples_split=2,  # samples needed to split node
                       min_samples_leaf=1,  # samples needed for a leaf
                       min_weight_fraction_leaf=0.0,  # weight of samples needed for a node
                       max_features=None,  # number of features to look for when splitting
                       max_leaf_nodes=None,  # max nodes
                       min_impurity_decrease=1e-07, #early stopping
                       random_state = 10) #random seed


# In[ ]:


np.random.seed(seed=2468) #13579
features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Title', 'Age', 'Agegroup', 'Family_Size', 'Fare_p_Person']
features_labels = features + ['Survived']

train_idx = np.random.choice(range(len(train_set)), int(len(train_set) * .8), replace=False)
test_idx = list(set(range(len(train_set))) - set(list(train_idx)))

train_x = train_set.loc[train_idx, features]
train_y = train_set.loc[train_idx, 'Survived']
test_x = train_set.loc[test_idx, features]
test_y = train_set.loc[test_idx, 'Survived']


# In[ ]:


dt_model_new = dt_classifier_new.fit(train_x, train_y)
print('Test accuracy:')
print(dt_model_new.score(test_x, test_y))
print('\n''Feature importance:')
for i in range(len(features)):
    print(features[i], dt_model_new.feature_importances_[i])


# In[ ]:


fig, ax = plt.subplots()
ind = np.arange(10)

plt.bar(ind, dt_model_new.feature_importances_)
ax.set_xticks(ind)
ax.set_xticklabels(features)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.set_ylim([0, 0.6])
ax.set_ylabel('Percent of impact')
ax.set_title('Which feature was the most impactful?')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show


# In[ ]:


dot_data_1 = tree.export_graphviz(dt_model_new, out_file=None,
                                max_depth=2, 
                         feature_names=features,  
                         class_names=train_y.name,  
                         filled=True, rounded=True,
                         special_characters=True)  
graph_1 = graphviz.Source(dot_data_1)  
graph_1


# #### Conclusion:
# Title remains the most important feature, while Fare and Fare_p_Person had a smaller impact this time, even smaller than Pclass which is the second to most important feature

# ### 2.3 Sensitivity analysis
# The built-in algorithm we are using has several parameters which we can tune. Let's see how the choice of these parameters affects performance.
# 
# First, let's see how max_depth affects train and test accuracy. On a single axis, we'll plot train and test accuracy as a function of max_depth.
# 
# Second, let's see how test accuracy relates to both max_depth and min_samples_leaf. We'll create a 3-D plot where the x-axis will be max_depth, the y-axis - min_samples_leaf, and the z-axis - accuracy. What combination of max_depth and min-samples_leaf achieves the highest accuracy? How sensitive are the results to these two parameters?

# In[ ]:


from sklearn.model_selection import cross_validate


# In[ ]:


chart1_train = []
chart1_test = []
chart2_z = []
chart2_x = []
chart2_y = []

for i in range(4,10):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    scores = cross_validate(clf, train_set[features], train_set['Survived'], cv=5, return_train_score=True)
    chart1_train.append(scores['train_score'].mean())
    chart1_test.append(scores['test_score'].mean())
    for l in range(1,4):
        clf = tree.DecisionTreeClassifier(max_depth=i, min_samples_leaf=l)
        scores = cross_validate(clf, train_set[features], train_set['Survived'], cv=5, return_train_score=False)
        chart2_z.append(scores['test_score'].mean())
        chart2_x.append(i)
        chart2_y.append(l)


# In[ ]:


fig, ax = plt.subplots()
plt.plot(chart1_train)
plt.plot(chart1_test)
plt.legend(['train score', 'test score'], loc='upper left')
ax.set_ylabel('accuracy')
ax.set_xlabel('max_depth')
ax.set_title('Accuracy scores of the Decision Tree depending on the max_depth value')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data.
X = range(4,10)
Y = range(1,4)
X, Y = np.meshgrid(X, Y)
Z = np.asarray(chart2_z).reshape(3,6)

# Plot the surface.
ax.scatter(X, Y, Z, s=100)

# Customize the z axis.
ax.set_zlim(0.75, 0.83)

ax.yaxis.set_major_locator(LinearLocator(4))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.zaxis.set_major_locator(LinearLocator(4))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('max_depth')
ax.set_ylabel('min-samples-leaf')
plt.show()


# #### Conclusion:
# 
# According to the 3D-plot, the combination of max_depth = 4 and min-samples_leaf = 1.2 achieves the highest accuracy (a little over 0.80). The result is moderately sensitive to these two parameters, becase when we're changing max_depth to 9 and min-samples-leaf to 3.10, the accuracy drops only 2%, to 0.79%

# ### 2.4 Logistic Regression, for comparison
# It's 10-fold cross-validated logistic regression. 

# In[ ]:


# Enter your code here
from sklearn.linear_model import LogisticRegression, LassoCV

logit = LogisticRegression(penalty='l2', 
                   dual=False, 
                   tol=0.0001, 
                   C=1.0, 
                   fit_intercept=True, 
                   intercept_scaling=1, 
                   class_weight=None,
                   random_state=None, 
                   solver='liblinear', 
                   max_iter=100, 
                   multi_class='ovr', 
                   verbose=0, warm_start=False, n_jobs=1)

model_logit = logit.fit(train_x, train_y, sample_weight=None)
#tr_score = model_logit.score(train_x, train_y, sample_weight=None)
#test_score = model_logit.score(test_x, test_y, sample_weight=None)


# In[ ]:


scores = cross_validate(logit, train_set[features], train_set['Survived'], cv=10, return_train_score=True)


# In[ ]:


test_score_logit = scores['test_score'].mean()
train_score_logit = scores['train_score'].mean()

print('train: ' + str(train_score_logit) + '\ntest: ' + str(test_score_logit))
print('\n''Feature importance:')
print(logit.coef_)


# #### Conclusion:
# 
# In my implementation, the accuracy of this model is similar to the accuracy of the decision tree (however as we'll see further these two models do not correlate much and have different strengths and weaknesses).
# 

# ## Part 3: Putting together the pieces 
# 
# Here, we will explore how well different algorithms can predict survival on the Titanic.
# 

# ### 3.1 Horseraces
# 
# In addition to logistic regression and the single decision tree, we'll use the following algorithms to predict survival: LASSO regression, k-Nearest Neighbors, [random forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), a [multi-layer perceptron](http://scikit-learn.org/stable/modules/neural_networks_supervised.html) and [Gradient Boosting](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).
# 
# The barplot further below shows how well each algorithm compares relative to the others, separately for the training and testing datsets. What algorithm performs best?
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

names = ['Logistic \nRegression', 'Decision \nTree', 'LASSO', 'k-Nearest \nNeighbors', 
         'Random \nForest', 'm-layer \nPerceptron', 'Gradient \nBoosting'] 

estimators = [logit, dt_classifier_new, LassoCV(),  #lassoCV finds the best parameters for lasso
            KNeighborsClassifier(n_neighbors=5), #K=5 turned out optimal
            RandomForestClassifier(bootstrap=True, max_depth=5),
            MLPClassifier(hidden_layer_sizes=(25, 2), max_iter=1000, shuffle=False),
            GradientBoostingClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_depth=3)]


# In[ ]:


from sklearn.metrics import zero_one_loss, precision_score, recall_score, accuracy_score

CV = pd.DataFrame(data=train_set, columns=features_labels)
arr = np.arange(len(CV))
np.random.shuffle(arr)
shuffled = pd.DataFrame(data=CV, index=arr)
cv_data = np.array_split(shuffled, 10)
len(CV)


# In[ ]:


train_recalls = []
train_precisions = []
train_misclassified = []

test_recalls = []
test_precisions = []
test_misclassified = []

all_errors = {}
all_predictions = {}

for name, estimator in zip(names, estimators):
    test = []
    train = []
    errors = []
    
    sum_misclass_train = 0
    sum_misclass_test = 0

    sum_precision_train = 0
    sum_precision_test = 0

    sum_recall_train = 0
    sum_recall_test = 0

    sum_accuracy_train = 0
    sum_accuracy_test = 0
        
    test_predictions = []
    all_predictions[name] = test_predictions
    

    for i in range(10):
        data_copy = copy.deepcopy(cv_data)
        test = data_copy.pop(i)

        train = pd.concat(data_copy)
        model = estimator.fit(train[features], train.Survived)
        
        y_pred_train = model.predict(train[features]).round()
        misclass_train = zero_one_loss(train.Survived, y_pred_train, normalize=False)
        sum_misclass_train+=misclass_train
        accuracy_train = accuracy_score(train.Survived, y_pred_train)
        sum_accuracy_train+=accuracy_train
        recall_train = recall_score(train.Survived, y_pred_train)
        sum_recall_train+=recall_train
        precision_train = precision_score(train.Survived, y_pred_train)
        sum_precision_train+=precision_train
        
        y_pred_test = model.predict(test[features]).round()
        misclass_test = zero_one_loss(test.Survived, y_pred_test, normalize=False)
        sum_misclass_test+=misclass_test
        accuracy_test = accuracy_score(test.Survived, y_pred_test)
        sum_accuracy_test+=accuracy_test
        recall_test = recall_score(test.Survived, y_pred_test)
        sum_recall_test+=recall_test
        precision_test = precision_score(test.Survived, y_pred_test)
        sum_precision_test+=precision_test
        er = y_pred_test-test.Survived
        error = er.iloc[er.nonzero()[0]]
        errors.append(error)
        
        test_predictions.append(y_pred_test)
        
        
        i+=1
    print(name)
    print("average number of people misclassified for train set: {0:.0f}".format(sum_misclass_train/10))
    print("average accuracy for train set: {0:.3f}".format(sum_accuracy_train/10))
    print("average number of people misclassified for test set: {0:.0f}".format(sum_misclass_test/10))
    print("average accuracy for test set: {0:.3f}".format(sum_accuracy_test/10))

    train_misclassified.append(sum_misclass_train/10)
    train_recalls.append(sum_recall_train/10)
    train_precisions.append(sum_precision_train/10)

    test_misclassified.append(sum_misclass_test/10)
    test_recalls.append(sum_recall_test/10)
    test_precisions.append(sum_precision_test/10)

    print()
    all_errors[name] = error.index.values
    

    


# In[ ]:


ind = np.arange(7)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, train_precisions, width, color='r')
rects2 = ax.bar(ind + width, train_recalls, width, color='y')
ax.set_ylabel('Scores')
ax.set_title('Train set: precision and recall per classifier')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(names)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.legend((rects1[0], rects2[0]), ('precision', 'recall'), loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

fig, ax = plt.subplots()
rects3 = ax.bar(ind, test_precisions, width, color='r')
rects4 = ax.bar(ind + width, test_recalls, width, color='y')
ax.set_ylabel('Scores')
ax.set_title('Test set: precision and recall per classifier')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(names)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.legend((rects1[0], rects2[0]), ('precision', 'recall'), loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


# In[ ]:


ind = np.arange(7)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
plt.plot(train_misclassified)
plt.plot(test_misclassified)
plt.legend(['train (643 people)', 'test (71)'], loc='upper left')
ax.set_ylabel('number of people')
ax.set_title('Number of misclassified people for train and test data')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(names)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


# #### Conclusion:
# 
# Decision Trees performs best on testing set, followed by GB and Random Forest.

# ### 3.2 Error analysis
# 
# An error analysis comes handy when you need to gain some intuition for where your models are not performing well. Are our models all having problems with the same individuals?
# 
# We'll first do this manually, and the using a correlation matrix which will show the correlation in predictions between each model. 

# In[ ]:


all_errors


# Semms like the most models are having problems with the same individuals (with some variations). And Logistic Regression is doung very differently - it will be a great help for our ensemble model.
# 
# Below are the passenger misclassified by all the models. What do they have in common?

# In[ ]:


l = set.intersection(*(set(all_errors[name]) for name in names if name in all_errors))
train_set.iloc[list(l)]


# Well, all of them survived against all odds! They all were men, and either travelled 3rd class or fell into age groups 3 and 4 - all three are strong predictors against survival. Apparently, our model doesn't like outliers.
# 
# Now, let's compute an N-M matrix where the output of each of the models as a binary vector with one entry for each of the test data, and show how these prediction vectors relate to each other by plotting the M x M matrix of correlation coefficients.

# In[ ]:


s = (7,714)
M_M_matrix = np.zeros(s)


# In[ ]:


estimator_index = {'Decision \nTree': 0, 'Gradient \nBoosting': 1, 'LASSO': 2, 
                   'Logistic \nRegression': 3, 'Random \nForest': 4, 
                   'k-Nearest \nNeighbors': 5, 'm-layer \nPerceptron': 6}
for key, value in all_errors.items():
    M_M_matrix[estimator_index[key], value] = 1


# In[ ]:


np.dot(M_M_matrix,np.transpose(M_M_matrix))


# In[ ]:


#MM = [np.correlate(N_M_matrix[i], N_M_matrix[j]) for i in range(7) for j in range(7)]
np.corrcoef(M_M_matrix)


# #### Conclusion:
# 
# The correlation matrix says Gradient Boosting and Random Forest's outputs are not higly correlated with the rest of the estimators'. 

# ### 3.3 Develop an ensemble
# 
# This an ensemble learner created by hand. Using all of the training data, we fit each of the models in 3.1. Then, we classified each instance in the test data by taking the majority vote of all of our different fitted models. How does the new survival accuracy compares to our earlier results? 

# In[ ]:


def ensemble_train(train_x, train_y, estimators):
    for cls in estimators:
        cls.fit(train_x, train_y)
    return estimators

def ensemble_predict(test_x, test_y, estimators):
    predicted_y = np.array(test_y)  # just to initialize, will be overwritten
 
    j = 0
    for i, x in test_x.iterrows():
        binary_votes = np.array([0,0])
        for cls in estimators:
            vote = cls.predict(x.values.reshape((1, test_x.shape[1]))).round()
            binary_votes[int(vote)] += 1
        y = np.argmax(binary_votes)
        predicted_y[j] = y
        j += 1
    acc = accuracy_score(test_y, predicted_y)
    #pd.DataFrame(predicted_y).to_csv('predictions.csv')
    return acc 

estimators = ensemble_train(train_x, train_y, estimators)
acc = ensemble_predict(test_x, test_y, estimators)
print(acc)


# #### Conclusion:
# 
# The ensemble accuracy is 0.81 - 0.84, and it's in line with the best result so far (DT with 0.82).
