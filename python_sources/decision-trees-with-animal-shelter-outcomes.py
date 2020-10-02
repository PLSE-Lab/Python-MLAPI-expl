#!/usr/bin/env python
# coding: utf-8

# # Decision trees with animal shelter outcomes
# 
# ## Introduction
# 
# Decision trees are a flexible supervised learning algorithm which works by approaching a classification problem as a sequence of decisions. At each decision point, the data is split into a subset passing the decision, and a subset failing the decision. The rule that gets used to make that decision is simple: do whatever will result in the most distinct subclasses.
# 
# For example, if neutered dogs are significantly more likely to be adopted than intact ones, then we can logically expect a decision tree to use that to make a decision: neutered dogs in this bucket, and unneutered ones in this one.
# 
# By sequencing "enough" of these decisions in a row (subject to overfitting/underfitting), a decision tree arrives at distinct subsets which it can (hopefully with decent confidence!) assign to one bin or to another one.
# 
# ## Application
# 
# Here's an application of decision trees to animal shelter outcomes. Can we predict the outcomes for a set of animals interned to a shelter based on what we know about them?
# 
# First build the feature matrix.

# In[22]:


import pandas as pd
outcomes = pd.read_csv("../input/aac_shelter_outcomes.csv")


# In[62]:


df = (outcomes
      .assign(
         age=(pd.to_datetime(outcomes['datetime']) - pd.to_datetime(outcomes['date_of_birth'])).map(lambda v: v.days)
      )
      .rename(columns={'sex_upon_outcome': 'sex', 'animal_type': 'type'})
      .loc[:, ['type', 'breed', 'color', 'sex', 'age']]
)
df = pd.get_dummies(df[['type', 'breed', 'color', 'sex']]).assign(age=df['age'] / 365)
X = df
y = outcomes.outcome_type.map(lambda v: v == "Adoption")


# In[41]:


df.head()


# Now let's run the classifier.
# 
# The `sklearn` classifier implementation is `DecisionTreeClassifier`.

# In[71]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
y_hat = clf.predict(df)
accuracy_score(y, y_hat)


# A big advantage of decision trees is that they are easily understood. If X, Y, Z hold, then put the observation in such-and-such class. If a different set of conditions holds, then we put the observation in a different class.
# 

# In[72]:


import graphviz
from sklearn.tree import export_graphviz
from IPython.display import display

dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=['Not Adopted', 'Adopted'], filled=True)
graph = graphviz.Source(dot_data)
display(graph)
# graph.render("shelter_outcomes") 


# Can you read the classification results off of the outcomes chart?
# 
# Now a quick cross-validation check.

# In[90]:


from sklearn.model_selection import KFold
import numpy as np

clf = DecisionTreeClassifier(max_depth=3)
for train, test in KFold(n_splits=5).split(X):
    X_train, X_test = X.loc[train], X.loc[test]
    y_train, y_test = y.loc[train], y.loc[test]
    clf.fit(X_train, y_train)
    print(accuracy_score(y_test, clf.predict(X_test)))


# ## Discussion
# 
# There are two decision tree algorithms available in `sklearn`, `DecisionTreeClassifier` and `DecisionTreeRegressor`. The latter is a simple adoptation of decision trees to regression problems; the only difference is that instead of outputting classes, these decision trees output numbers.
# 
# It's easy to build a decision tree that's overfitted to the data. The default settings used by `sklearn` for determining the decision tree stopping point are remarkably overfit-happy. Luckily, there are tons of switches and dials you can use to control when the decision tree will continue to branch or not branch. Figuring out when to stop and using what toggles is an "art" thing; you really just have to play with the algorithm to see what is optimal.
# 
# I find overfitted decision trees to be helpful for EDA. Because they are so easy to inspect and understand, they make it simple to determine places where there are artifacts in your data which will have unexpected consequences on other algorithms you apply to the dataset. For example, in this dataset all `Other` animals are given an `Age` of 365 days. I figured this out by generating a further split-out tree and noticing 1.01 years gets picked up as a break point in several places, and then inspecting the raw data to see why that is. Particularly for small classes, which won't show up on a big scatter blot very well, this is a highly useful capacity to have!
# 
# Although the CART algorithm used to determine the break points is complex, there are two things to keep in mind about it.
# 
# One is that it's not technically feasible to find the globally optimal breakpoint at each decision boundary; e.g. it's not computationally tractable to find the "best possibe splitting rule" for each rule. Hence a local minimum is used and found instead. One way of checking for alternative splitting rules, in marginal cases when you really need one, is to retrain the algorithm many times, and determine whether or not the rules used change.
# 
# The other is that, by default, the leaves that are generated are ones which minimize the Gini coefficient. Gini is a well-accepted measure of inequality whose origin is in econometrics but which is well-loved for cases such as this one in ML. A discussion of the Gini coefficient is another notebook, though. There's also an entropy option available, which is functionally little different from using Gini (and a bit more computationally expensive; see [here](https://datascience.stackexchange.com/questions/20415/what-should-be-the-order-of-class-names-in-sklearn-tree-export-function-beginne)).
