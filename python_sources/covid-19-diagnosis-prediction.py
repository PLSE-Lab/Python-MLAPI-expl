#!/usr/bin/env python
# coding: utf-8

# # COVID-19 diagnosis prediction using machine learning
# 
# An _attempt_ to diagnose COVID-19 in patients using different models trained with patients' health markers.

# ## Reading the dataset

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('../input/covid19/dataset.xlsx')
data.columns = map(lambda x: x.lower().strip().replace('\xa0', '_').replace(' ', '_'), data.columns)
data.head()


# # Finding a subset of useful samples
# 
# As other people who already analysed the provided dataset, a lot of samples have _null_ values. Thus, we need to find a subset of the dataset that has useful information.

# In[ ]:


import missingno as msno
msno.matrix(data, labels=True, fontsize=9)


# As we can see, the features the have greatest number of valid values are the ones related to tests of other diseases. Although, these information are not really useful to diagnose the presence of the SARS-COV2 in a person's body. Thus, the features that we can use are the ones related to blood markers. 

# In[ ]:


useful_data = data[['sars-cov-2_exam_result', 'hematocrit', 'hemoglobin', 'platelets', 'mean_platelet_volume', 'red_blood_cells', 'lymphocytes', 'mean_corpuscular_hemoglobin_concentration_(mchc)', 'leukocytes', 'basophils', 'mean_corpuscular_hemoglobin_(mch)', 'eosinophils', 'mean_corpuscular_volume_(mcv)', 'monocytes', 'red_blood_cell_distribution_width_(rdw)']].dropna()

data_length = len(data)
useful_data_length = len(useful_data)
print('Number of samples: {}'.format(data_length))
print('Number of useful samples: {}'.format(useful_data_length))
print('Useful portion of the dataset: {:.2f}%'.format((useful_data_length / data_length) * 100.0))


# # A problem: small number of samples
# 
# As we could see, a lot of samples were lost after the selection of a useful subset of the dataset. The major problem of this is that these samples probably don't reflect the patterns of a population, leading to models that trained based on a tiny part of the population.
# 
# This's a problem we need work with, thought. By now, we need to focus on generating models could be general enough the possibly predict a diagnose of COVID-19.

# # Another problem: imbalanced data
# 
#  But there's another problem: the remaining samples are highly unbalanced (considering their classes). The reason why this is a problem is because, if this data is used without prior processing for traning a model, they will generate a bias, favoring one class over to another.
# 
# A plot of the number of instances per class is shown right below.

# In[ ]:


sns.countplot(x='sars-cov-2_exam_result', data=useful_data)


# ## Solving the imbalancing problem
# 
# There are different alternatives to tackle the problem of unbalanced data. We could, for example, _undersample_ the majority class, or even _oversample_ the minority class (which is probably not a good idea for this case), and then train the desired model. Another approach could be _change the "class weights"_ of one of the classes. 
# 
# As the models are trained, the different approaches are going to be described too.

# # Models

# ### Separating the data and the classifications

# In[ ]:


# Here we separate the data into a matrix X and a vector y (data and target)
X = useful_data.loc[:, useful_data.columns != 'sars-cov-2_exam_result']
y = useful_data[['sars-cov-2_exam_result']]

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rus = RandomUnderSampler(random_state=42, sampling_strategy='majority')
X_test, y_test = rus.fit_resample(X_test, y_test) # Here, we undersample the majority class of the test set


# # Decision Tree
# 
# A well know model that could be used to try to predict the class of new instances is the [decision tree](https://en.wikipedia.org/wiki/Decision_tree). The advantages of this model is that the resultant tree could give us some good insights on what are the features that contribute the most on predicting a class.
# 
# In the first approach, we're going to set different weights to the samples that belong to the minority class. This can be done setting the _class_weight_ parameter to "balanced".

# In[ ]:


from sklearn import tree
from sklearn.model_selection import cross_val_score

decision_tree_classifier = tree.DecisionTreeClassifier(max_depth=3, class_weight='balanced')
decision_tree_scores = cross_val_score(decision_tree_classifier, X, y, cv=5)
print("Scores: {}".format(decision_tree_scores))
print("Mean: %0.2f" % decision_tree_scores.mean())
print("Standard deviation: %0.2f" % decision_tree_scores.std())


# ## Confusion matrix

# In[ ]:


from sklearn.metrics import plot_confusion_matrix

decision_tree_classifier.fit(X_train, y_train)
plot_confusion_matrix(decision_tree_classifier, X_test, y_test, normalize='true')


# Decision trees are prone to overfitting, specially when it's deep...

# ## A different approach
# 
# A possible different approach could be training the model considering an undersample of the majority class, but, for now, I won't do this.
# 

# ## Tree visualization
# 
# As said earlier, the trained model may provide some good insights about the importance of the features. The plot of the tree is shown right below.

# In[ ]:


import graphviz
decision_tree_classifier.fit(X, y)
dot_data = tree.export_graphviz(decision_tree_classifier, out_file=None, 
                     rotate=True,
                     feature_names=X.columns,
                     class_names=decision_tree_classifier.classes_,
                     filled=True, rounded=True,  
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph


# # Random Forest
# 
# Another approach could be the use of a random forest, which is basically a set of trained decision trees trained with different subsamples of the dataset. In this case, we'll set the number of estimators (trees) to thirty, since the dataset is small. We set
# 
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier(n_estimators=30, max_depth=3, class_weight='balanced_subsample')
random_forest_scores = cross_val_score(random_forest_classifier, X, y.values.ravel(), cv=5)
print("Scores: {}".format(random_forest_scores))
print("Mean: %0.2f" % random_forest_scores.mean())
print("Standard deviation: %0.2f" % random_forest_scores.std())


# ## Confusion matrix

# In[ ]:


from sklearn.metrics import plot_confusion_matrix

random_forest_classifier.fit(X_train, y_train)
plot_confusion_matrix(random_forest_classifier, X_test, y_test, normalize='true')


# ## A different approach
# 
# A possible different approach could be the usage of balanced random forests. With this approach, we could undersample the majority class for each decision tree that is used to form the random forest. We can do this by setting the _sampling_strategy_ parameter of the balanced random forest to _"majority"_.

# In[ ]:


from imblearn.ensemble import BalancedRandomForestClassifier

balanced_random_forest_classifier = BalancedRandomForestClassifier(n_estimators=30, max_depth=3, sampling_strategy='majority')
balanced_random_forest_scores = cross_val_score(balanced_random_forest_classifier, X, y.values.ravel(), cv=5)
print("Scores: {}".format(balanced_random_forest_scores))
print("Mean: %0.2f" % balanced_random_forest_scores.mean())
print("Standard deviation: %0.2f" % balanced_random_forest_scores.std())


# ## Confusion matrix

# In[ ]:


from sklearn.metrics import plot_confusion_matrix

balanced_random_forest_classifier.fit(X_train, y_train)
plot_confusion_matrix(balanced_random_forest_classifier, X_test, y_test, normalize='true')


# # Notes
# 
# ### About the models
# - These trained models are, probably, very limited to the patterns of the set of data that was used to train them. In other words, they're probably not general enough, although they can be used as an initial step, given that dataset is probably growing.
# 
# ### Possible improvements
# - Different test sets could be used to plot different confusion matrices. Then, we could better analyse the accuracy of the models considering the different classes
# - False-positives should be favored over false-negatives, since one of the utilities of these trained models could be to do triage of patients
# 
# ### About the data
# - More data instances could have **basic health markers** (blood and urine markers, for example)
# - A **chest X-ray dataset** could also be made available
# - **Commorbities** might be useful for predicting patient's future clinical conditions
# 
# #### To be continued...
