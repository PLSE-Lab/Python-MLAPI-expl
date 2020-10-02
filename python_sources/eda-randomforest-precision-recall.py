#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Loading the data

# In[ ]:


train, test = pd.read_csv('../input/titanic/train.csv'), pd.read_csv('../input/titanic/test.csv')


# # A quick glance at the data

# In[ ]:


train.head()


# The **info** function basicly shows information about the data. column names, types..etc
# 
# **Something to notice here is that some values are missing, so we will have to take care of these later.**

# In[ ]:


train.info()


# Some of the columns are of type **object**, meaning that those values could be any Python object. However, since the data are coming from a CSV file, they must be **text attributes**, known as **categorical attributes**.  
# 
# 
# To look closely at the categorical attributes for now we may use the function **value_counts**.

# In[ ]:


train['Sex'].value_counts()


# Similarly, to have a closer look at the numerical attributes, we do the following: 

# In[ ]:


train.describe()


# Looks like some columns are pretending to be **numerical** but actually they are not! E.g. the attribute **Pclass**'s value counts are as shown below. The values belong to one of 3 **categories**, classes (1, 2, 3). Similar attributes are **Survived** which is our target label. We will take care of these later. 

# In[ ]:


train['Pclass'].value_counts()


# Furthomore, we may look at the distributions of the attributes.

# In[ ]:


train.hist(bins=50, figsize=(15, 10))


# * The histogram of the attribute **Pclass** proves what we mentioned earlier. Pclass is indeed a categorical attribute, or **ordinal** attribute. Though we need to if the order does actually matter.
# 
# * Suprisinlgy two more columns could be considered as categorical attribures: **Parch** and **SibSp**
# 
# 
# * Finally, we will need to have all the values to be on the same scale.

# # Creating a Test Set
# 
# At this point, before touching the data we would've splitted the data and put our test set aside. However, there is no need for that, Kaggle, did this step for us. 

# # Discovering the Data to Gain Insights
# 
# We will create a copy to work on and keep our original data clean. 

# In[ ]:


train_c = train.copy()


# The attribute **Pclass** contains 3 categories or classes. 
# * Let's see the number of passengers that fall in each class. 
# * The number of passengers that suvived from each class.
# * The survival ratio of each class: Survived/PassengerCount.
# 
# The following function does that for us. 

# In[ ]:


def cat_survival_rate(column_name):
    """
    Counting the people survived in each class. And calculating the survial ratio for each. 
    """
    cat_survived = train_c.groupby(column_name).agg({'PassengerId':'count', 'Survived':'sum'}
                                                   ).reset_index()
    cat_survived['survival_rate'] = cat_survived.Survived/cat_survived.PassengerId
    return cat_survived.rename(columns={'PassengerId':'PassengerCount'})


# In[ ]:


pclass_survival_ratio = cat_survival_rate('Pclass')
pclass_survival_ratio


# Let's graph this information.

# In[ ]:


def plot_cat_survived(df, colum_name):
    fig = plt.figure(figsize=(8, 5))
    plt.bar(df[colum_name]-.2, df.PassengerCount, width=.3,label='Passengers Count')
    plt.bar(df[colum_name]+.1, df.Survived, width=.3, label='Survived')
    plt.legend()
    plt.show()
plot_cat_survived(pclass_survival_ratio, 'Pclass')


# Somethings to notice:
# * There are more passengers in class 3 than class 2, and class 2 has more passengers than class 1. This could tell us that class 1's ticket is **expensive**. 
# * The **survival ratio** is different for each class. Apparently, Many of class 1 survived. Let's see the follwoing graph to see this.

# In[ ]:


def plot_cat_survival_ratio(df, column_name):
    f, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[column_name], df.survival_rate,label='Survial Ratio', marker='s')
#     ax.set_xticks([1, 2, 3])
#     ax.set_xticklabels(['class 1', 'class 2', 'class 3'])
    plt.legend()
    plt.show()
plot_cat_survival_ratio(pclass_survival_ratio, 'Pclass')


# * Class 1's survival ratio is close to **.6**. 
# * Class 2's survival ratio is close to **.5**. 
# * Class 3's survival ratio is close to **.25**.
# 
# **We can conclude that the higher the class(low in value), the higher the survival ratio is.**
# 

# Let us investigate a couple of more columns the same way with the same functions we created. 

# In[ ]:


parch_survival_ratio = cat_survival_rate('Parch')
parch_survival_ratio


# In[ ]:


plot_cat_survival_ratio(parch_survival_ratio, 'Parch')


# For the attribute **Parch**(parent/children), the line isn't always decreasing. However, still the higher the ratio, the higher the possibility of the passenger to not survive. 

# In[ ]:


sibsp_survival_ratio = cat_survival_rate('SibSp')
sibsp_survival_ratio


# In[ ]:


plot_cat_survival_ratio(sibsp_survival_ratio, 'SibSp')


# The ratio in the above graph is decreaseing all the way up(except for the first category). Looks like the more sibligs you have(since the denominator is either 1 or 0, the attribute just means the number of sibilings of a passenger) given that you have one spouse on board, the lower the chance of you surviving.

# In[ ]:


train_c.groupby('Pclass').sum()


# Enough huh? Time to look for correlations!

# # Looking For Correlations
# 
# 
# Let's compute the standard correlation coefficient between every pair. 

# In[ ]:


corr_matrix = train_c.corr()


# In[ ]:


corr_matrix


# Let's see how every attribute is related to our target label **Survived**.

# In[ ]:


corr_matrix['Survived'].sort_values(ascending=False)


# Pretty much what we expected for some attributes. The **Pclass** is showing a high correlation. It is negative because the lower the value(1st class) the higher the survival ratio. Also, automatically the Pclass attribute will be related to the **Fare** attribute.
# 
# **We may go on and try some meaningful combinations to get more features!**

# In[ ]:


train_c['Parch_SibSp'] = train_c['SibSp'] + train_c['Parch']
train_c['age_parch'] = train_c['Parch']/train_c['Age']
train_c['age_Sibsp'] = train_c['Age']*train_c['SibSp']


# In[ ]:


corr_matrix = train_c.corr()
corr_matrix['Survived'].sort_values(ascending=False)


# **Great**! We got more features and even some of them are more related to our target attribute!

# # Data Cleaning
# 
# 
# First, we will go on a copy the original data again and drop the target attribute. And save the labels to a variable as well.
# 

# In[ ]:


train_c = train.drop('Survived', axis=1)
train_c_labels = train['Survived'].copy()


# ### Missing Values
# 
# We need to fill the missing values in our data. We may do any of the following to fix this: 
# 1. Drop the whole attribute.
# 2. Drop those training exampls.(passengers)
# 3. Fill in those values with some value(e.g. mean, median, etc.)
# 
# Let's show what attributes have missing value again. 

# In[ ]:


train_c.info()


# Only three attributes: 
# * Age: 177 missing values.
# * Cabin: 687 missing values. 
# * Embarked: 2 missing values. 
# 
# Let's start with the first one: **Age**. It has 177 missing values. Therefore, we won't drop the whole attribute or even drop the corresponding rows. We will go with option three "filling with some value". To choose that value, it is worthy looking at the distribution of that attribute. 

# In[ ]:


train_c.Age.hist()
plt.show()


# Filling the missing values with the **mean** seems reasonable. However, one could go deeper to understand why those values are missing in the first place, and make assumptions.
# 
# When it comes to the next attribute, **Cabin**. Dropping the whole attribute makes sense, since approximatly 80% of the values are misssing. 
# 
# Finally, for the the last one, **Embarked**. It is only missing 2 values! We can just fill these with the most frequent one. 
# 
# For this task. We will be using the **SimpleImputer** from Sklearn.

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')


# In[ ]:


num_attrs = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
imputer.fit(train_c[num_attrs])  
train_num = imputer.transform(train_c[num_attrs])


# In[ ]:


# converting the output back to a dataframe. 
train_num = pd.DataFrame(train_num, columns=train_c[num_attrs].columns, index=train_c[num_attrs].index)


# In[ ]:


train_num.info()


# Now, as you see. The **Age** attribute does not have missing values anymore. 
# 
# Let's Drop the column **Cabin**.

# In[ ]:


train_c.drop('Cabin', axis=1)


# And fill the 2 missing values in **Embarked** with the most frequent value.
# 
# We do this on all the categorical values that we are planning to feed to the ML algorithm.

# In[ ]:


imputer_cat = SimpleImputer(strategy='most_frequent')
cat_attrs = ['Sex', 'Embarked']
train_cat = imputer_cat.fit_transform(train_c[cat_attrs])
train_cat = pd.DataFrame(train_cat, columns=train_c[cat_attrs].columns, index=train_c[cat_attrs].index)
train_cat.info()


# ### Handling Text and Categorical Attributes
# 
# We have been ignoring the categorical attributes. Now it is the time to handle them. Convert them from text to numbers. 
# 
# We will only consider two categorical attributes: 
# * Sex
# * Embarked
# 
# We will use **OneHotEncoder** to convert the categorical values to one hot vectors. 

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
cat_ecnoded = cat_encoder.fit_transform(train_cat)
cat_ecnoded.toarray()


# ### Transformers & Pipelines
# 
# **Great**! We have transformed our data in different ways, but in different places. This could be more organized. Let's user Transformers and pipelines to do this.
# 
# 
# We will build a transformer to perform the combination of the attributes we did eariler.

# In[ ]:


train_c.head()


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, idx):
        self.idx = idx
    def fit(self, X, y=None):
        return self
    def transform(self, X):        
        parch_SibSp = X[:,self.idx[1]]*X[:, self.idx[2]]
        age_parch = X[:, self.idx[2]]/X[:, self.idx[0]]
        age_Sibsp = X[:, self.idx[2]]*X[:, self.idx[1]]
        return np.c_[X, parch_SibSp, age_parch, age_Sibsp]
        
attr_adder = CombinedAttributesAdder(idx=[4, 5, 6])
train_extra_attrs = attr_adder.transform(train_c.values)


# At this moment. We used transformers provided by Sklearn, and we built ours. One important thing to do is to do these transformations in order. To make things easier, we will use pipelines. First, we start with a pipeline for the numerical then the categorical data.

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('attrs_adder', CombinedAttributesAdder(idx=[0, 1, 2])),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])


# Now, to handle all the columns together, we are going to use **ColumnTransformer** from Sklearn.

# In[ ]:


from sklearn.compose import ColumnTransformer
num_attrs = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']
cat_attrs = ['Sex', 'Embarked']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attrs),
    ('cat', cat_pipeline, cat_attrs)
])

train_prepared = full_pipeline.fit_transform(train_c)
test_prepared = full_pipeline.fit_transform(test)


# That is it! The data is ready for the machine learning algorithms. Time to select & train a mode.

# # Select and Train a Model
# 
# We will start with a **Stochastic Gradient Descent** (SGD) classifier. It deals with training instances independently. Also, we will evaluate the model using cross vlidation. 

# In[ ]:


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier()


# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, train_prepared, train_c_labels, cv=3, scoring='accuracy')


# If you run this cell multiple times, you will end up with different scores. However, the score is usually in the seventies. 
# 
# 
# What if we build a classifier than classifes every instance as no Survived (0)? Our classifier's accuracy would be 61. which is close to what we are getting already. That's why we better not trust accuracy as a performance measure for classifiers. 

# In[ ]:


train['Survived'].value_counts() #549/(342 + 549) ~= 61


# #### Confusion Matrix
# 
# 
# Since accuracy is not a preferred performance measure for this problem. We will look at somethign called **confusion matrix**. Each row in a confusion matrix represents an actual class whereas each column represents a predicted class. To calculate our confusion matrix, we need to have predictions for the target label.
# 

# In[ ]:


from sklearn.model_selection import cross_val_predict
preds = cross_val_predict(sgd_clf, train_prepared, train_c_labels, cv=3)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(train_c_labels, preds)


# * The first row is for the negative class (did not survived). 450 of them were correctly classified, while 99 were not. 
# * The next row, however, is for the positive class  survived). 131 of them were wrongly classified, while 211 were correctly classified. 
# 
# 
# 
# #### The confusion matrix gives us lots of information. We need more concise matrices. Like: 
# * Precision: the accuracy of the positive predictions. 
# * Recall: the ratio of the positive instances(survived) that are detected correctly.

# In[ ]:


from sklearn.metrics import precision_score, recall_score
precision_score(train_c_labels, preds)


# In[ ]:


recall_score(train_c_labels, preds)


# A good way to combine both scores is the F1 score. which is the harmoic mean(gives more weight to the low values) of both. F1 score then prefers classifiers that have similar precisoin and recall. However, sometimes a problem may give more importance to the precision or the recall. 

# In[ ]:


from sklearn.metrics import f1_score
f1_score(train_c_labels, preds)


# #### Precision/Recall Trade-off
# 
# Increasing precision reduces recal, and vice versa. Usually, a threshold decides this. We will compute precisions and recalls for all possible thresholds. 

# In[ ]:


scores = cross_val_predict(sgd_clf, train_prepared, train_c_labels, cv=3, method='decision_function')


# In[ ]:


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(train_c_labels, scores)


# In[ ]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend()
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# Let's get a classifier that has 80% precision.

# In[ ]:


threshold_80_precision = thresholds[np.argmax([precisions >= .80])]
preds_80_precision = (scores >= threshold_80_precision)


# In[ ]:


precision_score(train_c_labels, preds_80_precision)


# In[ ]:


recall_score(train_c_labels, preds_80_precision)


# Very low recall as expected! Let us try another algorithm.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)


# In[ ]:


preds = cross_val_predict(forest_clf, train_prepared, train_c_labels, cv=3)
precision_score(train_c_labels, preds)


# In[ ]:


recall_score(train_c_labels, preds)


# In[ ]:


f1_score(train_c_labels, preds)


# This is a huge improvement! The f1 score of the RandomForestClassifier is much better than the SGDClassifier. 
# 
# 
# Let us try another algorithm: **LogisticRegression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
preds = cross_val_predict(LogisticRegression(), train_prepared, train_c_labels, cv=3)
precision_score(train_c_labels, preds)


# In[ ]:


recall_score(train_c_labels, preds)


# In[ ]:


f1_score(train_c_labels, preds)


# Great! This is even higher than what RandomForestClassifier achieved.
# 
# ## Lets Fine-Tune two of our models
# 
# Finally, we will fine tune our models and test them on the test set. 

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid={"C":np.logspace(-3,3,7), "penalty":["l2"]}
logistic_clf = GridSearchCV(LogisticRegression(), param_grid, cv=5, verbose=0)
logistic_clf_grid = logistic_clf.fit(train_prepared, train_c_labels)


# In[ ]:


logistic_clf_grid.best_params_


# In[ ]:


lg_best = LogisticRegression(C = 0.1, penalty='l2')
preds = cross_val_predict(lg_best, train_prepared, train_c_labels, cv=3)
f1_score(train_c_labels, preds)


# In[ ]:


logistic_final_preds = logistic_clf_grid.predict(test_prepared)


# In[ ]:


submit = pd.read_csv('../input/titanic/gender_submission.csv')
submit['Survived'] = logistic_final_preds
submit.to_csv('logistic_submission.csv', index=False)


# In[ ]:


param_grid = { 
    'n_estimators': [80, 100, 120, 140, 150],
    'max_features': ['auto', 'sqrt', 'log2']
}
forest_clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, verbose=0)
forest_clf_grid = forest_clf.fit(train_prepared, train_c_labels)


# In[ ]:


forest_clf_grid.best_params_


# In[ ]:


forest_best = RandomForestClassifier(max_features='sqrt', n_estimators=140)
preds = cross_val_predict(forest_best, train_prepared, train_c_labels, cv=3)
f1_score(train_c_labels, preds)


# In[ ]:


forest_final_preds = forest_clf_grid.predict(test_prepared)


# In[ ]:


submit = pd.read_csv('../input/titanic/gender_submission.csv')
submit['Survived'] = forest_final_preds
submit.to_csv('forest_submission.csv', index=False)


# # Conclusion
# 
# The purpose of this notebook is to get you started and to help you practice some of the basic data analysis and machine learning skills. We have not done everything possible to increase the accuracy of the classifier. We could have spent more time understanding our features and adding more to score better. This **Forest** submission's score is **0.77272**. 
# 
# Thank you!
