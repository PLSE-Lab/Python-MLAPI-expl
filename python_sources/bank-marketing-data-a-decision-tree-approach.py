#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing Data - A Decision Tree Approach

# ## Aim:
# The aim of this attempt is to predict if the client will subscribe (yes/no) to a term deposit, by building a classification model using Decision Tree.
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load data file
bank=pd.read_csv('../input/bank.csv')
bank.head()


# ## Summay of data
# 
# ### Categorical Variables :
# **[1] job      :** admin,technician, services, management, retired, blue-collar, unemployed, entrepreneur,
#                housemaid, unknown, self-employed, student
# <br>**[2] marital  :** married, single, divorced
# <br>**[3] education:** secondary, tertiary, primary, unknown
# <br>**[4] default  :** yes, no
# <br>**[5] housing  :** yes, no
# <br>**[6] loan     :** yes, no 
# <br>**[7] deposit  :** yes, no ** (Dependent Variable)**
# <br>**[8] contact  :** unknown, cellular, telephone
# <br>**[9] month    :** jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec
# <br>**[10] poutcome:** unknown, other, failure, success
# 
# ### Numerical Variables:
# **[1] age 
# <br>[2] balance
# <br>[3] day
# <br>[4] duration
# <br>[5] campaign
# <br>[6] pdays
# <br>[7] previous **

# In[ ]:


# Check if the data set contains any null values - Nothing found!
bank[bank.isnull().any(axis=1)].count()


# In[ ]:


bank.describe()


# In[ ]:


# Boxplot for 'age'
g = sns.boxplot(x=bank["age"])


# In[ ]:


# Distribution of Age
sns.distplot(bank.age, bins=100)


# In[ ]:


# Boxplot for 'duration'
g = sns.boxplot(x=bank["duration"])


# In[ ]:


sns.distplot(bank.duration, bins=100)


# ### Convert categorical data

# In[ ]:


# Make a copy for parsing
bank_data = bank.copy()


# #### ------------------------------ job ------------------------------

# In[ ]:


# Explore People who made a deposit Vs Job category
jobs = ['management','blue-collar','technician','admin.','services','retired','self-employed','student',        'unemployed','entrepreneur','housemaid','unknown']

for j in jobs:
    print("{:15} : {:5}". format(j, len(bank_data[(bank_data.deposit == "yes") & (bank_data.job ==j)])))


# In[ ]:


# Different types of job categories and their counts
bank_data.job.value_counts()


# In[ ]:


# Combine similar jobs into categiroes
bank_data['job'] = bank_data['job'].replace(['management', 'admin.'], 'white-collar')
bank_data['job'] = bank_data['job'].replace(['services','housemaid'], 'pink-collar')
bank_data['job'] = bank_data['job'].replace(['retired', 'student', 'unemployed', 'unknown'], 'other')


# In[ ]:


# New value counts
bank_data.job.value_counts()


# #### ------------------------------ poutcome ------------------------------

# In[ ]:


bank_data.poutcome.value_counts()


# In[ ]:


# Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'
bank_data['poutcome'] = bank_data['poutcome'].replace(['other'] , 'unknown')
bank_data.poutcome.value_counts()


# #### ------------------------------ contact ------------------------------

# In[ ]:


# Drop 'contact', as every participant has been contacted. 
bank_data.drop('contact', axis=1, inplace=True)


# #### ------------------------------ default ------------------------------

# In[ ]:


# values for "default" : yes/no
bank_data["default"]
bank_data['default_cat'] = bank_data['default'].map( {'yes':1, 'no':0} )
bank_data.drop('default', axis=1,inplace = True)


# #### ------------------------------ housing ------------------------------

# In[ ]:


# values for "housing" : yes/no
bank_data["housing_cat"]=bank_data['housing'].map({'yes':1, 'no':0})
bank_data.drop('housing', axis=1,inplace = True)


# #### ------------------------------ loan ------------------------------

# In[ ]:


# values for "loan" : yes/no
bank_data["loan_cat"] = bank_data['loan'].map({'yes':1, 'no':0})
bank_data.drop('loan', axis=1, inplace=True)


# #### ------------------------------ month, day ------------------------------

# In[ ]:


# day  : last contact day of the month
# month: last contact month of year
# Drop 'month' and 'day' as they don't have any intrinsic meaning
bank_data.drop('month', axis=1, inplace=True)
bank_data.drop('day', axis=1, inplace=True)


# #### ------------------------------ deposit ------------------------------

# In[ ]:


# values for "deposit" : yes/no
bank_data["deposit_cat"] = bank_data['deposit'].map({'yes':1, 'no':0})
bank_data.drop('deposit', axis=1, inplace=True)


# #### ------------------------------ pdays ------------------------------

# In[ ]:


# pdays: number of days that passed by after the client was last contacted from a previous campaign
#       -1 means client was not previously contacted

print("Customers that have not been contacted before:", len(bank_data[bank_data.pdays==-1]))
print("Maximum values on padys    :", bank_data['pdays'].max())


# In[ ]:


# Map padys=-1 into a large value (10000 is used) to indicate that it is so far in the past that it has no effect
bank_data.loc[bank_data['pdays'] == -1, 'pdays'] = 10000


# In[ ]:


# Create a new column: recent_pdays 
bank_data['recent_pdays'] = np.where(bank_data['pdays'], 1/bank_data.pdays, 1/bank_data.pdays)

# Drop 'pdays'
bank_data.drop('pdays', axis=1, inplace = True)


# In[ ]:


bank_data.tail()


# ### ------------------------------ Convert to dummy values ------------------------------

# In[ ]:


# Convert categorical variables to dummies
bank_with_dummies = pd.get_dummies(data=bank_data, columns = ['job', 'marital', 'education', 'poutcome'],                                    prefix = ['job', 'marital', 'education', 'poutcome'])
bank_with_dummies.head()


# In[ ]:


bank_with_dummies.shape


# In[ ]:


bank_with_dummies.describe()


# ### Observations on whole population

# In[ ]:


# Scatterplot showing age and balance
bank_with_dummies.plot(kind='scatter', x='age', y='balance');

# Across all ages, majority of people have savings of less than 20000.


# In[ ]:


bank_with_dummies.plot(kind='hist', x='poutcome_success', y='duration');


# #### Analysis on people who sign up for a term deposite

# In[ ]:


# People who sign up to a term deposite
bank_with_dummies[bank_data.deposit_cat == 1].describe()


# In[ ]:


# People signed up to a term deposite having a personal loan (loan_cat) and housing loan (housing_cat)
len(bank_with_dummies[(bank_with_dummies.deposit_cat == 1) & (bank_with_dummies.loan_cat) & (bank_with_dummies.housing_cat)])


# In[ ]:


# People signed up to a term deposite with a credit default 
len(bank_with_dummies[(bank_with_dummies.deposit_cat == 1) & (bank_with_dummies.default_cat ==1)])


# In[ ]:


# Bar chart of job Vs deposite
plt.figure(figsize = (10,6))
sns.barplot(x='job', y = 'deposit_cat', data = bank_data)


# In[ ]:


# Bar chart of "previous outcome" Vs "call duration"

plt.figure(figsize = (10,6))
sns.barplot(x='poutcome', y = 'duration', data = bank_data)


# > ## Classification

# In[ ]:


# make a copy
bankcl = bank_with_dummies


# In[ ]:


# The Correltion matrix
corr = bankcl.corr()
corr


# In[ ]:


# Heatmap
plt.figure(figsize = (10,10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .82})
plt.title('Heatmap of Correlation Matrix')


# In[ ]:


# Extract the deposte_cat column (the dependent variable)
corr_deposite = pd.DataFrame(corr['deposit_cat'].drop('deposit_cat'))
corr_deposite.sort_values(by = 'deposit_cat', ascending = False)


# > ## Build the Data Model

# In[ ]:


# Train-Test split: 20% test data
data_drop_deposite = bankcl.drop('deposit_cat', 1)
label = bankcl.deposit_cat
data_train, data_test, label_train, label_test = train_test_split(data_drop_deposite, label, test_size = 0.2, random_state = 50)


# In[ ]:


# Decision tree with depth = 2
dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=2)
dt2.fit(data_train, label_train)
dt2_score_train = dt2.score(data_train, label_train)
print("Training score: ",dt2_score_train)
dt2_score_test = dt2.score(data_test, label_test)
print("Testing score: ",dt2_score_test)


# In[ ]:


# Decision tree with depth = 3
dt3 = tree.DecisionTreeClassifier(random_state=1, max_depth=3)
dt3.fit(data_train, label_train)
dt3_score_train = dt3.score(data_train, label_train)
print("Training score: ",dt3_score_train)
dt3_score_test = dt3.score(data_test, label_test)
print("Testing score: ",dt3_score_test)


# In[ ]:


# Decision tree with depth = 4
dt4 = tree.DecisionTreeClassifier(random_state=1, max_depth=4)
dt4.fit(data_train, label_train)
dt4_score_train = dt4.score(data_train, label_train)
print("Training score: ",dt4_score_train)
dt4_score_test = dt4.score(data_test, label_test)
print("Testing score: ",dt4_score_test)


# In[ ]:


# Decision tree with depth = 6
dt6 = tree.DecisionTreeClassifier(random_state=1, max_depth=6)
dt6.fit(data_train, label_train)
dt6_score_train = dt6.score(data_train, label_train)
print("Training score: ",dt6_score_train)
dt6_score_test = dt6.score(data_test, label_test)
print("Testing score: ",dt6_score_test)


# In[ ]:


# Decision tree: To the full depth
dt1 = tree.DecisionTreeClassifier()
dt1.fit(data_train, label_train)
dt1_score_train = dt1.score(data_train, label_train)
print("Training score: ", dt1_score_train)
dt1_score_test = dt1.score(data_test, label_test)
print("Testing score: ", dt1_score_test)


# #### Compare Training and Testing scores for various tree depths used

# In[ ]:


print('{:10} {:20} {:20}'.format('depth', 'Training score','Testing score'))
print('{:10} {:20} {:20}'.format('-----', '--------------','-------------'))
print('{:1} {:>25} {:>20}'.format(2, dt2_score_train, dt2_score_test))
print('{:1} {:>25} {:>20}'.format(3, dt3_score_train, dt3_score_test))
print('{:1} {:>25} {:>20}'.format(4, dt4_score_train, dt4_score_test))
print('{:1} {:>25} {:>20}'.format(6, dt6_score_train, dt6_score_test))
print('{:1} {:>23} {:>20}'.format("max", dt1_score_train, dt1_score_test))


# It could be seen that, higher the depth, training score increases and matches perfects with the training data set. However higher the depth the tree goes, it overfit to the training data set. So it's no use keep increasing the tree depth. According to above observations, tree with a depth of 2 seems more reasonable as both training and test scores are reasonably high.

# In[ ]:


# Let's generate the decision tree for depth = 2
# Create a feature vector
features = bankcl.columns.tolist()

# Uncomment below to generate the digraph Tree.
#tree.export_graphviz(dt2, out_file='tree_depth_2.dot', feature_names=features)


# ****Contents of "tree_depth_2.dot":  ****  
# digraph Tree {  
# node [shape=box] ;  
# 0 [label="duration <= 206.5\ngini = 0.4986\nsamples = 8929\nvalue = [4700, 4229]"] ;  
# 1 [label="poutcome_failure <= 0.5\ngini = 0.3274\nsamples = 3612\nvalue = [2867, 745]"] ;  
# 0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;  
# 2 [label="gini = 0.2733\nsamples = 3380\nvalue = [2828, 552]"] ;  
# 1 -> 2 ;  
# 3 [label="gini = 0.2797\nsamples = 232\nvalue = [39, 193]"] ;  
# 1 -> 3 ;  
# 4 [label="duration <= 441.5\ngini = 0.4518\nsamples = 5317\nvalue = [1833, 3484]"] ;  
# 0 -> 4 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;  
# 5 [label="gini = 0.4996\nsamples = 2762\nvalue = [1340, 1422]"] ;  
# 4 -> 5 ;  
# 6 [label="gini = 0.3114\nsamples = 2555\nvalue = [493, 2062]"] ;  
# 4 -> 6 ;  
# }

# Thee decision tree for depth =2 can be found at below link:
# (I wasn't successful in attaching the image to Kaggle)
# https://i.imgur.com/YML2E7h.png
# 

# Based on the decision tree results, it could be seen that higher the "duration", bank is able to sign up more people to term deposites.

# In[ ]:


# Two classes: 0 = not signed up,  1 = signed up
dt2.classes_


# In[ ]:


# Create a feature vector
features = data_drop_deposite.columns.tolist()

features


# In[ ]:


# Investigate most important features with depth =2

dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=2)

# Fit the decision tree classifier
dt2.fit(data_train, label_train)

fi = dt2.feature_importances_

l = len(features)
for i in range(0,len(features)):
    print('{:.<20} {:3}'.format(features[i],fi[i]))


# ## Predictions

# In[ ]:


# According to feature importance results, most importtant feature is the "Duration"
# Let's calculte statistics on Duration
print("Mean duration   : ", data_drop_deposite.duration.mean())
print("Maximun duration: ", data_drop_deposite.duration.max())
print("Minimum duration: ", data_drop_deposite.duration.min())


# In[ ]:


# Predict: Successful deposite with a call duration = 371 sec

print(dt2.predict_proba(np.array([0, 0, 371, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]).reshape(1, -1)))
print(dt2.predict(np.array([0, 0, 371, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]).reshape(1, -1)))
# column 0: probability for class 0 (not signed for term deposite) & column 1: probability for class 1
# Probability of Successful deposite = 0.51484432


# In[ ]:


# Predict: Successful deposite with a maximun call duration = 3881 sec

print(dt2.predict_proba(np.array([0, 0, 3881, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]).reshape(1, -1)))
print(dt2.predict(np.array([0, 0, 3881, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]).reshape(1, -1)))


# In[ ]:


# Get a row with poutcome_success = 1
#bank_with_dummies[(bank_with_dummies.poutcome_success == 1)]
data_drop_deposite.iloc[985]


# In[ ]:


# Predict: Probability for above

print(dt2.predict_proba(np.array([46,3354,522,1,1,0,1,0,0.005747,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0]).reshape(1, -1)))
#print(ctree.predict(np.array([46,3354,522,1,1,0,1,0,0.005747,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0]).reshape(1, -1)))


# In[ ]:


# Make predictions on the test set
preds = dt2.predict(data_test)

# Calculate accuracy
print("\nAccuracy score: \n{}".format(metrics.accuracy_score(label_test, preds)))

# Make predictions on the test set using predict_proba
probs = dt2.predict_proba(data_test)[:,1]

# Calculate the AUC metric
print("\nArea Under Curve: \n{}".format(metrics.roc_auc_score(label_test, probs)))

