#!/usr/bin/env python
# coding: utf-8

# [Link for description of dataset](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)<br>
# [Research paper](http://users.fs.cvut.cz/ivo.bukovsky/PROJEKT/Data/Realna/BIO/EEG/reference/PRE61907.pdf)<br>
# [Kaggle](https://www.kaggle.com/harunshimanto/epileptic-seizure-recognition)
# 
# Electroencephalography (EEG) is an electrophysiological monitoring method to record electrical activity of the brain.

# ### Short description of dataset

# |Data Set Characteristics||Attribute Characteristics||Associated Tasks||Number of Instances||Number of Attributes||Missing Values?|
# |---||---||---||---||---||---|
# |Multivariate, Time-Series||Integer, Real||Classification, Clustering||11500||179||N/A|

# ### Plan
#     - To be introduced with dataset description
#     - Change the y target column (make a binary classification task)
#     - Remove Unnamed: 0 column (additionaly check the importance of it)
#     - EDA + Smart visualisation of data
#     - Make pipelines for all the approaches for binary classification task + make a comperison table of results
#     ----------------
#     - PCA or ICA (from mne) --> reduce size of data
#     - Use previous pipelines for reduced data + make a comperison table of results
#     - Approaching with mne library
#     - Model tuning!!!
#     

# In[ ]:


#THIS IS DEFAULT KAGGLE CELL, WHICH DOWNLOAD OUR DATA TO PATH: /kaggle/input/epileptic-seizure-recognition/Epileptic Seizure Recognition.csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# First of all let us import all necessary libraries

# In[ ]:


import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

import plotly
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import init_notebook_mode, iplot

import imblearn


import seaborn as sns

init_notebook_mode(connected=True)
pd.set_option('display.max_columns', 100)


# Let us got to the directory where we have our data (.csv file, named: Epileptic Seizure Recognition.csv)

# In[ ]:


cd /kaggle/input/epileptic-seizure-recognition/


# Or alternatevily read data directly from web application (additionally we can download data locally)

# In[ ]:


#data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv')


# In[ ]:


#Read our data (from Kaggle)
data = pd.read_csv('Epileptic Seizure Recognition.csv')


# In[ ]:


#Have a look on first five rows
data.head()


# In[ ]:


#Have a look on last five rows
data.tail()


# In[ ]:


#Check dataframe shape (11500 rows and 180 columns(features))
data.shape


# So, after quick looking we observe weird names of features.<br>
# Let us look throught description of features and target (y) feature also.

# <b>From data description:</b> we divided and shuffled every 4097 data points into 23 chunks, each chunk contains 178 data points for 1 second, and each data point is the value of the EEG recording at a different point in time.<br>
# 
# So now we have 23 x 500 = 11500 pieces of information(row), each information contains 178 data points for 1 second(column), the last column represents the label y {1,2,3,4,5}.

# Let us look more preciesly on <b>y</b> column and it's value's importance for our case.<br>
# 
# <b>y</b> contains the category of the 178-dimensional input vector. Specifically y in {1, 2, 3, 4, 5}:
#  - 5 - eyes open, means when they were recording the EEG signal of the brain the patient had their eyes open
#  - 4 - eyes closed, means when they were recording the EEG signal the patient had their eyes closed
#  - 3 - Yes they identify where the region of the tumor was in the brain and recording the EEG activity from the healthy brain area
#  - 2 - They recorder the EEG from the area where the tumor was located
#  - 1 - Recording of seizure activity
# All subjects falling in classes 2, 3, 4, and 5 are subjects who did not have epileptic seizure. Only subjects in class 1 have epileptic seizure.

# From the first view we can assume we need to solve **multi-classification task**, but, after accurate exploaring definitions of classes of <b>y</b>, we can realeyes we can *reform* our **multi-classification task** to **binary classification task**.<br>
# For that we can just combine {2,3,4,5} classes as 0 class (not epileptic seizure) and keep {1} class as 1 (epileptic seizure).

# In[ ]:


#Before joining the classes, let us check y values for balancing
data['y'].value_counts()


# In[ ]:


data.y.hist();


# By the chance we faced with plotting, let us quickly plot few curves.<br>
# As we can observe, it seems we have few types of curves. Let us keep it in our minds and analyse it during further steps.

# In[ ]:


plt.figure(figsize=(50,4))
plt.subplot(131)
[plt.plot(data.values[i][1:-1]) for i in range(23)];


# ### Change the y target column (make a binary classification task)

# In[ ]:


dic = {5: 0, 4: 0, 3: 0, 2: 0, 1: 1}
data['y'] = data['y'].map(dic)


# In[ ]:


#Check the difference in dataframe in general

#Check which values do we have in y column
print(data['y'].value_counts())

data.head()


# ### "Remove Unnamed" column (it has information which we don't need)

# In[ ]:


data = data.drop('Unnamed', axis = 1)


# ### Let us shuffle data because of previous manipulations

# In[ ]:


data = shuffle(data)


# So, for now let us have a look on the description of our data. <br>
# We can do it using several approaches:
#  - 1) ususal: pd.description(), pd.info()
#  - 2) ff.create_table (approach from plotly - in our case unnecessary, but we will use it for interesting)

# In[ ]:


# table_cat = ff.create_table(data.describe().T, index=True, index_title='Signals')
# iplot(table_cat)


# In[ ]:


data.describe()


# In[ ]:


data.info()


# Let us go further:

# In[ ]:


#Let us group all the Epileptic occureses and Non Epileptic
print('Number of records of Non Epileptic {0} VS Epilepttic {1}'.format(len(data[data['y'] == 0]), len(data[data['y'] == 1])))


# In[ ]:


#Description of Non Epileptic
data[data['y'] == 0].describe().T


# In[ ]:


#Description of Epileptic

data[data['y'] == 1].describe().T


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


print('Totall Mean VALUE for Epiletic: {}'.format((data[data['y'] == 1].describe().mean()).mean()))
print('Totall Std VALUE for Epiletic: {}'.format((data[data['y'] == 1].describe().std()).std()))


# In[ ]:


print('Totall Mean VALUE for NON Epiletic: {}'.format((data[data['y'] == 0].describe().mean()).mean()))
print('Totall Std VALUE for NON Epiletic: {}'.format((data[data['y'] == 0].describe().std()).std()))


# We can see quiet big difference, probably, we wiil demand to normalize/scale our data. 

# ### Let us plot and have a look on EPILIPTIC and NOT EPILETTIC occureses

# In[ ]:


#Few cases of Not Epileptic case
[(plt.figure(figsize=(8,4)), plt.title('Not Epileptic'), plt.plot(data[data['y'] == 0].iloc[i][0:-1])) for i in range(5)];


# In[ ]:


#Few cases of Epileptic case
[(plt.figure(figsize=(8,4)), plt.title('Epileptic'), plt.plot(data[data['y'] == 1].iloc[i][0:-1])) for i in range(5)];


# So, as we can observe, records of Epileptic seusizes are more smooth and looks like have a tendency.

# ### Let us make a scatter plot of values for Epiletpic and Not Epileptic occureses

# In[ ]:


#lists of arrays containing all data without y column
not_epileptic = [data[data['y']==0].iloc[:, range(0, len(data.columns)-1)].values]
epileptic = [data[data['y']==1].iloc[:, range(0, len(data.columns)-1)].values]

#We will create and calculate 2d indicators in order plot data in 2 dimensions;

def indic(data):
    """Indicators can be different. In our case we use just min and max values
    Additionally, it can be mean and std or another combination of indicators"""
    max = np.max(data, axis=1)
    min = np.min(data, axis=1)
    return max, min

x1,y1 = indic(not_epileptic)
x2,y2 = indic(epileptic)

fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(111)

ax1.scatter(x1, y1, s=10, c='b', label='Not Epiliptic')
ax1.scatter(x2, y2, s=10, c='r', label='Epileptic')
plt.legend(loc='lower left');
plt.show()


# In[ ]:


#Just Epileptic
x,y = indic(data[data['y']==1].iloc[:, range(0, len(data.columns)-1)].values)
plt.figure(figsize=(14,4))
plt.title('Epileptic')
plt.scatter(x, y, c='r');


# In[ ]:


#Just Not Epileptic
x,y = indic(data[data['y']==0].iloc[:, range(0, len(data.columns)-1)].values)
plt.figure(figsize=(14,4))
plt.title('NOT Epileptic')
plt.scatter(x, y);


# ### After all, let us go further with ML models. 

# As we realyesed earlier, we can try to normalize data. Let us do it. But before that we will use undersampling approach in order to prevent imbalanced issue

# In[ ]:





# In[ ]:


# define oversampling strategy
oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X, y = oversample.fit_resample(data.drop('y', axis=1), data['y'])

X.shape, y.shape


# Check the balance for y

# In[ ]:


#Let us group all the Epileptic occureses and Non Epileptic
print('Number of records of Non Epileptic {0} VS Epilepttic {1}'.format(len(y == True), len(y == False)))


# Normalizing

# In[ ]:


# X = data.drop('y', axis=1)
# y = data['y']

normalized_df = pd.DataFrame(normalize(X))
normalized_df


# In[ ]:


#Concat back in order to check description:
normalized_df['y'] = y

print('Normalized Totall Mean VALUE for Epiletic: {}'.format((normalized_df[normalized_df['y'] == 1].describe().mean()).mean()))
print('Normalized Totall Std VALUE for Epiletic: {}'.format((normalized_df[normalized_df['y'] == 1].describe().std()).std()))

print('Normalized Totall Mean VALUE for NOT Epiletic: {}'.format((normalized_df[normalized_df['y'] == 0].describe().mean()).mean()))
print('Normalized Totall Std VALUE for NOT Epiletic: {}'.format((normalized_df[normalized_df['y'] == 0].describe().std()).std()))


# In[ ]:


#Let us split our dataset on train and test and than invoke validation approach

X = normalized_df.drop('y', axis=1)
y = normalized_df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

#Check the shapes after splitting
he = X_train, X_test, y_train, y_test
[arr.shape for arr in he]


# Make a pipeline for Classification models:
# - LogisticRegression
# - Support Vector Machines - linear and rbf
# - K-nearest Classifier
# - Decision Tree Classifier
# - Gradient Bossting Classifier

# In[ ]:


#Define set of classifiers for input
models = [LogisticRegression(), SVC(),
          DecisionTreeClassifier(),
          RandomForestClassifier(), 
          GradientBoostingClassifier(),
          KNeighborsClassifier()]

#Check the correctness of list of classifiers and also 
model_name = [type(model).__name__ for model in models]
print(model_name)

# all parameters are not specified are set to their defaults
def classifiers(models):
    columns = ['Score', 'Predictions']
    df_result = pd.DataFrame(columns=columns, index=[type(model).__name__ for model in models])

    for model in models:
        clf = model
        print('Initialized classifier {} with default parameters \n'.format(type(model).__name__))    
        clf.fit(X_train, y_train)
        #make a predicitions for entire data(X_test)
        predictions = clf.predict(X_test)
        # Use score method to get accuracy of model
        score = clf.score(X_test, y_test)
        print('Score of classifier {} is: {} \n'.format(type(model).__name__, score))
        df_result['Score']['{}'.format(type(model).__name__)] = str(round(score * 100, 2)) + '%' 
        df_result['Predictions']['{}'.format(type(model).__name__)] = predictions
    return df_result


# In[ ]:


classifiers(models)


# Tuning Hyperparameters

# In[ ]:


from sklearn.model_selection import KFold
### LogisticRegression
# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = KFold(n_splits=10, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


### KNeighborsClassifier
# define models and parameters
model = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = KFold(n_splits=3, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


### SVC
# define model and parameters
model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = KFold(n_splits=3, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


### RandomForestClassifier
# define models and parameters
model = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = KFold(n_splits=3, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


### GradientBoostingClassifier
# define models and parameters
model = GradientBoostingClassifier()
n_estimators = [10, 100, 1000]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
# define grid search
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = KFold(n_splits=3, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:




