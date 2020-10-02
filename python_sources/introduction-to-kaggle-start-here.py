#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
Image("../input/images/welcome.png")


# <h1><center>**An Introduction to Applied Machine Learning with Kaggle!**</center></h1>
# <h2><center>**Learn the tools needed to start competing in data science competitions!**</center></h2>
# 

# # Packages we'll be using
# <center><img src="https://i.pinimg.com/originals/93/50/03/9350031e26b208efe5e53165022deecc.png" width="150px"/></center>

# * **Pandas** is a versatile package built for data manipulation and analysis. Many Kaggle competitions work with datasets stored in the form of `.csv`, `.xlxs` or `.tsv` files. Pandas gives us the ability to import these kinds of datasets into our workspace, to begin our exploratory data analysis, and to build intuition about how we should proceed with the data.

# <center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/NumPy_logo.svg/1200px-NumPy_logo.svg.png" width="200px"/></center>

# * We will also be using a package called **NumPy** which is a strong package that allows python objects (including Pandas arrays!) to be stored as N-dimensional arrays, or _tensors_.
# <br>
#     * Vectors are 1-D tensors, and Matrices are 2-D tensors.
# <br>
# * This is important because later we might be manipulating one data structure of image data that has 3 channels (RGB), that each have a height and a width, and we will be passing these through our classifier in "mini batches" to perform our analysis on multiple images at once. Being able to store data in high dimensions is _fundamental_ to data science!
# <img src="data:image/jpeg;base64,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" width="400px"/>

# **Matplotlib** is the most widely-used python package for plotting that is based on Matlab's plotting interface. We'll be using this for all sorts of purposes. We'll use it during exploratory data analysis, while evaluating our models, and when tuning hyperparameters for our models in this demo, but the uses are limitless! Another good alternative for plotting in python is ggplot2 (which we won't be using here but is worth checking out!), which is a plotting package originally from R. Also, **seaborn** works well with Pandas objects and is built on top of matplotlib, so it has some advanced plotting functions like heatmaps, dendrograms, and more!

# # Importing data and using pandas

# In[ ]:


# Import the necessary packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Magic function that elimates the necessity of using plt.show() after plotting with matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# To import data with pandas, we use the `.read_csv()` function. Once we have our data stored into a pandas DataFrame, we can ****print out the first five rows**** with the `.head()` method for DataFrame objects. Similarly, we can print out the *last* five rows with `.tail()`.

# In[ ]:


#Importing our data using pandas
df = pd.read_csv('../input/qclkaggle/train.csv') # df is now our Pandas DataFrame!
df.head() # print the first 5 rows of df


# In[ ]:


print(df.shape, '\n')
print(df.columns.to_list()) # returns a list of the column names


# In[ ]:


print(df.describe()) # gives a summary of statistics for each column


# In[ ]:


print(df.info()) # Returns information about data types and missing entries


# If we want to select all entries in the DataFrame for a particular column, we can use either `df[column name]` or `df.column_name`

# In[ ]:


(df.Survived == df['Survived']).head() # Both of these are methods for selecting one column


# In[ ]:


# To select more than one column, we must pass a list of the columns we want!
cols = ['PassengerId', 'Sex', 'Parch']
df[cols].head()


# # Exploratory Data Analysis

# In[ ]:


# `pairplot()` may become very slow with the SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")


# In[ ]:


# First step exploratory data analysis. Shows interactions of all features.
sns.pairplot(df)


# In[ ]:


# Countplot will give a barchart of the frequencies of nominal data
sns.countplot('Pclass', data=df)
# Try changing Pclass to PassengerId (a non-nominal feature) to see what happens!


# In[ ]:


# Using the "hue" parameter allows us to visualize interactions between two features
sns.countplot('Sex', hue='Survived', data=df)


# In[ ]:


# We can grab a subset of the data by using logical indexing, loc, or iloc
df[df.Survived == 0].head()


# In[ ]:


# loc and iloc methods use square brackets, not parentheses!
# loc searches for a specified row in the index, then returns the value of the given column, or returns all values for that row if no column given
print(df.loc[0])
print('\n')
print(df.loc[0, 'Name'])


# In[ ]:


df.index = [f'Passenger_{a}' for a in range(len(df))]
print(df.loc['Passenger_0'])


# In[ ]:


df.head()


# In[ ]:


df.index = list(range(len(df)))


# In[ ]:


df.iloc[:50, :].head()


# In[ ]:


print(df.iloc[:50, :].shape)


# In[ ]:


print(df[df.Survived == 1].Age.mean())
print(df[df.Survived == 0].Age.mean())
df[df.Survived == 1].Age.plot.hist()


# In[ ]:


print(df[df.Survived == 1]['Age'].mean())
print(df[df.Survived == 0]['Age'].mean())
df[df.Survived == 1]['Age'].plot.hist()


# In[ ]:


# using the .plot.PLOT_TYPE() method works on pandas DataFrames and allows for quick, easy plotting
df[df.Survived == 0].Age.plot.hist(bins=30, color='orange')
df[df.Survived == 1].Age.plot.hist(bins=30, color='blue', alpha=0.6)


# In[ ]:


# Whereas a countplot will tell you HOW MANY of each feature interactions there are, a barplot will tell you what the AVERAGE interaction is
sns.barplot('Sex','Survived',hue='Pclass',data=df)


# In[ ]:


# using the .corr() method on pandas DataFrames returns on array of correlations between variables
print(df.corr())
sns.heatmap(df.corr())


# In[ ]:


# as the .astype() method to change data storage types in dataframes for more efficient storage
# we will drop the target variable, and features that are not too useful for prediction
target = df['Survived'].astype('int16')
df = df.drop(['Ticket','Survived','PassengerId','Cabin'], axis=1)
target.head()


# In[ ]:


# Hardly any machine learning algorithm can handle missing values (Go RandomForest!!) so we have to do something about these entries
df.isnull().sum()


# In[ ]:


# Figure out which Embarked value is most common
sns.countplot('Embarked', data=df)


# # Feature Engineering

# In[ ]:


df['Age'] = df['Age'].fillna(np.nanmedian(df['Age'])) # Impute missing values with average age
df['Embarked'] = df['Embarked'].fillna('S') # Impute missing values with most common value for Embarked column


# In[ ]:


df['Name'] = df['Name'].str.len() # Replace the actual name with the length of the name. Turns out to be a good predictor!
df['FamilySize'] = df['SibSp'] + df['Parch'] # SibSp = Number of siblings/spouses and Parch = number of parents/children
df['isAlone'] = [1 if p == 0 else 0 for p in df['FamilySize']]
df['Embarked'] = df['Embarked'].astype('str')
df.head()


# In[ ]:


from sklearn import preprocessing
# We can use question marks to get information about modules in jupyter! Double question marks takes us right to the source code.
get_ipython().run_line_magic('pinfo2', 'preprocessing.label')


# In[ ]:


# Using LabelEncoder on numerical data
encoder = preprocessing.LabelEncoder()
Pclass = df['Pclass'].astype('str')
print(Pclass.head(10))
print(encoder.fit_transform(Pclass)[:10])


# In[ ]:


# Using LabelEncoder on alphabetical data
encoder = preprocessing.LabelEncoder()
Embarked = df['Embarked'].astype('str')
print(Embarked.head(10))
print(encoder.fit_transform(Embarked)[:10])


# In[ ]:


# Convert categorical variables to numerical
cat_names = ['Sex','Embarked']
col_names = df.columns.to_list()
col_names = [i for i in col_names if i not in cat_names]
encoders = []
scalers = []
for i in cat_names:
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(df[i].astype('str'))
    df[i] = encoder.transform(df[i].astype('str'))
    encoders.append(encoder)
for i in col_names:
    scaler = preprocessing.StandardScaler()
    scaler = scaler.fit(np.array(df[i]).reshape(-1,1))
    df[i] = scaler.transform(np.array(df[i]).reshape(-1,1))
    scalers.append(scaler)
df.head()


# # Model Selection
# There are a ton of machine learning algorithms, and each of them has its use! There is no model that will perform the best in all scenarios. No free lunch!!

# <img src="https://miro.medium.com/max/477/1*KFQI59Yv7m1f3fwG68KSEA.jpeg" width=400>

# In[ ]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron


# In[ ]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('PCT', Perceptron()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('XGB', XGBClassifier(n_estimators=300, n_jobs=-1)))
models.append(('RF', RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)))


# <img src="https://media.geeksforgeeks.org/wp-content/cdn-uploads/20190523171229/overfitting_1.png" width=700>
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/K-fold_cross_validation_EN.svg/1200px-K-fold_cross_validation_EN.svg.png" width=500>"

# In[ ]:


import time
results = []
names = []
times = []
scoring = 'roc_auc'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    time0 = time.time()
    cv_results = model_selection.cross_val_score(model, df, target, cv=kfold, scoring=scoring)
    time1 = time.time()
    results.append(cv_results)
    names.append(name)
    times.append(time1-time0)
    msg = f"{name} / {scoring}: {round(cv_results.mean(),3)} / StDev: {round(cv_results.std(),4)} / Time: {round(time1-time0,4)}"
    print(msg)


# In[ ]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(121)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel('Accuracy')
plt.xlabel('Model')
ax1 = fig.add_subplot(122)
plt.bar(names,times)
plt.ylabel('Training time')
plt.xlabel('Model')
fig = plt.gcf()
fig.set_size_inches(16, 9)
plt.show()


# # Hyperparameter optimization
# Hyperparmeters are parameters of our model that change performance, but that we have to code ourselves. Other parameters for the model will be determined by training.

# In[ ]:



# max_depth = [6, 8, 10, 12]
# n_estimators = [100, 250]
# subsample = np.linspace(0,1,5)
# colsample_bytree = np.linspace(0,1,5)
# models = []
# for depth in max_depth:
#     for sub in subsample:
#         for col in colsample_bytree:
#             for n in n_estimators:
#                 models.append(XGBClassifier(n_estimators=n, max_depth=depth, subsample=sub, colsample_bytree=col))
# results = []
# scoring = 'accuracy'
# for model in models:
#     kfold = model_selection.KFold(n_splits=5, random_state=42)
#     cv_results = model_selection.cross_val_score(model, df, target, cv=kfold, scoring=scoring)
#     results.append(cv_results.mean())
# best = models[np.argmax(results)]
# print(best)


# In[ ]:


# Now we have to load our test data in and apply the same transformations to it as we did our training data
test_df = pd.read_csv('../input/qclkaggle/test.csv')
test_df.drop(['PassengerId', 'Cabin', 'Ticket'],axis=1, inplace=True)
test_df['Name'] = test_df['Name'].str.len() # Replace the actual name with the length of the name. Turns out to be a good predictor!
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] # SibSp = Number of siblings/spouses and Parch = number of parents/children
test_df['isAlone'] = [1 if p == 0 else 0 for p in test_df['FamilySize']]
test_df['Age'] = test_df['Age'].fillna(np.nanmedian(df['Age'])) # Impute missing values with average age
test_df['Embarked'] = test_df['Embarked'].fillna('S') # Impute missing values with most common value for Embarked column
for k, i in enumerate(cat_names):
    test_df[i] = encoders[k].transform(test_df[i].astype('str'))
for k, i in enumerate(col_names):
    test_df[i] = scalers[k].transform(np.array(test_df[i]).reshape(-1,1))
test_df.head()


# # Prediction on test set and submission!

# In[ ]:


# define a new model based on the best performance from our grid search
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1.0, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=12,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=0.5, verbosity=1)
model.fit(df, target)
predictions = model.predict(test_df)
print(model.score(df, target))


# In[ ]:


sub = pd.read_csv('../input/qclkaggle/sample_submission.csv')
sub.columns = ['Id', 'Category']
sub.Category = predictions
sub.to_csv('first_predictions.csv', index=False)


# ## Congratulations!! You just finished your first Kaggle pipeline! Here are a few extra resources to get going with data science. 
# * [Andrew Ng's Machine Learning MOOC (Stanford)](https://www.coursera.org/learn/machine-learning?ranMID=40328&ranEAID=vedj0cWlu2Y&ranSiteID=vedj0cWlu2Y-E_.kzMGyB_9JlJJnMqKeIw&siteID=vedj0cWlu2Y-E_.kzMGyB_9JlJJnMqKeIw&utm_content=10&utm_medium=partners&utm_source=linkshare&utm_campaign=vedj0cWlu2Y)
# * [Jeremy Howard's collection of fast.ai MOOCs](http://course.fast.ai)
# * [mlcourse.ai has videos, Jupyter notebooks, and practice assignments](http://mlcourse.ai)
# * Tons of stuff on YouTube, Udacity, Coursera, EdX, etc.

# In[ ]:





# In[ ]:




