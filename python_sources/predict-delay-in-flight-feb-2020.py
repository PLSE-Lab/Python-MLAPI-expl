#!/usr/bin/env python
# coding: utf-8

# ## Background
# Let's say we have to develop an AI model that can predict flight delays during the
# flight duration based on data about previous delay/turbulence encountered.

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import emoji
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)


# In[ ]:


data = pd.read_csv('/kaggle/input/feb-2020-us-flight-delay/feb-20-us-flight-delay.csv')


# ### Data Format
# Data Source : [US Bureau of Transportation Statistics](https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236)
# 
# After carefully analyzing each data points, I decided to manually pick 9 variable to predict if there will be a delay in the flight.
# - __MONTH__ - Month
# - __DAY_OF_MONTH__ - Day of Month
# - __DAY_OF_WEEK__ - Day of Week
# - __OP_UNIQUE_CARRIER__ - Unique Carrier Code
# - __ORIGIN__ - Origin airport location
# - __DEST__ - Destination airport location
# - __DEP_TIME__ - Actual Departure Time (local time: hhmm)
# - __DEP_DEL15__ - Departure Delay Indicator, 15 Minutes or More (1=Yes, 0=No) [TARGET VARIABLE]
# - __DISTANCE__ - Distance between airports (miles)

# In[ ]:


data.head()


# ## Data Preprocessing

# We might have an extra column in our dataset, let's get rid of it first

# In[ ]:


data = data.drop(['Unnamed: 9'], axis=1)


# Let's find out the distribution of our target variable

# In[ ]:


data['DEP_DEL15'].value_counts()


# We can see that we have highly imbalanced data, as we there are only __14.43%__ rows with the value of 1.0 (Delay in flight).
# 
# We will drop a significant amount of rows where our target variable is 0.0 (No delay in flight).

# In[ ]:


# Split the data into positive and negative
positive_rows = data.DEP_DEL15 == 1.0
data_pos = data.loc[positive_rows]
data_neg = data.loc[~positive_rows]

# Merge the balanced data
data = pd.concat([data_pos, data_neg.sample(n = len(data_pos))], axis = 0)

# Shuffle the order of data
data = data.sample(n = len(data)).reset_index(drop = True)


# Let's quickly remove the NULL values if present any

# In[ ]:


data.isna().sum()


# There are around __~0.5%__ NULL values present in __DEP_TIME__ and __DEP_DEL15__

# In[ ]:


data = data.dropna(axis=0)


# In[ ]:


data.info()


# We see that our target variable __DEP_DEL15__ has the datatype of _float64_.
# 
# Let's convert it into _int_.

# In[ ]:


data['DEP_DEL15'] = data['DEP_DEL15'].astype(int)


# Now let's have a look at the number of columns and rows in our dataset.

# In[ ]:


print(f"There are {data.shape[0]} rows and {data.shape[1]} columns in our dataset.")


# ## Exploratory Data Analysis
# Let's uncover some meaningful and hidden insights out of our dataset.

# In[ ]:


data.describe()


# Apart from the statistics given in the table above, we can also that there are 6 numerical and 3 categorical variables in our dataset.

# Let's quickly visualize the distribution of __DISTANCE__ variable

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(data['DISTANCE'], hist=False, color="b", kde_kws={"shade": True})
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.title("Distribution of distance")
plt.show()


# We can see that our __DISTANCE__ variable is positively skewed.
# I am just curious to find out the correlation between the distance and delay of a flight.

# In[ ]:


print(emoji.emojize("Let's find it out :fire:"))


# Though, there is no possible way to find correlation between a continuous and categorical variable, I'll try to find the average distance for __DEP_DEL15__ variable.

# In[ ]:


print(f"Average distance if there is a delay {data[data['DEP_DEL15'] == 1]['DISTANCE'].values.mean()} miles")
print(f"Average distance if there is no delay {data[data['DEP_DEL15'] == 0]['DISTANCE'].values.mean()} miles")


# Let's visualize the categorical variables.

# ### Count of carriers in the dataset

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x=data['OP_UNIQUE_CARRIER'], data=data)
plt.xlabel("Carriers")
plt.ylabel("Count")
plt.title("Count of unique carrier")
plt.show()


# ### Count of origin and destination airport

# In[ ]:


plt.figure(figsize=(10,70))
sns.countplot(y=data['ORIGIN'], data=data, orient="h")
plt.xlabel("Airport")
plt.ylabel("Count")
plt.title("Count of Unique Origin Airports")
plt.show()


# In[ ]:


plt.figure(figsize=(10,70))
sns.countplot(y=data['DEST'], data=data, orient="h")
plt.xlabel("Airport")
plt.ylabel("Count")
plt.title("Count of Unique Destination Airports")
plt.show()


# ## Modelling
# 

# Our __MONTH__ variable is constant so it will not have any effect on in the training.
# It's better to remove it. Also, let's rename the __DEP_DEL15__ column name to __TARGET__ to avoid confusion between predictors and target variable.

# In[ ]:


data = data.rename(columns={'DEP_DEL15':'TARGET'})


# __Encoding the categorical variable__

# In[ ]:


def label_encoding(categories):
    """
    To perform mapping of categorical features
    """
    categories = list(set(list(categories.values)))
    mapping = {}
    for idx in range(len(categories)):
        mapping[categories[idx]] = idx
    return mapping


# In[ ]:


data['OP_UNIQUE_CARRIER'] = data['OP_UNIQUE_CARRIER'].map(label_encoding(data['OP_UNIQUE_CARRIER']))


# In[ ]:


data['ORIGIN'] = data['ORIGIN'].map(label_encoding(data['ORIGIN']))


# In[ ]:


data['DEST'] = data['DEST'].map(label_encoding(data['DEST']))


# In[ ]:


data.head()


# In[ ]:


data['TARGET'].value_counts()


# In[ ]:


X = data[['DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'DEP_TIME', 'DISTANCE']].values
y = data[['TARGET']].values


# In[ ]:


# Splitting Train-set and Test-set
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=41)

# Splitting Train-set and Validation-set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=41)


# ### Choosing the evaluation metric
# Here we will go with the __Accuracy__ metric for our predicted values because we have already balanced our dataset.
# So, accuracy is the best metric to evaluate any binary classification problem if it is performed on a balanced dataset.

# In[ ]:


# Formula to get accuracy
def get_accuracy(y_true, y_preds):
    # Getting score of confusion matrix
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_true, y_preds).ravel()
    # Calculating accuracy
    accuracy = (true_positive + true_negative)/(true_negative + false_positive + false_negative + true_positive)
    return accuracy


# ### Creating some baseline models

# __Logistic Regression__

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0).fit(X_train, y_train)


# __CatboostClassifier__

# In[ ]:


# Initialize CatBoostClassifier
catboost = CatBoostClassifier(random_state=0)
catboost.fit(X_train, y_train, verbose=False)


# __Naive Bayes__

# In[ ]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)


# __Random Forest Classifier__

# In[ ]:


rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)


# __KNN Classifier__

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)


# __XGBoost Classifier__

# In[ ]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)


# ### Evaluation of accuracy on validation dataset

# In[ ]:


models = [lr, catboost, gnb, rf, knn, xgb]
acc = []
for model in models:
    preds_val = model.predict(X_val)
    accuracy = get_accuracy(y_val, preds_val)
    acc.append(accuracy)


# In[ ]:


model_name = ['Logistic Regression', 'Catboost', 'Naive Bayes', 'Random Forest', 'KNN', 'XGBoost']
accuracy = dict(zip(model_name, acc))


# In[ ]:


plt.figure(figsize=(15,5))
ax = sns.barplot(x = list(accuracy.keys()), y = list(accuracy.values()))
for p, value in zip(ax.patches, list(accuracy.values())):
    _x = p.get_x() + p.get_width() / 2
    _y = p.get_y() + p.get_height() + 0.008
    ax.text(_x, _y, round(value, 3), ha="center") 
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model vs. Accuracy")
plt.show()


# We tried to fit our data on default parameters of different algorithms for binary classification.
# 
# Surprisingly, __KNN Classifier__ turned out to best in terms of validation set accuracy.
# 
# Now we'll try to find the best possible parameters for K-Nearest Neighbors Algorithm.

# ### Accuracy on Test set with KNN before hyperparameter tuning

# In[ ]:


test_preds = knn.predict(X_test)
get_accuracy(y_test, test_preds)


# ### Hyperparameter tuning for KNN

# In[ ]:


leaf_size = list(range(1,5))
n_neighbors = list(range(1,3))
p=[1,2]

hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

knn_2 = KNeighborsClassifier()

clf = GridSearchCV(knn_2, hyperparameters, cv=2)

best_model = clf.fit(X_train,y_train)

print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])


# In[ ]:


knn_best = KNeighborsClassifier(leaf_size=3, p=1, n_neighbors=1)


# In[ ]:


knn_best.fit(X_train, y_train)
test_preds_1 = knn_best.predict(X_test)


# ### Accuracy on Test set with KNN after hyperparameter tuning

# In[ ]:


get_accuracy(y_test, test_preds_1)


# ## Conclusion

# #### We see an increment of __~ 6.1%__ in the accuracy of our model after tuning the hyperparamter.
# 
# #### Accuracy can be improved further if time and resources are given.

# **Also, this is my first kernel on Kaggle. Feel free to correct me if I am wrong anywhere :)**
