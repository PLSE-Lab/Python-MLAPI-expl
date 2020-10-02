#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **1. Import the csv dataset from https://www.kaggle.com/uciml/adult-census-income**

# In[ ]:


adult_df = pd.read_csv("../input/adult-census-income/adult.csv")


# **2. Identify the presence of missing values, fill the missing values with mean for numerical attributes and mode value for categorical attributes.**

# In[ ]:


adult_df.isnull().sum()


# In[ ]:


adult_df.describe()


# In[ ]:


adult_df.head(10)
#here we get that null values are represented with "?"


# In[ ]:


adult_df = adult_df.replace("?",np.NaN)


# In[ ]:


adult_df.isnull().sum()


# In[ ]:


adult_df.info()


# In[ ]:


#Converting object type data to category
adult_df[['workclass', 'education', 'marital.status','occupation','relationship','race','sex','native.country','income']].apply(lambda x: x.astype('category'))


# In[ ]:


#Category: Missing values imputation using SimpleImputer as Imputer class is deprecated
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy="most_frequent")
adult_df_category = adult_df[['workclass','occupation','native.country']]
imputer = imputer.fit(adult_df_category[['workclass','occupation','native.country']])
adult_df_category = imputer.transform(adult_df_category[['workclass','occupation','native.country']])
adult_df_category = pd.DataFrame(data=adult_df_category , columns=[['workclass','occupation','native.country']])
adult_df_category.head()


# In[ ]:


#Numeric: Missing values imputation using SimpleImputer as Imputer class is deprecated
imputer1 = SimpleImputer(missing_values = np.nan, strategy="mean")
adult_df_numeric = adult_df[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']]
imputer1 = imputer1.fit(adult_df_numeric[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']])
adult_df_numeric = imputer1.transform(adult_df_numeric[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']])
adult_df_numeric = pd.DataFrame(data=adult_df_numeric , columns=[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']])


# In[ ]:


#merging back to original dataframe
adult_df[['workclass','occupation','native.country']] = adult_df_category[['workclass','occupation','native.country']]
adult_df[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']] = adult_df_numeric[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']]


# In[ ]:


import seaborn as sns
categorical_attributes = adult_df[['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']]

for i, attribute in enumerate(categorical_attributes):
    # Set the width and height of the figure
    plt.figure(figsize=(16,6))
    plt.figure(i)
    sns.countplot(categorical_attributes[attribute])
    plt.xticks(rotation=90)


# **3. Extract X as all columns except the Income column and Y as Income column**

# In[ ]:


X = adult_df.drop(['income'], axis=1)
y = adult_df['income']
# here Y variable is binary >=50, <=50


# **4. Split the data into training set and testing set**

# In[ ]:


y.replace(('<=50K', '>50K'), (0, 1), inplace = True)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# <h3>**Feature Engineering**</h3>
# 
# Encoding Categorical Values

# In[ ]:


from sklearn import preprocessing
categorical_variables = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical_variables:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])


# <h3>**Scaling variables using StandardScalar **</h3>

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

cols = X_train.columns

temp_train = X_train.copy()
temp_test = X_test.copy()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = cols)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = cols)

print(X_train.head())
print("----------------------------------------------------------------------------")
print(X_test.head())


# **<h2>MODEL BUILDING </h2>**

# <h3>**5(a).GaussianNB model**</h3>

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gaussian = gnb.predict(X_test)


# <h3>**Accuracy and Confusion matrix for GaussianNB**</h3>

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print("CONFUSION MATRIX:\n")
print(confusion_matrix(y_test, y_pred_gaussian),"\n\n")
print("Accuracy of gaussian:\n")
print(accuracy_score(y_test, y_pred_gaussian))


# <h3>**5(b).BernoulliNB model**</h3>

# In[ ]:


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bernoulli = bnb.predict(X_test)


# <h3>**Accuracy and Confusion matrix for BernoulliNB**</h3>

# In[ ]:


print("CONFUSION MATRIX:\n")
print(confusion_matrix(y_test, y_pred_bernoulli),"\n\n")
print("Accuracy of gaussian:\n")
print(accuracy_score(y_test, y_pred_bernoulli))


# <h3>**5(c).MultinomialNB  model**</h3>

# MultinomialNB model does not takes negative values so we are changing the scaling technique to min-max scaling which changes the values to positive, in range [0,1].

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

X_train_minmax = pd.DataFrame(minmax.fit_transform(temp_train), columns = X.columns)

X_test_minmax = pd.DataFrame(minmax.transform(temp_test), columns = X.columns)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train_minmax, y_train)
y_pred_multinomial = mnb.predict(X_test_minmax)


# In[ ]:


print("CONFUSION MATRIX:\n")
print(confusion_matrix(y_test, y_pred_multinomial),"\n\n")
print("Accuracy of gaussian:\n")
print(accuracy_score(y_test, y_pred_multinomial))


# In[ ]:


# Visualising the Training set results
def plot_decisionboundary(classifier_plt, xx, yy):
    colors = "rgb"
    Z = classifier_plt.predict(np.c_[xx.ravel(), yy.ravel()])
    pd.DataFrame(Z,columns=['hi'])['hi'].value_counts()
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    cs = plt.contourf(xx,yy,  Z, cmap = ListedColormap(('red', 'green', 'blue', 'yellow')))
    plt.subplot(1,2,2)
    cs = plt.contourf(xx,yy,  Z, cmap = ListedColormap(('red', 'green', 'blue', 'yellow')))


    # Plot also the training points
    for i, color in zip(classifier_plt.classes_, colors):
        idx = np.where(y_train == i)
        plt.subplot(1,2,1)
        plt.scatter(X_train.values[idx, 0], X_train.values[idx, 1], c=color, label=['0', '1'][i],cmap = ListedColormap(('red', 'green', 'blue', 'yellow')), edgecolor='black', s=20)
        plt.axis('tight')
        plt.subplot(1,2,2)
        idx = np.where(y_test == i)
        plt.scatter(X_test.values[idx, 0], X_test.values[idx, 1], c=color, label=['0', '1'][i],cmap = ListedColormap(('red', 'green', 'blue', 'yellow')), edgecolor='black', s=20)

    plt.title("Plot the decision boundary, visualize training and test results")
    plt.legend()


# In[ ]:


from matplotlib.colors import ListedColormap


# In[ ]:


h = 0.01 # step size in the mesh
x = 1

x1_min, x1_max = X_train.values[:, 0].min() - x, X_train.values[:, 0].max() + x
y1_min, y1_max = X_train.values[:, 1].min() - x, X_train.values[:, 1].max() + x

xx1, yy1 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(y1_min, y1_max, h))


# In[ ]:


gnb_t = GaussianNB()
gnb_t.fit(X_train.values[:, :2], y_train)
plot_decisionboundary(gnb_t, xx1, yy1)


# In[ ]:


bnb_t = BernoulliNB()
bnb_t.fit(X_train.values[:, [2, 11]], y_train)
plot_decisionboundary(bnb_t, xx1, yy1)


# In[ ]:


mnb_t = MultinomialNB()
mnb_t.fit(X_train_minmax.values[:, :2], y_train)
plot_decisionboundary(mnb_t, xx1, yy1)


# ### Create an output .csv file consisting actual Test set values of Y (column name: Actual) and Predictions of Y(column name: Predicted). (1 points)

# In[ ]:


y_pred_gaussian.shape


# In[ ]:


y_pred_gaussian
y_pred_bernoulli
y_pred_multinomial
y_test.values


# In[ ]:


dataset = pd.DataFrame({'Actual': y_test.values[:,], 'Gaussian_Predicted': y_pred_gaussian[:,], 'Bernoulli_Predicted': y_pred_bernoulli[:, ], 'Multinomial_Predicted': y_pred_multinomial[:, ]})


# In[ ]:


dataset.to_csv('/kaggle/working/output_test.csv')


# In[ ]:




