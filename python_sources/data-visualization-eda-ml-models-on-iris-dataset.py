#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# # 1. Load dataset

# In[ ]:


df = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/iris.csv")


# In[ ]:


df.head()


# #  2. Analyze Data

# In[ ]:


df.info()


# no null values, great!

# In[ ]:


df.isnull().sum()


# just double checking

# In[ ]:


iris = df
iris.describe()


# Here, 25% -> quantile value and thus, 25% of the flowers have sepallength value lower than or equal to 5.1 and similarly we can check for others.
# 
# 50% -> median value and this,50% of the flowers have sepallength lower than or equal to 5.8 and similarly we can check for others.
# 

# In[ ]:


print(df.shape)


# In[ ]:


print("What categories are there and how many instances for each category?\n")
print(iris["variety"].value_counts())
print("\n\nWhat are the unique categories?")
print(iris["variety"].unique())
# How many unique values are there
print("\n\nHow many unique categories there are?")
print(iris["variety"].nunique())
print("\n\nWhat is the shape of our dataframe?")
print(iris.shape)


# Let's make the species simpler without the prefix Iris

# In[ ]:


iris.loc[iris["variety"] == "Iris-setosa", ["variety"]] = "Setosa"
iris.loc[iris["variety"] == "Iris-virginica", ["variety"]] = "Virginica"
iris.loc[iris["variety"] == "Iris-versicolor", ["variety"]] = "Versicolor"
iris.head()


# In[ ]:


import seaborn as sns
sns.countplot(x='variety', data=iris)


# So what did we learn so far?
# 
#     We have a dataset of 150 rows and 5 columns
# 
#     Only one column is categorical and the rest are numerical.
# 
#     There are 3 different categories and they all have equal number of instances, 50.
# 
#     There are no null values.
# 
#     25% of the flowers have sepallength value lower than or equal to 5.1 and similarly we can check for others.
# 
# 

# **Visualization**
# 

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
sns.scatterplot(x = 'sepal.length' ,y='sepal.width', data=iris)


# Roughly, we can say that sepal.length and sepal.width are positively correlated(as width increases with length)

# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='sepal.length', y='sepal.width', data=iris, hue='variety')


# We can observe that
# 
# For setosa, max. sepal length is less than 6 cm -> smaller sepal length However, sepalwidth is largest than other two categories
# 
# For Versicolor, majority of sepal length is between 5.5 and 6.5 -> quite smaller sepal length as well
# 
# For Virginica, sepal length is the largest
# 

# In[ ]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
sns.scatterplot(x='sepal.length', y='sepal.width', hue='variety', data=iris)
plt.subplot(1,2,2)
sns.scatterplot(x='sepal.width', y='sepal.length', hue='variety', data=iris)


# In[ ]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
sns.scatterplot(x='petal.length', y='petal.width', hue='variety', data=iris)
plt.subplot(1,2,2)
sns.scatterplot(x='petal.width', y='petal.length', hue='variety', data=iris)


# Both petal length and width tend to have positive corelation as well !
# 
# Setosa -> smallest length
# 
# Versicolor -> length is larger than setosa but smaller than virginica
# 
# Verginica -> largest length
# 

# In[ ]:


# height parameter decides the size of the plot here
sns.lmplot(x="sepal.length", y="sepal.width", hue="variety", data=iris, height=8, markers=["o", "x", "^"])


# It gives the regression line
# 
# We can say from the graph that the value of sepal.width for Setosa changes rapidly with respect to sepal.length than the other two categories
# 

# In[ ]:


sns.lmplot(x="petal.length", y="petal.width", hue="variety", data=iris, height=8)


# Here, the slope is steeper for Versicolor than the other two

# In[ ]:


#We will use median of each feature and try to compare them 
species = ["Setosa", "Versicolor", "Virginica"]
features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
d = {"Median":[], "Features":[],  "Species":[]}
for s in species:
    for f in features:
        d["Median"].append(iris[iris["variety"] == s][f].mean())
        d["Features"].append(f)
        d["Species"].append(s)

        
new_df = pd.DataFrame(data=d)
new_df


# What we have done is that for every Species and for every Feature (sepal's and petal's dimensions) we have created a value which is median of that feature.

# In[ ]:


plt.figure(figsize=(12, 6))
sns.lineplot(x="Features", y="Median", hue="Species", data=new_df)


# In[ ]:


plt.figure(figsize=(12, 6))
sns.pointplot(x="Features", y="Median", hue="Species", data=new_df)


# What we can observe is that the median of lengths is greater than the median of widths
# 
# Sepal length, Petal length, Petal width:
# 
# Virginica -> Versicolor -> Setosa
# 
# Sepal width: Setosa -> Virginica -> Versicolor
# 
# Clearly, sepals are larger for Iris flowers than petals
# 

# In[ ]:


#univariate analysis
plt.figure(figsize=(8, 6))
sns.stripplot(x="variety", y="sepal.length", data=iris, jitter=True, size=7)


# In[ ]:


plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
sns.stripplot(x="variety", y="petal.length", data=iris, jitter=True, size=7)
plt.subplot(2, 2, 2)
sns.stripplot(x="variety", y="petal.width", data=iris, jitter=True, size=7)
plt.subplot(2, 2, 3)
sns.stripplot(x="variety", y="sepal.length", data=iris, jitter=True, size=7)
plt.subplot(2, 2, 4)
sns.stripplot(x="variety", y="sepal.width", data=iris, jitter=True, size=7)


# In[ ]:


plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
sns.stripplot(x="variety", y="petal.length", data=iris, jitter=False, size=7)
plt.subplot(2, 2, 2)
sns.stripplot(x="variety", y="petal.width", data=iris, jitter=False, size=7)
plt.subplot(2, 2, 3)
sns.stripplot(x="variety", y="sepal.length", data=iris, jitter=False, size=7)
plt.subplot(2, 2, 4)
sns.stripplot(x="variety", y="sepal.width", data=iris, jitter=False, size=7)


# In[ ]:


plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
sns.swarmplot(x="variety", y="petal.length", data=iris, size=7)
plt.subplot(2, 2, 2)
sns.swarmplot(x="variety", y="petal.width", data=iris, size=7)
plt.subplot(2, 2, 3)
sns.swarmplot(x="variety", y="sepal.length", data=iris, size=7)
plt.subplot(2, 2, 4)
sns.swarmplot(x="variety", y="sepal.width", data=iris, size=7)


# In swarmplot, all the points are visible and pasted side by side. It shows that which particular values are coming how many times. For a dataset small (and cute) like Iris, its actually good. Swarmplot might not be able to do justice to larger datasets.

# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
sns.violinplot(x="variety", y="petal.length", data=iris, size=7)
plt.subplot(2, 2, 2)
sns.violinplot(x="variety", y="petal.width", data=iris, size=7)
plt.subplot(2, 2, 3)
sns.violinplot(x="variety", y="sepal.length", data=iris, size=7)
plt.subplot(2, 2, 4)
sns.violinplot(x="variety", y="sepal.width", data=iris, size=7)


# In[ ]:


plt.figure(figsize=(10, 10))
binsize = 10
plt.subplot(2, 2, 1)
sns.distplot(a=iris["petal.length"], bins=binsize)
plt.subplot(2, 2, 2)
sns.distplot(a=iris["petal.width"], bins=binsize)
plt.subplot(2, 2, 3)
sns.distplot(a=iris["sepal.length"], bins=binsize)
plt.subplot(2, 2, 4)
sns.distplot(a=iris["sepal.width"], bins=binsize)


# In[ ]:


sns.jointplot(x="sepal.length", y="sepal.width", kind='hex', data=iris[iris["variety"] == "Setosa"])


# Wherever you see darker colour, that means there have been high density of points there. 

# # 3. Model Evaluation

# Declaring features and target variable

# In[ ]:


X = iris.drop(['variety'], axis=1)
y = iris['variety']
# print(X.head())
print(X.shape)
# print(y.head())
print(y.shape)


# Splitting train and test data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# **KNN**

# In[ ]:


k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    

print(metrics.accuracy_score(y_test, y_pred))


# **Logistic Regression**

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# **SVM**

# In[ ]:


from sklearn import svm
model = svm.SVC()
model.fit(X_train,y_train) 
prediction=model.predict(X_test) 
print(metrics.accuracy_score(y_test,y_pred))


# **Decision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train) 
prediction=model.predict(X_test) 
print(metrics.accuracy_score(y_test,y_pred))


# This is a basic implementation of Data visualization and EDA techniques, ofcourse I have added only those techniques that were suitable and needed for this dataset. There are many other techniques that are out there which can be more useful! This is just something you can start off with.
# 
# I have used https://www.kaggle.com/gadaadhaarigeek/another-eda-on-iris-dataset as a reference, if you have time to check their notebook out as well!
# 
# Thank you for going through this notebook. I hope you found this useful and if you did, please upvote it!
# 
