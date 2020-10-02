#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# ## Import Dataset

# In[ ]:


data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.head(10)


# ## Exploratory Data Analysis

# * age [29-77]
# * sex [1=male, 0=female]
# * chest pain type (4 values) [0, 1, 2, 3]
# * resting blood pressure
# * serum cholestoral in mg/dl
# * fasting blood sugar > 120 mg/dl [1=true, 0=false]
# * resting electrocardiographic results (values 0,1,2)
# * maximum heart rate achieved => thalach
# * exercise induced angina [1=yes, 0=no]
# * oldpeak = ST depression induced by exercise relative to rest
# * the slope of the peak exercise ST segment [0, 1, 2]
# * number of major vessels (0-3) colored by flourosopy [0, 1, 2, 3, 4]
# * thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# * target [1=Have Heart Disease, 0=Don't have Heart Disease]

# In[ ]:


print("Total Rows in a Data: ", data.shape[0])
print("Total Columns in a Data: ", data.shape[1])


# In[ ]:


print("\t****************")
print("\tData Information:")
print("\t****************\n")
data.info()


# All the values in a dataset are numerical.

# In[ ]:


print("\t****************")
print("\tData Describe:")
print("\t****************\n")
data.describe()


# In[ ]:


print("Null Values in each column:")
print(data.isna().any())


# Has from the above output there's no missing values in any column so we don't have to do data cleaning and data filling.

# In[ ]:


print("Unique Values in each column:")
print("----------------------------\n")
cols = list(data.columns)
for c in cols:
    print(c.upper(), ":", data[c].unique(), "\n")


# ### Exploring Age Column

# In[ ]:


print("The most younger person in a data:")
print(data.iloc[data.age.idxmin()])


# the most youngest person is a male and has a heart disease.

# In[ ]:


print("The most older person in a data:")
print(data.iloc[data.age.idxmax()])


# the most older person is a male and not has a heart disease.

# In[ ]:


fig, ax = plt.subplots(figsize=(14, 8))
sns.countplot(y=data.age)
# sns.countplot(y=data['age'], order=data['age'].value_counts().index)
plt.title("Age Count")
plt.xlabel("Count")
plt.ylabel("Age")
ax.set(xticks=range(0, 21))
plt.show()


# As we can see from the above plot, mostly people age are between 50-60

# In[ ]:


sns.boxplot(data.age)
plt.show()


# There is no outlier in a dataset as we can see from the above plot. The average person age is between 50-60.

# In[ ]:


fig, ax = plt.subplots(figsize=(14, 5))
ax = plt.plot(data.groupby('age')['target'].mean())
plt.xticks(range(min(data.age)-1, max(data.age)+1))
plt.show()


# People from the age between 29 to 32 and 70 to 76 has the highest chance of having a heart disease.

# In[ ]:


data[data.age == 61]


# In[ ]:


data[data.age > 70]


# As from the above chart people above 70 has highest chance of having a heart disease.

# ### Exploring Gender Column

# In[ ]:


print("Number of Unique Values in Gender Column: ", data.sex.nunique(), "\n")
print("Unique Values count are:")
print(data.sex.value_counts())


# In this dataset mostly people are Males and some are Females

# In[ ]:


ax = sns.countplot(data.sex, palette='Set3')
plt.xlabel("Sex")
plt.title("Gender Occuring in a Dataset")
plt.show()


# As we can see from the above plot Males are the most in a data

# In[ ]:


data.groupby('sex')['target'].value_counts()


# Male has a high heart disease rate than female and they are occuring 93 times while female has 72 times occuring

# In[ ]:


fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x='sex', hue='target', data=data, palette='Set2')
plt.xlabel("Gender (0=Female, 1=Male)")
plt.show()


# As we can see from the above plot, that Male has a higher rate of heart disease.

# In[ ]:


ax = sns.catplot(x='target', col='sex', data=data, kind='count')


# ### Exploring Chest Pain Column

# In[ ]:


print("Number of Unique Values in Chest Pain Type Column: ", data.cp.nunique(), "\n")
print("Unique Values count are:")
print(data.cp.value_counts())


# Here 0 means no pain and 3 means extreme pain

# In[ ]:


sns.countplot(data.cp)
plt.title("Chest Pain Count")
plt.xlabel("Chest Pain Type")
plt.show()


# Mostly people have no Chest Pain

# In[ ]:


data.groupby('cp')['target'].value_counts()


# Person who has a Chest Pain Type 1 and 2 has the highest chance to have Heart Disease.

# In[ ]:


ax = sns.countplot(x='cp', data=data, hue='target')
plt.xlabel("Chest Pain Type (lowest to highest)")
plt.ylabel("Count")
plt.show()


# In this plot, as we can see a Person who has a chest pain type 2 has mostly chances to have heart disease.

# ### Exploring Blood Sugar Column

# In[ ]:


print("Number of Unique Values in Fasting Blood Sugar Column: ", data.fbs.nunique(), "\n")
print("Unique Values count are:")
print(data.fbs.value_counts())


# In[ ]:


data.groupby('fbs')['target'].value_counts()


# A person who don't have a Blood Sugar has the highest chance of having Heart Disease.

# In[ ]:


sns.countplot(x='fbs', hue='target', data=data, palette='Set2')
plt.xlabel("Fasting Blood Sugar (0=False, 1=True)")
plt.ylabel("Count")
plt.show()


# ### Exploring Electrographic Column

# In[ ]:


print("Number of Unique Values in Resting Electrocardiographic Column: ", data.restecg.nunique(), "\n")
print("Unique Values count are:")
print(data.restecg.value_counts())


# In[ ]:


data.groupby('restecg')['target'].value_counts()


# In[ ]:


sns.countplot(x=data['restecg'])
plt.xlabel("Resting Electrocardiographic Results")
plt.show()


# ### Exploring Thalach Column

# Thalach = maximum heart rate achieved

# In[ ]:


print("Number of Unique Values in Thalach Column: ", data.thalach.nunique(), "\n")
print("Unique Values count are:")
print(data.thalach.value_counts())


# Thalach is a continous column so we have to draw distribute plot

# In[ ]:


sns.distplot(data.thalach)
plt.xlabel("Heart Rate")
plt.show()


# It is left skewed means negative skewed. [For Further Know](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/skewed-distribution/)

# In[ ]:


sns.scatterplot(x='age', y='thalach', data=data)
plt.xlabel('Age')
plt.ylabel('Heart Rate')
plt.show()


# ### Exploring Exercise Induced Angina Column

# [Open this link for further knowing of Exercise Induced Angina](https://www.mayoclinic.org/diseases-conditions/angina/symptoms-causes/syc-20369373)

# In[ ]:


print("Number of Unique Values in Exercise Induced Angina Column: ", data.exang.nunique(), "\n")
print("Unique Values count are:")
print(data.exang.value_counts())


# This result shows us that 204 people have no Exercise Induced Angina and 99 people have it.

# In[ ]:


data.groupby('exang')['target'].value_counts()


# As we can see from the above output people who don't have Induced Angina have a higher chance of having Heart Disease.

# In[ ]:


sns.countplot(y=data.exang)
plt.xlabel("Count")
plt.ylabel("Exercised Induced Angina")
plt.show()


# Now let visualize the above graph with respect to Target variable

# In[ ]:


ax = sns.countplot(x='exang', data=data, hue='target')
plt.xlabel("Exercised Induced Angina (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()


# Above graph is illustrating that person who don't have Induced Angina has Heart Disease.

# ### Exploring Slope Column

# In[ ]:


plt.title("The Slope of the Peak Exercise ")
sns.countplot(x=data['slope'])
plt.xlabel("Slope")
plt.ylabel("Count")
plt.show()


# As we can see 1 and 2 are the highest occuring slopes in a dataset

# In[ ]:


plt.title("The Slope of the Peak Exercise ")
sns.countplot(x=data['slope'], hue=data['target'], data=data)
plt.xlabel("Slope")
plt.ylabel("Count")
plt.show()


# The person who has a type 2 slope has a higher chance of having Heart Disease

# ### Exploring Further Columns

# In[ ]:


sns.countplot(x=data['ca'])
plt.title("Number of Major Vessels")
plt.show()


# In ca column there are 5 categories.

# In[ ]:


sns.countplot(x=data['thal'])
plt.title("Blood Disorder called Thalassemia")
plt.show()


# ### Exploring Target Column

# In[ ]:


sns.countplot(x=data['target'])
plt.title("Target Count")
plt.show()


# There are more people who have a heart disease in this data.

# ## Data Correlation

# In[ ]:


data.corr()['target'].sort_values(ascending=False)


# Column: cp, thalach, slope are +ve correlated to a target while other columns are -ve correlated.

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True)
plt.show()


# In[ ]:


sns.pairplot(data[['cp', 'restecg', 'thalach', 'slope', 'target']])
plt.show()


# ## Prediction

# In[ ]:


X1 = data.iloc[:, :-1]
y1 = data.iloc[:, -1]

df = data[['cp', 'restecg', 'thalach', 'slope', 'target']]
X2 = df.iloc[:, :-1]
y2 = df.iloc[:, -1]

print("Columns in Data1: ", list(data.columns))
print("Data1 Shape: ", data.shape)
print("X1 Shape: ", X1.shape)
print("y1 Shape: ", y1.shape)
print()

print("Columns in Data2: ", list(df.columns))
print("Data2 Shape: ", df.shape)
print("X2 Shape: ", X2.shape)
print("y2 Shape: ", y2.shape)


# ### Splitting Data

# ### Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
train_X1, test_X1, train_y1, test_y1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
train_X2, test_X2, train_y2, test_y2 = train_test_split(X2, y2, test_size=0.2, random_state=42)


# For prediction, i'm going to use 
# * Decision Tree Classifier
# * Random Forest Classifier 
# * Logistic Regression
# * Support Vector Machine 
# * Stochastic Gradient Descent
# * K-Nearest Neighbor Classifier
# * Adaboost Classifier

# ### Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(train_X1, train_y1)
pred_dtc1 = DTC.predict(test_X1)
score_dtc1 = round(DTC.score(test_X1, test_y1)*100, 2)
print("Accuracy of Data 1 with Decision Tree Classifier is: ", score_dtc1, "%")
print()
DTC.fit(train_X2, train_y2)
pred_dtc2 = DTC.predict(test_X2)
score_dtc2 = round(DTC.score(test_X2, test_y2)*100, 2)
print("Accuracy of Data 2 with Decision Tree Classifier is: ", score_dtc2, "%")


# ### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier() # n_estimators = 100
RFC.fit(train_X1, train_y1)
pred_rfc1 = RFC.predict(test_X1)
score_rfc1 = round(RFC.score(test_X1, test_y1)*100, 2)
print("Accuracy of Data 1 with Random Forest Classifier is: ", score_rfc1, "%")
print()
RFC.fit(train_X2, train_y2)
pred_rfc2 = RFC.predict(test_X2)
score_rfc2 = round(RFC.score(test_X2, test_y2)*100, 2)
print("Accuracy of Data 2 with Random Forest Classifier is: ", score_rfc2, "%")


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(train_X1, train_y1)
pred_lr1 = LR.predict(test_X1)
score_lr1 = round(LR.score(test_X1, test_y1)*100, 2)
print("Accuracy of Data 1 with Logistic Regression is: ", score_lr1, "%")
print()
LR.fit(train_X2, train_y2)
pred_lr2 = LR.predict(test_X2)
score_lr2 = round(LR.score(test_X2, test_y2)*100, 2)
print("Accuracy of Data 2 with Logistic Regression is: ", score_lr2, "%")


# ### Support Vector Machine

# In[ ]:


from sklearn.svm import SVC
SC = SVC()
SC.fit(train_X1, train_y1)
pred_sc1 = SC.predict(test_X1)
score_sc1 = round(SC.score(test_X1, test_y1)*100, 2)
print("Accuracy of Data 1 with Support Vector Classifier is: ", score_sc1, "%")
print()
SC.fit(train_X2, train_y2)
pred_sc2 = SC.predict(test_X2)
score_sc2 = round(SC.score(test_X2, test_y2)*100, 2)
print("Accuracy of Data 2 with Support Vector Classifier is: ", score_sc2, "%")


# ### Stochastic Gradient Descent

# In[ ]:


from sklearn.linear_model import SGDClassifier
SGD = SGDClassifier()
SGD.fit(train_X1, train_y1)
pred_sgd1 = SGD.predict(test_X1)
score_sgd1 = round(SGD.score(test_X1, test_y1)*100, 2)
print("Accuracy of Data 1 with Stochastic Gradient Descent Classifier is: ", score_sgd1, "%")
print()
SGD.fit(train_X2, train_y2)
pred_sgd2 = SGD.predict(test_X2)
score_sgd2 = round(SGD.score(test_X2, test_y2)*100, 2)
print("Accuracy of Data 2 with Stochastic Gradient Descent Classifier is: ", score_sgd2, "%")


# ### K-Nearest Neighbor Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=10) # default n_neighbors = 5
KNN.fit(train_X1, train_y1)
pred_knn1 = KNN.predict(test_X1)
score_knn1 = round(KNN.score(test_X1, test_y1)*100, 2)
print("Accuracy of Data 1 with K-Nearest Neighbor Classifier is: ", score_knn1, "%")
print()
KNN.fit(train_X2, train_y2)
pred_knn2 = KNN.predict(test_X2)
score_knn2 = round(KNN.score(test_X2, test_y2)*100, 2)
print("Accuracy of Data 2 with K-Nearest Neighbor Classifier is: ", score_knn2, "%")


# ### AdaBoost Classifier

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
ABC = AdaBoostClassifier() # n_estimators = 50 default
ABC.fit(train_X1, train_y1)
pred_abc1 = ABC.predict(test_X1)
score_abc1 = round(ABC.score(test_X1, test_y1)*100, 2)
print("Accuracy of Data 1 with AdaBoost Classifier is: ", score_abc1, "%")
print()
ABC.fit(train_X2, train_y2)
pred_abc2 = ABC.predict(test_X2)
score_abc2 = round(ABC.score(test_X2, test_y2)*100, 2)
print("Accuracy of Data 2 with AdaBoost Classifier is: ", score_abc2, "%")


# ## RESULT:

# In[ ]:


model1 = pd.DataFrame(
    {
        'Models': [
            'Decision Tree Classifier',
            'Random Forest Classifier',
            'Logistic Regression',
            'Support Vector Machine',
            'Stochastic Gradient Descent',
            'K-Nearest Neighbors',
            'AdaBoost Classifier'
        ],
        'Scores': [
            score_dtc1,
            score_rfc1,
            score_lr1,
            score_sc1,
            score_sgd1,
            score_knn1,
            score_abc1
        ],
    }
)
model2 = pd.DataFrame(
    {
        'Models': [
            'Decision Tree Classifier',
            'Random Forest Classifier',
            'Logistic Regression',
            'Support Vector Machine',
            'Stochastic Gradient Descent',
            'K-Nearest Neighbors',
            'AdaBoost Classifier'
        ],
        'Scores': [
            score_dtc2,
            score_rfc2,
            score_lr2,
            score_sc2,
            score_sgd2,
            score_knn2,
            score_abc2
        ],
    }
)


# In[ ]:


print("Models who are Train on Data 1:")
model1


# In this result, Decision Tree, Random Forest, Logistic Regression, AdaBoost are the best choice for Data 1

# In[ ]:


print("Models who are Train on Data 2:")
model2


# In this result, Stochastic Gradient Descent, Logistic Regression are the best choice for Data 2
