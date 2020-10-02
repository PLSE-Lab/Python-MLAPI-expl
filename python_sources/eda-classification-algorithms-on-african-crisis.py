#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# > #### Objectives: 
# * To explore the data, look for trends and draw inferences.
# * To classify whether a country is suffering from inflation crisis or not using different classification algorithms such as Random Forest Classifer, Decision Tree, K Nearest Neighbors & Logistic Regression.
# * To compare K-Fold scores of different models.
# * To check and compare the accuracy of the different models and confirm our findings using the Confusion Matrix.

# In[ ]:


df = pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
df.head()


# In[ ]:


df.describe().T[1:10]


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ## Exploratory Analysis

# ### Percentages of Different Types of Crises

# In[ ]:


fig,ax = plt.subplots(1,2, figsize=(15,10))
vc=df['inflation_crises'].value_counts()
#print(vc)
vc.plot(kind='pie',autopct='%1.1f', ax = ax[0])

vc=df['banking_crisis'].value_counts()
#print(vc)
vc.plot(kind='pie',autopct='%1.1f',ax = ax[1])
ax[0].set_title('Percentage of Inflation Crisis')
ax[1].set_title('Percentage of Banking Crisis')
plt.show()


# In[ ]:


fig,ax = plt.subplots(1,2, figsize=(15,10))
vc=df['currency_crises'].value_counts()
vc.plot(kind='pie',autopct='%1.1f', ax = ax[0])

vc=df['systemic_crisis'].value_counts()
vc.plot(kind='pie',autopct='%1.1f',ax = ax[1])
ax[0].set_title('Percentage of Currency Crisis')
ax[1].set_title('Percentage of Systemic Crisis')
plt.show()


# * The percentage of cases with inflation crises is highest at 12.9%, 
# * The percentage of cases with systemic crisis is the lowest at 7.7%.

# ### Correlation Between Variables

# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(),annot=True, cmap='YlGnBu')
plt.show()


# * The highest correlation exists between Domestic Debt in Default and Sovereign External Debt Default.
# * The lowest correlation is between Case and Exchange Used.

# ### No. of Cases With Respect to Country

# In[ ]:


plt.figure(figsize=(12,7))
sns.barplot(data=df, x='country', y='case')
plt.show()


# ### Scatter of Inflation Crises

# In[ ]:


plt.figure(figsize=(14,8))
sns.scatterplot(data = df, y='year', x = 'inflation_crises',hue='inflation_crises')
plt.show()


# > #### The number of years that had inflation crisis = 1 is not in continuum. It means that inflation crisis did not occur continuously from 1860 to 2020.

# ### Distribution of Exchange Rate of Countries w.r.t US Dollar

# In[ ]:


plt.figure(figsize=(10,7))
sns.distplot(df['exch_usd'], bins=20, hist=True, kde=True, color='r')
plt.show()


# > #### The kernel density is maximum between 0 to 25 (approximate range) implying that the maximum number of countries have an exchange rate in this range.

# ### Relation Between Exchange Rate and Inflation Crisis

# In[ ]:


plt.figure(figsize=(10,7))
sns.pointplot(y=df['exch_usd'] , x= df['inflation_crises'])
plt.title('Exchange Rate vs. Inflation Crisis')
plt.show()


# > #### The countries with inflation crisis have a lower exchange rate (in US Dollars).

# ### Top Five Countries of Crisis

# In[ ]:


topfive = df.groupby('country')['inflation_crises'].agg('sum').nlargest(5)

df['banking_crisis'] = df['banking_crisis'].map(lambda x : 1 if x == 'crisis' else 0)
topbank = df.groupby('country')['banking_crisis'].agg('sum').nlargest(5)


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(15,6))
topfive.plot.bar(color='m', ax = ax[0])
topbank.plot.bar(color='g', ax = ax[1])
ax[0].set_title('Top Five Countries of Inflation Crisis')
ax[1].set_title('Top Five Countries of Banking Crisis')
plt.show()


# In[ ]:


curr = df.groupby('country')['currency_crises'].agg('sum').nlargest(5)
sys = df.groupby('country')['systemic_crisis'].agg('sum').nlargest(5)


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(15,6))
curr.plot.bar(color='r', ax = ax[0])
sys.plot.bar(color='k', ax = ax[1])
ax[0].set_title('Top Five Countries of Currency Crisis')
ax[1].set_title('Top Five Countries of Systemic Crisis')
plt.show()


# > #### Zimbabwe and Nigeria are among the top five countries in all four kinds of crises.

# In[ ]:


sns.regplot(data=df, y='inflation_crises', x='exch_usd')
plt.show()


# > #### As the above plot shows, the relationship between Inflation Crises and Exchange Rate is not profoundly linear. Thus, linear regression algoritm is not well-application

# ### Relationship Between Exchange Rate and Annual Inflation CPI

# In[ ]:


plt.figure(figsize=(10,7))
df.set_index('exch_usd')['inflation_annual_cpi'].plot.line(color='r')
plt.show()


# ### Comparing Cross Validation Scores of Different Algorithms

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# In[ ]:


le = LabelEncoder()
df['country'] = le.fit_transform(df['country'])


# In[ ]:


df['cc3'] = le.fit_transform(df['cc3'])


# In[ ]:


X = df.drop('inflation_crises', axis=1)
y = df['inflation_crises']
cv_scr = cross_val_score(RandomForestClassifier(), X, y, cv = 10)
cv_scr


# In[ ]:


sum1 = 0
for i in cv_scr:
    sum1 = sum1+i
    i +=1
print("The average accuracy of Random Forest algorithm with 10 folds is", sum1*100/10, "percent")


# In[ ]:


cv_scr1 = cross_val_score(LogisticRegression(), X, y, cv = 10)
sum2 = 0
for i in cv_scr1:
    sum2 = sum2+i
    i +=1
print("The average accuracy of Logistic Regression algorithm with 10 folds is", sum2*100/10, "percent")


# In[ ]:


cv_scr2 = cross_val_score(DecisionTreeClassifier(), X, y, cv = 10)
sum3 = 0
for i in cv_scr2:
    sum3 = sum3+i
    i +=1
print("The average accuracy of Decision Tree algorithm with 10 folds is", sum3*100/10, "percent")


# In[ ]:


cv_scr3 = cross_val_score(KNeighborsClassifier(), X, y, cv = 10)
sum4 = 0
for i in cv_scr3:
    sum4 = sum4+i
    i +=1
print("The average accuracy of KNN algorithm with 10 folds is", sum4*100/10, "percent")


# In[ ]:


indices = ['Random Forest','Logistic Regression','Decision Tree', 'KNN']
bar = pd.DataFrame([sum1*10,sum2*10, sum3*10, sum4*10], index = indices)


# In[ ]:


bar.plot.bar(color='m')
plt.show()


# > #### As seen in the bar chart, the cross validation score of Logistic Regression and Random Forest Classifer is almost the same, while the score for Decision Tree is the lowest.

# ### Random Forest

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)


# In[ ]:


model = RandomForestClassifier()
model.fit(X_train,y_train)


# In[ ]:


print("The accuracy of the model is",model.score(X_test,y_test)*100,"percent")


# > #### The accuracy score of Random Forest Classifier is 98.74 percent.

# In[ ]:


plt.figure(figsize=(10,7))
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()


# ### Interpretation of Confusion Matrix
# 
# > ### Note: 0 - No Inflation Crisis     |     1 - Inflation Crisis
# 
# * The number of True Positives is 37 (column 2, row 2). This implies that these values are correctly predicted as '1' or they have inflation crisis.
# * The number of False Negatives is 3 (column 2, row 1). This implies that 3 values that are predicted as '0' are actually '1', i.e. these values are predicted as no having inflation crisis while actually they do have inflation crisis.
# * The number of False Positives is 1 (column 1, row 2). This implies that these values that are predicted as '1' are actually '0', i.e. they are predicted as having inflation crisis but actually they do not have inflation crisis.
# * The number of True Negatives is 2.8e+02 (column 1, row 1). This implies that these values that are predicted as '1' are actually '1', i.e. they are predicted as having inflation crisis and they actually do have inflation crisis.

# ### Logistic Regression

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# In[ ]:


print("The accuracy of the model is",logreg.score(X_test,y_test)*100,"percent")


# In[ ]:


plt.figure(figsize=(10,7))
y_pred = logreg.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()


# > #### The confusion matrix obtained is almost the same as the one obtained in Random Forest implying that both Random Forest and Logistic Regression predict with almost the same accuracy (as can be confirmed by the model score above).

# ### Decision Tree 

# In[ ]:


gini = DecisionTreeClassifier()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)
gini.fit(X_train, y_train)
y_pred = gini.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report
print('Accuracy Score: ',accuracy_score(y_test, y_pred)*100,"percent")


# In[ ]:


print('Classification Report')
print(classification_report(y_test, y_pred))


# > #### The cross-validation score of Decision Tree showed an accuracy of approximately 92% but on applicatiion of the Decision Tree model, we obtain an accuracy of 98.74%.

# In[ ]:


plt.figure(figsize=(10,7))
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()


# ### K Nearest Neighbours

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)


# #### Hyperparametric Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors':np.arange(1,25)}
knn_cv = GridSearchCV(knn, grid, cv=5)
knn_cv.fit(X,y)
print("Tuned Parameters are: {}".format(knn_cv.best_params_))
print("Best Score is: {}".format(knn_cv.best_score_))


# #### Since after tuning, the best number of parameters has come out to be no. of neighbors = 5. We shall use KNN model with n_neighbors = 5 on our dataset.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("The accuracy score is", knn.score(X_test, y_test)*100,"percent")


# #### Out of the four methods that we have applied, KNN gives the lowest accuracy.

# In[ ]:


plt.figure(figsize=(10,7))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()


# #### The confusion matrix also confirms the fact that the accuracy of KNN is lower than the other three models because the number of False Positives and False Negatives in this case are higher than all the other three models.

# In[ ]:




