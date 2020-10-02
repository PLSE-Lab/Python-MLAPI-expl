#!/usr/bin/env python
# coding: utf-8

# # The Competition

# - This is my first work of machine learning using Python. This is a kernel in progress.
# 
# 
# - In this challenge it is necessary to predict which people will survive or not based on actual data from the shipwreck.
# 
# 
# - This is a simplified analysis, but it does contain some relevance for the purpose of characterizing and exploring different data visualization and modeling tools that may be useful for others initiating data analysis and machine learning to gain insight into their own studies.
# 
# 
# - Comments, criticisms and suggestions are always welcome.
# 
# 
# WTFPL license

# ***

# # Introduction

# **Imports and Parameters:**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.2f}'.format

get_ipython().run_line_magic('matplotlib', 'inline')


# **Load dataset:**

# In[ ]:


trainset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')
dataset = trainset.append(testset , ignore_index = True )
titanic = dataset[ :891 ]


# **Observe the dataset:**

# In[ ]:


trainset.head(2)


# In[ ]:


testset.head(2)


# In[ ]:


dataset.head(2)


# In[ ]:


titanic.head(2)


# ## DATA DICTIONARY:
# 
# ##### The **titanic** file contains actual information about the passengers organized as follows:
# 
# - **Age:** Age in years.
# 
# 
# - **Cabin:** Cabin number.
# 
# 
# - **Embarked:** Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
# 
# 
# - **Fare:** Passenger fare
# 
# 
# - **Name:** Passenger name. They contain titles and some people can share the same surname; indicating family relationships.
# 
# 
# - **Parch:** Number of parents / children on board
# 
# 
# - **PassengerId:** Race index and whether or not this passenger has survived (1) or not (0)
# 
# 
# - **Pclass:** Entry class 1 = 1st - Superior, 2 = 2nd - Medium, 3 = 3rd - lower
# 
# 
# - **Sex:** Sex of the passenger
# 
# 
# - **Sibsp:** Number of siblings / spouses on board
# 
# 
# - **Survived:** 0 = No, 1 = Yes
# 
# 
# - **Ticket:** Number of boarding pass

# ***

# # Knowing the DataSet

# **Analyzing the dimensions of the dataset:**

# In[ ]:


print('This DataSet has rows:', titanic.shape[0])
print('This DataSet has columns:', titanic.shape[1])


# It means that this database does not contain the information of all people aboard the Titanic. Contains information of only 891 people.

# **Statistical summary of the DataFrame, with quartiles, median, among others:**

# In[ ]:


titanic.describe()


# **Analyzing information about data type, including index and dtype, column types, non-null values, and memory usage:**

# In[ ]:


titanic.info()


# **Analyzing the data type:**

# In[ ]:


titanic.dtypes


# **Checking the number of values in each column:**

# In[ ]:


titanic.count()


# We can see that the Age, Cabin, and Embarked columns are smaller than the other columns. Meaning that there are null values.

# **Peer correlation:**

# In[ ]:


titanic.mean()


# In[ ]:


titanic.std()


# In[ ]:


titanic.sem()


# ***

# # Processing Empty Data

# ### Calculations involving numeric columns with missing data can be impacted. Is it possible to tell if there is missing data in the dataset? If so, what and how many would these data be? Filling the missing data in a way that does not influence future operations.

# **Checking for null data:**

# In[ ]:


titanic.isnull()


# **Checking amount of null data:**

# In[ ]:


titanic.isnull().sum()


# **Changing Nan Data from Age to Mean of Existing Values:**

# In[ ]:


titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic.head(5)


# **Changing other Nan values to 0:**

# In[ ]:


titanic.fillna(0, inplace = True)
titanic.head(5)


# **Checking amount of null data:**

# In[ ]:


titanic.isnull().sum()


# ***

# # Data Visualization

# ### How many women and how many men were on board, according to the dataset?

# **Number people separated by Sex:**

# In[ ]:


titanic.groupby(['Sex']).size().reset_index(name='Quantity')


# ### How many passengers survived and how many did not survive?

# **How many passengers survived and how many did not survive:**
# - 0.00 = No 
# - 1.00 = Yes

# In[ ]:


titanic.groupby(['Survived']).size().reset_index(name='Quantity')


# **Visualization:**

# In[ ]:


titanic['Survived'].value_counts().plot.pie(colors=('tab:red', 'tab:green'), 
                                       title='Percentage of Surviving and Non-surviving Persons', 
                                       fontsize=12, shadow=True, startangle=90, autopct='%1.1f%%', 
                                       labels=('Not Survived','Survived')).set_ylabel('')


# ### How many women did not survive?

# **Before that was created a column did not survive to facilitate the questions below:**

# In[ ]:


titanic['Not Survived'] = titanic['Survived'].map({0:1,1:0})
titanic.head(2)


# **Women Not Survivors**

# In[ ]:


titanic[titanic['Sex'] == 'female'].groupby('Sex')['Not Survived'].apply(lambda x: np.sum(x == 1))


# ### Proportionally, did more men or more women survive? 

# **Analyzing the amount of Survivors by Gender:**

# In[ ]:


df_survive = titanic[titanic['Survived'] == 1].groupby('Sex')[['Survived']].count()
df_survive


# **Applying the ratio:**

# In[ ]:


plot = df_survive.apply(lambda x: (x / x.sum(axis=0))*100)['Survived']
plot


# **Visualization:**

# In[ ]:


plot.plot.pie(colors=('tab:red', 'tab:green'), 
                                       title='Percentage of passengers by Sex', 
                                       fontsize=12, shadow=True, startangle=90, autopct='%1.1f%%', 
                                       labels=('Woman','Men')).set_ylabel('')


# ### Given the age of the passengers, what is the age and number of people with the highest number of dead?

# **As previously identified, checking for any unfilled ages:**

# In[ ]:


print('Passengers without age filled:',titanic['Age'].isnull().sum())
print('Passengers with full age:',(~titanic['Age'].isnull()).sum())


# **Now we will visualize the amount of the highest number of deaths by age:**

# In[ ]:


titanic['Age'].value_counts().sort_values(ascending = [False]).nlargest(1)


# **Visualization:**

# In[ ]:


plt.figure();
titanic.hist(column='Age', color=('green'), alpha=0.5, bins=10)
plt.title('Age Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency')


# Most of the passengers were approximately 18 and 32 years old.

# In[ ]:


df_hist = pd.DataFrame({'Total': titanic['Age'],
                           'Not Survived': titanic[titanic['Not Survived'] == 1]['Age'], 
                           'Survived':titanic[titanic['Survived'] == 1]['Age']},                       
                    
                          columns=['Total','Not Survived', 'Survived'])

plt.figure();

df_hist.plot.hist(bins=10, alpha=0.5, figsize=(10,5), color=('red','tab:blue','green'), 
                     title='Histograms (Total, Survived and Not Survived) by Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# We can observe that the age range from 0 to 10 has a high survival rate.

# In[ ]:


ax = sns.kdeplot(df_hist['Not Survived'], shade=True, color="r")
ax = sns.kdeplot(df_hist['Survived'], shade=True, color="g")
plt.title('Density of Survived and Not Survived by Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.xticks((0, 10, 20, 30, 40, 50, 60, 70, 80))


# According to the graph above, the age range between 0 and approximately 10 years was of all the range that had more survivors.

# ### What is the average age of surviving men?

# **Average Age of Man:**

# In[ ]:


titanic[(titanic['Sex'] == 'male') & (titanic['Survived'] == 1)].groupby('Sex').mean()['Age']


# ### Taking into account priority passengers (women and children up to 15 years of age regardless of gender) what is the proportion of survivors by sex?

# **Creating a Dataframe that Contains Only Women and Children:**

# In[ ]:


df_priority = (titanic['Age'] <= 15) & (titanic['Age'] > 0) | (titanic['Sex'] == 0)
df_priority = titanic[df_priority]
df_priority.head(2)


# **Grouping the data of women and children who survived:**

# In[ ]:


df_priority.groupby('Sex')['Survived'].apply(lambda x: np.mean(x ==  1)*100)


# In[ ]:


df_priority[df_priority['Age'] <= 15].groupby('Sex')['Survived'].apply(lambda x: np.mean(x == 1)*100)


# ### How many passengers per class?

# **Number of people per class:**

# In[ ]:


pd.pivot_table(titanic, values='Name', index='Pclass', aggfunc='count')


# **Percentage of Survivors by Class (in relation to total survivors):**
# - 1.0 = Yes

# In[ ]:


pd.crosstab(titanic['Pclass'],titanic['Survived'])[[1]].apply(lambda x: (x / x.sum(axis=0))*100)


# **Visualization:**

# In[ ]:


titanic.pivot_table(index='Pclass',  values='Name', aggfunc='count').plot(kind='bar', legend=None,
                                                                     title='Number of people per Class', 
                                                                     color='blue', rot=0).set_xlabel('Class')
plt.ylabel('Quantity')


# In[ ]:


titanic[titanic['Survived'] == 1].groupby('Pclass').sum()['Survived'].plot(kind='bar',
                                                      title='Number of survivors per Class', rot=0).set_xlabel('Classe')
plt.ylabel('Quantity')


# ### Demonstrating the number of survivors and non-survivors, grouped by sex and class.

# **Number of Men and Women by Class:**

# In[ ]:


titanic.pivot_table('Survived', ["Pclass"], 'Sex', aggfunc='count')


# **Number of Survivors Men and Women by Class:**
# - 0.0 = No
# - 1.0 = Yes

# In[ ]:


titanic.pivot_table('Not Survived', ["Sex","Pclass"], 'Survived', aggfunc='count')


# **Visualization:**

# In[ ]:


titanic.pivot_table('PassengerId', ['Pclass'], 'Sex', aggfunc='count').sort_index().plot(kind='barh', stacked=True, 
                                            title='Number of Men and Women by Class').legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('Quantity')


# In[ ]:


titanic[titanic['Survived'] == 1].pivot_table('PassengerId', ['Pclass'], 'Sex', aggfunc='count').plot(kind='barh', 
                                                              title='Number of Survivors Men and Women by Class')\
                                                              .legend(bbox_to_anchor=(1.0, 1.0))
plt.xlabel('Quantity')


# ### Peer Correlation.

# **Checking if there is any dependency relation between them.**

# **Visualization:**

# In[ ]:


sns.heatmap(titanic.corr(),annot=True,cmap=sns.diverging_palette(220, 10, as_cmap = True),linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# ### CONCLUSION (Insights):
# 
# 
# The exploratory analysis aimed to identify attributes of some people to know if they had a greater chance of survival than others.
# 
# 
# We identified factors such as Class, Gender, and Age that actually influenced the increase or decrease in survival chances.
# 
# 
# Women were the ones that had the greatest chances of survival. In summary, the greatest chances of survival were for women in the first and second class. And the lowest chances of survival were for the third-class men.
# 
# 
# It was also identified that the majority of people who did not survive were in the third class and the children in the third class 60% died.
# 
# 
# This study can be continued with the analysis of other variables, exploring and finding new insights using the information from this base to generate new variables, such as for example to know if it is possible to identify the crew.
# Age itself as already commented could be explored through predictive techniques to better identify how this information impacted survival rate.
# 
# 
# ***IMPORTANT:*** It is important to emphasize that the conclusions identified are not definitive, as we are not using statistical techniques to perform this study.

# ***

# # Preparing the Data

# In the original dataset the sex of the passengers is defined as:
# - male
# - female
# 
# In order to use this information in classifiers we will convert it to a numeric value. Being:
# 
# - 1 = male
# - 0 = women

# **Modifying the content of the field that identifies the Gender:**

# In[ ]:


titanic['Sex'] = titanic['Sex'].map({'female': 0,'male': 1})


# In[ ]:


titanic.head(2)


# **Visualization:**

# In[ ]:


sns.countplot(x = 'Sex', hue ='Survived',data = titanic, palette = 'viridis'); 


# In[ ]:


sns.countplot(x = 'Pclass', hue ='Survived',data = titanic, palette = 'viridis');


# In[ ]:


sns.countplot(x = 'Pclass', hue ='Sex',data = titanic, palette = 'viridis');


# ***

# # Converting String Values into Numeric

# **Embarked Feature:**

# In[ ]:


embark = pd.get_dummies(dataset.Embarked , prefix='Embarked')


# In[ ]:


embark.head(2)


# **Class Feature:**

# In[ ]:


classify = pd.get_dummies(dataset.Pclass , prefix='Pclass')


# In[ ]:


classify.head(2)


# **Sex Feature:**

# In[ ]:


gender = pd.Series(np.where(dataset.Sex == 'male' , 1 , 0) , name = 'Sex')


# In[ ]:


gender.head(2)


# **Cabin Feature:**

# **Replaces missing cabin data with U (unrecognized):**

# In[ ]:


booth = pd.DataFrame()
booth['Cabin'] = dataset.Cabin.fillna('U')


# **Mapping each Cabin value with the cabin letter:**

# In[ ]:


booth['Cabin'] = booth['Cabin'].map(lambda c : c[0])
booth = pd.get_dummies(booth['Cabin'] , prefix = 'Cabin')


# In[ ]:


booth.head(2)


# **Below the values are filled using a median:**

# In[ ]:


entry = pd.DataFrame()
entry['Age'] = dataset.Age.fillna(dataset.Age.mean())
entry['Fare'] = dataset.Fare.fillna(dataset.Fare.mean())
entry['SibSp'] = dataset.SibSp.fillna(dataset.SibSp.mean())
entry['Parch'] = dataset.Parch.fillna(dataset.Parch.mean())


# In[ ]:


entry.head(2)


# **Summary Feature:**

# In[ ]:


featured_data = pd.concat([entry , embark , classify , gender, booth], axis=1)


# In[ ]:


featured_data.tail(2)


# ***

# # Training Model

# ### Normalizing data:

# **Applying Normalizing:**

# In[ ]:


from sklearn.model_selection import train_test_split

featured_data_final = featured_data.apply(lambda x:(x - np.mean(x)) / (np.max(x) - np.min(x)))
featured_data['Age'] = featured_data_final['Age']
featured_data['Fare'] = featured_data_final['Fare']
featured_data['SibSp'] = featured_data_final['SibSp']
featured_data['Parch'] = featured_data_final['Parch']


# **Separating Normalized Data:**

# **Creating the dataset to train, validate, and test the models:**

# In[ ]:


training_data_final = featured_data[0:891]
training_data_valid = titanic.Survived
featuring_data_test = featured_data[891:]
train_data, test_data, train_labels, test_labels = train_test_split(training_data_final, training_data_valid, train_size=.7)


# ***

# # Testing Different Models

# **Gaussian Naive Bayes:**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(train_data, train_labels)
y_pred = gaussian.predict(test_data)
acc_gaussian = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_gaussian)


# **Logistic Regression:**

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(train_data, train_labels)
y_pred = logreg.predict(test_data)
acc_logreg = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_logreg)


# **Support Vector Machines:**

# In[ ]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(train_data, train_labels)
y_pred = svc.predict(test_data)
acc_svc = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_svc)


# **Linear SVC:**

# In[ ]:


from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(train_data, train_labels)
y_pred = linear_svc.predict(test_data)
acc_linear_svc = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_linear_svc)


# **Perceptron:**

# In[ ]:


from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(train_data, train_labels)
y_pred = perceptron.predict(test_data)
acc_perceptron = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_perceptron)


# **Multi-layer Perception:**

# Training using the MLP classifier with Gradient Stochastic Descending (SGD) algorithm and 6 neurons in the hidden layer.

# In[ ]:


from sklearn.neural_network import MLPClassifier

mlperceptron = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(6, 2), random_state=1)
mlperceptron.fit(train_data, train_labels)
y_pred = mlperceptron.predict(test_data)
acc_mlperceptron = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_mlperceptron)


# **Decision Tree:**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(train_data, train_labels)
y_pred = decisiontree.predict(test_data)
acc_decisiontree = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_decisiontree)


# **Adaboost:**

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.fit(train_data, train_labels)
y_pred = adaboost.predict(test_data)
acc_adaboost = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_adaboost)


# **Random Forest:**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(train_data, train_labels)
y_pred = randomforest.predict(test_data)
acc_randomforest = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_randomforest)


# **KNN:**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(train_data, train_labels)
y_pred = knn.predict(test_data)
acc_knn = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_knn)


# **Bagged KNN:**

# In[ ]:


from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

baggedknn = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
baggedknn.fit(train_data, train_labels)
y_pred = baggedknn.predict(test_data)
result = accuracy_score(y_pred, test_labels)
cross = cross_val_score(baggedknn,train_data,train_labels,cv=10,scoring='accuracy')
acc_baggedknn = (cross.mean() * 100)
print(acc_baggedknn)


# **Stochastic Gradient Descent:**

# In[ ]:


from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(train_data, train_labels)
y_pred = sgd.predict(test_data)
acc_sgd = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_sgd)


# **Gradient Boosting Classifier:**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(train_data, train_labels)
y_pred = gbk.predict(test_data)
acc_gbk = round(accuracy_score(y_pred, test_labels) * 100, 2)
print(acc_gbk)


# ### Comparing Model Pedictions:

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Bagged KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Multi-layer Perception', 'Linear SVC', 
              'Decision Tree', 'Adaboost', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_baggedknn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron, acc_mlperceptron,
              acc_linear_svc, acc_decisiontree, acc_adaboost,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# * As the Best Score being the Gradient Boosting Classifier. He was chosen for test data.

# ***

# # Deployment

# Creating submission file to upload to the Kaggle competition!

# In[ ]:


test_final = featuring_data_test.as_matrix()

predictions  = gbk.predict(test_final)
predictions  = predictions.flatten().round().astype(int)
passenger_id = dataset[891:].PassengerId
output = pd.DataFrame({'PassengerId': passenger_id, 'Survived': predictions })
output.shape
output.head()
output.to_csv('submission.csv', index = False)


# ***

# ##### INSTALLED VERSIONS

# In[ ]:


pd.show_versions()


# ***
