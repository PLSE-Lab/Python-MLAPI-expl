#!/usr/bin/env python
# coding: utf-8

# Hi!

# This is my first Kernel, hope you'll love it!

# For questions and remarks - please write me a comment.

# For those interested in looking at the full notebook - https://github.com/NadavKiani/Students-Performance-in-Exams
# 

# # Student Performence in Exams - Writing Score

# In[ ]:


# Import Libreries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/StudentsPerformance.csv')


# In[ ]:


data.head()


# In[ ]:


# Some statistics about the data


# In[ ]:


data.describe()


# In[ ]:


data.info()


# We can see that there are no missing values

# Now let's convert the data to numeric so the machine learning algorithms could be trained and make predictions.

# In[ ]:


data['gender'].value_counts()


# In[ ]:


sns.catplot(x='gender',kind='count',data=data,height=4.5,palette='viridis')
plt.title('Gender')


# In[ ]:


data['gender'].replace({'male':'0','female':'1'},inplace=True)


# In[ ]:


data['race/ethnicity'].value_counts()


# In[ ]:


data["race/ethnicity"].sort_values()
sns.catplot(x='race/ethnicity',kind='count',data=data,height=4.5,palette='viridis',
            order=['group A','group B','group C','group D','group E'])


# In[ ]:


data['race/ethnicity'].replace({'group A':'1','group B':'2', 'group C':'3',
                               'group D':'4','group E':'5'},inplace=True)


# In[ ]:


data['lunch'].value_counts()


# In[ ]:


sns.catplot(x='lunch',kind='count',data=data,height=4.5,palette='viridis')


# In[ ]:


data['lunch'].replace({'free/reduced':'0','standard':'1'},inplace=True)


# In[ ]:


data['test preparation course'].value_counts()


# In[ ]:


sns.catplot(x='test preparation course',kind='count',data=data,height=4.5,palette='viridis')


# In[ ]:


data['test preparation course'].replace({'none':'0','completed':'1'},inplace=True)


# In[ ]:


data['parental level of education'].value_counts()


# In[ ]:


data["race/ethnicity"].sort_values()
sns.catplot(x='parental level of education',kind='count',data=data,height=4.5,aspect=2,palette='viridis',
            order=["some high school","high school","associate's degree","some college",
                   "bachelor's degree","master's degree"],)


# In[ ]:


data['parental level of education'].replace({'some high school':'1','high school':'1',"associate's degree":'2',
                                        'some college':'3',"bachelor's degree":'4',"master's degree":'5'},inplace=True)


# In[ ]:


data.head()


# Now, when all our fields are numeric, we can continue with the project and make some exploratory data analysis

# ***

# ## Exploratory Data Analysis

# In[ ]:


# Now we will look a little deeper on some interesting plots of our data, in order to get some insights


# In[ ]:


sns.set(rc={'figure.figsize':(20,6)})
sns.countplot(x='writing score', hue='test preparation course',data=data, palette='viridis')
plt.title('Writing Score by Test Preparation Course')


# We can see from this plot that most of the students who got a high score at the writing test, __study__ at the test preparation course. We can understand that the preparation course is __helping__ the students at the writing test

# ***

# In[ ]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='math score',y='reading score', hue='gender',data=data, palette='viridis')
plt.title('Math score VS Readind Score by Gender')


# We can see that there is a __nice correlation__ between the math score and the reading score. We can also see that __female__ (Green) has a better score at the reading exams

# ***

# In[ ]:


# Take a look at our objectives distribution
plt.figure(figsize=(8, 4))
plt.hist(x='writing score',bins=10,data=data)


# We can see that the scores distribution is a __Normal distrution__, where most of the student's score is between __65 to 80__

# ***

# In[ ]:


sns.set(rc={'figure.figsize':(20,6)})
sns.countplot(x='writing score', hue='lunch',data=data, palette='viridis')


# We can see that the students who __ate lunch__, had a __better__ score at the writing test.

# ***

# # Machine Learning

# ## Linear Regression##

# We will first split the data to our predictors variables and our criterion variable

# In[ ]:


X = data[['gender','race/ethnicity','parental level of education','lunch','test preparation course','math score','reading score']]


# In[ ]:


y = data['writing score']


# In[ ]:


# Split the data to train and test
from sklearn import model_selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


# implementation of Linear Regression model using scikit-learn and K-fold for stable model
from sklearn.linear_model import LinearRegression
kfold = model_selection.KFold(n_splits=10)
lr = LinearRegression()
scoring = 'r2'
results = model_selection.cross_val_score(lr, X, y, cv=kfold, scoring=scoring)
lr.fit(X_train,y_train)
lr_predictions = lr.predict(X_test)
print('Coefficients: \n', lr.coef_)


# In[ ]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, lr_predictions))
print('MSE:', metrics.mean_squared_error(y_test, lr_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lr_predictions)))


# In[ ]:


from sklearn.metrics import r2_score
print("R_square score: ", r2_score(y_test,lr_predictions))


# ***

# ***

# ## Decision Trees

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(X_train,y_train)
dtr_predictions = dtr.predict(X_test) 

# R^2 Score
print("R_square score: ", r2_score(y_test,dtr_predictions))


# ***

# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100)
rfr.fit(X_train,y_train)
rfr_predicitions = rfr.predict(X_test) 

# R^2 Score
print("R_square score: ", r2_score(y_test,rfr_predicitions))


# ***

# ## Gardient Boosting

# In[ ]:


from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')
clf.fit(X_train, y_train)
clf_predicitions = clf.predict(X_test) 
print("R_square score: ", r2_score(y_test,clf_predicitions))


# ***

# Now we will show a comparison between all the models that we got

# In[ ]:


y = np.array([r2_score(y_test,lr_predictions),r2_score(y_test,dtr_predictions),r2_score(y_test,rfr_predicitions),
           r2_score(y_test,clf_predicitions)])
x = ["LinearRegression","RandomForest","DecisionTree","Grdient Boost"]
plt.bar(x,y)
plt.title("Comparison of Regression Algorithms")
plt.ylabel("r2_score")
plt.show()


# The Linear Regression has the highest R^2 Score.
