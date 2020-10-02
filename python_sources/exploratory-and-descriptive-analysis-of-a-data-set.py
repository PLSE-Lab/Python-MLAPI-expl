#!/usr/bin/env python
# coding: utf-8

# In this kernel, I will look into a dataset containing student performance data acquired from the University of Minho, Portugal. I will be using the student-mat portion of the Dataset.#

# In[ ]:


#Importing the important libraries for algebra and data processing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # statistical visualization

import matplotlib.pyplot as plt #matlab plots
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)

#Loading data
data = pd.read_csv("../input/student-mat.csv", sep = ';') #Load the clean training data. Splitter is the semi-colon character


# Now we will start exploring the data to see what it has and look at what we can do with it.

# In[ ]:


data.head()


# In[ ]:


print ('The data has {0} rows and {1} columns'.format(data.shape[0],data.shape[1]))


# In[ ]:


data.info()


# In[ ]:


data.describe() #to look at the numerical fields and their describing mathematical values.


# From the info available on the data, we can tell that the quality of the data is quite decent as there aren't any columns with null values and every cell has a single piece of data. This will significanlty simplify the processing stage of the data as we would not be required to compensate for null values or split dynamic data. However, there are many categorical fields in the data set and that requires some additional processing to generate better results from.

# Next up, we will take the fields (columns) one by one to analyze their importance and effect on the G3 value:
# 
# 1 school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
# 2 sex - student's sex (binary: 'F' - female or 'M' - male)
# 3 age - student's age (numeric: from 15 to 22)
# 4 address - student's home address type (binary: 'U' - urban or 'R' - rural)
# 5 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
# 6 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
# 7 Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
# 8 Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
# 9 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# 10 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# 11 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
# 12 guardian - student's guardian (nominal: 'mother', 'father' or 'other')
# 13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# 14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# 15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
# 16 schoolsup - extra educational support (binary: yes or no)
# 17 famsup - family educational support (binary: yes or no)
# 18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# 19 activities - extra-curricular activities (binary: yes or no)
# 20 nursery - attended nursery school (binary: yes or no)
# 21 higher - wants to take higher education (binary: yes or no)
# 22 internet - Internet access at home (binary: yes or no)
# 23 romantic - with a romantic relationship (binary: yes or no)
# 24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# 25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# 26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# 27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 29 health - current health status (numeric: from 1 - very bad to 5 - very good)
# 30 absences - number of school absences (numeric: from 0 to 93)

# In[ ]:


sns.distplot(data['G3']) #Plotting the distribution of the final grades.


# From this we can tell that the distribution of the grades is decent and doesn't require any further skewness correction yet. We can go with this distribution for now to analyze the data and create a primitive model and it's error rate first. We can look into data processing of the G3 field afterwards if the results aren't satisfactory. 

# In[ ]:


corr = data.corr() # only works on numerical variables.
sns.heatmap(corr)


# In[ ]:


print (corr['G3'].sort_values(ascending=False), '\n')


# From the correlation graph above, we can look at the numerical fields to know the values that affect the end result the most. Obviously G2 and G1 are the most correlated fields to G3 as they are part of the calculation formula for G3 so they will have the greatest effect on our prediction. 
# Another thing we can see is the negative correlation between failures and the G3 result. This also makes quite a lot of sense as more failures tend to negatively affect your end score.
# Absences and free time seem to not be very relevant in the dataset that are analyzing which can be a flag that may help us further understand the data in the future. 

# Now that we have analyzed the numerical data slightly and figured out the most correlated fields, we now have to take a look at the categorical data to figure out how useful the fields may be and how to introduce them into the prediction model. The simplest way to analyze those fields is to compare the means accross the categories. 

# In[ ]:


groupColumns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup'
               , 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

avgColumns = ['G3', 'G2', 'G1']


# In[ ]:


school = data.groupby(groupColumns[0])[avgColumns].mean()
school.head()


# From this, we see that Gabriel Pereira students generally do better than Mousinho da Silveira students. The same analysis can be done for a few more fields:

# In[ ]:


sex = data.groupby(groupColumns[1])[avgColumns].mean()
sex.head()


# In[ ]:


address = data.groupby(groupColumns[2])[avgColumns].mean()
address.head()


# In[ ]:


famsize = data.groupby(groupColumns[3])[avgColumns].mean()
famsize.head()


# In[ ]:


Pstatus = data.groupby(groupColumns[4])[avgColumns].mean()
Pstatus.head()


# In[ ]:


Mjob = data.groupby(groupColumns[5])[avgColumns].mean()
Mjob.head() #interesting results here. Children of fathers working in the health industry are doing significantly better than children
            #of fathers at home or other.


# In[ ]:


Fjob = data.groupby(groupColumns[6])[avgColumns].mean()
Fjob.head()


# In[ ]:


reason = data.groupby(groupColumns[7])[avgColumns].mean()
reason.head()


# In[ ]:


guardian = data.groupby(groupColumns[8])[avgColumns].mean()
guardian.head()


# In[ ]:


schoolsup = data.groupby(groupColumns[9])[avgColumns].mean()
schoolsup.head()


# In[ ]:


famsup = data.groupby(groupColumns[10])[avgColumns].mean()
famsup.head()


# In[ ]:


paid = data.groupby(groupColumns[11])[avgColumns].mean()
paid.head()


# In[ ]:


activities = data.groupby(groupColumns[12])[avgColumns].mean()
activities.head()


# In[ ]:


nursery = data.groupby(groupColumns[13])[avgColumns].mean()
nursery.head()


# In[ ]:


higher = data.groupby(groupColumns[14])[avgColumns].mean()
higher.head() #another interesting field. 


# In[ ]:


internet = data.groupby(groupColumns[15])[avgColumns].mean()
internet.head()


# In[ ]:


romantic = data.groupby(groupColumns[16])[avgColumns].mean()
romantic.head()


# We can also generate an aggregate summary of the means of the most valuable fields we found: internet, guardian and Fjob

# In[ ]:


focusGroupColumns = ['internet', 'guardian', 'Fjob']
aggs = data.groupby(focusGroupColumns)[avgColumns].mean()
print(aggs.to_string())


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# **Model**
# Now we will start working on a regression model that we can train and use to predict future records.

# In[ ]:


X = data.drop('G3', axis=1)
Y = data.G3
X = pd.get_dummies(X) # to convert categorical data to a format that can be used in regression. This isn't the best method to use as it increases the
                      # dimensionality of the dataset but it is a valid place to start
X.info()


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42) # splitting data into 80% test and 20% train since the data is quite small. Usually it's best to use 60:40 or something similar
                                                                                              # with the possibility of validation data for certain types of regression models to avoid overfitting.


# Lets start with a general linear regression model and see how it goes from there

# In[ ]:


from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

predicted = regr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Grade',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))


# Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(max_features='auto')
dtr.fit(X_train, Y_train)
predicted = dtr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Grade',fontsize=25)
plt.ylabel('Error',fontsize=25)


plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))


# XGBooster Regression

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

gbr = GradientBoostingRegressor(loss ='huber', max_depth=6)
gbr.fit (X_train, Y_train)
predicted = gbr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Grade',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))


# KNN Regression

# In[ ]:


from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=6)
knn.fit(X_train, Y_train)

predicted = knn.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Grade',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))


# Since the KNN Model was the best model we had so far with a variance score of 0.79, we can select it as our current prediction model.

# In[ ]:


#Exporting
from sklearn.externals import joblib
#joblib.dump(knn, 'model.pkl') #This will produce a model file that we can import later in a web based python script and possibly take input from a web/mobile application and predict the G3 score


# 
# 
