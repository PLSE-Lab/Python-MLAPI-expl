#!/usr/bin/env python
# coding: utf-8

# # Prediction of the average final grade

# The data used is from a Portuguese secondary school. The data includes academic and personal characteristics of the students as well as final grades.

# G1 - first period grade (numeric: from 0 to 20)  
# G2 - second period grade (numeric: from 0 to 20)  
# G3 - final grade (numeric: from 0 to 20, output target)  
# We will add a fourth column 'G_avg' which is the average of G1, G2 and G3. 'G_avg' will be our target column.

# In our model we will visualize the relation between the average grade and all other features that we will divide in sections.  
# We will predict the average grade using Regression.  
# Then we will add a new binary column (pass or not) depending on the average grade and perform Logistic regression.

# ### Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


# ### Uploading dataset

# In[ ]:


data= pd.read_csv('../input/student-grade-prediction/student-mat.csv')
data


# ### Adding the new column Grades Average

# In[ ]:


data['G_avg']= round((data['G1']+data['G2']+data['G3'])/3, 2)
data.head()


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum().sum()


# In[ ]:


data.describe()


# # Data Visualization

# In[ ]:


plt.hist(data['G_avg'], bins=20, range=[0,20])
plt.title('Distribution of grades average of students')
plt.xlabel('Grades average')
plt.ylabel('Count')
plt.show()


# In[ ]:


ax= sns.boxplot(data['G_avg'])
ax.set_title('Boxplot of average grades of students')
ax.set_xlabel('Average Grades')
plt.show()


# The majority of the students (75% of them) scored an average between 8 and 13.  
# The most common average is 9.

# ### Personal Info

# In[ ]:


ax = sns.countplot('age',hue='sex', data=data)
ax.set_title('Students distribution according to age and sex')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
plt.show()


# In[ ]:


ax = sns.swarmplot(x='age', y='G_avg',hue='sex', data=data)
ax.set_title('Age and sex relation with average grades')
ax.set_xlabel('Age')
ax.set_ylabel('Average grades')
plt.show()


# The majority of students are aged between 15 and 18, equally divided between the two genders (the number of females is slightly higher).  
# Most importantly age and sex do not have a clear influence on the average grades.

# ### Family

# In[ ]:


ax = sns.swarmplot(x='famsize', y='G_avg', data=data)
ax.set_title('Age and sex relation with average grades')
ax.set_xlabel('Age')
ax.set_ylabel('Average grades')
plt.show()


# In[ ]:


Pedu = data['Fedu'] + data['Medu'] 
ax = sns.swarmplot(x=Pedu,y=data['G_avg'])
ax.set_title('Parents education effect to child grades')
ax.set_xlabel('Parents education')
ax.set_ylabel('Average Grades')
plt.show()


# In[ ]:


ax = sns.boxplot(x=data['Fjob'],y=data['G_avg'])
ax.set_title('Father job effect to child grades')
ax.set_xlabel('Father job')
ax.set_ylabel('Average Grades')
plt.show()


# In[ ]:


ax = sns.boxplot(x=data['Mjob'],y=data['G_avg'])
ax.set_title('Mother job effect to child grades')
ax.set_xlabel('Mother job')
ax.set_ylabel('Average Grades')
plt.show()


# In[ ]:


ax = sns.swarmplot(x=data['Pstatus'],y=data['G_avg'])
ax.set_title('Parents status effect on grades')
ax.set_xlabel('A= apart, T= living together')
ax.set_ylabel('Average Grades')
plt.show()


# In[ ]:


ax = sns.swarmplot(x=data['famrel'], y=data['G_avg'])
ax.set_title('family relations effect to child grades')
ax.set_xlabel('Relationship with family scale')
ax.set_ylabel('Average Grades')
plt.show()


# Most families have more than 3 members but this doesn't affect the child's grades.  
# Students with educated parents have slightly higher grades.  
# Students with fathers working as teacher and mothers working in health score better than others.  
# Most students live with both their parents and have a good relationship with them but this doesn't affect the grades.

# ### Location

# In[ ]:


ax = sns.boxplot(x='traveltime', y='G_avg',hue='address', data=data)
ax.set_title('Address and travel time to school')
ax.set_xlabel('Travel time (1: <15min, 2: 15min-30min, 3: 30min-1h, 4: >1h)')
ax.set_ylabel('Average grades')
plt.show()


# Students living more than 1 hour far from school score less than others.  
# Also students living in urban side score more than those living in the rural side.

# ### Activities and friends

# In[ ]:


ax = sns.swarmplot(x=data['romantic'],y=data['G_avg'])
ax.set_title('Students having a romantic relationship')
ax.set_xlabel('Romantic')
ax.set_ylabel('Average Grades')
plt.show()


# In[ ]:


b = sns.boxplot(x=data['freetime'], hue=data['activities'], y=data['G_avg'])
b.set_title('Freetime and extra activities')
b.set_xlabel('Freetime')
b.set_ylabel('Average Grades')
plt.show()


# In[ ]:


ax = sns.boxplot(x=data['goout'],y=data['G_avg'])
ax.set_title('Students going out')
ax.set_xlabel('Go out times per week')
ax.set_ylabel('Average Grades')
plt.show()


# In[ ]:


alc= data['Walc'] + data['Dalc'] 
ax = sns.swarmplot(x=alc,y=data['G_avg'])
ax.set_title('Alcohol consumption')
ax.set_xlabel('Alcohol')
ax.set_ylabel('Average Grades')
plt.show()


# Students having a romantic relationship have lower grades.  
# Freetime and extra activities do not affect grades.
# Students that go out a lot score less.
# Alcohol consumption resulted in lower grades.

# ### Discipline

# In[ ]:


ax= sns.violinplot(x=data['studytime'],y=data['G_avg'])
ax.set_title('Study time in relation with gradess')
ax.set_xlabel('Study time')
ax.set_ylabel('Average Grades')
plt.show()


# In[ ]:


ax = sns.violinplot(x=data['failures'],y=data['G_avg'])
ax.set_title('Past subjects failures')
ax.set_xlabel('Failures')
ax.set_ylabel('Average Grades')
plt.show()


# In[ ]:


ax= sns.swarmplot(x=data['absences'],y=data['G_avg'])
ax.set_title('Absence effect on results')
ax.set_xlabel('Number of absence')
ax.set_ylabel('Average Grades')
plt.show()


# Higher study time results in better grades.  
# Students with past failures have lower grades.  
# Absence is not affecting grades.

# ### Support

# In[ ]:


ax= sns.swarmplot(x=data['schoolsup'],y=data['G_avg'])
ax.set_title('School support and grades')
ax.set_xlabel('School support')
ax.set_ylabel('Average Grades')
plt.show()


# In[ ]:


ax= sns.swarmplot(x=data['paid'],y=data['G_avg'])
ax.set_title('Extra paid courses effect to grades')
ax.set_xlabel('paid course')
ax.set_ylabel('Average Grades')
plt.show()


# In[ ]:


ax= sns.swarmplot(x=data['internet'],y=data['G_avg'])
ax.set_title('internet access')
ax.set_xlabel('internet')
ax.set_ylabel('Average Grades')
plt.show()


# School support affect positevely the grades.  
# Paid courses and internet do not affect the grades.

# ### Ambitions

# In[ ]:


ax= sns.boxplot(x=data['higher'],y=data['G_avg'])
ax.set_title('Students who aim to go to university')
ax.set_xlabel('higher education')
ax.set_ylabel('Average Grades')
plt.show()


# Students aiming to join universities later and have a higher education have better grades.

# ## Convert categorical variables to numerical ones

# In[ ]:


data['school']=data['school'].map({'GP':0, 'MS':1})
data['sex']=data['sex'].map({'M':0 ,'F':1})
data['address']=data['address'].map({'R':0 ,'U':1})
data['famsize']=data['famsize'].map({'LE3':0 ,'GT3':1})
data['Pstatus']=data['Pstatus'].map({'A':0 ,'T':1})
data['Mjob']=data['Mjob'].map({'at_home':0 ,'services':1, 'teacher':2, 'health':3, 'other':4})
data['Fjob']=data['Fjob'].map({'at_home':0 ,'services':1, 'teacher':2, 'health':3, 'other':4})
data['famsup']=data['famsup'].map({'no':0, 'yes':1})
data['reason']=data['reason'].map({'course':0 ,'home':1, 'reputation':2, 'other':3})
data['guardian']=data['guardian'].map({'mother':0 ,'father':1, 'other':2})
data['schoolsup']=data['schoolsup'].map({'no':0, 'yes':1})
data['paid']=data['paid'].map({'no':0, 'yes':1})
data['activities']=data['activities'].map({'no':0, 'yes':1})
data['nursery']=data['nursery'].map({'no':0, 'yes':1})
data['higher']=data['higher'].map({'no':0, 'yes':1})
data['internet']=data['internet'].map({'no':0, 'yes':1})
data['romantic']=data['romantic'].map({'no':0, 'yes':1})


# ### Correlation map

# In[ ]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# ### We drop the grades G1, G2 and G3 because they will affect the prediction of the average. 

# In[ ]:


data= data.drop(['G1','G2','G3'], axis=1)
data


# # Predict the Grades Average with Regression

# In[ ]:


import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ### Linear regression 

# We chose our X to be number of failures since it has the highest correlation

# In[ ]:


X = data[["failures"]]
y = data["G_avg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
model = LinearRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
print ("MSE :", metrics.mean_squared_error(y_test,predicted))
print("R squared :", metrics.r2_score(y_test,predicted))


# In[ ]:


plt.scatter(X, y)
plt.title("Linear Regression")
plt.xlabel("Failures")
plt.ylabel("Grade average")
plt.plot(X, model.predict(X), color="r")
plt.show()


# ### Polynomial regression

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
X = data[["failures"]]
y = data["G_avg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)
lg = LinearRegression()
poly = PolynomialFeatures()
X_train_fit = poly.fit_transform(X_train)
lg.fit(X_train_fit, y_train)
X_test_fit = poly.fit_transform(X_test)
predicted = lg.predict(X_test_fit)
print ("MSE :", metrics.mean_squared_error(y_test,predicted))
print("R squared :", metrics.r2_score(y_test,predicted))


# ### Multilinear regression

# In[ ]:


X= data.drop(["G_avg"], axis=1)
y= data["G_avg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=30)
model = LinearRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
print ("MSE :", metrics.mean_squared_error(y_test, predicted))
print("R squared :", metrics.r2_score(y_test, predicted))


# # Prediction model with Classification

# We will add a new column 'pass', it will take binary values. If the Grades average is equal or higher than 10, the student will pass, otherwise he/she will not.

# In[ ]:


data['pass']= np.where(data['G_avg']<10, 0, 1)
data


# In[ ]:


print(data['pass'].value_counts())


# In[ ]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# We will use only the highest correlated features in our prediction model.

# In[ ]:


import sklearn as sk
from sklearn.model_selection import train_test_split
Y = data["pass"]
X = data[["failures","schoolsup","Medu","Fedu","higher","goout","internet"]]
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=20)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg= LogisticRegression(solver="lbfgs")
logreg.fit(X_train, Y_train)
Y_pred= logreg.predict(X_test)
print("Accuracy = {:.2f}".format(logreg.score(X_test, Y_test)))


# In[ ]:


Y_pred1= logreg.predict([[0,1,3,4,1,1,1],[3,0,2,3,1,5,1]])
print(Y_pred1)


# In[ ]:


confusion_matrix= pd.crosstab(Y_test, Y_pred, rownames=["Actual"], colnames=["predict"])
print(confusion_matrix)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


# Accuracy(0.72): The model made 57 correct predictions out of 79 observations. Recall: Out of 48 students that passed, the model predicted 44 correctly. Precision: Out of 62 students that we predicted to pass, 44 actually passed.
