#!/usr/bin/env python
# coding: utf-8

# # Predicting the final grade of a student
# 
# The data used is from a Portuguese secondary school. The data includes academic and personal characteristics of the students as well as final grades. The task is to predict the final grade from the student information. (Regression)
# 
# ### [Link to dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)
# 
# ### Citation:
# 
# P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
# [Web Link](http://www3.dsi.uminho.pt/pcortez/student.pdf)
# 
# ### Reference [article](/home/dipamvasani7/Desktop/Ubuntu/jupyter_notebooks/data)

# ### Import the relevant modules

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# Splitting data into training/testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

# Distributions
import scipy


# In[ ]:


student = pd.read_csv('../input/student-mat.csv')
student.head()


# # Some basic analysis

# In[ ]:


print('Total number of students:',len(student))


# ## Checking the final grade

# In[ ]:


student['G3'].describe()


# ### Grades according to the number of students who scored them

# In[ ]:


plt.subplots(figsize=(8,12))
grade_counts = student['G3'].value_counts().sort_values().plot.barh(width=.9,color=sns.color_palette('inferno',40))
grade_counts.axes.set_title('Number of students who scored a particular grade',fontsize=30)
grade_counts.set_xlabel('Number of students', fontsize=30)
grade_counts.set_ylabel('Final Grade', fontsize=30)
plt.show()


# 
# 
# This plot does not tell us much. What we should really plot is the distribution of grade.
# 
# 

# # Final grade distribution

# In[ ]:


b = sns.countplot(student['G3'])
b.axes.set_title('Distribution of Final grade of students', fontsize = 30)
b.set_xlabel('Final Grade', fontsize = 20)
b.set_ylabel('Count', fontsize = 20)
plt.show()


# ## Hmmmmm!
# 
# Something seems off here. Apart from the high number of students scoring 0, the distribution is normal as expected.
# Maybe the value 0 is used in place of null. Or maybe the students who did not appear for the exam, or were not allowed to sit for the exam due to some reason are marked as 0. We cannot be sure. Let us check the table for null values

# In[ ]:


student.isnull().any()


# ### None of the variables has null values so maybe grade 0 does not mean null after all

# ## Next let us take a look at the gender variable

# In[ ]:


male_studs = len(student[student['sex'] == 'M'])
female_studs = len(student[student['sex'] == 'F'])
print('Number of male students:',male_studs)
print('Number of female students:',female_studs)


# ## Checking the distribution of Age along with gender

# In[ ]:


b = sns.kdeplot(student['age'], shade=True)
b.axes.set_title('Ages of students', fontsize = 30)
b.set_xlabel('Age', fontsize = 20)
b.set_ylabel('Count', fontsize = 20)
plt.show()


# ### Histogram might be more useful to compare different ages

# In[ ]:


b = sns.countplot('age',hue='sex', data=student)
b.axes.set_title('Number of students in different age groups',fontsize=30)
b.set_xlabel("Age",fontsize=30)
b.set_ylabel("Count",fontsize=20)
plt.show()


# The ages seem to be ranging from 15 - 19. The students above that age may not necessarily be outliers but students with year drops. Also the gender distribution is pretty even.

# ## Does age have anything to do with the final grade?

# In[ ]:


b = sns.boxplot(x='age', y='G3', data=student)
b.axes.set_title('Age vs Final', fontsize = 30)
b.set_xlabel('Age', fontsize = 20)
b.set_ylabel('Final Grade', fontsize = 20)
plt.show()


# ### Plotting the distribution rather than statistics would help us better understand the data

# In[ ]:


b = sns.swarmplot(x='age', y='G3',hue='sex', data=student)
b.axes.set_title('Does age affect final grade?', fontsize = 30)
b.set_xlabel('Age', fontsize = 20)
b.set_ylabel('Final Grade', fontsize = 20)
plt.show()


# We see that age 20 has only 3 data points hence the inconsistency in statistics. Otherwise there seems to be no clear relation of age or gender with final grade

# ## Count of students from urban and rural areas

# In[ ]:


b = sns.countplot(student['address'])
b.axes.set_title('Urban and rural students', fontsize = 30)
b.set_xlabel('Address', fontsize = 20)
b.set_ylabel('Count', fontsize = 20)
plt.show()


# ## Most students are from urban ares, but do urban students perform better than rurual students?

# In[ ]:


# Grade distribution by address
sns.kdeplot(student.loc[student['address'] == 'U', 'G3'], label='Urban', shade = True)
sns.kdeplot(student.loc[student['address'] == 'R', 'G3'], label='Rural', shade = True)
plt.title('Do urban students score higher than rural students?', fontsize = 20)
plt.xlabel('Grade', fontsize = 20);
plt.ylabel('Density', fontsize = 20)
plt.show()


# The graph shows that on there is not much difference between the scores based on location.

# ## Reason to choose this school

# In[ ]:


b = sns.swarmplot(x='reason', y='G3', data=student)
b.axes.set_title('Reason vs Final grade', fontsize = 30)
b.set_xlabel('Reason', fontsize = 20)
b.set_ylabel('Final Grade', fontsize = 20)
plt.show()


# ## Other features
# 
# It might not be wise to analyse every feature so I will find the features most correlated to the final grade and spend more time on them.

# ## Correlation
# 
# Next we find the correlation between various features and the final grade.
#  
# ### Note: This correlation is only between numeric values

# In[ ]:


student.corr()['G3'].sort_values()


# # Encoding categorical variables
# 
# A machine learning model cannot deal with categorical variables (except for some models). Therefore we need to find a way to encode them (represent as numbers) before handing them to the model.
# 
# ## Label encoding
# 
# This method involves assigning one label for each category
# 
# | Occupation    | Label         |
# | ------------- |:-------------:|
# | programmer    | 0             |
# | data scientist| 1             |
# | Engineer      | 2             |
# 
# 
# 
# The problem with label encoding is that the assignment of integers is random and changes every time we run the function. Also the model might give higher priority to larger labels. Label encoding can be used when we have only 2 unique values.
# 
# ## One hot encoding
# 
# The problem with label encoding is solved by one hot encoding. It creates a new column for each category and uses only binary values. The downside of one hot encoding is that the number of features can explode if the categorical variables have many categories. To deal with this we can perform PCA (or other dimensionality reduction methods) followed by one hot encoding.
# 
# | Occupation    | Occupation_prog| Occupation_ds | Occupation_eng |
# | ------------- |:-------------: |:-------------:|:-------------: |
# | programmer    | 1              | 0             | 0              |
# | data scientist| 0              | 1             | 0              |
# | Engineer      | 0              | 0             | 1              |

# ### Example of one hot encoding

# In[ ]:


# Select only categorical variables
category_df = student.select_dtypes(include=['object'])

# One hot encode the variables
dummy_df = pd.get_dummies(category_df)

# Put the grade back in the dataframe
dummy_df['G3'] = student['G3']

# Find correlations with grade
dummy_df.corr()['G3'].sort_values()


# ## Applying one hot encoding to our data and finding correlation again!
# 
# 
# ### Note: 
# Although G1 and G2 which are period grades of a student and are highly correlated to the final grade G3, we drop them. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful because we want to find other factors affect the grade.

# In[ ]:


# selecting the most correlated values and dropping the others
labels = student['G3']

# drop the school and grade columns
student = student.drop(['school', 'G1', 'G2'], axis='columns')
    
# One-Hot Encoding of Categorical Variables
student = pd.get_dummies(student)


# In[ ]:


# Find correlations with the Grade
most_correlated = student.corr().abs()['G3'].sort_values(ascending=False)

# Maintain the top 8 most correlation features with Grade
most_correlated = most_correlated[:9]
most_correlated


# In[ ]:


student = student.loc[:, most_correlated.index]
student.head()


# # Now we will analyse these variables and then train a model

# ### Student with less previous failures usually score higher

# In[ ]:


b = sns.swarmplot(x=student['failures'],y=student['G3'])
b.axes.set_title('Students with less failures score higher', fontsize = 30)
b.set_xlabel('Number of failures', fontsize = 20)
b.set_ylabel('Final Grade', fontsize = 20)
plt.show()


# In[ ]:


family_ed = student['Fedu'] + student['Medu'] 
b = sns.boxplot(x=family_ed,y=student['G3'])
b.axes.set_title('Educated families result in higher grades', fontsize = 30)
b.set_xlabel('Family education (Mother + Father)', fontsize = 20)
b.set_ylabel('Final Grade', fontsize = 20)
plt.show()


# There seems to be a slight trend that with the increase in family education the grade moves up (apart from the unusual high value at family_ed = 1 (maybe students whose parents did not get to study have more motivation)
# 
# ### Note:
# 
# I prefer swarm plots over box plots because it is much more useful to see the distribution of data (and also to spot outliers)

# In[ ]:


b = sns.swarmplot(x=family_ed,y=student['G3'])
b.axes.set_title('Educated families result in higher grades', fontsize = 30)
b.set_xlabel('Family education (Mother + Father)', fontsize = 20)
b.set_ylabel('Final Grade', fontsize = 20)
plt.show()


# As we can see there are only 2 points in family_ed = 1 hence our conclusion was faulty.

# ## Higher education
# 
# Higher education was a categorical variable with values yes and no. Since we used one hot encoding it has been converted to 2 variables. So we can safely eliminate one of them (since the values are compliments of each other). We will eliminate higher_no, since higher_yes is more intuitive.

# In[ ]:


student = student.drop('higher_no', axis='columns')
student.head()


# In[ ]:


b = sns.boxplot(x = student['higher_yes'], y=student['G3'])
b.axes.set_title('Students who wish to go for higher studies score more', fontsize = 30)
b.set_xlabel('Higher education (1 = Yes)', fontsize = 20)
b.set_ylabel('Final Grade', fontsize = 20)
plt.show()


# ### Going out with friends

# In[ ]:


b = sns.countplot(student['goout'])
b.axes.set_title('How often do students go out with friends', fontsize = 30)
b.set_xlabel('Go out', fontsize = 20)
b.set_ylabel('Count', fontsize = 20)
plt.show()


# Most students have an average score when it comes to going out with friends. (normal distribution)

# In[ ]:


b = sns.swarmplot(x=student['goout'],y=student['G3'])
b.axes.set_title('Students who go out a lot score less', fontsize = 30)
b.set_xlabel('Going out', fontsize = 20)
b.set_ylabel('Final Grade', fontsize = 20)
plt.show()


# The graph shows a slight downward trend

# ## Does having a romantic relationship affect grade?
# 
# Again because of one hot encoding we have our variable called romantic_no which is slightly less intuitive but I am going to stick with it. Keep in mind that:
# 
# - romantic_no = 1 means NO romantic relationship
# - romantic_no = 0 means romantic relationship

# In[ ]:


b = sns.swarmplot(x=student['romantic_no'],y=student['G3'])
b.axes.set_title('Students with no romantic relationship score higher', fontsize = 30)
b.set_xlabel('Romantic relationship (1 = None)', fontsize = 20)
b.set_ylabel('Final Grade', fontsize = 20)
plt.show()


# # Modeling
# 
# ### We can create a model in 3 ways
# 
# 1. Binary classification
#     - G3 > 10: pass
#     - G3 < 10: fail
# 2. 5-level classification based on Erasmus grade conversion system
#     - 16-20: very good
#     - 14-15: good
#     - 12-13: satisfactory
#     - 10-11: sufficient
#     -  0-9 : fail
# 3. Regression (Predicting G3)
# 
# ### We will be using the 3rd type

# In[ ]:


# splitting the data into training and testing data (75% and 25%)
# we mention the random state to achieve the same split everytime we run the code
X_train, X_test, y_train, y_test = train_test_split(student, labels, test_size = 0.25, random_state=42)


# In[ ]:


X_train.head()


# ### MAE - Mean Absolute Error
# ### RMSE - Root Mean Square Error

# In[ ]:


# Calculate mae and rmse
def evaluate_predictions(predictions, true):
    mae = np.mean(abs(predictions - true))
    rmse = np.sqrt(np.mean((predictions - true) ** 2))
    
    return mae, rmse


# ### Naive baseline is the median prediction

# In[ ]:


# find the median
median_pred = X_train['G3'].median()

# create a list with all values as median
median_preds = [median_pred for _ in range(len(X_test))]

# store the true G3 values for passing into the function
true = X_test['G3']


# In[ ]:


# Display the naive baseline metrics
mb_mae, mb_rmse = evaluate_predictions(median_preds, true)
print('Median Baseline  MAE: {:.4f}'.format(mb_mae))
print('Median Baseline RMSE: {:.4f}'.format(mb_rmse))


# In[ ]:


# Evaluate several ml models by training on training set and testing on testing set
def evaluate(X_train, X_test, y_train, y_test):
    # Names of models
    model_name_list = ['Linear Regression', 'ElasticNet Regression',
                      'Random Forest', 'Extra Trees', 'SVM',
                       'Gradient Boosted', 'Baseline']
    X_train = X_train.drop('G3', axis='columns')
    X_test = X_test.drop('G3', axis='columns')
    
    # Instantiate the models
    model1 = LinearRegression()
    model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model3 = RandomForestRegressor(n_estimators=100)
    model4 = ExtraTreesRegressor(n_estimators=100)
    model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
    model6 = GradientBoostingRegressor(n_estimators=50)
    
    # Dataframe for results
    results = pd.DataFrame(columns=['mae', 'rmse'], index = model_name_list)
    
    # Train and predict with each model
    for i, model in enumerate([model1, model2, model3, model4, model5, model6]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Metrics
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        
        # Insert results into the dataframe
        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse]
    
    # Median Value Baseline Metrics
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))
    
    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]
    
    return results


# In[ ]:


results = evaluate(X_train, X_test, y_train, y_test)
results


# In[ ]:


plt.figure(figsize=(12, 8))

# Root mean squared error
ax =  plt.subplot(1, 2, 1)
results.sort_values('mae', ascending = True).plot.bar(y = 'mae', color = 'b', ax = ax, fontsize=20)
plt.title('Model Mean Absolute Error', fontsize=20) 
plt.ylabel('MAE', fontsize=20)

# Median absolute percentage error
ax = plt.subplot(1, 2, 2)
results.sort_values('rmse', ascending = True).plot.bar(y = 'rmse', color = 'r', ax = ax, fontsize=20)
plt.title('Model Root Mean Squared Error', fontsize=20) 
plt.ylabel('RMSE',fontsize=20)

plt.show()


# ### We see that linear regression is performing the best in both cases

# In[ ]:




