#!/usr/bin/env python
# coding: utf-8

# ## Data Analysis And Machine Learning on Campus Placement Data
# 
# - **Explatory Data Analysis**
# - **Prediction of wheather student gets placed or not (Binary Classification)**
# - **Determining characteristics affecting placement**
# - **Predition of Salary secured by a student (Regression)**
# - **Determining characteristics affecting salary**

# ### Common Questions
# * **Does GPA affect placement?**
# * **Does Higher Secondary School's Percentage still affect campus placement?**
# * **Is work experience required for securing good job?**
# * **What factor affect the salary?**
# 
# **Let's find out**

# # Library Imports

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling


# # Loading Data

# In[ ]:


data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
#Remove Serial Number
data.drop("sl_no", axis=1, inplace=True)


# # Exploratory Data Analysis

# ### Pandas Profiler's Interactive Report  

# In[ ]:


data.profile_report(title='Campus Placement Data - Report', progress_bar=False)


# 
# 
# 
# * 67 Missing values in Salary for students who didn't get placed. **NaN Value needs to be filled**.
# 
# * **Data is not scaled**. Salary column ranges from 200k-940k, rest of numerical columns are percentages.
# 
# * 300k at 75th Percentile goes all the way up to 940k max, in Salary (high skewnwss). Thus, **outliers at high salary end**.

# ## Exploring Data by each Features

# ### Feature: Gender
# #### Does gender affect placements?

# In[ ]:


data.gender.value_counts()
# Almost double


# In[ ]:


sns.countplot("gender", hue="status", data=data)
plt.show()


# In[ ]:


#This plot ignores NaN values for salary, igoring students who are not placed
sns.kdeplot(data.salary[ data.gender=="M"])
sns.kdeplot(data.salary[ data.gender=="F"])
plt.legend(["Male", "Female"])
plt.xlabel("Salary (100k)")
plt.show()


# In[ ]:


plt.figure(figsize =(18,6))
sns.boxplot("salary", "gender", data=data)
plt.show()


# #### Insights
# 
# * We have samples of 139 Male studets and 76 Female students.
# * 30 Female and 40 Male students are not placed. Male students have comparatively higher placemets. 
# * More outliers on Male -> Male students are getting high CTC jobs.
# * Male students are offered slightly greater salary than female on an average. 
# 

# ### Feature: ssc_p (Secondary Education percentage), ssc_b (Board Of Education)
# #### Does Secondary Education affect placements?

# In[ ]:


#Kernel-Density Plot
sns.kdeplot(data.ssc_p[ data.status=="Placed"])
sns.kdeplot(data.ssc_p[ data.status=="Not Placed"])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Secondary Education Percentage")
plt.show()


# * All students with Secondary Education Percentage above 90% are placed
# * All students with Secondary Education Percentage below 50% are not-placed
# * **Students with good Secondary Education Percentage are placed on average.**

# In[ ]:


sns.countplot("ssc_b", hue="status", data=data)
plt.show()


# * Board Of Education does not affect Placement Status much

# In[ ]:


plt.figure(figsize =(18,6))
sns.boxplot("salary", "ssc_b", data=data)
plt.show()


# * Outliers on both, but students from Central Board are getting the highly paid jobs.

# In[ ]:


sns.lineplot("ssc_p", "salary", hue="ssc_b", data=data)
plt.show()


# * No specific pattern (correlation) between Secondary Education Percentage and Salary.
# * Board of Education is Not Affecting Salary

# ### Feature: hsc_p (Higher Secondary Education percentage), hsc_b (Board Of Education), hsc_s (Specialization in Higher Secondary Education)
# #### Does Higher Secondary School affect Placements?

# In[ ]:


#Kernel-Density Plot
sns.kdeplot(data.hsc_p[ data.status=="Placed"])
sns.kdeplot(data.hsc_p[ data.status=="Not Placed"])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Higher Secondary Education Percentage")
plt.show()


# * Overlap here too. More placements for percentage above 65%
# * Straight drop below 60 in placements -> Perntage must be atleast 60 for chance of being placed

# In[ ]:


sns.countplot("hsc_b", hue="status", data=data)
plt.show()


# Education Board again, doesn't affect placement status much

# In[ ]:


sns.countplot("hsc_s", hue="status", data=data)
plt.show()


# * We have very less students with Arts specialization.
# * Around 2:1 placed:unplaced  ratio for both Science and Commerse students
# 

# In[ ]:


plt.figure(figsize =(18,6))
sns.boxplot("salary", "hsc_b", data=data)
plt.show()


# * Outliers on both, board doesn't affect getting highly paid jobs. Highest paid job was obtailed by student from Central Board though.

# In[ ]:


sns.lineplot("hsc_p", "salary", hue="hsc_b", data=data)
plt.show()


# * High salary from both Central and Other. 
# * High salary for both high and low percentage.
# * Thus, both these feature doesnot affect salary.

# In[ ]:


plt.figure(figsize =(18,6))
sns.boxplot("salary", "hsc_s", data=data)
plt.show()


# * We can't really say for sure due to only few samples of students with Arts Major, but they aren't getting good salaries.
# 
# * Commerse students have slightly better placement status.

# In[ ]:


sns.lineplot("hsc_p", "salary", hue="hsc_s", data=data)
plt.show()


# * **Student with Art Specialization surprisingly have comparatively low salary**

# ### Feature: degree_p (Degree Percentage), degree_t (Under Graduation Degree Field)
# #### Does Under Graduate affect placements? 

# In[ ]:


#Kernel-Density Plot
sns.kdeplot(data.degree_p[ data.status=="Placed"])
sns.kdeplot(data.degree_p[ data.status=="Not Placed"])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Under Graduate Percentage")
plt.show()


# * Overlap here too. But More placements for percentage above 65.
# * UG Percentage least 50% to get placement

# In[ ]:


sns.countplot("degree_t", hue="status", data=data)
plt.show()


# * We have very less students with "Other". We cant make decision from few cases.
# * Around 2:1 placed:unplaced  ratio for both Science and Commerse students

# In[ ]:


plt.figure(figsize =(18,6))
sns.boxplot("salary", "degree_t", data=data)
plt.show()


# * Science&Tech students getting more salary on average
# * Management stidents are getting more highly paid dream jobs. 

# In[ ]:


sns.lineplot("degree_p", "salary", hue="degree_t", data=data)
plt.show()


# * Percentage does not seem to affect salary.
# * Commerce&Mgmt students occasionally get dream placements with high salary

# ### Feature: workex (Work Experience)
# #### Does Work Experience affect placements?

# In[ ]:


sns.countplot("workex", hue="status", data=data)
plt.show()


# * **This affects Placement.** Very few students with work experience not getting placed!

# In[ ]:


plt.figure(figsize =(18,6))
sns.boxplot("salary", "workex", data=data)
plt.show()


# * Outliers (High salary than average) on bith end but **students with experience getting dream jobs**
# * Average salary as well as base salary high for students with work experience.

# ### Feature: etest_p  (Employability test percentage)

# In[ ]:


#Kernel-Density Plot
sns.kdeplot(data.etest_p[ data.status=="Placed"])
sns.kdeplot(data.etest_p[ data.status=="Not Placed"])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Employability test percentage")
plt.show()


# * High overlap -> It does not affect placement status much
# * More "Not Placed" on percentage 50-70 range and more placed on 80% percentage range

# In[ ]:


sns.lineplot("etest_p", "salary", data=data)
plt.show()


# **This feature surprisingly does not affect placements and salary much**

# ### Feature: specialisation (Post Graduate Specialization)

# In[ ]:


sns.countplot("specialisation", hue="status", data=data)
plt.show()


# * This feature affects Placement status.
# * Comparitively very low not-placed students in Mkt&Fin Section

# In[ ]:


plt.figure(figsize =(18,6))
sns.boxplot("salary", "specialisation", data=data)
plt.show()


# * **More Highly Paid Jobs for Mkt&Fin students **

# ### Feature: mba_p (MBA percentage)
# #### Does MBA Percentage affect placements?

# In[ ]:


sns.boxplot("mba_p", "status", data=data)
plt.show()


# In[ ]:


sns.lineplot("mba_p", "salary", data=data)
plt.show()


# MBA Percentage also deos not affect salary much

# # Feature Selection
# 
# Using Only following features (Ignoring Board of Education -> they didnt seem to have much effect)
# * Gender
# * Secondary Education percentage
# * Higher Secondary Education Percentsge
# * Specialization in Higher Secondary Education
# * Under Graduate Dergree Percentage
# * Under Graduation Degree Field
# * Work Experience
# * Employability test percentage
# * Specialization
# * MBA Percentage
# 
# Will compute feature importance later on.
# 

# # Data Pre-Processing

# In[ ]:


data.drop(['ssc_b','hsc_b'], axis=1, inplace=True)


# ## Feature Encoding

# In[ ]:


data.dtypes
# We have to encode gender,hsc_s, degree_t, workex, specialisation and status


# In[ ]:


data["gender"] = data.gender.map({"M":0,"F":1})
data["hsc_s"] = data.hsc_s.map({"Commerce":0,"Science":1,"Arts":2})
data["degree_t"] = data.degree_t.map({"Comm&Mgmt":0,"Sci&Tech":1, "Others":2})
data["workex"] = data.workex.map({"No":0, "Yes":1})
data["status"] = data.status.map({"Not Placed":0, "Placed":1})
data["specialisation"] = data.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})


# # Problem Statement
# 
# * Predicting If Students gets placed or not (Binary Classification Problem)
# * Predicting Salary of Student (Regression Problem)

# In[ ]:


#Lets make a copy of data, before we proceeed with specific problems
data_clf = data.copy()
data_reg = data.copy()


# ## Binary Classification Problem

# ### Decision Tree Based Models

# **Using Decision Tree based Algorithm does not  require feature scaling, and works great also in presence of categorical columns without ONE_HOT Encoding**

# In[ ]:


# Library imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ### Dropping Salary Feature
# 
# Filling 0s for salary of students who didn't get placements would be bad idea as it would mean student gets placement if he earns salary.

# In[ ]:


# Seperating Features and Target
X = data_clf[['gender', 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p', 'degree_t', 'workex','etest_p', 'specialisation', 'mba_p',]]
y = data_clf['status']


# In[ ]:


#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


#Using Random Forest Algorithm
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# ### Feature Importance (Percentage)
# 
# Tree based algorithms can be used to compute feature importance 
# 
# Checking feature importance obtained from these:

# In[ ]:


rows = list(X.columns)
imp = pd.DataFrame(np.zeros(6*len(rows)).reshape(2*len(rows), 3))
imp.columns = ["Classifier", "Feature", "Importance"]
#Add Rows
for index in range(0, 2*len(rows), 2):
    imp.iloc[index] = ["DecisionTree", rows[index//2], (100*dtree.feature_importances_[index//2])]
    imp.iloc[index + 1] = ["RandomForest", rows[index//2], (100*random_forest.feature_importances_[index//2])]


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot("Feature", "Importance", hue="Classifier", data=imp)
plt.title("Computed Feature Importance")
plt.show()


# hsc_s -> Specialization in Higher Secondary Education
# 
# degree_t -> Under Graduation(Degree type)- Field of degree education
# 
# specialisation -> Post Graduation(MBA)- Specialization
# 
# **Field of study does not seem to affect much**
# 
# Optionally we can remove these least important features and re-clssify data.

# ### Binary Classification with Logistic Regression

# ### One Hot Encoding
# 
# Encoding Categorical Features 

# In[ ]:


# Seperating Features and Target
X = data_clf[['gender', 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p', 'degree_t', 'workex','etest_p', 'specialisation', 'mba_p',]]
y = data_clf['status']
#Reverse Mapping and making Categorical
X["gender"] = pd.Categorical(X.gender.map({0:"M",1:"F"}))
X["hsc_s"] = pd.Categorical(X.hsc_s.map({0:"Commerce",1:"Science",2:"Arts"}))
X["degree_t"] = pd.Categorical(X.degree_t.map({0:"Comm&Mgmt",1:"Sci&Tech",2:"Others"}))
X["workex"] = pd.Categorical(X.workex.map({0:"No",1:"Yes"}))
X["specialisation"] = pd.Categorical(X.specialisation.map({0:"Mkt&HR",1:"Mkt&Fin"}))


# In[ ]:


#One-Hot Encoding
X = pd.get_dummies(X)
colmunn_names = X.columns.to_list()


# ### Feature Scaling
# 
# * Percentages are on scale 0-100 
# * Categorical Features are on range 0-1 (By one hot encoding)
# * High Scale for Salary -> Salary is heavily skewed too -> SkLearn has RobustScaler which might work well here
# 
# **Scaling Everything between 0 and 1 (This wont affect one-hot encoded values)**

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:


#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
y_pred = logistic_reg.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# ### [Computating Feature importance by Mean Decrease Accuracy (MDA)](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html)
# 
# **Since Logistic Regression performed well, Lets run another method for determining fearure importance here.**
# 

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(logistic_reg).fit(X_test, y_test)
eli5.show_weights(perm)


# In[ ]:


plt.figure(figsize=(30, 10))
plt.bar(colmunn_names , perm.feature_importances_std_ * 100)
plt.show()


# **From Feature Importance of Tree-based Algorithms and MDA we can conclude that:**
# * Academic performance affects placement (All percentages had importantance)
# * Work Experience Effects Placement
# * Gender and Specialization in Commerse (in higher-seondary and undergraduate) also has effect on placements.

# ## Prediction of Salary (Regression Analysis)

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score


# ### Data Preprocessing

# In[ ]:


#dropping NaNs (in Salary)
data_reg.dropna(inplace=True)
#dropping Status = "Placed" column
data_reg.drop("status", axis=1, inplace=True)


# In[ ]:


data_reg.head()


# In[ ]:


#Seperating Depencent and Independent Vaiiables
y = data_reg["salary"] #Dependent Variable
X = data_reg.drop("salary", axis=1)
column_names = X.columns.values


# In[ ]:


#Scalizing between 0-1 (Normalization)
X_scaled = MinMaxScaler().fit_transform(X)


# ### Feature Selection

# ** Not all features are significant. Thus, let's perform a feature selection procedure**
# 
# ![Sequential Forward Feature Selection](https://quantifyinghealth.com/wp-content/uploads/2019/10/backward-stepwise-algorithm.png)

# **Determining Least Significant Variable**
# 
# The least significant variable is a variable which:
# 
# - has the highest p-value
# - Removing it reduces R2 to lowest value compared to other features
# - Removing it has least increment in residuals-sum-of-squares (RSS)
# 

# ### Outliers' Removal
# 
# Feature Selecton cannot perform well in presence of outliers. Lets identy and remove outliers before proceding

# In[ ]:


#PDF of Salary
sns.kdeplot(y)
plt.show()


# It is clear that very few students have salary greater than 400,000 (hence outliers)

# In[ ]:


#Selecting outliers
y[y > 400000]
# 9 records


# In[ ]:


#Removing these Records from data
X_scaled = X_scaled[y < 400000]
y = y[y < 400000]


# In[ ]:


#PDF of Salary without outliers. Still skewed though
sns.kdeplot(y)
plt.show()


# ### 1. Determining Least Significant Variable by R2 Score

# In[ ]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# In[ ]:


linreg = LinearRegression()
sfs = SFS(linreg, k_features=1, forward=False, scoring='r2',cv=10)
sfs = sfs.fit(X_scaled, y)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')

plt.title('Sequential Backward Elimination')
plt.grid()
plt.show()
#From Plot its clear that, many features actually decrease the performance


# In[ ]:


# Lets see the top 5 most significant features
top_n = 5
sfs.get_metric_dict()[top_n]


# In[ ]:


#Top N Features
top_n_indices = list(sfs.get_metric_dict()[top_n]['feature_idx'])
print(f"Most Significant {top_n} Features:")
for col in column_names[top_n_indices]:
    print(col)


# In[ ]:


#Select these Features only
X_selected = X_scaled[: ,top_n_indices]
lin_reg = LinearRegression()
lin_reg.fit(X_selected, y)
y_pred = lin_reg.predict(X_selected)
print(f"R2 Score: {r2_score(y, y_pred)}")
print(f"MAE: {mean_absolute_error(y, y_pred)}")


# This is the best I could do with Linear Regression

# ### 2. Determining Least Significant Variable by P-Value
# 
# If the base model gives 0.7 R2 score and the model without a feature gives 0.75 R2 score, we cannot conclude that feature makes the difference, as the score may vary in another trial; in 10 trials the R2 score might change in +/- 0.05. However, if model only varies in +/- 0.01, we can then say that removing a feature made the model better.
# 
# Our null hypothesis is that there is no difference between the two samples of R2 scores.
# 
# P-value is the probability that you would arrive at the same results as the null hypothesis. One of the most commonly used p-value is 0.05. If the calculated p-value turns out to be less than 0.05, the null hypothesis is considered to be false, or nullified (hence the name null hypothesis). And if the value is greater than 0.05, the null hypothesis is considered to be true.
# 
# For a feature, a small p-value indicates that it is unlikely we will observe a relationship between the predictor (feature) and response (salary in our case) variables due to chance.
# 
# Thus, we start with all features. We compute the P-values. We eliminate frature with highest p-value until p-values of all features reach below threshold: 0.05. 
# 
# ![Determining Least Significant Variable by P-Value](https://miro.medium.com/max/1400/1*Jub_nEYtN0htxFpTRzRtBQ.png)
# 
# 

# In[ ]:


#Converting to DF for as  column names gives readibility
X_scaled = pd.DataFrame(X_scaled, columns=column_names)
y = y.values

# We must add a constants 1s for intercept before doing Linear Regression with statsmodel
X_scaled = sm.add_constant(X_scaled)
X_scaled.head()
#Constants 1 added for intercept term


# In[ ]:


# Step 1: With all Features
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()


# In[ ]:


# Identify max P-value (P>|t|) column
# Feature ssc_p has 0.995
#drop ssc_p
X_scaled = X_scaled.drop('ssc_p', axis=1)
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()


# In[ ]:


# specialisation has max P-Value and is greater than 0.05
# Identify max P-value (P>|t|) column
# Feature specialisation has 0.759
#drop specialisation
X_scaled = X_scaled.drop('specialisation', axis=1)
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()


# In[ ]:


# degree_p has max P-Value and is greater than 0.05
# Increase in Adjusted R2
# Feature degree_p has 0.657
#drop degree_p
X_scaled = X_scaled.drop('degree_p', axis=1)
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()


# In[ ]:


# workex has max P-Value and is greater than 0.05
# Increase in Adjusted R2
# Feature workex has 0.337
#drop workex
X_scaled = X_scaled.drop('workex', axis=1)
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()


# In[ ]:


# hsc_p has max P-Value and is greater than 0.05
# Increase in Adjusted R2
# Feature hsc_p has 0.444
#drop hsc_p
X_scaled = X_scaled.drop('hsc_p', axis=1)
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()


# In[ ]:


# etest_p has max P-Value and is greater than 0.05
# Slight Decrease in Adjusted R2..
# Feature etest_p has 0.2
#drop etest_p
X_scaled = X_scaled.drop('etest_p', axis=1)
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()


# In[ ]:


# hsc_s has max P-Value and is greater than 0.05
# Drastic Decrease in Adjusted R2..
# Feature hsc_s has 0.09
#drop hsc_s
X_scaled = X_scaled.drop('hsc_s', axis=1)
model = sm.OLS(y, X_scaled)
results = model.fit()
results.summary()


# In[ ]:


#Stopping here. mba_p -> 0.06 only a bit higher than 0.05


# Thus, Top 5 Features affecting salary we identified are:
# * gender -> Gender
# * degree_t -> Under Graduation(Degree type)- Field of degree education
# * mba_p -> MBA percentage
# * hsc_s -> Specialization in Higher Secondary Education
# * etest_p -> Employability test percentage 
# 
# 
# (Same as with Sequential Feature Selection with mlxtend considering R2)

# #### Do Upvote if you like my notebook.
# #### Thanks
