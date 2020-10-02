#!/usr/bin/env python
# coding: utf-8

# ## 0.1. Problem Statement: 
# 
# The Client is a large MNC and have 9 broad verticals across the organisation. 
# One of the problem this client is facing is around identifying the right people for promotion (only for manager position and below) and prepare them in time. 
# 
# Currently the process, they are following is:
# 
# 1. They first identify a set of employees based on recommendations/ past performance
# 2. Selected employees go through the separate training and evaluation program for each vertical. These programs are based on the required skill of each vertical
# 3. At the end of the program, based on various factors such as training performance, KPI completion (only employees with 
# 
# For above mentioned process, the final promotions are only announced after the evaluation and this leads to delay in transition to their new roles. Hence, company needs your help in identifying the eligible candidates at a particular checkpoint so that they can expedite the entire promotion cycle.

# ## 0.2. Objective:
# 
# * To analyse multiple attributes around Employee's past and current performance along with demographics from the dataset, and find the important features among available features that can be used to predict promotion eligibility.
# 
# * To predict whether a potential promotee at checkpoint in the test set will be promoted or not after the evaluation process.

# ## 1. Data Understanding

# * The training dataset has:
# * Number of instances or data points: 54808
# * Number of variables or attributes: 14
# 
# ----------------------------------------------------------------------------------------
#                                         # DATA DICTIONARY #
# ----------------------------------------------------------------------------------------
# | Variable | Definition |  
# |:---|:---|  
# | employee_id: | Unique ID for employee |
# | department: | Department of employee |    
# | region: | Region of employment (unordered) |  
# | education: | 	Education Level |
# | gender: | Gender of Employee |  
# | recruitment_channel: | 	Channel of recruitment for employee |
# | no_of_trainings: | 	no of other trainings completed in previous year on soft skills, technical skills etc. | 
# | age: | 	Age of Employee | 
# | previous_year_rating: | 	Employee Rating for the previous year | 
# | length_of_service: | 	Length of service in years | 
# | KPIs_met >80%: | 	if Percent of KPIs(Key performance Indicators) >80% then 1 else 0 | 
# | awards_won?: | 	if awards won during previous year then 1 else 0 | 
# | avg_training_score: | 	Average score in current training evaluations | 
# | is_promoted: | 	(Target) Recommended for promotion | 
#  
# #### References:
# https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018/

# ## 2. Data Preparation
# 

# In[ ]:


# To import all the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from sklearn import model_selection
from sklearn.cross_validation import train_test_split 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
import warnings

# To ignore or hide all the warnings
warnings.filterwarnings('ignore');


# In[ ]:


# To load the training dataset into a pandas dataFrame
emp_train_df = pd.read_csv('../input/train_LZdllcl.csv')
emp_test_df = pd.read_csv('../input/test_2umaH9m.csv')

# To add is_test_set column in training dataset to identify the type of dataset
emp_train_df['is_test_set']=0

# To add is_train_test column in test dataset
emp_test_df['is_test_set']=1

# To add is_promoted column in test dataset
emp_test_df['is_promoted']=np.nan


# In[ ]:


# To see columns of train dataset
emp_train_df.columns


# In[ ]:


# To change sequence of columns to match as that from test dataset column sequence
emp_train_df = emp_train_df[['employee_id', 'department', 'region', 'education', 'gender',
       'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating',
       'length_of_service', 'KPIs_met >80%', 'awards_won?',
       'avg_training_score', 'is_test_set', 'is_promoted']]


# In[ ]:


# Let's have a look at first few records of train dataset
emp_train_df.head()


# In[ ]:


# To see columns of test dataset
emp_test_df.columns


# In[ ]:


# Let's have a look at first few records of test dataset
emp_test_df.head()


# In[ ]:


# Combining train and test data using pandas
emp_df = emp_train_df.append(emp_test_df)


# In[ ]:


emp_df.columns


# In[ ]:


# High level statistics of the dataset 

# To check the number of datapoints and number of attributes or features available in the training dataset
print(emp_train_df.shape) 


# In[ ]:


# High level statistics of the dataset 

# To check the number of datapoints and number of attributes or features available in the test dataset
print(emp_test_df.shape) 


# In[ ]:


# High level statistics of the dataset 

# To check the number of datapoints and number of attributes or features available in the merged dataset
print(emp_df.shape) 


# **Observation(s):**
# * Number of datapoints: 78298
# * Number of features or attributes: 15

# In[ ]:


# To see column names in the dataset
print(emp_df.columns)

# To see first few data points in the dataset
emp_df.head()


# In[ ]:


# check the data frame info
emp_df.info()


# In[ ]:


# Let's have look at the different column values
emp_df.describe().transpose()


# **Observation(s):**
# * no_of_trainings range between 1 to 10
# * age of employee range from 20 to 60
# * An employee maximum survived for 37 years
# * KPI is either 1 or 0 where 1 indicates an employee Percent of KPIs >80% otherwise 0
# * awards_won = 1 means an employee was awarded during previous year otherwise 0
# * Average score in current training evaluations range between 39 to 99
# * is_test_test, here 1 indicate the record belong to test dataset, and 0 indicates the record belong to train dataset
# * The previous_year_rating attribute has 4124 missing values in the dataset   

# In[ ]:


# To check duplicate employee_id values if any
emp_df[emp_df.duplicated(['employee_id'], keep=False)] #No duplicate employee_id found


# In[ ]:


# To check distinct values for different attributes
print("Department:\n{0}\n\nRegion:\n{1}\n\nEducation:\n{2}\n\nGender:\n{3}\n\nRecruitment_channel:\n{4}\n\nno_of_trainings:\n{5}\n\nAge:\n{6}\n\nprevious_year_rating:\n{7}\n\nlength_of_service:\n{8}\n\nKPIs_met >80%:\n{9}\n\nawards_won?:\n{10}\n\navg_training_score:\n{11}\n\nis_promoted:\n{12}\n\n"      .format(emp_df["department"].unique(),sorted(emp_df["region"].unique()),emp_df["education"].unique(),emp_df["gender"].unique(),emp_df["recruitment_channel"].unique(),sorted(emp_df["no_of_trainings"].unique()),sorted(emp_df["age"].unique()),sorted(emp_df["previous_year_rating"].unique()),sorted(emp_df["length_of_service"].unique()),emp_df["KPIs_met >80%"].unique(),emp_df["awards_won?"].unique(),sorted(emp_df["avg_training_score"].unique()),emp_df["is_promoted"].unique())); 


# **Observation(s):**
# * There are only two columns having NaN values which are: education and previous_year_rating

# In[ ]:


# To check count of nan values in each column of trating dataset

#count_nan = len(emp_df) - emp_df.count()
#count_nan

emp_df.isnull().sum(axis = 0)


# **Observation(s):**
# * education column has 3443 nan values
# * previous_year_rating column has 5936 nan values
# * We will assign the predicted value(0,1) for is_promoted for the test dataset once the final model has been created

# In[ ]:


# To replace the nan values with zero for easy manupulation
emp_df['education']=emp_df['education'].fillna('Not_Known')

# To replace the previous_year_rating : nan values with 2 (taking average rating) for easy data manupulation
emp_df['previous_year_rating']=emp_df['previous_year_rating'].fillna(2)

emp_df.head()


# In[ ]:


# To check number of classes in is_promoted
emp_df["is_promoted"].value_counts()


# **Observation(s):**
# * There are 4668 employee eligible for promotion in the training dataset.

# In[ ]:


# Let's classify the data based on is_promoted status
promoted = emp_df[emp_df["is_promoted"]==1];
not_promoted = emp_df[emp_df["is_promoted"]==0];


# In[ ]:


# To verify the above classified variables value looking into first few records
print("Employees eligible for promotion:")
promoted.head()


# In[ ]:


print("\n\nEmployees not eligible for promotion:")
not_promoted.head()


# In[ ]:


# To remove region column values prefiex as below
emp_df['region'] = emp_df['region'].str.replace('region_','')
#df['range'].str.replace(',','-')
emp_df.head()


# In[ ]:


# Creating dummy variables for categorical datatypes
emp_df_dummies = pd.get_dummies(emp_df, columns=['department','region','education','recruitment_channel'])
emp_df_dummies.head()


# In[ ]:


# To replace gender categorical variable value as 1 for 'm' and 0 for 'f'
gender_mapping = {'m': 1, 'f': 0}
emp_df_dummies['gender'] = emp_df['gender'].map(gender_mapping) 
emp_df_dummies.head()


# ## 4. Exploratory Data Analysis

# ### 4.1. Univaraite Analysis
# 
# ### 4.1.1. Plotting using PDF and CDF values

# In[ ]:


#Probability Density Functions (PDF)
#Cumulative Distribution Function (CDF)

##Let's Plots for PDF and CDF of age for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["age"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["age"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "middle left")
plt.xlabel("Employee Age") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html


# ** Observation(s)**
# * No significant information can be derived due to huge overlappling

# In[ ]:


##Let's Plots for PDF and CDF of length_of_service for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["length_of_service"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["length_of_service"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "middle left")
plt.xlabel("Employee service (No. of yrs)") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html


# ** Observation(s)**
# * No significant information can be derived due to huge overlappling

# In[ ]:


##Let's Plots for PDF and CDF of no_of_trainings for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["no_of_trainings"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["no_of_trainings"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "middle left")
plt.xlabel("no_of_trainings") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html


# ** Observation(s)**
# * No significant information can be derived

# In[ ]:


##Let's Plots for PDF and CDF of KPIs_met >80% for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["KPIs_met >80%"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["KPIs_met >80%"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "upper left")
plt.xlabel("KPIs_met >80%") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html


# ** Observation(s)**
# * No significant information can be derived

# In[ ]:


##Let's Plots for PDF and CDF of previous_year_rating for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["previous_year_rating"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["previous_year_rating"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "upper left")
plt.xlabel("previous_year_rating") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html


# ** Observation(s)**
# * No significant information can be derived due to huge overlapping of pdf

# In[ ]:


##Let's Plots for PDF and CDF of awards_won for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["awards_won?"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["awards_won?"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "middle left")
plt.xlabel("awards_won?\n(0 - No, 1 - Yes)") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html


# ** Observation(s)**
# * No significant information can be derived

# In[ ]:


#Probability Density Functions (PDF)
#Cumulative Distribution Function (CDF)

##Let's Plots for PDF and CDF of avg_training_score for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["avg_training_score"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["avg_training_score"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "upper left")
plt.xlabel("avg_training_score") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html


# ** Observation(s)**
# * No significant information can be derived

# ### 4.1.2. Box plot and Whiskers

# In[ ]:


## Box-plot for no_of_trainings
ax = sbn.boxplot(x="is_promoted", y="no_of_trainings", hue = "is_promoted", data=emp_df)  
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["Yes", "No"], loc = "upper center")
plt.xlabel("Promotion Status? (1=Yes, 0=No)") 
plt.ylabel("no_of_trainings") 
plt.show()


# ** Observation(s)**
# * No significant information could be derived

# ### 4.1.3. Violin Plots

# In[ ]:


### A violin plot combines the benefits of Box-plot and PDF
#Let's have a look at employee age wise Violin Plot
sbn.violinplot(x="is_promoted", y="age", data=emp_df, width=0.9)
plt.xlabel("Employee Promotion Status? \n[1 = Yes, 0 = No]") 
plt.ylabel("Employee Age") 
plt.show()


# ** Observation(s)**
# * Some employee b/w age 45-50 yrs have more chances to not being promoted

# In[ ]:


#Let's have a look at employee previous_year_rating wise Violin Plot
sbn.violinplot(x="is_promoted", y="previous_year_rating", data=emp_df, width=0.9)
plt.xlabel("Employee Promotion Status? \n[1 = Yes, 0 = No]") 
plt.ylabel("previous_year_rating") 
plt.show()


# ** Observation(s)**
# * Employee with previous year rating as 3 or 5 has more chances being promoted
# * Employee with previous year rating as 1,2, or 4 has more chances NOT being promoted

# In[ ]:


#Let's have a look at employee avg_training_score wise Violin Plot
sbn.violinplot(x="is_promoted", y="avg_training_score", data=emp_df, width=0.9)
plt.xlabel("Employee Promotion Status? \n[1 = Yes, 0 = No]") 
plt.ylabel("avg_training_score") 
plt.show()


# ** Observation(s)**
# * Employee with avg_training_score > 85 definitely gets promoted
# * Employee with avg_training_score b/w 46-54 and b/w 59-61 has more chances NOT being promoted

# ### 4.2. Bi-varaite Analysis

# ### 4.2.1.  2-D Scatter Plot

# In[ ]:


# Let's see promotion behaviour using 2D scatter plot
sbn.set_style("whitegrid");
sbn.FacetGrid(emp_df, hue="is_promoted", size=4)   .map(plt.scatter, "age", "length_of_service")   .add_legend();
plt.show(); 


# ** Observation(s)**
# * Employee with age b/w 23 to 45 yrs and lenght of service b/w 3 to 8 yrs have more chance of promotion

# In[ ]:


# Compare b/w avg_training_score and KPIs_met >80%
sbn.set_style("whitegrid");
sbn.FacetGrid(emp_df, hue="is_promoted", size=4)   .map(plt.scatter, "KPIs_met >80%", "avg_training_score")   .add_legend();
plt.show(); 


# ** Observation(s)**
# * Hard to derive any significant information.

# In[ ]:


# Compare b/w previous_year_rating and no_of_trainings
sbn.set_style("whitegrid");
sbn.FacetGrid(emp_df, hue="is_promoted", size=4)   .map(plt.scatter, "previous_year_rating", "no_of_trainings")   .add_legend();
plt.show();


# ** Observation(s)**
# * Due to considerable overlaps, No significant information can be derived

# ### 4.2.2. Pair-plot

# In[ ]:


# Let's see all the possible combination using pair plot
plt.close();
sbn.set_style("whitegrid"); #white, dark, whitegrid, darkgrid, ticks
sbn.pairplot(emp_df, hue="is_promoted", vars=['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service',                                               'KPIs_met >80%', 'awards_won?', 'avg_training_score'], size=5);
plt.show();


# ** Observation(s)**
# * length_of_service, age, avg_training_score, no_of_trainings are more important variables.
# * Employee with length_of_service b/w 1-8 and avg_training_score > 45 has more chances to get promoted

# ### 4.2.3. Histogram (with PDF)

# In[ ]:


#Employee Age wise Promotion status
sbn.FacetGrid(emp_df, hue="is_promoted", size=5)   .map(sbn.distplot, "age")   .add_legend();

plt.show();


# ** Observation(s)**
# * Employee b/w age 33-36 has more promotion chances

# In[ ]:


#Employee previous_year_rating wise Promotion status
sbn.FacetGrid(emp_df, hue="is_promoted", size=5)   .map(sbn.distplot, "previous_year_rating")   .add_legend();

plt.show();


# ** Observation(s)**
# * Employee with rating 5 has high chance of promotion

# In[ ]:


#Employee avg_training_score wise Promotion status
sbn.FacetGrid(emp_df, hue="is_promoted", size=5)   .map(sbn.distplot, "avg_training_score")   .add_legend();

plt.show();


# ** Observation(s)**
# * Employee with avg_training_score b/w 45-55 has less chance of promotion
# * Employee with avg_training_score > 85 has more chance of promotion

# ## 5. Data cleanup - remaing part
# 

# In[ ]:


# To change the data type of previous_year_rating as int
emp_df_dummies[['previous_year_rating']] = emp_df_dummies[['previous_year_rating']].astype(int) 


# In[ ]:


# To validate the data types of all the variables
emp_df_dummies.info()


# In[ ]:


#To see the total number of columns in final dataframe
len(emp_df_dummies.columns)


# In[ ]:


# To check the all column names
emp_df_dummies.columns


# In[ ]:


# To change the sequence of columns and store the data points into final_df

final_df = emp_df_dummies[['gender', 'no_of_trainings', 'age',        'previous_year_rating', 'length_of_service', 'KPIs_met >80%',        'awards_won?', 'avg_training_score',        'department_Analytics', 'department_Finance', 'department_HR',        'department_Legal', 'department_Operations', 'department_Procurement',        'department_R&D', 'department_Sales & Marketing', 'department_Technology',        'region_1', 'region_2', 'region_3', 'region_4', 'region_5',        'region_6', 'region_7', 'region_8', 'region_9', 'region_10', 'region_11',        'region_12', 'region_13', 'region_14', 'region_15', 'region_16',        'region_17', 'region_18', 'region_19', 'region_20',        'region_21', 'region_22', 'region_23', 'region_24', 'region_25',        'region_26', 'region_27', 'region_28', 'region_29', 'region_30',        'region_31', 'region_32', 'region_33', 'region_34',        'education_Below Secondary', 'education_Bachelor\'s',        'education_Master\'s & above', 'education_Not_Known',        'recruitment_channel_referred', 'recruitment_channel_sourcing',        'recruitment_channel_other', 'is_test_set', 'is_promoted']];


# In[ ]:


final_df.columns


# In[ ]:


len(final_df.columns)


# ## 6. Model Building and Evaluation
# 

# In[ ]:


# Let's divide final_df into train and test dataset
train = final_df[final_df["is_test_set"] == 0]
test = final_df[final_df["is_test_set"] == 1]

# Remove is_test_set column from both train and test
del train['is_test_set']
del test['is_test_set']

# To check location of dependent variable/column
train.columns.get_loc("is_promoted")


# In[ ]:


# Assigning default value as Zero for now to test dataset for is_promoted column
test['is_promoted'] = 0.0

array = train.values
X_train = array[:,0:58] 
Y_train = array[:,58]

test_array = test.values
X_test = test_array[:,0:58] 
Y_test = test_array[:,58]


# In[ ]:


# To set test options and evaluation metric
seed = 7
scoring = 'accuracy'


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# ** Observation **
# 
# **Accurecy**
# LR: 0.931743 (0.003295)
# LDA: 0.933915 (0.002671)
# KNN: 0.921708 (0.003300)
# CART: 0.898226 (0.003911)
# NB: 0.533317 (0.049943)
# QDA: 0.680213 (0.179284)
# 
# * We can see that it looks like LDA has the largest estimated accuracy score.

# In[ ]:


# To Compare Algorithms
# To create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. 

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ** Observation **
# 
# * Linear Discriminant Analysis acheiving the best accuracy

# In[ ]:


# Assigning default value as Zero for now to test dataset is_promoted
test['is_promoted'] = 0.0

# Make predictions on test dataset
test_array = test.values
X_test = test_array[:,0:58]
Y_test = test_array[:,58] 

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_test)

print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


# ** Observation **
# 
# * Linear Discriminant Analysis accuracy on test dataset: 0.969348659004

# In[ ]:


#To store the final predicted result for test dataset into sample_submission.csv

sample_submission_lda = emp_df_dummies.loc[(emp_df_dummies.is_test_set == 1), ['employee_id']]
sample_submission_lda['is_promoted'] = list(predictions)

sample_submission_lda.to_csv('sample_submission_lda.csv',index=False)


# In[ ]:


# To check the number of eligible employees in the predicted test dataset
sample_submission_lda[sample_submission_lda['is_promoted']==1].count()


# ** Observation **
# 
# * 720 employee promoted from the test dataset as per final prediction
# 
# Note: Any comments and improvements, guidence are appretiated for the model.

# In[ ]:




