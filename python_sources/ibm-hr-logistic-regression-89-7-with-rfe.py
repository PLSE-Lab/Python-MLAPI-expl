"""
IBM HR Attrition Prediction Model Using Logistic Regression (89.7%)

In this data, we notice that a lot of variables are binn-able or dummy encodable in a way
that can introduce great seperability for Logistic Regression. It is devoid of decimal features
entirely while having numerical features that either represent categories or have innate clusters that 
makes bins effective. Using some creative preprocessing, we can create a simple model that runs incredibly
fast and accurate, which is nice for production implementations.
"""

import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn import metrics
from sklearn import feature_selection
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

# Set seed for reproducability
seed=16

# Format floats without scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load in our data
data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Some Feature Engineering and Discretizing 'Some' features into bins
# I chose bin sizes either by every 10th percentile or by viewing macro trends in the data and picking suitable bins to reflect them
data['StandardWeeklyHours'] = data.StandardHours / 2  # Add Weekly hours along with biweekly hours
data['Salary'] = data.DailyRate * 5 * 52  # Their Yearly Salary (Assumes 5 days a week)
data['AgeBin'] = pd.cut(data.Age, [x for x in range(17, 70, 2)])
data['DailyRateQBin'] = pd.qcut(data.DailyRate, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
data['SalaryQBin'] = pd.qcut(data.Salary, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
data['DistanceFromHomeBin'] = pd.cut(data.DistanceFromHome, [x for x in range(0, 33, 3)])
data['YearsAtCompanyBin'] = pd.cut(data.YearsAtCompany, [x for x in range(0, 44, 2)], right=False) # used left include to include 0
data['PercentSalaryHikeQBin'] = pd.qcut(data.PercentSalaryHike, [0, .2, .4, .6, .8, 1])
data['TotalWorkingYearsBin'] = pd.cut(data.TotalWorkingYears, [x for x in range(0, 44, 2)], right=False) # used left include to include 0

# Drop Some Features
"""
Reasons For Drop
Age = Binned - AgeBin
DailyRate = Binned - DailyRateQBin
DistanceFromHome = DistanceFromHomeBin
EmployeeCount = Variable that just counts employees, No predictive Value
EmployeeNumber = Unique ID for employee, No Predictive Value
HourlyRate = Binned - Similar to DailyRateQBin
MonthlyIncome = Binned - Similiar to SalaryQBin
MonthlyRate = Binned - Similiar to SalaryQBin
Over18 = Only Yes, No Predictive Value
StandardHours = All were 80, No Predictive Value
StandardWeeklyHours = All were 40, No Predictive Value
YearsAtCompany = Binned - YearsAtCompanyBin
PercentSalaryHike = Binned - PercentSalaryHikeQBin
TotalWorkingYears = Binned - TotalWorkingYearsBin
Salary = Binned - SalaryQBin
"""
data = data.drop(['Age', 'DailyRate', 'DistanceFromHome', 'EmployeeCount', 'EmployeeNumber', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'Over18', 'StandardHours', 'StandardWeeklyHours', 'YearsAtCompany', 'PercentSalaryHike', 'TotalWorkingYears', 'Salary'], axis=1)

# Convert Object to Categorical
data.Attrition = data.Attrition.astype('category')
data.BusinessTravel = data.BusinessTravel.astype('category')
data.Department = data.Department.astype('category')
data.Education = data.Education.astype('category')
data.EducationField = data.EducationField.astype('category')
data.EnvironmentSatisfaction = data.EnvironmentSatisfaction.astype('category')
data.Gender = data.Gender.astype('category')
data.JobInvolvement = data.JobInvolvement.astype('category')
data.JobLevel = data.JobLevel.astype('category')
data.JobRole = data.JobRole.astype('category')
data.JobSatisfaction = data.JobSatisfaction.astype('category')
data.MaritalStatus = data.MaritalStatus.astype('category')
# NumCompaniesWorked kept as Int64
data.OverTime = data.OverTime.astype('category')
data.PerformanceRating = data.PerformanceRating.astype('category')
data.RelationshipSatisfaction = data.RelationshipSatisfaction.astype('category')
data.StockOptionLevel = data.StockOptionLevel.astype('category')
# TrainingTimesLastYear kept as Int64
data.WorkLifeBalance = data.WorkLifeBalance.astype('category')
# YearsInCurrentRole kept as Int64
# YearsSinceLastPromotion kept as Int64
# YearsWithCurrManager kept as Int64

# I keep some variables as ints because they are already discretized, have a small range, and a level of sparsity that keep them from being effective dummy variables

# Seperate Out Categorical Features and Give TThem A Dummy Encoding
y = data.Attrition
X = data.drop(['Attrition'], axis=1)

numerical_inds = [11, 16, 18, 19, 20]
categorical_inds = list(set([x for x in range(0, X.shape[1])]) - set(numerical_inds))
X_numerical = X.iloc[:, numerical_inds]
X_categorical = X.iloc[:, categorical_inds]

# Get Dummy Variables
X_categorical = pd.get_dummies(X_categorical)                     

X = pd.concat([X_numerical, X_categorical], axis=1)

# Split Train/Test by Stratification because we have a class imbalance
# This ensures that we have the same ratio of yes/no target variables in train and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=seed, stratify=y)

# Declare our model
model = LogisticRegression(random_state=seed)

# Fit Recursive Feature Extraction with our model to the data
# This intuitively tries combinations of 26 features that maximizes our accuracy
selector = feature_selection.RFE(model, 26, step=1) # I did an exhaustive search of all the number of features to try. 26 features is just the best
selector = selector.fit(X_train, y_train)

# Get Dropped Features
dropped_col_ind = list(set([x for x in range(0, X_train.shape[1])]) - set(selector.get_support(indices=True)))

# Drop Not Selected Features
X_train = X_train.drop(X_train.columns[dropped_col_ind], axis=1)
X_test = X_test.drop(X_test.columns[dropped_col_ind], axis=1)
print("Kept Features:", X_train.columns.values)

# Fit Model to new feature subset
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Get Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)