#!/usr/bin/env python
# coding: utf-8

# ## 1. Importing datasets and libraries

# In[ ]:


# importing necessary libraries:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as ss
from collections import Counter
import math 
from scipy import stats


# In[ ]:


# converting the CSV files to data frames:

emp_gen = pd.read_csv("../input/general_data.csv") # main dataset
emp_sur = pd.read_csv("../input/employee_survey_data.csv")  # employee survey data
emp_man_sur = pd.read_csv("../input/manager_survey_data.csv") # manager survey data
emp_in_time = pd.read_csv("../input/in_time.csv") # login time data
emp_out_time = pd.read_csv("../input/out_time.csv") # logout time data

# data_dictionary as a bonus:
emp_desc = pd.read_excel("../input/data_dictionary.xlsx")

#assigning names to each data frame:

emp_sur.name = 'Employee Survey data'
emp_gen.name = 'General data'
emp_in_time.name = 'Log in time data'
emp_out_time.name = 'Log out time data'
emp_man_sur.name = 'Manager Survey data'


# ## 2. Exploratory Data Analysis

# ### 2.1. Basic exploratory data analysis

# In[ ]:


emp_gen.head(5) #5 top rows of the main dataset


# Many categorical variables have been already encoded. It makes the work easier, but there are still a few ordinal and nominal variables that require to be encoded later.

# In[ ]:


gen_and_surveys = [emp_gen,emp_sur,emp_man_sur]  # list of 3 'main' datasets (general + surveys). I will take care of login/logout datasets separately
log_time = [emp_in_time, emp_out_time]


# In[ ]:


# basic info about datasets:

for dataset in gen_and_surveys:
    print(dataset.name + ':\n')
    print(dataset.info())
    print('-'*50)
    
for dataset in log_time:
    print(dataset.name + ':\n')
    print(dataset.info())
    print('-'*50)


# In[ ]:


emp_gen.describe().T # basic descriptive statistics for main dataset


# Let's see if somehow we can combine all the datasets. The 'EmployeeID' seems to be a key, as it appears in every dataset.

# In[ ]:


# let's see if all the values are unique:

for dataset in gen_and_surveys:
    print(dataset.name +':')  
    print(len(set(dataset['EmployeeID'])))  # for each dataset 'EmployeeID' column contains 4410 unique values
    print('-'*20)  


# In[ ]:


# let's see if the log_time datasets have also EmployeeID column:
for dataset in log_time:
    print(dataset.name + ':\n')
    print(dataset.columns)
    print('-'*50)

    # we see in the output that both datasets have the same column names (1 unnamed + dates).


# In[ ]:


for dataset in log_time:
    print(dataset.name +':')  
    print(len(set(dataset['Unnamed: 0'])))  # for each dataset 'EmployeeID' column contains 4410 unique values. It seems it's our desired key column
    print('-'*20)  


# In[ ]:


# Let's replace our unnamed columns with "EmployeeID"
for dataset in log_time:
    dataset.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True) #now in all 5 datasets we have common 'EmployeeID' column containing 4410 unique values.


# In[ ]:


# let's make sure that in every set we have the same 'EmployeeID' values:

print(len(set(emp_sur['EmployeeID']).intersection(emp_gen['EmployeeID']).intersection(emp_in_time['EmployeeID']).intersection(emp_man_sur['EmployeeID']).intersection(emp_out_time['EmployeeID'])))

# we have 4410 (all) common unique values in 'EmployeeID' column for all 5 datasets. Now, we can set the 'EmployeeID' column as an index:


# In[ ]:


for dataset in log_time:
    dataset.set_index('EmployeeID', inplace=True)

for dataset in gen_and_surveys:
    dataset.set_index('EmployeeID', inplace=True)


# Now we are almost ready to merge all the datasets, but firstly let's explore the login/logout datasets and pull out usefull information from there.

# In[ ]:


# Previously we saw that both sets from our 'log_time' list contain 261 features. To be sure, I will check the number of common header names:

print(len(emp_in_time.columns.intersection(emp_out_time.columns))) # all (261) variables are common


# In[ ]:


# transposing dataframes to perform calculations:
emp_out_time_transposed = emp_out_time.T
emp_in_time_transposed = emp_in_time.T


# In[ ]:


# changing format of indexes and variables
emp_out_time_transposed.index = pd.to_datetime(emp_out_time_transposed.index)
emp_in_time_transposed.index = pd.to_datetime(emp_in_time_transposed.index)

emp_out_time_transposed = emp_out_time_transposed.apply(pd.to_datetime, errors='raise')
emp_in_time_transposed = emp_in_time_transposed.apply(pd.to_datetime, errors='raise')


# In[ ]:


work_time =  emp_out_time_transposed.sub(emp_in_time_transposed)
work_time.head()


# In[ ]:


avg_work_time = work_time.mean()   # this variable will help us to create a new feature ('Overtime') a bit later
avg_work_time.head()


# ### 2.2 Concatenating datasets

# We have already pulled out the necessary information from the login / logout time datasets, so we don't need them anymore. Let's combine the 3 others.

# In[ ]:


main_df = pd.concat(gen_and_surveys,1)


# ### 2.3. Exploratory Data Analysis of combined dataset

# In[ ]:


main_df.head(5) # checking the first observations for the new dataset


# **Our goal is to predict employees' churn, that's why the 'Attrition' is our dependent variable**. Around 16% of the employees has already left the company:

# In[ ]:


print(round(main_df['Attrition'].value_counts(normalize = True),2))
sns.countplot(x='Attrition',data=main_df)


# In[ ]:


sns.pairplot(main_df[['Age','MonthlyIncome','DistanceFromHome','Attrition']],hue = 'Attrition')


# In[ ]:


sns.pairplot(main_df[['Age','MonthlyIncome','DistanceFromHome','Gender']],hue = 'Gender',hue_order=['Male','Female'], palette={'Male':'black','Female':'magenta'},plot_kws={'alpha':0.1},height=4)


# In[ ]:


print("The youngest employee is {} years old.\nThe oldest employee is {} years old.\nThe range of ages in the company: {}".format(main_df.Age.min(), main_df.Age.max(), main_df.Age.max() - main_df.Age.min()))


# In[ ]:


print('Frequency of travels (in %): \n')
print(round(main_df['BusinessTravel'].value_counts(normalize = True)*100,2))
print('\nAttrition rate by Frequency of travels \n')
print(round(main_df['BusinessTravel'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['BusinessTravel'].value_counts()*100,2))


# Employees from the Humar Resources department (the smallest one) were more likely to leave the company:

# In[ ]:


print('Number of Employees in department (in %): \n')
print(round(main_df['Department'].value_counts(normalize = True)*100,2))  
print('\nAttrition rate by Department \n')
print(round(main_df['Department'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['Department'].value_counts()*100,2))


# In[ ]:


print('Number of Employees by Education Level (in %): \n')
print(round(main_df['Education'].value_counts(normalize = True)*100,2).sort_index())
print('\nAttrition rate by Education Level: \n')
print(round(main_df['Education'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['Education'].value_counts()*100,2).sort_index())


# In[ ]:


print('Number of Employees by Education Field (in %): \n')
print(round(main_df['EducationField'].value_counts(normalize = True)*100,2).sort_index())

print('\nAttrition rate by Education Field: \n')
print(round(main_df['EducationField'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['EducationField'].value_counts()*100,2).sort_index())


# In[ ]:


fig = plt.figure(figsize=(15,12))
fig.subplots_adjust(top=0.85, wspace=0.7,hspace = 0.6)

ax1 = fig.add_subplot(2,2,1)
ax1.set_title("Percentage share of employees coming from \ndifferent fields of educations, by department")
sns.heatmap(pd.crosstab(main_df.Department, main_df.EducationField, normalize = 'columns'),
            cmap="coolwarm", annot=True, cbar=False, center=0.5)



ax2 = fig.add_subplot(2,2,2)
ax2.set_title("Percentage share of employees coming from \ndifferent fields of educations, by JobRole")
sns.heatmap(pd.crosstab(main_df.JobRole, main_df.EducationField, normalize = 'columns'),
            cmap="coolwarm", annot=True, cbar=False, center=0.5)


# In[ ]:


print('\nAttrition rate by Marital Status: \n')
print(round(main_df['MaritalStatus'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['MaritalStatus'].value_counts()*100,2))
(main_df['MaritalStatus'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['MaritalStatus'].value_counts()*100).plot.bar(color = 'blue')


# In[ ]:


print('\nAttrition rate by Total number of companies the employee has worked for: \n')
print(round(main_df['NumCompaniesWorked'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['NumCompaniesWorked'].value_counts()*100,2))

(main_df['NumCompaniesWorked'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['NumCompaniesWorked'].value_counts()*100).plot.bar(color = 'blue')

# base to create a new feature: 4 or less companies? 


# In[ ]:


print('\Attrition rate by Salary Hike in %: \n')
print(round(main_df['PercentSalaryHike'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['PercentSalaryHike'].value_counts()*100,2))

(main_df['PercentSalaryHike'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['PercentSalaryHike'].value_counts()*100).plot.bar(color = 'blue')


# In[ ]:


print('Number of Employees by JobRole (in %): \n')
print(round(main_df['JobRole'].value_counts(normalize = True)*100,2).sort_index())

print('\nAttrition rate by JobRole: \n')
print(round(main_df['JobRole'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['JobRole'].value_counts()*100,2).sort_index())


# In[ ]:


print('Number of Employees by Job Satisfaction Level(in %): \n')
print(round(main_df['JobSatisfaction'].value_counts(normalize = True)*100,2).sort_index())

print('\nAttrition rate by Job Satisfaction Level: \n')
print(round(main_df['JobSatisfaction'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['JobSatisfaction'].value_counts()*100,2).sort_index())


# In[ ]:


print('Number of Employees by Work Life Balance Level(in %): \n')
print(round(main_df['WorkLifeBalance'].value_counts(normalize = True)*100,2).sort_index())

print('\nAttrition rate by Work Life Balance Level: \n')
print(round(main_df['WorkLifeBalance'][main_df['Attrition'] == 'Yes'].value_counts()/main_df['WorkLifeBalance'].value_counts()*100,2).sort_index())


# In[ ]:


fig = plt.figure(figsize=(15,12))
fig.subplots_adjust(top=0.85, wspace=0.3,hspace = 0.3)

ax1 = fig.add_subplot(2,2,1)
ax1.set_title("Share of job satisfaction of employees by work life balance")
sns.heatmap(pd.crosstab(main_df.JobSatisfaction, main_df.WorkLifeBalance, normalize = True),
            cmap="coolwarm", annot=True, cbar=False)


ax2 = fig.add_subplot(2,2,2)
ax2.set_title("Share of job satisfaction of employees by environment satisfaction balance")
sns.heatmap(pd.crosstab(main_df.JobSatisfaction, main_df.EnvironmentSatisfaction, normalize = True),
            cmap="coolwarm", annot=True, cbar=False)

ax2 = fig.add_subplot(2,2,3)
ax2.set_title("Share of job involvment of employees by job satisfaction")
sns.heatmap(pd.crosstab(main_df.JobInvolvement, main_df.JobSatisfaction, normalize = True),
            cmap="coolwarm", annot=True, cbar=False)

ax2 = fig.add_subplot(2,2,4)
ax2.set_title("Share of job satisfaction of employees by work life balance")
sns.heatmap(pd.crosstab(main_df.JobInvolvement, main_df.WorkLifeBalance, normalize = True),
            cmap="coolwarm", annot=True, cbar=False)


# ## 3. Data preprocessing

# ### 3.1 Dealing with missing values

# Let's see where do we have missing values (and how many):

# In[ ]:


print(main_df.isnull().sum())


# There are five columns with missing values, the number of missing values is relatively low.  I am going to replace all of them with median.

# In[ ]:


main_df['TotalWorkingYears'].fillna(main_df.groupby(['Age'])['TotalWorkingYears'].transform('median'), inplace=True)
main_df['NumCompaniesWorked'].fillna(main_df.groupby(['TotalWorkingYears'])['NumCompaniesWorked'].transform('median'), inplace=True)
main_df["EnvironmentSatisfaction"].fillna(main_df["EnvironmentSatisfaction"].median(), inplace=True)
main_df["JobSatisfaction"].fillna(main_df["JobSatisfaction"].median(), inplace=True)
main_df["WorkLifeBalance"].fillna(main_df["WorkLifeBalance"].median(), inplace=True)


# ### 3.2 Creating new variables

# As we have filled all the missing values for "TotalWorkingYears" and "NumCompaniesWorked" variables, we can now create a new feature:

# In[ ]:


main_df['Avg_time_in_company'] = main_df['TotalWorkingYears'] / (main_df['NumCompaniesWorked'] + 1)


# Now it's to time to use previously created variable 'avg_work_time' to build a new feature:

# In[ ]:


main_df['Overtime'] = (avg_work_time.astype('timedelta64[s]') / 3600)- main_df['StandardHours'] # adding average overtime in hours variable (float)


# In[ ]:


sns.pairplot(main_df[['Avg_time_in_company','Overtime','Attrition']],hue = 'Attrition', height = 4)


# ### 3.3 Removing usuless variables

# Many of the numerical variables we could treat as categorical ones. The number of unique values for each category can tell us more. Here, we need to remember that we have created few 'artificial' values when we were replacing missing values with median (in some cases median was not an integer).

# In[ ]:


unique_counts = pd.DataFrame.from_records([(col, main_df[col].nunique()) for col in main_df.columns],
                          columns=['Variable_Name', 'Num_Unique_Vals']).sort_values(by=['Num_Unique_Vals'])
print('Number of unique values' +':\n') 
print(unique_counts)


# The 'Over18' , 'StandardHours' and 'EmployeeCount' columns have just one unique values - they are not useful for purpose of this analysis anymore. The dataset contains many variables with just a few unique values - we will treat theme as either ordinal or nominal variables. The other variables we will bin into small groups and we will treat theme as ordinal ones.

# In[ ]:


# dropping unnecessary variables:

main_df.drop(['Over18', 'StandardHours', 'EmployeeCount'], axis = 1, inplace = True)


# ### 3.4 Correlation matrix

# Internet is full of correlation examples for continuous data (mostly Pearson's equation), ufortunatelly there are no many examples of associations between continuous-categorical or categorical-categorical features). The article https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9 and the 'dython' package are here to help.

# In[ ]:


# Functions required to calculate the correlation/strength-of-association of features (as a part of dython package)

import scipy.stats as ss
from collections import Counter
import math 
from scipy import stats


def convert(data, to):
    converted = None
    if to == 'array':
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == 'list':
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == 'dataframe':
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data),to))
    else:
        return converted
    
def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    :param x: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :param y: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :return: float
    """
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def correlation_ratio(categories, measurements):
    """
    Calculates the Correlation Ratio (sometimes marked by the greek letter Eta) for categorical-continuous association.
    Answers the question - given a continuous value of a measurement, is it possible to know which category is it
    associated with?
    Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement, and 1 means
    a category can be determined with absolute certainty.
    Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    categories : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    measurements : list / NumPy ndarray / Pandas Series
        A sequence of continuous measurements
    """
    categories = convert(categories, 'array')
    measurements = convert(measurements, 'array')
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

def associations(dataset, nominal_columns=None, mark_columns=False, theil_u=False, plot=True,
                          return_results = False, **kwargs):
    """
    Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
    continuous features using:
     - Pearson's R for continuous-continuous cases
     - Correlation Ratio for categorical-continuous cases
     - Cramer's V or Theil's U for categorical-categorical cases
    :param dataset: NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    :param nominal_columns: string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
        columns are categorical, or None (default) to state none are categorical
    :param mark_columns: Boolean (default: False)
        if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
        continuous), as provided by nominal_columns
    :param theil_u: Boolean (default: False)
        In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
    :param plot: Boolean (default: True)
        If True, plot a heat-map of the correlation matrix
    :param return_results: Boolean (default: False)
        If True, the function will return a Pandas DataFrame of the computed associations
    :param kwargs:
        Arguments to be passed to used function and methods
    :return: Pandas DataFrame
        A DataFrame of the correlation/strength-of-association between all features
    """

    dataset = convert(dataset, 'dataframe')
    columns = dataset.columns
    if nominal_columns is None:
        nominal_columns = list()
    elif nominal_columns == 'all':
        nominal_columns = columns
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0,len(columns)):
        for j in range(i,len(columns)):
            if i == j:
                corr[columns[i]][columns[j]] = 1.0
            else:
                if columns[i] in nominal_columns:
                    if columns[j] in nominal_columns:
                        if theil_u:
                            corr[columns[j]][columns[i]] = theils_u(dataset[columns[i]],dataset[columns[j]])
                            corr[columns[i]][columns[j]] = theils_u(dataset[columns[j]],dataset[columns[i]])
                        else:
                            cell = cramers_v(dataset[columns[i]],dataset[columns[j]])
                            corr[columns[i]][columns[j]] = cell
                            corr[columns[j]][columns[i]] = cell
                    else:
                        cell = correlation_ratio(dataset[columns[i]], dataset[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                else:
                    if columns[j] in nominal_columns:
                        cell = correlation_ratio(dataset[columns[j]], dataset[columns[i]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                    else:
                        cell, _ = ss.pearsonr(dataset[columns[i]], dataset[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = ['{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col) for col in columns]
        corr.columns = marked_columns
        corr.index = marked_columns
    if plot:
        plt.figure(figsize=(20,20))#kwargs.get('figsize',None))
        sns.heatmap(corr, annot=kwargs.get('annot',True), fmt=kwargs.get('fmt','.2f'), cmap='coolwarm')
        plt.show()
    if return_results:
        return corr


# In[ ]:


numcols = ['Age', 'NumCompaniesWorked','Avg_time_in_company','TotalWorkingYears','YearsAtCompany','YearsWithCurrManager','YearsSinceLastPromotion','DistanceFromHome','Education',
           'JobLevel','MonthlyIncome','PercentSalaryHike','Overtime','EnvironmentSatisfaction', 'JobSatisfaction','WorkLifeBalance','PerformanceRating','JobInvolvement']
catcols = ['BusinessTravel','TrainingTimesLastYear','Department','EducationField', 'Gender', 'JobRole', 'MaritalStatus','StockOptionLevel', 'Attrition']


# In[ ]:


main_df_corr = main_df[numcols + catcols]
results = associations(main_df_corr,nominal_columns=catcols,return_results=True)


# ### 3.5 Encoding variables

# In[ ]:


main_df['Overtime'] = np.where(main_df['Overtime']>0, 1, 0)


# In[ ]:


main_df['Attrition'] = main_df['Attrition'].map({'Yes': 1, 'No': 0})
main_df['Gender'] = main_df['Gender'].map({'Female': 1, 'Male': 0})
main_df['BusinessTravel'] = main_df['BusinessTravel'].map({'Travel_Frequently':2 , 'Travel_Rarely': 1, 'Non-Travel': 0})


# In[ ]:


# As we saw previously, the monthly income distribution is skewed to the right.
# That's why, we will use pd.qcut (quantile-based) function, to devide distribution into intervals

main_df['Income_band'] = pd.qcut(main_df['MonthlyIncome'], 4)

main_df[['Income_band', 'Attrition']].groupby(['Income_band'], as_index=False).mean().sort_values(by='Income_band', ascending=True)


# In[ ]:


Income_intervals = sorted(main_df['Income_band'].unique())
   
main_df.loc[main_df['MonthlyIncome'] <= Income_intervals[1].left, 'MonthlyIncome'] = 0
main_df.loc[(main_df['MonthlyIncome'] > Income_intervals[1].left) & (main_df['MonthlyIncome'] <= Income_intervals[1].right), 'MonthlyIncome'] = 1
main_df.loc[(main_df['MonthlyIncome'] > Income_intervals[2].left) & (main_df['MonthlyIncome'] <= Income_intervals[2].right), 'MonthlyIncome'] = 2
main_df.loc[ main_df['MonthlyIncome'] > Income_intervals[2].right, 'MonthlyIncome'] = 3


# In[ ]:


main_df.head()


# In[ ]:


main_df.drop('Income_band', axis = 1, inplace = True)


# In[ ]:


# I am going to use dummy encoding to have numerical representation of nominal variables:

dummy_cols = ['Department', 'MaritalStatus', 'EducationField', 'JobRole']
dummy_prefix = ['Dep','M_Stat', 'Edu_Field', 'J_Role']

df_with_dummies = pd.get_dummies(main_df, columns = dummy_cols, prefix = dummy_prefix, drop_first = True)


# In[ ]:


df_with_dummies.head(5)


# ### 4. Machine Learning

# In[ ]:


# importing necessary modules

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, confusion_matrix, roc_auc_score, roc_curve


# In[ ]:


# splitting data 

X = df_with_dummies.drop('Attrition', axis = 1) # independent features
y = df_with_dummies['Attrition'] # depentent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 44)


# In[ ]:


# creating model and fitting training data
logreg = LogisticRegression(solver='liblinear', random_state = 44)

logreg.fit(X_train,y_train)


# In[ ]:


# Obtain the predictions from our logistic regression model:
y_pred = logreg.predict(X_test)

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Print the ROC curve, classification report and confusion matrix
print("ROC AUC score:\n {} \n\n Classification report:\n{}\n\n Confusion matrix:\n {}".format(roc_auc_score(y_test, y_pred_prob),
                                                                                            classification_report(y_test, y_pred),
                                                                                              confusion_matrix(y_test, y_pred)))


# The model predicts 63 cases of attrition, out of which 35 employees actually have left the company. There are 28 false positives, what gives us pretty poor precision score = 0.56. Model also didn't catch 87 of employees who actually left the company, hence very low recall score = 0.29.  Let's try to improve that.

# In[ ]:


# ROC Curve

fpr, tpr, tresholds = roc_curve(y_test,y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[ ]:


# Calculate average precision and the PR curve
average_precision = average_precision_score(y_test, y_pred)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# --- plotting Precision-Recall curve -----
from inspect import signature


# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: avg precision={0:0.2f}'.format(average_precision))


# In[ ]:



# Setup the hyperparameter grid

c_log_space = np.logspace(-5, 8)
cl_weight = [{0:1,1:1},{0:1,1:1.5},{0:1,1:2},{0:1,1:2.5},{0:1,1:4},{0:1,1:6}, 'balanced']

param_grid = { 'C': c_log_space, 'class_weight': cl_weight}


# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(estimator = logreg, param_grid= param_grid, cv=5, scoring='roc_auc')

# Fit it to the data
logreg_cv.fit(X_train,y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))


# In[ ]:


# Obtain the predictions from our tuned model:
y_pred_cv = logreg_cv.predict(X_test)

# Compute predicted probabilities for tuned model: y_pred_prob_cv
y_pred_prob_cv = logreg_cv.predict_proba(X_test)[:,1]

# Print the ROC curve, classification report and confusion matrix for tuned model:
print("ROC AUC score:\n {} \n\n Classification report:\n{}\n\n Confusion matrix:\n {}".format(roc_auc_score(y_test, y_pred_prob_cv),
                                                                                            classification_report(y_test, y_pred_cv),
                                                                                              confusion_matrix(y_test, y_pred_cv)))


# The AUC score of tuned model is a bit higher than in the first attempt (0.8449 vs 0.8438), but the improvement is not satisfactory. Probably it's a good idea to check another tuning parameters for a model, check another alghoritms or, for example, create new 'artificial variables'.
# 
# The tuned model predicts 99 cases of attrition, out of which 50 employees actually have left the company. The 49 of false positives result in lower precision score (0.51 comparing to 0.56 in the first model). The tuned model didn't detect 'only' 72 of employees who actually left the company (first model - 87), what results in higher recall score = 0.41.
# 
