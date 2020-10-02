#!/usr/bin/env python
# coding: utf-8

# **Motivation:**
# 
# The goal of this notebook is to build my familiarty working with data using Pandas in Python. Specifically, I want to explore some of the 'College Scorecard' data published by the US department of education. In particular, I am interested in determining if some of the costs of higher education (tuition price, average faculty salary etc...) can be predicted based on types of data from the institutions. The notebook will be broken into three sections:
# 
# 1) Import and cleaning of the raw education data
# 
# 2) Analysis of the data and model development
# 
# 3) Model analysis and conclusions

# First lets set up our python environment.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from itertools import compress


# **Step 1: Import and clean the data**
# 
# Next we will load the raw data and do some initial data cleaning. The original dataset contains around 1900 descriptors for nearly 7k institutions. Due to the large size of the survey there are many missing values throughout the dataset. 
# 
# 1) We will first remove any institutions that do not supply tutiton data as this is one of the key values that we would like to predict. 
# 
# 2) Next we remove all columns that contain more then 10% of null values. 
# 
# 3) A large percentage of columns contain values that were suppressed due to privacy regulations. We will remove all columns that contain more then 10% of 'PrivacySuppressed' values and then in the remaining set change any instance of 'PrivacySuppressed' to the integer 0.
# 
# 4) Finally we will remove any remaining institutions that have null values within the cleaned set of columns.

# In[ ]:


df = pd.read_csv("../input/2015Ed.csv")

#First remove all rows that do not contain the target values of 'TUITIONFEE_OUT'
df = df[df['TUITIONFEE_OUT'].notna()]

#identify all columns that have above 10% null values
not_empty = []
for column in df.columns:
    if (df[column].isna().sum() / len(df[column])) <= 0.1:
        not_empty.append(column)
full = df[not_empty].copy()

#identify all columns that have 10% or below 'PrivacySuppressed' values
not_suppressed = []
for column in full.columns:
    if (full[column].apply(str).str.count('PrivacySuppressed').sum() / len(full[column])) <= 0.9:
        not_suppressed.append(column)
#create a new dataframe with those columns
cleaned = full[not_suppressed].copy()
#replace all values of 'PrivacySuppressed' to the number 0
cleaned = cleaned.replace(to_replace = 'PrivacySuppressed', value = 0)

#drop all rows that contain null values to create the final dataset
df = cleaned.dropna()


# In[ ]:


df.shape


# Our 'clean' dataset still contains over 3k institutions and 1261 features! Plenty of data to work with.

# In[ ]:


df.head()


# Lets define some functions that will assist us in our analysis. We will define a function linear_predict that takes a specific dataset and target value and trains a linear model. The function named epoch_test that will perform the modeling, prediction and create some graphical outputs displaying the quality of our model. Finally, a function named feature_compare_df will take a dictionary of feature sets and a list of target values and will perform linear predictions and return the outputs in a dataframe.

# In[ ]:


def linear_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    reg = linear_model.Ridge(alpha=.6)
    reg.fit(X_train,y_train)
    y_predict = reg.predict(X_test)
    return y_predict, y_test


# In[ ]:


def epoch_test(X, y, epochs = 100, print_output=False, graph_output = False):
    accuracy = []
    average = {}
    for i in range(epochs):
        y_predict, y_test = linear_predict(X,y)
        accuracy.append(np.sqrt(mean_squared_error(y_test,y_predict)))
        average[i] = np.mean(accuracy)
    if graph_output == True:
        plt.scatter(average.keys(),average.values())
        plt.title('Average RMSE over ' + str(epochs) + ' epochs')
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.show()
        plt.scatter(y_predict,y_test)
        plt.title("Predicted vs Actual " + y.name)
        plt.xlabel("Actual " + y.name)
        plt.ylabel("Predicted " + y.name)
        plt.show()
    if print_output == True:
        print('number of features: ' + str(len(X.columns)))
        print('number of examples: ' + str(len(X)))
        print('mean target value: ' + str(np.mean(y)))
        print('median target value: ' + str(np.median(y)))
        print('std. dev. of target value: ' + str(np.std(y)))
        print('final RMSE over ' + str(epochs) + ' epochs: ' + str(average[epochs-1]))
    return average[epochs-1]


# In[ ]:


def feature_compare_df(df, feature_list, target_list):
    output = pd.DataFrame()
    for target in target_list:
        comparison = {}
        for feature in feature_list.keys():
            comparison[str(feature)] = epoch_test(df[feature_list[feature]], df[target])
        comparison['Mean'] = np.mean(df[target])
        comparison['Median'] = np.median(df[target])
        comparison['StdDev'] = np.std(df[target])
        output = pd.concat([output, pd.DataFrame({str(target):pd.Series(comparison)})], axis = 1)
    return output


# Here is a list of the target financial features that we want to predict:
# 
# 1) The 4 year cost of attendence ('COSTT4_A')
# 
# 2) The out-of-state tuition fees per year ('TUITIONFEE_OUT')
# 
# 3) The net tutiton revenue per full time student ('TUITFTE')
# 
# 4) The instructional expenditures per full time student ('INEXPFTE')
# 
# 5) The average faculty salary ('AVGFACSAL')
# 
# 

# In[ ]:


target_features = ['COSTT4_A', 'TUITIONFEE_OUT', 'TUITFTE', 'INEXPFTE', 'AVGFACSAL']


# Let's take a look at the distribution of the different target values.

# In[11]:


figs, axs = plt.subplots(ncols=5, figsize=(20,5))
for ax in axs:
    ax.set_yticks([])
sns.set()
sns.distplot(df['TUITIONFEE_OUT'], ax=axs[0], color='r')
sns.distplot(df['COSTT4_A'], ax=axs[1], color='c')
sns.distplot(df['TUITFTE'], ax=axs[2], color='y')
sns.distplot(df['INEXPFTE'], ax=axs[3], color='g')
sns.distplot(df['AVGFACSAL'], ax=axs[4], color='m')


# From the plots we can see that the tution fees (TUITIONFEE_OUT and COSTT4_A) cover a broad distribution while the amount of money the institutions spend per student (TUITFTE, INEXPFTE and AVGFACSAL) are much more narrow.

# I am going to investigate how six different sets of features can be used to predict the financial targets.
# The six different sets are:
# 
# Social Factors:
#        
#     This includes the size and demographics of the student population, the average age of entry, the number that are first generation college attendees and the education level of their parents.
#        
# Family Income:
# 
#     These are features associated with income such as the median family income and the percent of students whose parents fall into different income tiers. This data also includes the percent of students that take out loans during attendance as this is a way to supplement income.
#     
# Debt Statistics:
#     
#     These features describe the average amount of debt that students take on when attending the school as well as repayment statstics.
#     
# Attendance and Completion:
# 
#     These featuers concern the enrollment, withdrawl and completion statistics for the institution.
#     
# Degree Types:
# 
#     This feature set describes the distribution of different types of degrees by percent of all degrees. 

# In[ ]:


social_factors = ['UGDS', 'UGDS_WHITE', 'UGDS_BLACK', 'UGDS_HISP', 'UGDS_ASIAN', 
            'UGDS_AIAN', 'UGDS_NHPI','UGDS_2MOR', 'UGDS_NRA', 'UGDS_UNKN', 'AGE_ENTRY', 'UGDS_MEN', 'UGDS_WOMEN'
           , 'FEMALE', 'MARRIED', 'DEPENDENT', 'VETERAN', 'FIRST_GEN', 'UG25ABV', 'PAR_ED_PCT_MS', 
            'PAR_ED_PCT_PS', 'PAR_ED_PCT_HS', 'PPTUG_EF', 'PFTFTUG1_EF', 'UGNONDS']


# In[ ]:


family_income = ['FAMINC', 'MD_FAMINC', 'FAMINC_IND', 'DEP_INC_AVG', 'IND_INC_AVG', 
                 'PCTPELL', 'PCTFLOAN', 'INC_PCT_LO','DEP_STAT_PCT_IND','DEP_INC_PCT_LO','IND_INC_PCT_LO',
                'INC_PCT_M1','INC_PCT_M2','INC_PCT_H1','INC_PCT_H2','DEP_INC_PCT_M1',
                 'DEP_INC_PCT_M2','DEP_INC_PCT_H1','DEP_INC_PCT_H2','IND_INC_PCT_M1','IND_INC_PCT_M2','IND_INC_PCT_H1',
                 'IND_INC_PCT_H2']


# In[ ]:


debt_statistics = []
keys = ['DEBT', 'RPY']
for name in df.columns:
    for key in keys:
        if key in name:
            debt_statistics.append(name)


# In[ ]:


attendance_statistics = []
keys = ['ENRL', 'COMP', 'WDRAW']
for name in df.columns:
    for key in keys:
        if key in name:
            attendance_statistics.append(name)


# In[ ]:


degree_types = list(df.columns[15:53])


# In[ ]:


total = social_factors + family_income + debt_statistics + attendance_statistics + degree_types


# Now lets create a dictionary called 'features' which will allow us to easily iterate over all of our feature sets.

# In[ ]:


features = {'Social Factors':social_factors, 'Family Income':family_income, 
            'Debt Statistics':debt_statistics, 'Attendance Statistics':attendance_statistics, 
            'Degree Types':degree_types}


# **Step 2: Data analysis and model development**
# 
# Now that we have prepared the data lets explore some of the relationships and predictive power of the different feature sets we have prepared.

# First, Lets take a look at the number of features in all of the sets.

# In[ ]:


for name, items in features.items():
    print('Length of ' + name + ' = ' + str(len(items)))


# Now let's do some analysis!
# We are going to attempt a linear fit of each feature set with each target value. We will be using ridge regression and also train the model for 100 epochs and average the RMSE over the training time to get an accurate view of the predicitive power.

# In[ ]:


analysis = feature_compare_df(df, features, target_features)
analysis


# Now lets look at the RMSE of our models after 100 epochs.

# In[ ]:


analysis


# Let's take a look at RMSE as a percentage of the standard deviation.

# In[ ]:


percent_analysis = analysis.copy()
for column in analysis.columns:
    percent_analysis[column] = percent_analysis[column] / percent_analysis.loc['StdDev'][column] * 100


# It seems that the Attendance Statistics are the strongest predictors of the target data (lowest RMSE). This could be due to this set containing a much larger number of features then the other sets (>600 compared to 23-133 for the others). 

# In[ ]:


percent_analysis[:5]


# Lets perform recursive feature elimination to get the large sets down to aprox. 25 features. We will use the target 'TUITIONFEE_OUT' for our RFE.

# In[ ]:


num_features = 25
rfe = RFE(linear_model.Ridge(alpha=.6), num_features)
rfe = rfe.fit(df[attendance_statistics], df['TUITIONFEE_OUT'])
list_a = attendance_statistics
fil = list(rfe.support_)
attendance_compressed = list(compress(list_a, fil))


# In[ ]:


num_features = 25
rfe = RFE(linear_model.Ridge(alpha=.6), num_features)
rfe = rfe.fit(df[debt_statistics], df['TUITIONFEE_OUT'])
list_a = debt_statistics
fil = list(rfe.support_)
debt_compressed = list(compress(list_a, fil))


# Now lets perform the same analysis but with the 'compressed' feature sets:

# In[ ]:


compressed_features = {'Social Factors':social_factors, 'Family Income':family_income, 
            'Debt Compressed':debt_compressed, 'Attendance Compressed':attendance_compressed, 
            'Degree Types':degree_types}


# In[ ]:


compressed_analysis = feature_compare_df(df, compressed_features, target_features)


# In[ ]:


compressed_analysis


# From this compressed feature sets we see that the attendance data is still the strongest predictor for the out-of-state tution fee, but this is not too suprising because we did optomize the features against this target value.

# In[ ]:


compressed_percent_analysis = compressed_analysis.copy()
for column in compressed_analysis.columns:
    compressed_percent_analysis[column] = compressed_percent_analysis[column] / compressed_percent_analysis.loc['StdDev'][column] * 100


# In[ ]:


compressed_percent_analysis[:5]


# Let's look at a plot of some of the predicted values vs actual values for the model that compares the compressed attendance features to the out-of-state tution fees.

# In[ ]:


epoch_test(df[attendance_compressed], df['TUITIONFEE_OUT'], graph_output = True)


# **Model analysis and conclusions**
# 
# From the initial analysis there are a few interesting things of note:
# 
# 1) From the current kernel, the social factors are the best predictor of the average faculty salary.
# 
# 2) Unsuprisingly, family income is one of the best preictors of the financial factors, except for facutly salary.
# 
# 3) The distribution of degree types seems to correlate reasonable well with the amount each institution spends on each student (INEXPFTE)
# 
# **Future Steps**
# 
# This initial investigation was successful in helping me to learn a lot about basic data manipluation and modeling in Python. For the next kernel I would like to try:
# 
# 1) Normalization of some of the features that take on large ranges of scalar values.
# 
# 2) A more detailed exploration of the features within the sets and their indiviudal contribution to the linear models as a whole.
# 
# **Feedback**
# 
# As I am just beginning my journey into data preparatin and analysis in Python I would love any constructive feedback on my code or approach. Thanks for taking the time to read my kernel!
