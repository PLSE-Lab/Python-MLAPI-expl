#!/usr/bin/env python
# coding: utf-8

# ## Objective: To demonstrate structured format of Performing Exploratory data Analysis.
# 

# ## Overview:
# 1. Introduction to problem statement
# 2. Hypothesis generation with respect to problem statement
# 3. Introduction to dataset
# 4. Importing dataset and first impressions
# 5. Variable Identification and Typecasting
# 6. Univariate Analysis : Numerical Variables
# 7. Univariate Analysis : Categorical Variables
# 8. Univariate Analysis : Missing Values
# 9. Univariate Analysis : Oulier Values
# 10. Summary of Univariate Analysis
# 
# Topics to be covered in succeeding Notebook
# * Bivariate Analysis
# * Multivariate Analysis
# * Summary of EDA

# ## 1. Introduction to problem statement:
# A Bank wants to take care of customer retention for their product; savings accounts. The bank wants you to identify customers likely to churn balances below the minimum balance. You have the customers information such as age, gender, demographics along with their transactions with the bank. Your task as a data scientist would be to predict the propensity to churn for each customer.

# ## 2. Hypothesis Generation for the problem statement:
# **Hypothesis generation is about preparing an exhaustive list of questions or possibilities which directly or indirectly affect the problem statement or the target variablw. It is a very important step as it prevents us from going down for a wild goose chase during EDA. It narrows down the process of performing EDA to the most essential aspects.**
# 
# **This step is performed before looking/gathering dataset**
# 
# To generate hypothesis, we require the following:
# 1. Common Sense or Rationality
# 2. Domain knowledge if possible
# 3. Communication with domain experts
# 
# Given below are the hypothesis we will be working with in this EDA
# 
# **On basis of Demographics**
# 1. Are females less likely to churn than males?
# 2. Are young customers more likely to churn?
# 3. Are customers in the lower income bracket more likely to churn?
# 4. Are customers with dependent(s) less likely to churn?
# 5. Customers with an average family size less than 4 are more likely to churn?
# 
# **On the basis of customer behaviour**
# 1. Are vintage customers less likely to churn?
# 2. Are customers with higher average balance less likely to churn?
# 3. Are customers dropping monthly balance highly likely to churn?
# 4. Are customers with no transaction is the last 3 months more likely to churn?
# 5. Are customers who have large withdrawal amounts in the last month more likely to churn?
# 6. Are customers who have large withdrawal amounts in the last quarter more likely to churn?
# 7. Customers who have not engaged with the bank in the last quarter are more likely to churn?

# ## 3. Introduction to Dataset
# 
# There are multiple variables in the dataset which can be cleanly divided in 3 categories:
# 
# ### Demographic information about customers
# 
# <b>customer_id</b> - Customer id
# 
# <b>vintage</b> - Vintage of the customer with the bank in number of days
# 
# <b>age</b> - Age of customer
# 
# <b>gender</b> - Gender of customer
# 
# <b>dependents</b> - Number of dependents
# 
# <b>occupation</b> - Occupation of the customer 
# 
# <b>city</b> - City of customer (anonymised)
# 
# 
# ### Bank Related Information for customers
# 
# 
# <b>customer_nw_category</b> - Net worth of customer (3:Low 2:Medium 1:High)
# 
# <b>branch_code</b> - Branch Code for customer account
# 
# <b>days_since_last_transaction</b> - No of Days Since Last Credit in Last 1 year
# 
# 
# ### Transactional Information
# 
# <b>current_balance</b> - Balance as of today
# 
# <b>previous_month_end_balance</b> - End of Month Balance of previous month
# 
# 
# <b>average_monthly_balance_prevQ</b> - Average monthly balances (AMB) in Previous Quarter
# 
# <b>average_monthly_balance_prevQ2</b> - Average monthly balances (AMB) in previous to previous quarter
# 
# <b>percent_change_credits</b> - Percent Change in Credits between last 2 quarters
# 
# <b>current_month_credit</b> - Total Credit Amount current month
# 
# <b>previous_month_credit</b> - Total Credit Amount previous month
# 
# <b>current_month_debit</b> - Total Debit Amount current month
# 
# <b>previous_month_debit</b> - Total Debit Amount previous month
# 
# <b>current_month_balance</b> - Average Balance of current month
# 
# <b>previous_month_balance</b> - Average Balance of previous month
# 
# <b>churn</b> - Average balance of customer falls below minimum balance in the next quarter (1/0)

# ## 4. Reading Files into Python And first Impressions

# In[ ]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings(action = 'ignore')


# In[ ]:


#importing data
data = pd.read_csv('../input/banking-churn-prediction/Banking_churn_prediction.csv')


# In[ ]:


#first 5 instances using "head()" function
data.head()


# In[ ]:


#last 5 instances using "tail()" function
data.tail()


# **This veriffies that the data was imported successfully (no abnormal absurd entries).**

# In[ ]:


#finding out the shape of the data using "shape" variable: Output (rows, columns)
data.shape


# **Data is fairly small with 28K rows observations, although the number of columns are 21**

# In[ ]:


#Printing all the columns present in data
data.columns


# **This verifies that the all the variables are present as claimed in the data dictionary**

# ## 5. Variable Identification and Typecasting
# 
# **This is one of the most important steps, Why?**
# 
# **Because pandas is not very good when it comes to recognising the datatype of theimported variables. So in this section, we will be analysing the datatypes of each variables and converting them to respective types.**

# In[ ]:


# A closer look at the data types present in the data
data.dtypes


# There are a lot of variables visible at one, so let's narrow this down by looking **at one datatype at once**. We will start with **int**
# 

# ### Integer Data Type

# In[ ]:


# Identifying variables with integer datatype
data.dtypes[data.dtypes == 'int64']


# Summary:
# 
# *    **Customer id** are a unique number assigned to customers. It is **Okay as Integer**.
# 
# *    **branch code** again represents different branches, therefore it should be **convereted to category**.
# 
# *    **Age** and **Vintage** are also numbers and hence we are okay with them as integers.
# 
# *    **customer_networth_category** is supposed to be an category, **should be converted to category**.
# 
# *    **churn** : 1 represents the churn and 0 represents not churn. However, there is no comparison between these two categories. This **needs to be converted to category datatype**.
# 

# In[ ]:


# converting churn to category
data['churn'] = data['churn'].astype('category')
data['branch_code'] = data['branch_code'].astype('category')
data['customer_nw_category'] = data['customer_nw_category'].astype('category')
data.dtypes[data.dtypes == 'int64']


# **The previous output verifies our conversion**

# ### Float Data Type

# In[ ]:


# Identifying variables with float datatype
data.dtypes[data.dtypes == 'float64']


# Summary:
# 
# *    **dependents** is expected to be a whole number. **Should be changed to integer type**
# 
# *    **city** variable is also a unique code of a city represented by some interger number. **Should be converted to Category type**
# 
# *    Rest of the variables like **credit, balance and debit** are best represented by the float variables.

# In[ ]:


# converting "dependents" and "city" to their respective types
data['dependents'] = data['dependents'].astype('Int64')
data['city'] = data['city'].astype('category')

# checking
data[['dependents','city']].dtypes


# **Previous output cell verifies the change in data types of dependents and city variables.**

# ### Object Data Type

# In[ ]:


data.dtypes


# *    **variables like 'gender', 'occupation' and 'last_transaction' are of type object**. This means that **Pandas was not able to recognise the datatype** of these three variables.

# In[ ]:


# Manually checking object types
data[['gender','occupation','last_transaction']].head(7)


# *    **gender** and **occupation** variables **belong to categorical data types**.
# *    **last_transaction** should be a  **datetime variable**.

# In[ ]:


# typecasting "gender" and "occupation" to category type
data['gender'] = data['gender'].astype('category')
data['occupation'] = data['occupation'].astype('category')

# checking
data[['gender','occupation']].dtypes


# **Last cell verifies that gender and occupation variables have been converted successfully**

# ### datetime Data Type

# In[ ]:


# creating an instance(date) of DatetimeIndex class using "last_transaction"
date = pd.DatetimeIndex(data['last_transaction'])


# In[ ]:


# extracting new columns from "last_transaction"

# last day of year when transaction was done
data['doy_ls_tran'] = date.dayofyear

# week of year when last transaction was done
data['woy_ls_tran'] = date.weekofyear

# month of year when last transaction was done
data['moy_ls_tran'] = date.month

# day of week when last transaction was done
data['dow_ls_tran'] = date.dayofweek


# In[ ]:


# checking new extracted columns using datetime
data[['last_transaction','doy_ls_tran','woy_ls_tran','moy_ls_tran','dow_ls_tran']].head()


# The first column is the complete date of the last transaction which was done by any given customer.
# 
# The next columns represent the day of year, week of year, month of year, day of week when the last transaction was done.
# 
# **Breaking down the date variable** into these granular information will **help us in understand when the last transaction was done from different perspectives**. Now that we have extracted the essentials from the last_transaction variables, we will drop it from the dataset.
# 
# 

# In[ ]:


# Removing the original datetime column
data = data.drop(columns = ['last_transaction'])
data.dtypes


# **So we can finally see that all the variables are now assigned their respective datatypes.**

# ## 6. Univariate Analysis: Numerical Variables
# When dealing with numerical variables, we have to check their properties like:
# * Mean 
# * Median 
# * Standard Deviation 
# * Kurtosis/skewness
# * distribution/range

# In[ ]:


# Numerical datatypes
data.select_dtypes(include=['int64','float64','Int64']).dtypes


# Now considering that we have 18 numerical variables and 5 properties associated with each, Performing univariate analysis can be tiresome. For this reasonn, it is always wise to form a cluster/group of variables which are similar to each other in nature. The variables can be grouped in many different ways.
# 
# In this EDA I am grouping variable into 5 groups:
# * customer_etails
# * current_month
# * previous_month
# * previous_quarters
# * transaction_date

# In[ ]:


# seggregating variables into groups
customer_details = ['customer_id','age','vintage']
current_month = ['current_balance','current_month_credit','current_month_debit','current_month_balance']
previous_month = ['previous_month_end_balance','previous_month_credit','previous_month_debit','previous_month_balance']
previous_quarters = ['average_monthly_balance_prevQ','average_monthly_balance_prevQ2']
transaction_date = ['doy_ls_tran','woy_ls_tran','moy_ls_tran','dow_ls_tran']


# **Now to visualise the variable groups at once with all the necessary descriptives, let's define a universal/reusable function to do that.**

# In[ ]:


# custom function for easy and efficient analysis of numerical univariate

def UVA_numeric(data, var_group):
  '''
  Univariate_Analysis_numeric
  takes a group of variables (INTEGER and FLOAT) and plot/print all the descriptives and properties along with KDE.

  Runs a loop: calculate all the descriptives of i(th) variable and plot/print it
  '''

  size = len(var_group)
  plt.figure(figsize = (7*size,3), dpi = 100)
  
  #looping for each variable
  for j,i in enumerate(var_group):
    
    # calculating descriptives of variable
    mini = data[i].min()
    maxi = data[i].max()
    ran = data[i].max()-data[i].min()
    mean = data[i].mean()
    median = data[i].median()
    st_dev = data[i].std()
    skew = data[i].skew()
    kurt = data[i].kurtosis()

    # calculating points of standard deviation
    points = mean-st_dev, mean+st_dev

    #Plotting the variable with every information
    plt.subplot(1,size,j+1)
    sns.kdeplot(data[i], shade=True)
    sns.lineplot(points, [0,0], color = 'black', label = "std_dev")
    sns.scatterplot([mini,maxi], [0,0], color = 'orange', label = "min/max")
    sns.scatterplot([mean], [0], color = 'red', label = "mean")
    sns.scatterplot([median], [0], color = 'blue', label = "median")
    plt.xlabel('{}'.format(i), fontsize = 20)
    plt.ylabel('density')
    plt.title('std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format((round(points[0],2),round(points[1],2)),
                                                                                                   round(kurt,2),
                                                                                                   round(skew,2),
                                                                                                   (round(mini,2),round(maxi,2),round(ran,2)),
                                                                                                   round(mean,2),
                                                                                                   round(median,2)))


# **Now that the stage is set, let's perform the univariate analysis and learn what we can.**

# ### customer_information

# In[ ]:


UVA_numeric(data,customer_details)


# **Summary of Customer_Information:**
# 
# *    **customer_id**:
#      *    variable is **unique for every customer, Hence uniform** distribution.
#      * This variable **does not contribute any information**
#      * Can be eliminated from data
# 
# 
# *    **age**:
#     *    Median Age = 46
#     *    **Most customers age between 30 to 66**
#     *    skewness +0.33 : customer age is **negligibly biased towards younger age**
#     *    **kurtosis = -0.17**; very less likely to have extreme/outlier values.
# 
# 
# *    **vintage:**
#     *    Most customers joined between 2100 and 2650 days from the day of data extraction.
#     *    **skewness** -1.42 : this is left skewed, **vintage variable is significantly biased towards longer association of customers.**
#     *    **Kurtosis = 2.93**: Extreme values and Outliers are very likely to be present in vintage.
#     *    Most of the customers are old, did rate of new customers decayed over time?
# 
# 
# 
# **Things to Investigate Further down the road:**
# *    The batch of **high number of very Old Age customers** in age variable.
# 

# ### current_month

# In[ ]:


UVA_numeric(data,current_month)


# **Summary**
# *    Considering the kurtosis and skewness value  for all 4 of these plots. Outliers/Extreme values are obvious.

# 
# **Need to Remove Outliers to visualise these plots**
# AS the bult distribution is not visible, let's trim down the outliers using the rule of standard deviation

# In[ ]:


# standard deviation factor
factor = 2

# copying current_month
cm_data = data[current_month]

# filtering using standard deviation (not considering obseravtions > mean + 3* standard deviation)
cm_data = cm_data[cm_data['current_balance'] < cm_data['current_balance'].mean() + factor*cm_data['current_balance'].std()]
cm_data = cm_data[cm_data['current_month_credit'] < cm_data['current_month_credit'].mean() + factor*cm_data['current_month_credit'].std()]
cm_data = cm_data[cm_data['current_month_debit'] < cm_data['current_month_debit'].mean() + factor*cm_data['current_month_debit'].std()]
cm_data = cm_data[cm_data['current_month_balance'] < cm_data['current_month_balance'].mean() + factor*cm_data['current_month_balance'].std()]

# checking how many points removed
len(data), len(cm_data)


# In[ ]:


UVA_numeric(cm_data,current_month)


# **Summary of current_month**
# *    After Removing extreme/outliers, plots are still very skewed.
# 
# **Things to investigate further down**
# 1.    **Is there thete any common trait/relation between the customers who are performing high transaction credit/debits?**
# 2.    **Customers who are performinng high amount of transactions, are they doinng it every month?**

# ### previous_month

# In[ ]:


UVA_numeric(data,previous_month)


# **Summary of previous_month**
# *    This looks very similar to current_month. Most of the customers perform low amount transactions.

# ### previous_quarters

# In[ ]:


UVA_numeric(data,previous_quarters)


# **Summary**
# The general trend still follows, it is crutial that we find the out if there is any common trait between the customers doing high high amount of transactions.

# ### transaction_date

# In[ ]:


UVA_numeric(data,transaction_date)


# **Summary**
# *    **Day_of_Year**:
#     *    most of the last transactions were made in the last 60 days from the extraction of data.
#     *    There are transactions which were made also an year ago.
# 
# *   **Week_of_year and Month_of_year**: these variable validate the findings from the **day_of_year**.
# *    **Day_of_Week**: Tuesdays are often the favoured day relative to others.
# 
# **Things to investigate further Down**
# *    **Customers whose last transaction was 6 months ago, did all of them churn?**

# ## 7. Univariate Analysis : Categorical Variables

# In[ ]:


data.select_dtypes(exclude=['int64','float64','Int64']).dtypes


# **Grouping Varibales**
# 
# Similar to what we did in the numerical section, we will form group of variables for the categorical variables. Moreover we will define a reusable function to visualise things.
# 
# * **customer_info**: gender, occupation, customer_nw_category
# * **account_info**: city, branch_code
# * **churn**

# In[ ]:


# Custom function for easy visualisation of Categorical Variables
def UVA_category(data, var_group):

  '''
  Univariate_Analysis_categorical
  takes a group of variables (category) and plot/print all the value_counts and barplot.
  '''
  # setting figure_size
  size = len(var_group)
  plt.figure(figsize = (7*size,5), dpi = 100)

  # for every variable
  for j,i in enumerate(var_group):
    norm_count = data[i].value_counts(normalize = True)
    n_uni = data[i].nunique()

  #Plotting the variable with every information
    plt.subplot(1,size,j+1)
    sns.barplot(norm_count, norm_count.index , order = norm_count.index)
    plt.xlabel('fraction/percent', fontsize = 20)
    plt.ylabel('{}'.format(i), fontsize = 20)
    plt.title('n_uniques = {} \n value counts \n {};'.format(n_uni,norm_count))


# ### customer_info

# In[ ]:


UVA_category(data, ['occupation', 'gender', 'customer_nw_category'])


# **Summary**
# * Occupation
#   * Majority of people are self_employed.
#   * There are extremely few Company Accounts. Might explain Outlier/Extreme values in credit/debit.
# 
# * Gender:
#   *  Males accounts are 1.5 times more than Female Accounts.
# 
# * customer_nw_category:
#   *  Half of all the accounts belong to the 3rd net worth category.
#   *  Less than 15% belong to the highest net worth category.
# 
# **Things to investigate further down:**
# * Possibility: Company accounts are the reason behind the outlier transactions.
# * Possibility: customers belonging to the highest net worth category may explain the skewness of the transactions.

# ### account_info

# In[ ]:


UVA_category(data, ['city', 'branch_code'])


# In[ ]:


#Plotting "city"
plt.figure(figsize = (5,5), dpi = 120)
city_count = data['city'].value_counts(normalize=True)
sns.barplot(city_count.index, city_count , order = city_count.index)
plt.xlabel('City')
plt.ylabel('fraction/percent')
plt.ylim(0,0.02)


# In[ ]:


#Plotting "branch_code"
plt.figure(figsize = (5,5), dpi = 120)
branch_count = data['branch_code'].value_counts()
sns.barplot(branch_count.index, branch_count , order = branch_count.index)
plt.xlabel('branch_code')
plt.ylabel('fraction/percent')
#plt.ylim(0,0.02)


# **Summary:**
# for both variable "city" and "branch_code", there are too many categories. There is clear relation that some branches and cities are more popular with customers and and this trend decreases rapidly.
# 
# **Things to investigate further Down**
# * Popular cities and branch code might be able to explain the skewness and outliers of credit/debit variables.
# * Possibility that cities and branch code with very few accounts may lead to churning.

# ### churn

# In[ ]:


UVA_category(data, ['churn'])


# **Summary**
# * Number of people who churned are 1/4 times of the people who did not churn in the given data.

# ## 8. Univariate: Missing Values
# **Missing values could be due to several reasons.**
# 
# Human error
# 
# Privacy issues etc

# In[ ]:


# finding number of missing values in every variable
data.isnull().sum()


# **Things to investigate further down:**
# * Gender: Do the customers with missing gender values have some common behaviour in-
#   * churn: do missing values have any relation with churn?
# 
# 
# 
# * Dependents:
#  * Missing values might be similar to zero dependents (people with 0 dependents might have left it blank)
#  * churn: do missing values have any relation with churn?
# 
# 
# 
# * Occupation:
#  * Do missing values have similar behaviour among themselves(unemployed?) or to any other occupation?
#  * do they have some relation with churn?
# 
# 
# 
# * city:
#   * the respective cities can be found using branch_code
# 
# 
# 
# * last_transaction:
#   * checking their previous month and current month and previous_quarter activity might give insight on their last transaction.

# ## 9. Univariate Analysis: Outliers

# **We suspected outliers in current_month and previous_month variable groups. We will verify that using box plots**

# In[ ]:


# custom function for easy outlier analysis

def UVA_outlier(data, var_group, include_outlier = True):
  '''
  Univariate_Analysis_outlier:
  takes a group of variables (INTEGER and FLOAT) and plot/print boplot and descriptives\n
  Runs a loop: calculate all the descriptives of i(th) variable and plot/print it \n\n

  data : dataframe from which to plot from\n
  var_group : {list} type Group of Continuous variables\n
  include_outlier : {bool} whether to include outliers or not, default = True\n
  '''

  size = len(var_group)
  plt.figure(figsize = (7*size,4), dpi = 100)
  
  #looping for each variable
  for j,i in enumerate(var_group):
    
    # calculating descriptives of variable
    quant25 = data[i].quantile(0.25)
    quant75 = data[i].quantile(0.75)
    IQR = quant75 - quant25
    med = data[i].median()
    whis_low = quant25-(1.5*IQR)
    whis_high = quant75+(1.5*IQR)

    # Calculating Number of Outliers
    outlier_high = len(data[i][data[i]>whis_high])
    outlier_low = len(data[i][data[i]<whis_low])

    if include_outlier == True:
      #Plotting the variable with every information
      plt.subplot(1,size,j+1)
      sns.boxplot(data[i], orient="v")
      plt.ylabel('{}'.format(i))
      plt.title('With Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                                                                                                   round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2)),
                                                                                                   (outlier_low,outlier_high)
                                                                                                   ))
      
    else:
      # replacing outliers with max/min whisker
      data2 = data[var_group][:]
      data2[i][data2[i]>whis_high] = whis_high+1
      data2[i][data2[i]<whis_low] = whis_low-1
      
      # plotting without outliers
      plt.subplot(1,size,j+1)
      sns.boxplot(data2[i], orient="v")
      plt.ylabel('{}'.format(i))
      plt.title('Without Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                                                                                                   round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2)),
                                                                                                   (outlier_low,outlier_high)
                                                                                                   ))


# ### current_month and previous_month

# In[ ]:


UVA_outlier(data, current_month,)


# In[ ]:


UVA_outlier(data, current_month, include_outlier=False)


# In[ ]:


UVA_outlier(data, previous_month)


# In[ ]:


UVA_outlier(data, previous_month, include_outlier=False)


# **Summary:**
# * If we look at corresponding plots in the outputs above, there seems to be a strong relation between the corresponding plots of previous_month and current_month variables.
# 
# * Outliers are significant in number and very similar in number between corresponding plots. Which indicates some inherent undiscovered behviour of Outliers.

# ### previous quarters

# In[ ]:


UVA_outlier(data,previous_quarters)


# In[ ]:


UVA_outlier(data,previous_quarters, include_outlier = False)


# Summary:
# * Outliers in previous two quarters are significantly large but very similar in number.

# ## 10. Summary of Univariate Analysis:
# 
# ### Investigation directions (for bivariate/multivariate)
# 1. customer_id variable can be dropped.
# 2.  Is there there any common trait/relation between the customers who are performing high transaction credit/debits?
#    * customer_nw_category might explain that.
#    * Occupation = Company might explain them
#    * popular cities might explain this
# 4.  Customers whose last transaction was 6 months ago, did all of them churn? 
# 5. Possibility that cities and branch code with very few accounts may lead to churning.
# 
# 
# ### Some Insights
# 1. Most of the customers lie in the Age between 30-66, but there is also significant customers who are very old(age>85)
# 2. Major bulk of the customers opened their account more than 4 years ago! (did customer signups detiorated in recent times?)
# 3. Major bulk of customers did their last transaction within last 100 days.
# 4. Majority of customers perform small scale transactions. But there are few who perform transactions of huge amounts, consistently.
# 

# 
# The succeeding notebook can be found using the following link:
#     <https://www.kaggle.com/lonewolf95/eda-102-bivariate-analysis-hypothesis-testing>

# In[ ]:




