#!/usr/bin/env python
# coding: utf-8

# ## Objective: To demonstrate structured format of Performing Exploratory data Analysis.
# 
# Note that this notebook only contains Bivariate analysis. the preceeding Univaiate analysis can be found in a separate notebook, I highly recommend that you go through that notebook first.
# 
# 
# https://www.kaggle.com/lonewolf95/eda-101-structured-univariate-analysis
# 

# ## Overview:
# 
# 1. Recapping the problem statement
# 2. Recapping generated hypothesis
# 3. Recapping investigation leads from univariate analysis
# 4. Importing dataset + variabe typecasting
# 5. Bivariate Analysis : Numerical Numerical
# 6. Bivariate Analysis : Numerical Categorical
# 7. Bivariate Analysis : Categorical Categorical
# 8. Summary of Bivariate analysis  

# ## 1. Recapping problem statement:
# A Bank wants to take care of customer retention for their product; savings accounts. The bank wants you to identify customers likely to churn balances below the minimum balance. You have the customers information such as age, gender, demographics along with their transactions with the bank. Your task as a data scientist would be to predict the propensity to churn for each customer.

# ## 2. Recapping generated hypothesis:
# During the univariate anlysis we saw that hypothesis testing was not possible because hypothesis testing primarily deals with the combination of some independent variable with the target variable. As we are not diving into bivariate analysis, **we will be performing hypothesis testing extensively.**
# 
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

# ## 3. Investigation leads from Univariate analysis:
# 
# 1. Is there there any common trait/relation between the customers who are performing high transaction credit/debits?
#     * customer_nw_category might explain that.
#     * Occupation = Company might explain them
#     * popular cities might explain this
# 2. Customers whose last transaction was 6 months ago, did all of them churn?
# 3. Possibility that cities and branch code with very few accounts may lead to churning.
# 
# 

# ## 4. Importing libraries + Datset + Variable identification and typecasting.

# In[ ]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action = 'ignore')

#importing data
data = pd.read_csv('../input/banking-churn-prediction/Banking_churn_prediction.csv')

# converting churn/brnch_code/customer_nw_category to category type
data['churn'] = data['churn'].astype('category')
data['branch_code'] = data['branch_code'].astype('category')
data['customer_nw_category'] = data['customer_nw_category'].astype('category')
data.dtypes[data.dtypes == 'int64']

# converting "dependents" and "city" to their respective types
data['dependents'] = data['dependents'].astype('Int64')
data['city'] = data['city'].astype('category')

# typecasting "gender" and "occupation" to category type
data['gender'] = data['gender'].astype('category')
data['occupation'] = data['occupation'].astype('category')

# creating an instance(date) of DatetimeIndex class using "last_transaction"
date = pd.DatetimeIndex(data['last_transaction'])

##### extracting new columns from "last_transaction"

# last day of year when transaction was done
data['doy_ls_tran'] = date.dayofyear

# week of year when last transaction was done
data['woy_ls_tran'] = date.weekofyear

# month of year when last transaction was done
data['moy_ls_tran'] = date.month

# day of week when last transaction was done
data['dow_ls_tran'] = date.dayofweek

# Removing the original datetime column
data = data.drop(columns = ['last_transaction'])


#dropping customer_id
data = data.drop(columns=['customer_id'])

#checking
data.dtypes


# In[ ]:


# seggregating variables into groups
customer_details = ['customer_id','age','vintage']
current_month = ['current_balance','current_month_credit','current_month_debit','current_month_balance']
previous_month = ['previous_month_end_balance','previous_month_credit','previous_month_debit','previous_month_balance']
previous_quarters = ['average_monthly_balance_prevQ','average_monthly_balance_prevQ2']
transaction_date = ['doy_ls_tran','woy_ls_tran','moy_ls_tran','dow_ls_tran']


# ## 5. Bivariate Analysis : Numerical-Numerical
# In this section we will be performing bivariate analysis for the Numerical Numerical combination of variables.
# 
# **Although we do not have have any hypothesis which falls under this combination of variables, but we will still perform the numerical numerical bivariate analysis and relation between the independent variables can be used during the preprocessing and feature engineering.**

# In[ ]:


# isolating numerical datatypes
numerical = data.select_dtypes(include=['int64','float64','Int64'])[:]
numerical.dtypes


# ### Correlation Matrix
# A straight forward goto method is to print the correlation matrix.

# In[ ]:


# calculating correlation
correlation = numerical.dropna().corr()
correlation


# **As number of variables are too large, correlation matrix is not much help.**

# ### Heatmap
# Heatmap will allow us to visually figure out the key correlation between variables and filter the down the essential variables so that we will have lesss to deal with during the scatter plots.
# 
# In order to have different perspectives on the correlation of the independent variables, we will be plotting the heatmaps using three methods of calculating the correlation.
# 
# 1. Pearson Correlation
# 2. Kendal's Tau
# 3. Spearman Correlation

# In[ ]:


# plotting heatmap usingl all methods for all numerical variables (peason, kendall, spearman)
plt.figure(figsize=(36,6), dpi=140)
for j,i in enumerate(['pearson','kendall','spearman']):
  plt.subplot(1,3,j+1)
  correlation = numerical.dropna().corr(method=i)
  sns.heatmap(correlation, linewidth = 2)
  plt.title(i, fontsize=18)


# * Kendall and Spearman correlation seem to have very similar pattern between them, except the slight variation in magnitude of correlation.
# *  Too many variables with insignificant correlation.
# *  Major correlation lies between the transaction variables and balance variables.
# 
# **As the there are are many variables with insignificant correlation, let's filter down to the most important ones.**

# In[ ]:


# extracting transaction information of current and previous months
var = []
var.extend(previous_month)
var.extend(current_month)
var.extend(previous_quarters)


# In[ ]:


# plotting heatmap usill all methods for all transaction variables
plt.figure(figsize=(36,6), dpi=140)
for j,i in enumerate(['pearson','kendall','spearman']):
  plt.subplot(1,3,j+1)
  correlation = numerical[var].dropna().corr(method=i)
  sns.heatmap(correlation, linewidth = 2)
  plt.title(i, fontsize=18)


# **Inferences:**
# 
# 
# 1.   Transaction variables like credit/debit have a strong correlation among themselves.
# 2.  Balance variables have strong correlation among themselves.
# 3.   Transaction variables like credit/debit have insignificant or no correlation with the Balance variables.
# 
# 

# ### Scatterplot
# **Now that we have a bird's eye view of the correlations, let's look over them closely with the help of scatter plots.**

# In[ ]:


# Grouping variables
transactions = ['current_month_credit','current_month_debit','previous_month_credit','previous_month_debit']
balance = ['previous_month_end_balance','previous_month_balance','current_balance','current_month_balance']


# In[ ]:


# scatter plot for transactional variables
plt.figure(dpi=140)
sns.pairplot(numerical[transactions])
plt.show()


# **the scatter plot is is not meaningful due to the presence of outliers**
# One way to visualise them is to take logarithm transform of every variable is to nullify the effect of outliers.

# In[ ]:


#taking log of every value to negate outliers
for column in var:
  mini=1
  if numerical[column].min()<0:
    mini =  abs(numerical[column].min()) + 1
  
  numerical[column] = [i+mini for i in numerical[column]]
  numerical[column] = numerical[column].map(lambda x : np.log(x))


# In[ ]:


# scatter plot for transactional variables
plt.figure(dpi=140)
sns.pairplot(numerical[transactions])
plt.show()


# **Inferences**
# 1.    This validates the high correlation between the transaction variables.
# 2.    This high correlation can be used for feature engineering during the later stages.

# In[ ]:


# balance variables
plt.figure(dpi=140, figsize = (20,20))
sns.pairplot(numerical[balance])
plt.show()


# **Inferences**
# 1.    This validates the high correlation between the balance variables.
# 2.    This high correlation can be used for feature engineering during the later stages.

# In[ ]:


# previous quarters
plt.figure(dpi=140)
sns.scatterplot(numerical['average_monthly_balance_prevQ'], numerical['average_monthly_balance_prevQ2'])
plt.show()


# **Inferences**
# 1.    This validates the high correlation between the two previous quarters
# 2.    This high correlation can be used for feature engineering during the later stages.
# 
# 
# **Key Insight**
# 
# We can generate dozens of new features from these highly correlated variables during the feature engineering phase, which should be able to explain the presence of outliers and may contribute to better model performance.

# ## 6. Bivariate Analysis: Continuous-Categorical u to plot the categorical mean and the categorical distribution.
# Moreover, in this section we will working with hypothesis testing. I you need a quick refresher on hypothesis testing and how p-value works, just follow along the this article.
# 
# https://medium.com/analytics-vihttps://medium.com/analytics-vidhya/everything-you-should-know-about-p-value-from-scratch-for-data-science-f3c0bfa3c4cc

# List of Hypothesis and investigation to perform under this combination.
# 
# 1.  Are vintage customers less likely to churn?for large number of observations.
# 2.  Are customers with higher average balance less likely to churn?
# 3.  Are customers dropping monthly balance highly likely to churn?
# 
# We will be performing the hypothesis testing as we go along plotting the graphs. This will save a lot of time in the long run. For this we will be making three functions.
# 1. Function for 2sample Z-Test
# 2. Function for 2 sample T-Test
# 3. Function for plotting which uses the above mentioned two functions.
# 
# Note that I am using both Z-test and T-test here to quantify that they perform similarly.

# In[ ]:


def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
  '''
  takes mean, standard deviation, and number of observations and returns p-value calculated for 2-sampled Z-Test
  '''
  from numpy import sqrt, abs, round
  from scipy.stats import norm
  ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
  z = (X1 - X2)/ovr_sigma
  pval = 2*(1 - norm.cdf(abs(z)))
  return pval


# In[ ]:


def TwoSampT(X1, X2, sd1, sd2, n1, n2):
  '''
  takes mean, standard deviation, and number of observations and returns p-value calculated for 2-sample T-Test
  '''
  from numpy import sqrt, abs, round
  from scipy.stats import t as t_dist
  ovr_sd = sqrt(sd1**2/n1 + sd2**2/n2)
  t = (X1 - X2)/ovr_sd
  df = n1+n2-2
  pval = 2*(1 - t_dist.cdf(abs(t),df))
  return pval


# In[ ]:


def Bivariate_cont_cat(data, cont, cat, category):
  #creating 2 samples
  x1 = data[cont][data[cat]==category][:]
  x2 = data[cont][~(data[cat]==category)][:]
  
  #calculating descriptives
  n1, n2 = x1.shape[0], x2.shape[0]
  m1, m2 = x1.mean(), x2.mean()
  std1, std2 = x1.std(), x2.mean()
  
  #calculating p-values
  t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
  z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

  #table
  table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc = np.mean)

  #plotting
  plt.figure(figsize = (20,4), dpi=140)
  
  #barplot
  plt.subplot(1,3,1)
  sns.barplot([str(category),'not {}'.format(category)], [m1, m2])
  plt.ylabel('mean {}'.format(cont))
  plt.xlabel(cat)
  plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                z_p_val,
                                                                table))

  # category-wise distribution
  plt.subplot(1,3,2)
  sns.kdeplot(x1, shade= True, color='blue', label = 'churned')
  sns.kdeplot(x2, shade= False, color='green', label = 'not churned', linewidth = 1)
  plt.title('categorical distribution')
    
  # boxplot
  plt.subplot(1,3,3)
  sns.boxplot(x=cat, y=cont, data=data)
  plt.title('categorical boxplot')


# ### 6.1 Are vintage customers less likely to churn?
# 
# 

# In[ ]:


Bivariate_cont_cat(data, 'vintage', 'churn', 1)


# **Inferences**
# 
# 1.    Vintage customers churned more, but results are not significantly different
# 2.    Boxplot shows very similar distribution with outliers on the lower end.
# 
# **Result**
# 
# p-value is >0.05, which means that the two samples are more or less similar to each other.
# 
# Thefore, we can safely reject the hypothesis that vintage customers are more likely to churn.

# ### 6.2 Are customers with higher average balance less likely to churn?

# In[ ]:


Bivariate_cont_cat(data, 'average_monthly_balance_prevQ', 'churn', 1)


# **Result**    
# 
# p-value < 0.05, the the two samples are significantly different.
# 
# Customers who churned have significantly higher balance during immediate preceeding quarter, which is contrary to what we were were testing but the result is significant.

# In[ ]:


Bivariate_cont_cat(data, 'average_monthly_balance_prevQ2', 'churn', 1)


# **Inferences**
# 
# We can see that people who churned actually had significantly higher balance during their previous two quarters.**This validates the previous plot conveying the message that people whoc hurned actually had higher balance.
# **

# #### previous month/current month

# In[ ]:


Bivariate_cont_cat(data, 'previous_month_balance', 'churn', 1)


# In[ ]:


Bivariate_cont_cat(data, 'current_month_balance', 'churn', 1)


# **Inferences**
# 
# > Customers who churned had significantly high balance throughout the previous two quarters and previous month. But their average balance reduced significantly in the current month. Moreover the customers who are maintaining higher balance are more prone to churning (so it seems)
# 

# ### 6.3 Are customers dropping monthly balance highly likely to churn?

# In[ ]:


# Extracting drop of balance in previous and current month
difference = data[['churn','previous_month_balance','current_month_balance']][:]
difference['bal_diff'] = difference['current_month_balance']-difference['previous_month_balance']


# In[ ]:


Bivariate_cont_cat(difference, 'bal_diff', 'churn', 1)


# This is a a very absurd plot to gain insight from, so what actually happening here is that.... 
# * the customers who churned have a negative average balance differenceand that too is a huge number.
# * Whereas the customers who did not churn slighly positive balance difference betwwen the previous month and the current month.

# **Inference**
# 
# Customers who churned had a very high drop in their balance which is signified by the negative value in this bar plot.
# 
# **This factor can be used generate a new feature.**

# ## Bivariate: Categorical Categorical
# In this section we will be working with the categorical categorical combination of variables. **Grouped bar plot and stacked bar plots are the 2 key ways to visualise them.** Also we will be performing the the hypothesis testing using chi-square.

# #### List of Hypothesis to check under this combination
# 1.   Are females less likely to churn than males?
# 2.   Are young customers more likely to churn?
# 3.   Are customers in the lower income bracket more likely to churn?
# 4.   Are customers with dependent(s) less likely to churn?
# 5.   Customers with an average family size less than 4 are more likely to churn?
# 6.   Customers whose last transaction was more than 6 months ago, do they have higher churn rate?
# 7.   Possibility that cities and branch code with very few accounts may lead to churning.
# 
# **Missing Values** - finding behaviour
# 
# **Gender**: 
#   *  Do missing values churn more?
# 
# **Dependents**:
#   *  Do missing values have any relation with churn?
# 
# **Occupation:**
#    * Do they have some relation with churn?

# In[ ]:


def BVA_categorical_plot(data, tar, cat):
  '''
  take data and two categorical variables,
  calculates the chi2 significance between the two variables 
  and prints the result with countplot & CrossTab
  '''
  #isolating the variables
  data = data[[cat,tar]][:]

  #forming a crosstab
  table = pd.crosstab(data[tar],data[cat],)
  f_obs = np.array([table.iloc[0][:].values,
                    table.iloc[1][:].values])

  #performing chi2 test
  from scipy.stats import chi2_contingency
  chi, p, dof, expected = chi2_contingency(f_obs)
  
  #checking whether results are significant
  if p<0.05:
    sig = True
  else:
    sig = False

  #plotting grouped plot
  sns.countplot(x=cat, hue=tar, data=data)
  plt.title("p-value = {}\n difference significant? = {}\n".format(round(p,8),sig))

  #plotting percent stacked bar plot
  #sns.catplot(ax, kind='stacked')
  ax1 = data.groupby(cat)[tar].value_counts(normalize=True).unstack()
  ax1.plot(kind='bar', stacked='True',title=str(ax1))
  int_level = data[cat].value_counts()


# #### 1. Are females less likely to churn than males?

# In[ ]:


BVA_categorical_plot(data, 'churn', 'gender')


# **Result:**
# 
# the difference between the males and females customer churning is significant. Males churn significantly more than females.
# 
# **this info can be used to generate new feature.**
# 

# #### 2. Are young customers more likely to churn?
# For this I will be making 4 segments:
# 1. young
# 2. adult
# 3. senior citizen
# 4. very old

# In[ ]:


# segregating customers into segments
churn = data[['churn','age']][:]
churn['age_group'] = 'str'
churn['age_group'][churn['age']>=80] = 'very old'
churn['age_group'][(churn['age']<80) & (churn['age']>=60)] = 'senior citizen'
churn['age_group'][(churn['age']<60) & (churn['age']>=18)] = 'adult'
churn['age_group'][churn['age']<18] = 'young'


# In[ ]:


BVA_categorical_plot(churn, 'churn', 'age_group')


# **Result**:
# Age group has significant effect on the churning rate. Each group has a significantly different churning rate with respect to the expected churning rate (20/80 approx)
# 
# **This info can be used to generate new features**

# #### 3. Customers from low income bracket more likely to churn

# In[ ]:


BVA_categorical_plot(data, 'churn', 'customer_nw_category')


# **Result:**
# 
# Different income brackets have significant effect on the churn rate.
# 
# **This information can be used for feature engineering**

# #### 4,5. Are customers with dependent(s) less likely to churn?
# For this I am making 4 segments:
# 1. single
# 2. small family
# 3. large family
# 6. joint family

# In[ ]:


# segregating dependents into categories
dependents = data[['churn','dependents']][:]
dependents.dropna(inplace=True)
dependents['dep_group'] = None
dependents['dep_group'][dependents['dependents']==0] = 'single'
dependents['dep_group'][(dependents['dependents']>=1) & (dependents['dependents']<=3)] = 'small family'
dependents['dep_group'][(dependents['dependents']>=4) & (dependents['dependents']<=9)] = 'large family'
dependents['dep_group'][(dependents['dependents']>=10)] = 'joint family'


# In[ ]:


BVA_categorical_plot(dependents, 'churn', 'dep_group')


# **Result:**
# 
# Number of dependents also play significant role in churning.

# #### 7. Possibility that cities and branch code with very few accounts may lead to churning.

# #### City : Isolating cities with less than 1% of total customers

# In[ ]:


# getting city codes which have less than 280 (1%) of accounts
tmp = data['city'].value_counts()[:]
cities = tmp[tmp<280].index

churn_acc = data[['churn','city']][:]
churn_acc['city_cat'] = None
churn_acc['city_cat'][churn_acc['city'].isin(cities[:])] = 'low accounts'
churn_acc['city_cat'][~churn_acc['city'].isin(cities[:])] = 'high accounts'


# In[ ]:


BVA_categorical_plot(churn_acc, 'churn', 'city_cat')


# **Result**
# 
# cities having less than 1 percent of the total have significantly lower churn rates as compared to the cities with more accounts. This is contrary to what we assumed.
# 
# this information can be used to generate new feature.

# #### Branch code: Isolating branches with less than 0.5%of accounts

# In[ ]:


# getting branch codes with more than 0.5% of total accounts
tmp = data['branch_code'].value_counts()[:]
branch = tmp[tmp<140].index

# making two segments
churn_acc = data[['churn','branch_code']][:]
churn_acc['branch_cat'] = None
churn_acc['branch_cat'][churn_acc['branch_code'].isin(branch[:])] = 'low accounts'
churn_acc['branch_cat'][~churn_acc['branch_code'].isin(branch[:])] = 'high accounts'


# In[ ]:


BVA_categorical_plot(churn_acc, 'churn', 'branch_cat')


# **Result**
# 
# Here we see that the branches with low number of accounts do not have any significant difference from the branches with higher number of accounts.

# #### Missing Values : Gender
# To check whether the missing values in gender have some common behaviour among themselves.

# In[ ]:


# isolating rows with missing gender
miss_gender = data[:]
miss_gender['missing_gender'] = 'not_missing'
miss_gender['missing_gender'][~miss_gender['gender'].isin(['Male','Female'])] = 'missing value'


# In[ ]:


BVA_categorical_plot(miss_gender, 'churn', 'missing_gender')


# **There is no diffrent behaviour of the missing values in gender wrt churn variable. Or in other words, male and female customers are equally likely to churn.**
# 

# #### Missing Values : Dependents
# Whether the missing values in dependents have some common behavior among themselves.

# In[ ]:


# isolating rows with missing gender
miss_dependents = data[:]
miss_dependents['missing_dependents'] = 'not_missing'
miss_dependents['missing_dependents'][~miss_dependents['dependents'].isin([0, 2, 3, 1, 7, 4,
                                                                           6, 5, 9, 52, 36, 50,
                                                                           8, 25, 32])] = 'missing value'


# In[ ]:


BVA_categorical_plot(miss_dependents, 'churn', 'missing_dependents')

**Result**
the missing values in gender have significantly higher churn rate.

We can make a new feature for this
# ### Missing values : Occupation

# In[ ]:


# isolating rows with missing gender
miss_occupation = data[:]
miss_occupation['missing_occupation'] = 'not_missing'
miss_occupation['missing_occupation'][~miss_occupation['occupation'].isin(['self_employed',
                                                                           'salaried',
                                                                           'retired',
                                                                           'student',
                                                                           'company'])] = 'missing value'


# In[ ]:


BVA_categorical_plot(miss_occupation, 'churn', 'missing_occupation')


# **Missing values in occupation does not have any significantly different relation with churn rate.**

# ## Summary of Bivariare Analysis:
# 
# ### numerical numerical
# 1. Transactional variables, balance variables, previous quarter variables have strong correlations among themselves. This can be used to generate a bunch of meaningful new features.
# 
# ### Numerical Categorical
# 1. Customers who are mantaining a higher balance in their accounts are actually more susceptible to churning.
# 2. The customers who churned drastically dropped their balance in the most recent month. (which validates the definition of churn).
# 
# ### Categorical categorical
# 1. Rate of churning among the different age groups vary significantly.
# 2. Different age brackets also affect the rate of churning.
# 3. Number of dependents also affect the churning rate significantly.
# 
# ### Things to investigate further in Multivariate Analysis:
# * Whether there are any independent (or combination of them) variables which can explain missing values or outliers. (this will govern the preprocessing step)

# The succeeding notebook addressing the multivariate analysis and other miscellaneous analysis can be found at...
# 
# <Link>
