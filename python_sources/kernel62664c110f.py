#!/usr/bin/env python
# coding: utf-8

# # Dataset name: Give Me Some Credit
# 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir('../input'))
get_ipython().run_line_magic('matplotlib', 'inline')


# Information on the columns available in the dataset:
# 
# **SeriousDlqin2yrs**                                        -    Person experienced 90 days past due delinquency or worse 
# 
# **RevolvingUtilizationOfUnsecuredLines**   -   Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits   
# 
# **age**                                                                -   Age of borrower in years
# 
# **NumberOfTime30-59DaysPastDueNotWorse**   -   Number of times borrower has been 30-59 days past due but no worse in the last 2 years.
# 
# **DebtRatio**   -   Monthly debt payments, alimony,living costs divided by monthy gross income   
# 
# **MonthlyIncome**   -   Monthly income
# 
# **NumberOfOpenCreditLinesAndLoans**   -   Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)
# 
# **NumberOfTimes90DaysLate**   -   Number of times borrower has been 90 days or more past due.
# 
# **NumberRealEstateLoansOrLines**   -   Number of mortgage and real estate loans including home equity lines of credit
# 
# **NumberRealEstateLoansOrLines**   -   Number of mortgage and real estate loans including home equity lines of credit
# 
# **NumberOfTime60-89DaysPastDueNotWorse**   -   Number of times borrower has been 60-89 days past due but no worse in the last 2 years.
# 
# **NumberOfDependents** - Number of dependents in family excluding themselves (spouse, children etc.)
# 
# 
#   

# # Preliminary inspection

# In[2]:


#reading data frame
df_train = pd.read_csv("../input/cs-training.csv")
df_test = pd.read_csv("../input/cs-test.csv")
df_train=df_train.append(df_test); #Appending test data too!


# Checking number of columns, and type of data we have (categorical or continuous)

# In[3]:


print("Number of columns = ",len(df_train.columns))
print("Number of rows = ",len(df_train))
df_train.head(10)


# Inspecting the table, we see that there are 12 columns which are all number with no categorical variables. 
# In a few columns we have missing data put as NaN, which we'll have to process before feeding into a model. But right now, we proceed with the inspection, and try to get insight into the statistical nature of the data. 
# 

# In[4]:


#Histogram plot of age
fig, axs = plt.subplots(3,3)
fig.set_size_inches(18.5, 12)
sns.distplot(df_train["age"],ax=axs[0,0]);
sns.distplot(df_train["RevolvingUtilizationOfUnsecuredLines"],ax=axs[0,1]);
sns.distplot(df_train["DebtRatio"],ax=axs[0,2]);

g=sns.distplot(df_train["MonthlyIncome"].dropna(),ax=axs[1,0]);
g.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

sns.distplot(df_train["NumberOfOpenCreditLinesAndLoans"].dropna().astype('int'),ax=axs[1,1]);
sns.distplot(df_train["SeriousDlqin2yrs"].dropna().astype('int'),ax=axs[1,2],kde=False);

sns.distplot(df_train["NumberOfTime30-59DaysPastDueNotWorse"].dropna(),ax=axs[2,0]);
sns.distplot((df_train["NumberOfDependents"].dropna()).astype('int'),ax=axs[2,1]);
sns.distplot(df_train["NumberOfTimes90DaysLate"].dropna().astype('int'),ax=axs[2,2]);


fig, axs = plt.subplots(1,2)
fig.set_size_inches(12,4)
sns.distplot(df_train["NumberRealEstateLoansOrLines"].dropna().astype('int'),ax=axs[0]);
sns.distplot(df_train["NumberOfTime60-89DaysPastDueNotWorse"].dropna().astype('int'),ax=axs[1]);


# We see some really bad plots for variables like MonthlyIncome, DebtRatio, etc. This is because there are abnormally large percentages of instances with very small values in these variables, i.e. the data has very high kurtosis, as seen in the next cell.

# In[5]:


for column in df_train.columns:
    print("Kurtosis  ",column, ": %f" % df_train[column].kurt())


# # Correlations
# It is important to find out correlations between the variables. This is helpful for filtering out redundant variables which helps to build a simpler and faster model. Also, the correlations can help us find abnormalities more accurately.
# 
# We now plot correlation matrix between different variables using spearman method. This method works better when there are huge standard deviations in the dataset, as is the case here.

# In[6]:


#Correlation heat map
corr = df_train.corr(method="spearman")
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, square=True, annot=True);


# We now examine top 3 pairs with respect to value of correlation coefficient

# In[34]:


fig,axs=plt.subplots(1,3)
fig.set_size_inches(20,4)
axs[0].scatter(df_train["NumberRealEstateLoansOrLines"],df_train["NumberOfOpenCreditLinesAndLoans"]);
axs[0].set_xlabel("NumberRealEstateLoansOrLines")
axs[0].set_ylabel("NumberOfOpenCreditLinesAndLoans")

axs[1].scatter(df_train["DebtRatio"],df_train["NumberRealEstateLoansOrLines"]);
axs[1].set_xlabel("DebtRatio")
axs[1].set_ylabel("NumberRealEstateLoansOrLines")

augdata=(df_train[["MonthlyIncome","NumberRealEstateLoansOrLines"]]).dropna()
axs[2].scatter(augdata["MonthlyIncome"],augdata["NumberRealEstateLoansOrLines"]);
axs[2].set_xlabel("MonthlyIncome")
axs[2].set_ylabel("NumberRealEstateLoansOrLines")
for tick in axs[2].get_xticklabels():
    tick.set_rotation(90)

plt.show()


# **Plot 1**
# 
# Variables NumberRealEstateLoansOrLines and NumberOfOpenCreditLinesAndLoans seem to exhibit interesting relationship, where the latter has a minimum value strongly correlated with the value of former. When we go into meaning of these variables, explanation becomes easy, because it is obvious that number of credit lines and loans opened for a person will be at least equal to the number of loans/lines opened for a real estate property. It is further deducible from the scatter plot that when number of real estate loans are at lower side, number of credit loans become strongly decorrelated with the variable, which is to say that as a person gets more number of real estate loans, his/her tendency to open other kinds of loans decreases.
# 
# **Plot 2**
# 
# Debt ratio seems to have pecuilar relationship with the variable NumberRealEstateLoansOrLines. High number of real estate loans are taken by only people with no DebtRatio, and very high debt ratio seems to give an upper limit on the NumberRealEstateLoansOrLines.
# 
# 
# **Plot 3**
# 
# This plot has nearly the same behavior as Plot 2, but between different variables. Number of real estate loans and mortgages become significantly small as the monthly income becomes above a certain threshold. This makes sense because a person with high income wouldn't like to take loans for real estate for which the interest rates are often high. A similiar, but less correlated, relationship is seen between montly income, and total number of open credit loans.
# 
# 
# **But wait**, shouldn't the correlation coefficient be negative between variables of Plot 2 and 3, because it looks like higher debt ratio means lower NumberOfTimes90DaysLate in Plot 2, and similarly in Plot 3 the two variables look to be inversely correlated. Maybe we aren't getting a good picture at this scale, to make sense of the positive correlation, we plot at different x and y limits the two graphs in the following cell.
# 

# In[54]:


fig,axs=plt.subplots(1,2)
fig.set_size_inches(11,4)

axs[0].scatter(df_train["DebtRatio"],df_train["NumberRealEstateLoansOrLines"]);
axs[0].set_xlim([100,40000])
axs[0].set_ylim([-1,10])

axs[1].scatter(df_train["MonthlyIncome"],df_train["NumberRealEstateLoansOrLines"]);
axs[1].set_xlim([0,100000])
axs[1].set_ylim([-1,50]);


# It is now clear why the correlation coefficient for the two set of the variables was positive.

# # More bivariate plots
# Correlation was fastest way to look for the important pairs of variables for analysis. However, the spearman correlation only outputs how well two variables are monotonically related, a rather complex relation between two variables (like quadratic relation) wouldn't be detected by the correlation matrix. Hence, we study more plots based on whatever intuition we have accumulated thus far.

# In[8]:


#Pair wise plot 
sns.set()
important_columns = ['MonthlyIncome', 'DebtRatio', 'age', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate']
figs=sns.pairplot(df_train[important_columns].dropna(), size = 2.5, aspect=1.5)
for i in range(5):
    for j in range(2):
        for ticks in figs.axes[i,j].get_xticklabels():
            ticks.set_rotation(90)
plt.show();


# From the univariate distributions present in diagonal plots, it can be seen that only the age and NumberOfOpenCreditLinesandLoans variables can be treated as having near gaussian distribution. 

# # Monthly Income
# We will now try to study effects of monthly income more closely.

# In[9]:


#Monthly income statistics
df_train["MonthlyIncome"].describe()


# It was seen earlier that MonthlyIncome has high kurosis. Here a large number of people are concentrated in a small range of monthly income, but the range of income extends to a very large value. Now we visualize univariate distribution again but now zooming in on the low income segment.

# In[10]:


#Monthly income univariate plot
fig, axs = plt.subplots(1,2)
fig.set_size_inches(12, 4)
#Zoomed out plot
g=sns.distplot(df_train["MonthlyIncome"].dropna(),ax=axs[0]);
g.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#Zoomed in plot
sns.distplot(df_train["MonthlyIncome"].loc[df_train["MonthlyIncome"]<0.5e5].dropna(),ax=axs[1]);


# We can have better representation of the variable at global scale by taking its log. We have to exclude all the 0 income people before doing that. 

# In[ ]:


df_train_mod=df_train.loc[df_train["MonthlyIncome"]!=0.0].dropna().copy()
df_train_mod["MonthlyIncome"]=np.log10(df_train_mod["MonthlyIncome"])
g=sns.distplot(df_train_mod["MonthlyIncome"]);
g.axes.set_xlabel("Log10 of MonthlyIncome");


# From the graph it looks like it will be useful to do analysis separately on the people with MonthlyIncome>0. We have separated the dataframe for all the employed people in df_train_mod. Now we can find the correlation matrix in this modified data frame.

# In[99]:


#Correlation heat map
corr = df_train_mod.corr(method="spearman")
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, square=True, annot=True);


# What can be seen is that there is now a stronger correlation between DebtRatio and NumberRealEstateLoansOrLines. Let's see what happens when we remove all the 0 values from DebtRatio and NumberRealEstateLoansOrLines from this modified data.

# In[100]:


df_train_mod2 = df_train_mod.loc[df_train_mod["DebtRatio"]!=0.0].copy()
df_train_mod2["DebtRatio"] = np.log10(df_train_mod2["DebtRatio"])


# In[101]:


df_train_mod3 = df_train_mod2.loc[df_train_mod2["NumberRealEstateLoansOrLines"]!=0.0].copy()
df_train_mod3["NumberRealEstateLoansOrLines"] = np.log10(df_train_mod3["NumberRealEstateLoansOrLines"])


# In[102]:


#Correlation heat map
corr = df_train_mod3.corr(method="spearman")
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, square=True, annot=True);


# The results are unexpected, removing all the zero values from MonthlyIncome, DebtRatio and NumberRealEstateLoansOrLines, there is no significant positive correlation between any two variables, while before doing that DebtRatio and MonthlyIncome were correlated with NumberRealEstateLoansOrLines. What this means is that if a person earns an income, and isn't debt free, his/her number of mortogages or real estate loans isn't directly correlated with any of the other two variables. Whereas, if a person is debt free, his number of real estate loans is positively correlated with his income. 
# 
# It can also be seen from the correlation matrix of the modified data frame, that for non-zero income, and non-zero debt ratio with non-zero number of real estate loans, the income and debt-ratio is highly negatively correlated. This can be a useful feature in building a predictive model.
# 

# In[106]:


plt.scatter(df_train_mod3['MonthlyIncome'],df_train_mod3['DebtRatio']);
plt.xlabel('log10 of MonthlyIncome')
plt.ylabel('log10 of DebtRatio');


# Let's look at bivariate plots between monthly income and some other variables

# In[93]:


fig,axs=plt.subplots(1,2)
fig.set_size_inches(16, 4)
axs[0].scatter(df_train['SeriousDlqin2yrs'],df_train['MonthlyIncome'])
axs[0].set_ylabel('MonthlyIncome')
axs[0].set_xlabel('SeriousDlqin2yrs')
axs[1].scatter(df_train['RevolvingUtilizationOfUnsecuredLines'],df_train['MonthlyIncome']);
axs[1].set_ylabel('MonthlyIncome')
axs[1].set_xlabel('RevolvingUtilizationOfUnsecuredLines');


# It can be gathered from the left plot  that above a certain threshold of monthly income, there are no SeriousDlqin2yrs. In plot 2, above a certain limit, no matter the monthly income RevolvingUtilizationOfUnsecuredLines is 0, and similarly above a certain limit of RevolvingUtilizationOfUnsecuredLines, monthly income is bounded to a low value.

# 

# In[ ]:




