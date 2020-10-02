#!/usr/bin/env python
# coding: utf-8

# # Give me some Credit

# Let's work with a fairly  complicated data to understand the nuances of Data preparation. This data is quite challenging to clean and you may not agree with my approach.
# Note following points:-
#     - I have not used the best of Python coding as I am new to Python. My approach was quite functional; check if the output is providing the answer or not.
#     - There are many approaches to data cleaning (treatment of missing values and outliers). Consider this approach as one of the many possible.
#     - The approach adopted is more connected to Analytics rather than Machine Learning. Hence, treatment is  'manual'!! You may notice that each of the variables are not treated in so much detail by ML professionals.
# 
# **Introduction**:- Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit. 
# 
# Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. The objective is  to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/cs-training.csv")


# In[ ]:


df.head(10)


# Note that some of the variables got missing values indicated by 'NaN'. But this is not reliable and we need a summary statistics. Let's take a look at list of variables.

# In[ ]:


df.info()


# The list of variables indicates that all variables are numeric and few of these got missing values. We will look at the summary of these variables.

# In[ ]:


df.rename(columns = {df.columns[0]:'ID'}, inplace = True) 

df.describe()


# The depdendent variable is 'SeriousDlqin2yrs'. There are also variables like 'NumberOfTime30_59DaysPastDueNotW',	'NumberOfTime60_89DaysPastDueNotW',	'NumberOfTimes90DaysLate'. These variables give info on how much customers were delayed in payment and frequency. In Financial Industry, these types of variables are the inputs for creating the dependent variable. Hence, these variables cannot be used as independent variables.
# 
# Moreover, the use of this model is to score a new customer and obviously these variables will not
# be  available for a new customer. Hence, let's remove these variables straight away.

# In[ ]:


df.drop(df.columns[[4, 8, 10]], axis=1, inplace=True)
df.head()


# In[ ]:


P = df.groupby('SeriousDlqin2yrs')['ID'].count().reset_index()

P['Percentage'] = 100 * P['ID']  / P['ID'].sum()

print(P)


# Freq table shows that there are no missing values and as expected it contains 0 and 1.
# Delinquents are 6.68%

# In[ ]:


df['SeriousDlqin2yrs'].value_counts(normalize=True).plot(kind='barh')


# ## RevolvingUtilizationOfUnsecuredLines

# In[ ]:


df['RevolvingUtilizationOfUnsecuredLines'].describe()


# there are no missing values. the lower value of 0 is fine but max value is ridiculous as it is rarley more than 1

# In[ ]:


df3=df.loc[df['RevolvingUtilizationOfUnsecuredLines'] <=1]
sns.distplot(df3['RevolvingUtilizationOfUnsecuredLines'])


# In[ ]:


len(df[(df['RevolvingUtilizationOfUnsecuredLines']>1)])


# this shows that about 3300 observations got values more than 1 and hence it 
# is not appropriate to consider all these as outliers and cap to 1. A better approach is to 
# make these missing and impute the values

# In[ ]:


df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].map(lambda x: np.NaN if x >1 else x)


# In[ ]:


df['RevolvingUtilizationOfUnsecuredLines'].describe()


# For imputation, we will use ffill method which will retain the distribution and mean of the variable.

# In[ ]:


df['RevolvingUtilizationOfUnsecuredLines'].fillna(method='ffill', inplace=True)


# In[ ]:


df['RevolvingUtilizationOfUnsecuredLines'].describe()


# ## age 
# 
# Let's take a look at univariate analysis and distribution.

# In[ ]:


df['age'].describe()


# there are no missing values. the lower value of 0 and max value of 109 are outliers.
# Typical age range is 18-80. 

# In[ ]:


sns.distplot(df['age'])


# this shows that about there are only very few observations outside 18-80 range.
# Hence, it is ok to cap the values

# In[ ]:


df.loc[df['age']>80, 'age']=80
df.loc[df['age']<18, 'age']=18


# In[ ]:


sns.distplot(df['age'])


# In[ ]:


df['age'].describe()


# ## DebtRatio

# In[ ]:


df['DebtRatio'].describe()


# This variable refers to debt to income ratio.there are no missing values. the lower value of 0 is fine but max value is ridiculous as it is rarley more than

# In[ ]:


df2=df[df['DebtRatio']<=1]
sns.distplot(df2['DebtRatio'])


# In[ ]:


df2=df[df['DebtRatio']>1]
df2['DebtRatio'].describe()


# Typical value of Debt Income ratio is 0.4. But almost 35000 observations got values
# 	higher than 1 and hence cannot be treated as outliers. Best approach is to consider
# 	it as missing and impute values

# In[ ]:


df.loc[df['DebtRatio']>1, 'DebtRatio']=np.NaN


# In[ ]:


df['DebtRatio'].describe()


# In[ ]:


df['DebtRatio'].fillna(method='ffill', inplace=True)


# In[ ]:


df['DebtRatio'].describe()


# In[ ]:


sns.distplot(df['DebtRatio'])


# ## NumberOfOpenCreditLinesAndLoans

# In[ ]:


df['NumberOfOpenCreditLinesAndLoans'].describe()


# there are no missing values. the lower value of 0 is fine. Max value
# 	of 58 seems to be an outlier as it is much higher than mean (max is 10*std away from mean)

# In[ ]:


sns.distplot(df['NumberOfOpenCreditLinesAndLoans'])


# the distribution indicate that it is continuous upto 30. Hence, let's cap at
# 	30

# In[ ]:


df.loc[df['NumberOfOpenCreditLinesAndLoans']>30, 'NumberOfOpenCreditLinesAndLoans']=30


# In[ ]:


df['NumberOfOpenCreditLinesAndLoans'].describe()


# In[ ]:


sns.distplot(df['NumberOfOpenCreditLinesAndLoans'])


# ## MonthlyIncome

# In[ ]:


df['MonthlyIncome'].describe()


# There are missing values and Max value is too large. Min value of 0 is not ok as finance industry expect a minimum income of 1000.

# In[ ]:


df['MonthlyIncome'].isnull().sum()


# In[ ]:


len(df[df['MonthlyIncome']<1000])


# Number of obs below 1000 is too large. hence, it is not ok to treat it as outliers and cap to 1000. Lets treat it as missing and then impute the values.

# In[ ]:


sns.distplot(df['MonthlyIncome'].dropna())


# The max value is too large and hence the plot is not making sense

# In[ ]:


df2=df[df['MonthlyIncome']<50000]
sns.distplot(df2['MonthlyIncome'].dropna())


# Distribution shows that income smoothly decreases upto 25000 and then few outliers 
# of huge values. 

# In[ ]:


df.loc[df['MonthlyIncome']>25000, 'MonthlyIncome']=25000
df['MonthlyIncome'].describe()


# In[ ]:


df.loc[df['MonthlyIncome']<1000, 'MonthlyIncome']=np.NaN
df['MonthlyIncome'].describe()


# In[ ]:


df['MonthlyIncome'].fillna(method='ffill', inplace=True)
df['MonthlyIncome'].describe()


# In[ ]:


sns.distplot(df['MonthlyIncome'])


# ## NumberRealEstateLoansOrLines    

# In[ ]:


df['NumberRealEstateLoansOrLines'].describe()


# There are no missing values but max value is too large. Min value of 0 is ok.

# In[ ]:


sns.distplot(df['NumberRealEstateLoansOrLines'])


# Distribution shows that the variable smoothly decreases upto 10 and then few outliers of large values. 

# In[ ]:


df2=df[df['NumberRealEstateLoansOrLines']<6]
sns.distplot(df2['NumberRealEstateLoansOrLines'].dropna())


# In[ ]:


df.loc[df['NumberRealEstateLoansOrLines']>5, 'NumberRealEstateLoansOrLines']=5
df['NumberRealEstateLoansOrLines'].describe()


# In[ ]:


sns.distplot(df['NumberRealEstateLoansOrLines'])


# ### NumOfDependents

# In[ ]:


df['NumberOfDependents'].describe()


# In[ ]:


sns.distplot(df['NumberOfDependents'].dropna())


# there are missing values. The distribution is continuous upto 5 and then few outliers

# In[ ]:


df.loc[df['NumberOfDependents']>5, 'NumberOfDependents']=5
df['NumberOfDependents'].describe()


# Since proportion of missing is large, imputation using mean is not appropriate as this will change the distribution too much. we will impute the missing values using ffill as this will preserve the mean and standard deviation.

# In[ ]:


#df['NumberOfDependents'].fillna(df['NumberOfDependents'].mean(), inplace=True)
df['NumberOfDependents'].fillna(method='ffill', inplace=True)
df['NumberOfDependents'].describe()


# In[ ]:


df.describe()


# Looks like the data is now clean. Lets save it as .pkl file

# In[ ]:


df.to_pickle("gmsc_clean.pkl")


# ## Exploratory Data Analysis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
sns.set(color_codes=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_pickle('gmsc_clean.pkl')
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# <h1 align="center">Univariate Analysis</h1> 

# The objective of univariate analysis is to examine each of the variables one by one. The focus will be on the distribution of the variable. Let's start with dependent variable.

# In[ ]:


sns.countplot(x='SeriousDlqin2yrs', data=df)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,6))
df['SeriousDlqin2yrs'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=False)
ax[0].set_title('SeriousDlqin2yrs')
ax[0].set_ylabel('')
sns.countplot('SeriousDlqin2yrs',data=df,ax=ax[1])
ax[1].set_title('SeriousDlqin2yrs')
plt.show()


# About 6.7% customers were delinquents. 
# 
# Let's create distribution charts for all independent variables together.

# In[ ]:


for column in df.columns[2:]:
    print(column)
    #s=df['column']
    s=df[column]
    mu, sigma =norm.fit(s)
    count, bins, ignored = plt.hist(s, 30, normed=True, color='g')
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=1, color='r')

    title = "Plot used: mu = %.2f,  std = %.2f" % (mu, sigma)
    plt.title(title, loc='right')

    plt.show()


# All these variables look fine. Although distribution of some of these variables are not close to a normal distribution, it is not very critical. The techniques we apply are not very sensitive to normality. We also have the option of converting some of these variables to categories for modeling (eg. RevolvingUtilizationOfUnsecuredLines). 
# 

# <h1 align="center">Bivariate Analysis</h1> 

# Under bivariate analysis we will examine the realtionship between Dependent variable and each of the independent variables. We will also check selected pairs of independent variables.

# ## **SeriousDlqin2yrs vs RevolvingUtilizationOfUnsecuredLines** 

# Easiest approach is to compare the means of RevolvingUtilizationOfUnsecuredL by two 
# categories of SeriousDlqin2yrs

# In[ ]:


df.groupby('SeriousDlqin2yrs')['RevolvingUtilizationOfUnsecuredLines'].agg(['count','mean'])


# In[ ]:


df['RevolvingUtilizationOfUnsecuredLines'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 


# As expected, delinquent customers got almost twice the utilization of unsecured lines

# Let's now explore the relationship in detail by categorising the varaible. To categorise the variable, we will choose 25, 50 and 75 percentile as cutoffs.  

# In[ ]:


df['RevolvingUtilizationOfUnsecuredLines'].describe()


# In[ ]:


def cat_ruul(ruul):
    if ruul <0.03:
        return 1
    elif 0.03<= ruul <0.14:
        return 2
    elif 0.14<= ruul <0.52:
        return 3
    else:
        return 4


# In[ ]:


df['ruul_cat'] = df['RevolvingUtilizationOfUnsecuredLines'].apply(cat_ruul)
df.head(3)


# In[ ]:


# lets check if the categorization was done correctly
df.groupby('ruul_cat')['RevolvingUtilizationOfUnsecuredLines'].agg(['min','max'])


# In[ ]:


pd.crosstab(df.SeriousDlqin2yrs, df.ruul_cat, normalize='columns')


# In[ ]:


sb=pd.crosstab(df.ruul_cat, df.SeriousDlqin2yrs, normalize=0)
sb.plot.bar(stacked=True)


# As expected, plot shows that there are more delinquents in the category of highest utilization. However there is not much difference between the fiirst two categories. 

# In[ ]:


df2=pd.crosstab(df.SeriousDlqin2yrs, df.ruul_cat)
chi2_contingency(df2)


# Chi-square test establish that there is significant dependency between utilization and delinquency;
# 

# ## **SeriousDlqin2yrs vs Age**

# Easiest approach is to compare the means of Age by two categories of SeriousDlqin2yrs.

# In[ ]:


df.groupby('SeriousDlqin2yrs')['age'].agg(['count','mean'])


# In[ ]:


df['age'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 


# Delinquent cusomters are younger than non-delinquent customers.

# let's now explore the relationship in detail by categorising age. To categorise the variable, we will choose 25, 50 and 75 percentile as cutoffs.  

# In[ ]:


df['age'].describe()


# In[ ]:


def cat_ruul(ruul):
    if ruul <41:
        return 1
    elif 41<= ruul <52:
        return 2
    elif 52<= ruul <63:
        return 3
    else:
        return 4


# In[ ]:


df['age_cat'] = df['age'].apply(cat_ruul)
df.head(3)


# In[ ]:


# lets check if the categorization was done correctly
df.groupby('age_cat')['age'].agg(['min','max'])


# In[ ]:


pd.crosstab(df.SeriousDlqin2yrs, df.age_cat, normalize='columns')


# In[ ]:


sb=pd.crosstab(df.age_cat, df.SeriousDlqin2yrs, normalize=0)
sb.plot.bar(stacked=True)


# Plot shows that there are more proportion of delinquents in the category of youngest age. Delinquency decreases as age increases uniformly. 

# In[ ]:


df2=pd.crosstab(df.SeriousDlqin2yrs, df.age_cat)
chi2_contingency(df2)


# chi-square test establish that there is significant dependency between age and delinquency;
# 

# ## **SeriousDlqin2yrs vs DebtRatio** 

# Easiest approach is to compare the means of DebtRatio by two 
# categories of SeriousDlqin2yrs

# In[ ]:


df.groupby('SeriousDlqin2yrs')['DebtRatio'].agg(['count','mean'])


# In[ ]:


df['DebtRatio'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 


# Delinquent customers got higher debtratio compared to non-delinquent customers.

# let's now explore the relationship in detail by categorising DebtRatio. To categorise the variable, we will choose 25, 50 and 75 percentile as cutoffs.  

# In[ ]:


df['DebtRatio'].describe()


# In[ ]:


def cat_ruul(ruul):
    if ruul <0.13:
        return 1
    elif 0.13<= ruul <0.27:
        return 2
    elif 0.27<= ruul <0.43:
        return 3
    else:
        return 4


# In[ ]:


df['DebtRatio_cat'] = df['DebtRatio'].apply(cat_ruul)
df.head(3)


# In[ ]:


# lets check if the categorization was done correctly
df.groupby('DebtRatio_cat')['DebtRatio'].agg(['min','max'])


# In[ ]:


pd.crosstab(df.SeriousDlqin2yrs, df.DebtRatio_cat, normalize='columns')


# In[ ]:


sb=pd.crosstab(df.DebtRatio_cat, df.SeriousDlqin2yrs, normalize=0)
sb.plot.bar(stacked=True)


# As expected it shows that there are more delinquents in the category of 
# highest DebtRatio.

# In[ ]:


df2=pd.crosstab(df.SeriousDlqin2yrs, df.DebtRatio_cat)
chi2_contingency(df2)


# Chi-square test establish that there is significant dependency between DebtRatio and delinquency.
# 

# ## **SeriousDlqin2yrs vs MonthlyIncome** 

# Simplest approach is to compare the means of MonthlyIncome by two 
# categories of SeriousDlqin2yrs

# In[ ]:


df.groupby('SeriousDlqin2yrs')['MonthlyIncome'].agg(['count','mean'])


# In[ ]:


df['MonthlyIncome'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 


# Delinquent cusomters got lower MonthlyIncome compared to non-delinquent customers.

# let's now explore the relationship in detail by categorising MonthlyIncome. To categorise the variable, we will choose 25, 50 and 75 percentile as cutoffs.  

# In[ ]:


df['MonthlyIncome'].describe()


# In[ ]:


def cat_ruul(ruul):
    if ruul <3600:
        return 1
    elif 3600<= ruul <5500:
        return 2
    elif 5500<= ruul <8333:
        return 3
    else:
        return 4


# In[ ]:


df['MonthlyIncome_cat'] = df['MonthlyIncome'].apply(cat_ruul)
df.head(3)


# In[ ]:


# lets check if the categorization was done correctly
df.groupby('MonthlyIncome_cat')['MonthlyIncome'].agg(['min','max'])


# In[ ]:


pd.crosstab(df.SeriousDlqin2yrs, df.MonthlyIncome_cat, normalize='columns')


# In[ ]:


sb=pd.crosstab(df.MonthlyIncome_cat, df.SeriousDlqin2yrs, normalize=0)
sb.plot.bar(stacked=True)


# As expected it shows that there are less delinquents in the category of 
# highest MonthlyIncome.

# In[ ]:


df2=pd.crosstab(df.SeriousDlqin2yrs, df.MonthlyIncome_cat)
chi2_contingency(df2)


# Chi-square test establish that there is significant dependency between MonthlyIncome and delinquency;
# 

# ## **SeriousDlqin2yrs vs NumberOfOpenCreditLinesAndLoans **

# Simplest approach is to compare the means of NumberOfOpenCreditLinesAndLoans by two 
# categories of SeriousDlqin2yrs

# In[ ]:


df.groupby('SeriousDlqin2yrs')['NumberOfOpenCreditLinesAndLoans'].agg(['count','mean'])


# In[ ]:


df['NumberOfOpenCreditLinesAndLoans'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 


# Delinquent cusomters got lower NumberOfOpenCreditLinesAndLoans compared to non-delinquent customers.

# let's now explore the relationship in detail by categorising NumberOfOpenCreditLinesAndLoans. To categorise the variable, we will choose 25, 50 and 75 percentile as cutoffs.  

# In[ ]:


df['NumberOfOpenCreditLinesAndLoans'].describe()


# In[ ]:


def cat_ruul(ruul):
    if ruul <5:
        return 1
    elif 5<= ruul <8:
        return 2
    elif 8<= ruul <11:
        return 3
    else:
        return 4


# In[ ]:


df['NOCLL_Cat'] = df['NumberOfOpenCreditLinesAndLoans'].apply(cat_ruul)
df.head(3)


# In[ ]:


# lets check if the categorization was done correctly
df.groupby('NOCLL_Cat')['NumberOfOpenCreditLinesAndLoans'].agg(['min','max'])


# In[ ]:


pd.crosstab(df.SeriousDlqin2yrs, df.NOCLL_Cat, normalize='columns')


# In[ ]:


sb=pd.crosstab(df.NOCLL_Cat, df.SeriousDlqin2yrs, normalize=0)
sb.plot.bar(stacked=True)


# As expected it shows that there are more delinquents in the category of 
# lowest NumberOfOpenCreditLinesAndLoans.

# In[ ]:


df2=pd.crosstab(df.SeriousDlqin2yrs, df.NOCLL_Cat)
chi2_contingency(df2)


# Chi-square test establish that there is significant dependency between NumberOfOpenCreditLinesAndLoans and delinquency;
# 

# ### **SeriousDlqin2yrs vs NumberRealEstateLoansOrLines** 

# Simplest approach is to compare the means of NumberRealEstateLoansOrLines by two 
# categories of SeriousDlqin2yrs

# In[ ]:


df.groupby('SeriousDlqin2yrs')['NumberRealEstateLoansOrLines'].agg(['count','mean'])


# In[ ]:


df['NumberRealEstateLoansOrLines'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 


# Delinquent cusomters got lower NumberRealEstateLoansOrLines compared to non-delinquent customers.

# let's now explore the relationship in detail by categorising NumberRealEstateLoansOrLines. To categorise the variable, we will choose 25, 50 and 75 percentile as cutoffs.  

# In[ ]:


df['NumberRealEstateLoansOrLines'].describe()


# In[ ]:


def cat_ruul(ruul):
    if ruul <=0:
        return 1
    elif 0< ruul <=1:
        return 2
    elif 1< ruul <=2:
        return 3
    else:
        return 4


# In[ ]:


df['NRELL_Cat'] = df['NumberRealEstateLoansOrLines'].apply(cat_ruul)
df.head(3)


# In[ ]:


# lets check if the categorization was done correctly
df.groupby('NRELL_Cat')['NumberRealEstateLoansOrLines'].agg(['min','max'])


# In[ ]:


pd.crosstab(df.SeriousDlqin2yrs, df.NRELL_Cat, normalize='columns')


# In[ ]:


sb=pd.crosstab(df.NRELL_Cat, df.SeriousDlqin2yrs, normalize=0)
sb.plot.bar(stacked=True)


# As expected it shows that there are more delinquents in the category of 
# lowest NumberRealEstateLoansOrLines.

# In[ ]:


df2=pd.crosstab(df.SeriousDlqin2yrs, df.NRELL_Cat)
chi2_contingency(df2)


# chi-square test establish that there is significant dependency between NumberRealEstateLoansOrLines and delinquency;
# 

# ## SeriousDlqin2yrs vs NumberOfDependents 

# Simplest approach is to compare the means of NumberOfDependents by two 
# categories of SeriousDlqin2yrs

# In[ ]:


df.groupby('SeriousDlqin2yrs')['NumberOfDependents'].agg(['count','mean'])


# In[ ]:


df['NumberOfDependents'].groupby(df.SeriousDlqin2yrs).mean().plot(kind='bar', color=['blue', 'green']) 


# Delinquent cusomters got higher NumberOfDependents compared to non-delinquent customers.

# let's now explore the relationship in detail by categorising NumberOfDependents. To categorise the variable, we will choose 25, 50 and 75 percentile as cutoffs.  

# In[ ]:


df['NumberOfDependents'].describe()


# In[ ]:


def cat_ruul(ruul):
    if ruul <=0:
        return 1
    elif 0< ruul <=1:
        return 2
    elif 1< ruul <=2:
        return 3
    else:
        return 4


# In[ ]:


df['NOD_Cat'] = df['NumberOfDependents'].apply(cat_ruul)
df.head(3)


# In[ ]:


# lets check if the categorization was done correctly
df.groupby('NOD_Cat')['NumberOfDependents'].agg(['min','max'])


# In[ ]:


pd.crosstab(df.SeriousDlqin2yrs, df.NOD_Cat, normalize='columns')


# In[ ]:


sb=pd.crosstab(df.NOD_Cat, df.SeriousDlqin2yrs, normalize=0)
sb.plot.bar(stacked=True)


# As expected it shows that there are more delinquents in the category of 
# highest NumberOfDependents.

# In[ ]:


df2=pd.crosstab(df.SeriousDlqin2yrs, df.NOD_Cat)
chi2_contingency(df2)


# chi-square test establish that there is significant dependency between NumberOfDependents and delinquency;
# 

# This discussion provided an overview of EDA. Ideally this should result in a professional presentation. The Charts created can be copied and pasted into the presentation software or the summarised values can be used to create charts. Each of this analyses should be supported with conclusions.

# In[ ]:




