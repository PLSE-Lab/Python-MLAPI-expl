#!/usr/bin/env python
# coding: utf-8

# ![](http://www.homecredit.net/~/media/Images/H/Home-Credit-Group/image-gallery/full/image-gallery-01-11-2016-b.png)

# Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.
# 
# Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
# 
# While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

# # Table of Content
# ## 1. Which one is riskier Female or male
# * ### 1.1 Credit amount of the loan (AMT_CREDIT)
# * ### 1.2  Who has the highest income Female vs Male ?
# 
# ## 2. The Highest and Lowest paying job
# * ### 2.1 Number of Borrower
# * ### 2.2 Average income total (AMT_INCOME_TOTA) 
# * ### 2.3 Credit amount of the loan (AMT_CREDIT)
# 
# ## 3. What ocupation has the highest default rate
# * ### 3.1 Ratio of AMT_CREDIT:AMT_INCOME
# * ### 3.2 Occupation default rate

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

pd.options.display.float_format = '{:,.3f}'.format

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


i="SK_ID_CURR"
t='TARGET'
c='CODE_GENDER'
c1='AMT_CREDIT'
c2='OCCUPATION_TYPE'
c3='AMT_INCOME_TOTAL'

app = '../input/application_train.csv'

application_train = pd.read_csv(app)
#application_train["OCCUPATION_TYPE"] = application_train["OCCUPATION_TYPE"].astype('category')

application_train["log_"+c1] = np.log1p(application_train[c1])
application_train["log_"+c3] = np.log1p(application_train[c3])

application_train.info()


# In[ ]:


def percentage(part, whole):
  return 100 * float(part-whole)/float(whole)


# # 1. Which one is riskier Female or male
# 
# 
# 
#  

# In[9]:


fig, ax =plt.subplots(1,2, figsize=(20,5))
ax[0].set_title("count") #0
sns.countplot(hue=t, x=c, data=application_train, ax=ax[0])

ax[1].set_title("sum("+c1+")") #1
sns.barplot(x=c, y=c1, hue=t, estimator=np.sum, data=application_train, ax=ax[1])


# In[82]:


#Count
group = pd.pivot_table(application_train, values=i, index=[c, t], aggfunc=[np.count_nonzero])
x = group.unstack()
x.columns = ['0','1'] # rename columns
x1 = x.loc[['F', 'M'],:].sum(axis=1).rename('total') # sum not-default+default
x1 = pd.concat([x.loc[['F', 'M'],:],x1], axis=1) # concat
x1['pct_default'] = x1['1']/x1['total'] # calculate percentage
x1


# The number of female clients is almost double the number of male clients. Looking to the percent of defaulted credits, males have a higher chance of not returning their loans (~10%), comparing with women (~7%).

# ## 1.1 Credit amount of the loan (AMT_CREDIT)

# In[12]:


# Let's look at the density of the AMT_INCOME_TOTAL of F vs M
temp1 = application_train.loc[application_train[c] != 'XNA'] # exclude XNA

#plt.figure(figsize=(20,10))
sns.FacetGrid(temp1, hue=c, size=6)    .map(sns.kdeplot, "log_"+c1)    .add_legend()
plt.title('KDE Plot of '+c1);


# AMT_CREDIT is balance

# In[5]:


#sum(AMT_CREDIT)
group = pd.pivot_table(application_train, values=c1, index=[c, t], aggfunc=[np.sum])
x = group.unstack()
x.columns = ['0','1'] # rename columns
x1 = x.loc[['F', 'M'],:].sum(axis=1).rename('total') # sum not-default+default
x1 = pd.concat([x.loc[['F', 'M'],:],x1], axis=1) # concat
x1['pct_default'] = x1['1']/x1['total'] # calculate percentage
x1


# If we see in value (AMT_CREDIT) Female took 120,004,436,385.00 compaire to  64,201,050,310.50 loan taken by Male.  Similar with above analysist (number of borrower) Looking to the percent of defaulted credits, males have a higher chance of not returning their loans (~9.2%), comparing with women (~6.6%).  

# ## 1.2  Who has the highest income Female vs Male ?

# In[41]:


#application_train["log_"+c3] = np.log1p(application_train[c3])

fig, ax =plt.subplots(1,2, figsize=(20,5))
ax[0].set_title("Distribution of " +c3) #0
sns.distplot(application_train[c3], ax=ax[0])

ax[1].set_title("Distribution of log1p("+c3+")") #1
sns.distplot(np.log1p(application_train[c3]), ax=ax[1])


# Becouse AMT_INCOME_TOTAL is not balance, we need to use log1p. On the chart shows that the average income received is 275,140.79  

# In[42]:


# Let's look at the density of the AMT_INCOME_TOTAL of F vs M
temp1 = application_train.loc[application_train[c] != 'XNA'] # exclude XNA

#plt.figure(figsize=(20,10))
sns.FacetGrid(temp1, hue=c, size=6)    .map(sns.kdeplot, "log_"+c3)    .add_legend()
plt.title('KDE Plot of '+c3);


# This is Interesting !!!, Female who had AMT_INCOME_TOTAL less then 162.753 (Low Income) had higger number borrower compaire to Men.  
#    
#    
# You can also look it as:
# * Female salary/AMT_INCOME_TOTAL is less then male
# 
# OR
# 
# * Low income male tend not to borrow money compaire to female
# * High Income male tend to borrow money
# 
# 
# 

# In[47]:


pd.pivot_table(application_train, values=c3, index=[c], aggfunc=[np.count_nonzero, np.mean, np.std, np.min, np.max], margins=True)


# **Average AMT_INCOME_TOTAL by ocupation & gender:**
# * Average AMT_INCOME_TOTAL is 168,797.919 with std 237,122.761
# * AMT_INCOME_TOTAL for Male is 193,396.482 and std 134,597.170
# * AMT_INCOME_TOTAL for Female is 156,032.309 and std 274,825.593
# 
# 
# Female borrower is 2 time more then male, 202.448(F) vs 105.059(M) borrower. but the average income for male (193.396) is 24% higher then female (156.032) .   

# ## Summary:
# * Male 42% more risk then Female
# * Male salary/AMT_INCOME_TOTAL is 24% higher then Female 

# # 2. The Highest and Lowest paying job
# 
# From analysis section 1 we know that Male has higher salary but also has higherst default reate. In this section we will analyst which ocupation that has higher total income (AMT_INCOME_TOTAL) and in section 3 we will analyst ocupation default rate.
# 

# In[53]:


temp1 = application_train

pd.pivot_table(temp1, values=c3, index=[c2, c], #add c to multi index by gender
               aggfunc=[np.count_nonzero, np.mean, np.std, np.min, np.max])#.sort_values(('mean', c3))


# ## 2.1 Number of Borrower per occupation

# In[40]:


# temp1 = application_train.groupby([application_train[c2], application_train[c]]).size().reset_index().rename({0:'count'}, axis=1)
temp1 = application_train
plt.figure(figsize=(20,5))
plt.xticks(rotation=45)

sns.countplot(x=temp1.OCCUPATION_TYPE, hue=temp1.CODE_GENDER, palette="Blues_d")


# ## 2.2 Average income total (AMT_INCOME_TOTA) per ocupation

# In[83]:


temp1 = application_train #[c3].groupby([application_train[c2], application_train[c]]).mean().reset_index()

plt.figure(figsize=(20,5))
plt.xticks(rotation=45)
sns.boxplot(x=temp1.OCCUPATION_TYPE, y=temp1['log_AMT_INCOME_TOTAL'], hue=temp1.CODE_GENDER, palette="Blues_d")


# In[67]:


temp1 = application_train[c3].groupby([application_train[c2], application_train[c]]).mean().reset_index()

plt.figure(figsize=(20,5))
plt.xticks(rotation=45)
sns.barplot(x=temp1.OCCUPATION_TYPE, y=temp1.AMT_INCOME_TOTAL, hue=temp1.CODE_GENDER, palette="Blues_d")


# **Average AMT_INCOME_TOTAL per occupation:**
# * There is 10 occupation have average AMT_INCOME_TOTAL above 200,00, Only 1 female wich is female Manager with 229,609.51
# * Highest AMT_INCOME_TOTAL is M Manager with 296,767.46	and std 205,760.54
# * Lowest AMT_INCOME_TOTAL is F Low-skill Laborers with 121,856.20	and std 52,809.40

# ## 2.3 Credit amount of the loan (AMT_CREDIT) per Occupation
# 
# *. Apakah yang memiliki gaji besar bisa mendapatkan kredit lebih besar

# In[85]:


temp1 = application_train #[c3].groupby([application_train[c2], application_train[c]]).mean().reset_index()

plt.figure(figsize=(20,5))
plt.xticks(rotation=45)
sns.boxplot(x=temp1.OCCUPATION_TYPE, y=temp1['AMT_CREDIT'], hue=temp1.CODE_GENDER, palette="Blues_d")


# In[18]:


#mean of AMT_CREDIT
temp1 = application_train[c1].groupby([application_train[c2], application_train[c]]).mean().reset_index()

plt.figure(figsize=(20,5))
plt.xticks(rotation=45)
sns.barplot(x=temp1.OCCUPATION_TYPE, y=temp1.AMT_CREDIT, hue=temp1.CODE_GENDER, palette="Blues_d")


# **Average AMT_CREDIT per occupation**
# * Mean AMT_CREDIT is 610,301.57
# * mean tertinggi di berikan kepada F manager  (780,908.68), M manager  (770,184.49) and F Accountants (710,616.03	)
# * Low-skill Laborers, Waiters/barmen staff, Cleaning staff in average only borrow < 500,000.00
# * Highest loan given to F & W Manager  (4,050,000.00) and F Accountan  (4,050,000.00)
# * Lowest loan given to XXX (45,000.00)

# In[ ]:


#TODO: Correlation map from above chart


# # 3. What ocupation has the highest default rate
# 
# AMT_TOTAL_INCOME or Total income had high correlation with default rate. Borrower who had total income less then 175,775.17 (average AMT_TOTAL_INCOME) have higher default rate. Low-skill Laborers, Waiters/barmen staff, Cleaning staff has default rate > 10%. Compaire to Manager, High skill tech staff and accountants who has default rate +- 6%. (Pleas see section 2 for analyses of AMT_TOTAL_INCOME)

# In[19]:


#Number of borrower
t1 = application_train[c2].value_counts().rename('t1')
t2 = application_train[c2].loc[(application_train.TARGET == 0)].value_counts().rename('t2') 
t3 = application_train[c2].loc[(application_train.TARGET == 1)].value_counts().rename('t3')

temp1 = pd.concat([t3, t2, t1], axis=1)

temp1['t3_pct'] = temp1.t3/temp1.t1
#temp1['t2_pct'] = temp1.t2/temp1.t1

#del t1;del t2; del t3

temp1.sort_values('t3_pct')


# Percentage of default per ocupation
# * The Highest default rate is Low-skill Laborers with 17% default
# * The Lowest default rate is Accountants with 5% default

# ## 3.1 Ratio of AMT_CREDIT:AMT_INCOME
# Whether that has a big paycheck can earn more credit

# In[18]:


#Ratio of "Total Income" : "Credit loan"
t1 = application_train[c1].groupby(application_train[c2]).mean()#.reset_index()
t2 = application_train[c3].groupby(application_train[c2]).mean()#.reset_index()
r = (t1/t2).rename('Ratio')

temp1 = pd.concat([t1,t2,r], axis=1)
temp1.sort_values('Ratio')


# Ratio of "Total Income" : "Credit loan"

# ## 3.2 Occupation default rate

# **Default rate for Female:**
# * The Highest default rate for Female is Low-skill Laborers	(15 %) 
# * The Lowest default rate for Female is Accountants (4.8%)
# 
# **Default rate for Male:**
# * The Highest default rate for Male is Realty agents & Low-Skill Laborers (17.5%)
# * The Lowest default rate for Male is Accountants & High skill tech staff (6%)

# In[30]:


# gender default/total loan by gender
# * Realty agent spreed is widest 

#top X OCCUPATION_TYPE (default, F)
t1 = application_train[c2].loc[(application_train.CODE_GENDER == 'F') 
                               & (application_train.TARGET == 1)].value_counts().rename('Default')
t1b = application_train[c2].loc[(application_train.CODE_GENDER == 'F') 
                               ].value_counts().rename('Total') #& (application_train.TARGET == 0)
fdr = 'FDefaultRate'
t1_pct = (t1/t1b).rename(fdr)

#top X OCCUPATION_TYPE (default, M)
t2 = application_train[c2].loc[(application_train.CODE_GENDER == 'M') 
                               & (application_train.TARGET == 1)].value_counts().rename('Default')
t2b = application_train[c2].loc[(application_train.CODE_GENDER == 'M') 
                               ].value_counts().rename('Total') #& (application_train.TARGET == 0)
mdr  = 'MDefaultRate'
t2_pct = (t2/t2b).rename(mdr)

temp1 = pd.concat([t1,t1b,t1_pct,t2, t2b, t2_pct], axis=1)
temp1['t_delta'] = abs(temp1[fdr] - temp1[mdr])
temp1.sort_values('t_delta')


# ## Summary
# 
# Ocupation  that has the lowest default rate is Manager, High skill tech staff and Accountants. AMT_TOTAL_INCOME or Total income had high correlation with default rate. High skill tech staff and Accountants has above average AMT_TOTAL_INCOME (175,775.17).

# In[ ]:




