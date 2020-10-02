#!/usr/bin/env python
# coding: utf-8

# Data Analysis to identify the potential customers who have a higher probability of purchasing the loan.

# The case is The Bank has a customers Data with various characteristics of the customers. The management built a new product - Personal Loan, and ran a small campaign towards selling the New Product to their clients. 
# After some time, 9% of customers have Personal Loan from The Bank.
# 
# 
# ### The GOAL IS!
# > - To sell more Personal Loan products to Bank customers.
# > - To devise campaigns to better target marketing to increase the success ratio with a minimal budget.
# > - To identify the potential customers who have a higher probability of purchasing the loan. 
# 
# Increase the success ratio of advertisement campaign while at the same time reduce the cost of the campaign.
# 
# 
# ### The Questions for Analysis
# As soon as we got 9% of customers who bought the Product, we got the following questions:
# 
# > - Is there some associations between personal characteristics and the fact that customer bought the Product? If so:
# >
# > - What are those Main Characteristics that have an association with the Product and what is the strength of the association?
# > - What are the Segments of Main Characteristics, that have a higher strength of association with the Product?
# > - What is the sample of Data with customers from Main Segments?
#  
# 
# ### Approach
# 
# We made the simple step-by-step analysis of customer's characteristics to identify patterns to effectively choose the subset of customers who have a higher probability to buy new product "Personal Loan" from The Bank. 
# <br><br>
# We performed the following steps:
# > - We check all twelve characteristics whether or not each of them has an association with the fact the product been sold.
# > - We find FIVE main characteristics that have higher than moderate strength of association with the product.
# > - We analyze main characteristics and get segments in each with different strength of association with the product.
# > - We tried to make a subset of customers with ideal characteristics who has the highest probability to buy the product. Unfortunately, our dataset does not contain such information. So...
# > - We build a simple algorithm to make a subset of data to get the customers IDs who have a high probability to buy the product.
# 
# ### Technologies
# 
# - Python
# - Pandas
# - Numpy
# - Seaborn
# - Matplotlib

# In[ ]:


import os
import pandas as pd
import numpy as np

#get rid of future warnings with seaborn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


# look at the file name
names = os.listdir('../input');
names


# In[ ]:


#get the path to the file
for name in names:
    if 'xlsx' in name:
        path = '../input/' + name
path


# In[ ]:


master = pd.read_excel('../input/Bank_Personal_Loan_Modelling.xlsx', 'Data')


# In[ ]:


master.head()


# ### Variables definition
# 
# 
# > - **ID** - Customer ID 
# > - **Age** - Customer's age in completed years 
# > - **Experience** - #years of professional experience 
# > - **Income** - Annual income of the customer - in thousands usd 
# > - **ZIPCode** - Home Address ZIP code. 
# > - **Family** - Family size of the customer 
# > - **CCAvg** - Avg. spending on credit cards per month - in thousands usd 
# > - **Education** - Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional 
# > - **Mortgage** - Value of house mortgage if any - in thousands usd  
# > - **Personal Loan**  - Did this customer accept the personal loan offered in the last campaign? 
# > - **Securities Account** - Does the customer have a securities account with the bank? 
# > - **CD Account** - Does the customer have a certificate of deposit (CD) account with the bank? 
# > - **Online** - Does the customer use internet banking facilities? 
# > - **CreditCard** - Does the customer uses a credit card issued by UniversalBank?

# In[ ]:


#for more convinient - reposition "Personal Loan"  column since it is our target column for research
a = master['Personal Loan']
master.drop('Personal Loan', axis = 1, inplace = True)
master['Personal Loan'] = a


# In[ ]:


master.head(1)


# # <br><br>
# 
# # Assess Data

# In[ ]:


df = master.copy()
df.info()


# In[ ]:


df.nunique()


# > **Observation**
# > - No null values
# > - No missing values
# > - Columns "ID", "ZIP Code" are categorical nominal variables. Should be in 'str' type

# In[ ]:


df.describe().transpose()


# > **Observation**
# > - Column "Experience" has some negative value. Need to fix
# > - Binary variables "Personal Loan", "CreditCard", "Online", "CD Account", "Securities Account" has clean data
# > - Ordinary Cat variables "Family" and "Education" are clean too  

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)


# In[ ]:


#cols = ['Experience', 'Mortgage']
fig, [ax0, ax1, ax2] = plt.subplots(1,3, figsize = (14,4))

ax0.hist(df.Mortgage)
ax0.set_xlabel('Mortgage distribution')
ax0.axvline(df.Mortgage.mean(), color = "black")

ax1.hist(df.Experience)
ax1.set_xlabel('Experience distribution')
ax1.axvline(0, color = "black");

ax2.hist(df.Income)
ax2.set_xlabel('Income distribution')
ax2.axvline(df.Income.mean(), color = "black");


# ### Summary Assess Data
# > - Columns "ID", "ZIP Code" are nominal variables. Should be in 'str' type
# > - Column "Experience" has some negative value. Need to fix
# 
# No bad tidiness issues
# 

# # <br>
# 
# ## Clean Data

# #### Define

# Columns "ID", "ZIP Code" are nominal variables

# #### Code

# In[ ]:


df[['ID','ZIP Code']] = df[['ID','ZIP Code']].astype('str')


# #### Test

# In[ ]:



df[['ID', 'ZIP Code']].dtypes


# # <br>
# 
# #### Define

# Column "Experience" has some negative value

# #### Code

# In[ ]:


#check the ammount of negative values
df[df['Experience'] < 0]['Experience'].value_counts()


# ##### Lets find the quantitive variable with strong association with 'Experience'

# In[ ]:


ncol = ['Age', 'Income','CCAvg', 'Mortgage']
grid = sns.PairGrid(df, y_vars = 'Experience', x_vars = ncol, height = 4)
grid.map(sns.regplot);


# 'Age' has a very strong association with 'Experience

# Get the subset of 'Age' data with negative values in 'Experience

# In[ ]:


df[df['Experience'] < 0]['Age'].value_counts()


# **Observation:**
# 
# The subset of each age with negative values in 'Experience' is definitely small. 
# 
# **Decision:**
# We can replace each negative 'Experience' value with the mean of positive 'Experience' value associated with the particular 'Age' value

# <br>
# Get a list of 'Age' values where we found some negative values in 'Experience'

# In[ ]:


ages = df[df['Experience'] < 0]['Age'].unique().tolist()
ages


# Get indexes of negative values in 'Experience'

# In[ ]:


indexes = df[df['Experience'] < 0].index.tolist()


# Replace nagative 'Experience' values with the means

# In[ ]:


for i in indexes:
    for x in ages:
        df.loc[i,'Experience'] = df[(df.Age == x) & (df.Experience > 0)].Experience.mean()


# #### Test

# In[ ]:


df[df['Experience'] < 0]['Age'].value_counts()


# In[ ]:


df.Experience.describe()


# In[ ]:


df.to_csv('df.csv', index = False)


# # <br>
# All Data is clean and we can start Analysis
# <br> 
# 
# # Analysis
# 
# ### Questions
# 
# > - Is there some association between personal characteristics and the fact that person obtained Personal Loan (Loan Fact)? If so:
# > - What are those Main Characteristics that has a higher association with Loan Fact and what the strength of correlation?
# > - What the Segments of Main Characteristics, that has a higher strength of association with  Personal Loan?
# > - What is the sample of Data with persons from Main Segments.

# ## Exploratory data analysis

# ###  Is there some association between personal characteristics and the fact that person obtained Personal Loan?

# Let's check what the values or group of values of each variable lies inside group that have 'Personal Loan' and don't have that.
# 
# Since we found strong association between 'Age' and 'Experience' we decided to exclud 'Experience' from analysis steps to avoid multicollinearity.

# #### QUANTATIVE VARIABLES

# ['Age', 'Income', 'CCAvg', 'Mortgage']

# In[ ]:


quant_df = df[['Personal Loan', 'Age', 'Income', 'CCAvg', 'Mortgage']].copy()


# #### Correlation Table

# In[ ]:


quant_df.corr()


# #### Heat map

# In[ ]:


cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(quant_df.corr(), cmap = cmap, annot = True);


# In[ ]:


# get association coefficients for 'Personal Loan' and exclude it's data from series
quant_df.corr()['Personal Loan'][1:]


# In[ ]:


quant_df.corr()['Personal Loan'][1:].plot.bar();


# **'Age'** and **'Mortgage'** both has very low cor.coef with Personal Loan. We may say that only **'Income'** and **'CCAvg**' has association with 'Personal Loan'
# 
# Let's check our confidense about this statment with logistic regression model:

# In[ ]:


import statsmodels.api as sm


# In[ ]:


quant_df['intercept'] = 1
log_mod = sm.Logit(quant_df['Personal Loan'], quant_df[['intercept', 'Age', 'Income', 'CCAvg', 'Mortgage']]).fit()


# In[ ]:


log_mod.summary()


# #### The bar chart of P-Values distribution and threshold line 

# In[ ]:


# exclude 'intercept'
log_mod.pvalues[1:5].plot.bar()
plt.axhline(y = 0.05);


# **We can say with confidence** that 'Income' and 'CCAvg' both has statisticaly significant association with 'Personal Loan', since  their p-value in logistic regression < 0.05

# #### The bar chart of coefficient distribution 

# In[ ]:


# exclude 'intercept'
log_mod.params[1:5].plot.bar();


# **'CCAvg'** has strongest association with 'Personal Loan'

# #### Filter columns with P-values less then 0.05 and store variables and it's coefficients into the dictionary

# In[ ]:


quant_df_main = {}
for i in log_mod.params[1:5].to_dict().keys():
    if log_mod.pvalues[i] < 0.05:
        quant_df_main[i] = log_mod.params[i]
    else:
        continue


# In[ ]:


quant_df_main


# #### Compute the odds

# In[ ]:


quant_df_main_odds = {k : np.exp(v) for k, v in quant_df_main.items()}


# In[ ]:


quant_df_main_odds


# ### Conclusion:
# 
# 'Personal Loan' has statisticaly significant association with:
# 
# > -  'Income' : coef = 0.03508
# > -  'CCAvg' : coef = 0.06879
# 
# Both variables are positively associated with 'Personal Loan'. As soon as both have one unit as $1000 we may say the following:
# 
# > - **For each $1000 increase in 'Income'** we expect the odds to sell Personal Loan to increase by 3.57%, holding everything else constant
# 
# > - **For each $1000 increase in 'CCAvg'** we expect the odds to sell Personal Loan to increase by 7.12%, holding everything else constant

# <br>
# 
# ### CATEGORICAL VARIABLES
# 
# 'ZIP Code', 'Family', 'Education'

# 'Family' and 'Education' are ordinal categorical variables so we may apply logistic regression direct to them. 'ZIP Code' is nominal, so we need to build dummy variables to check the association existence

# In[ ]:


cat_df = df[['ZIP Code', 'Family', 'Education', 'Personal Loan']].copy()


# <br>
# 
# ### 'Family' and  'Education'

# In[ ]:


cat_df.corr()


# In[ ]:


cat_df.corr()['Personal Loan'][0:2]


# In[ ]:


cat_df.corr()['Personal Loan'][0:2].plot.bar();


# **'Family'** and **'Education'** has low association with 'Personal Loan'
# 
# Let's check our confidence with logistic regretion
# 

# In[ ]:


cat_df['intercept'] = 1
log_mod = sm.Logit(cat_df['Personal Loan'], cat_df[['intercept', 'Family', 'Education']]).fit()


# In[ ]:


log_mod.summary()


# **We can say with confidence** that 'Family' and 'Education' both has statisticaly significant association with 'Personal Loan', since  their p-value in logistic regression < 0.05

# #### The bar chart of coefficient distribution 

# In[ ]:


# exclude 'intercept'
log_mod.params[1:3].plot.bar();


# **'Education'** has strongest association with 'Personal Loan'

# #### Filter columns with P-values less then 0.05 and store variables and it's coefficients into the dictionary

# In[ ]:


cat_df_main = {}
for i in log_mod.params[1:3].to_dict().keys():
    if log_mod.pvalues[i] < 0.05:
        cat_df_main[i] = log_mod.params[i]
    else:
        continue


# In[ ]:


cat_df_main


# #### Compute the odds

# In[ ]:


cat_df_odds = {k : np.exp(v) for k, v in cat_df_main.items()}


# In[ ]:


cat_df_odds


# In[ ]:





# ### Conclusion:
# 
# 'Personal Loan' has statisticaly significant association with:
# 
# > -  'Family' : coef = 0.16231
# > -  'Education' : coef = 0.54873
# 
# Both variables are positively associated with 'Personal Loan'. We may say the following:
# 
# > - **For each unit increase in 'Family'** we expect the odds to sell Personal Loan to increase by 17.62%, holding everything else constant
# 
# > - **For each unit increase in 'Education'** we expect the odds to sell Personal Loan to increase by 73.11%, holding everything else constant

# <br>
# 
# ### 'ZIP Code'

# In[ ]:


cat_df.head()


# In[ ]:


zip_df = cat_df[['Personal Loan', 'intercept','ZIP Code']].copy()


# In[ ]:


zip_df.head(2)


# Lets check how we can group the 'Zip Code' values to minimize the number of dummies

# In[ ]:


zip_df['ZIP Code'].nunique()


# In[ ]:


zip_df['ZIP Code'].str[0:3].nunique()


# In[ ]:


zip_df['ZIP Code'].str[0:2].nunique()


# In[ ]:


zip_df['ZIP Code'].str[0:2].value_counts()


# Guess this set is okay for the first view since we assume that the initial campaign of selling Personal Loans was evenly spreaded through all zip codes.
# 
# Let's get dummies...

# In[ ]:


dum_zip_df = zip_df.copy()


# In[ ]:


dum_zip_df['ZIP Code'] = dum_zip_df['ZIP Code'].str[0:2]


# In[ ]:


dum_zip_df.head(2)


# In[ ]:


dum_zip_df = pd.get_dummies(dum_zip_df, prefix = "Z", drop_first = True)


# In[ ]:


dum_zip_df.head(2)


# Fit a logic model

# In[ ]:


#exclude 'Personal Loan' from independ vars
dum_zip_df_columns = dum_zip_df.columns.drop('Personal Loan').tolist()


# In[ ]:


log_mod = sm.Logit(dum_zip_df['Personal Loan'], dum_zip_df[dum_zip_df_columns]).fit()


# In[ ]:


log_mod.summary()


# **We can say with confidence** that any ZIP Code does not have statisticaly significant association with 'Personal Loan', since  their p-value in logistic regression > 0.05

# # <br>
# 
# ### BINARY VARIABLES
# 
# 'Securities Account', 'CD Account', 'Online', 'Credit Card'

# In[ ]:


bin_df = df[['Personal Loan', 'Securities Account', 'CD Account', 'Online', 'CreditCard']].copy()


# In[ ]:


bin_df.head()


# In[ ]:


bin_df.corr()['Personal Loan']


# In[ ]:


bin_df.corr()['Personal Loan'][1:].plot.bar();


# **'CD Account'** - the only one variable with moderate association

# <br>
# Let's fit logistic regression

# In[ ]:


bin_df['intercept'] = 1
bin_df_colmn = bin_df.columns.drop('Personal Loan').tolist()
log_mod = sm.Logit(bin_df['Personal Loan'], bin_df[bin_df_colmn]).fit()


# In[ ]:


log_mod.summary()


# In[ ]:


log_mod = sm.Logit(bin_df['Personal Loan'], bin_df[['intercept', 'CD Account']]).fit()


# In[ ]:


log_mod.summary()


# In[ ]:


bin_odds = {'CD Account' : np.exp(log_mod.params[1])}


# In[ ]:


bin_odds


# ### Conclusion:
# 
# 'Personal Loan' has statisticaly significant **positive** association with only:
# 
# > -  'CD Account' : coef = 2.40
# 
# We may say the following:
# 
# > - **With customer been hold CD Account with The Bank** we expect the odds to sell Personal Loan to increase 10 times, holding everything else constant

# <br><br>
# 
# ## Summary Conclusion:
# 
# 'Personal Loan' has statisticaly significant association with:
# 
# > -  'CD Account' : coef = 2.40 : odds = 11.07
# > -  'Family' : coef = 0.16231 : odds = 1.176
# > -  'Education' : coef = 0.54873 : odds = 1.731
# > -  'Income' : coef = 0.03508 : odds = 1.036
# > -  'CCAvg' : coef = 0.06879 : odds = 1.071
# 
# Both variables are positively associated with 'Personal Loan'. We may say the following:
# 
# > - **With customer been hold CD Account with The Bank** we expect the odds to sell Personal Loan to increase **11 times**, holding everything else constant
# 
# > - **For each unit increase in 'Family'** we expect the odds to sell Personal Loan to increase **by 17.62%**, holding everything else constant
# 
# > - **For each unit increase in 'Education'** we expect the odds to sell Personal Loan to increase **by 73.11%**, holding everything else constant
# 
# > - **For each $1000 increase in 'Income'** we expect the odds to sell Personal Loan to increase **by 3.57%**, holding everything else constant
# 
# > - **For each $1000 increase in 'CCAvg'** we expect the odds to sell Personal Loan to increase **by 7.12%**, holding everything else constant

# As soon as we found that the 'Personal Loan' depends on FIVE main characteristics, let's subset our data frame and get a closer look at the data.

# # <br>
# 
# #  Explanatory analysis

# >>> ### What are those Main Characteristics that has a higher association  with Loan Fact and what the strength of association ?

# Here is a subset of the initial data frame with just characteristics that have a positive association  with 'Personal Loan' and the size of association  is higher than moderate

# In[ ]:


df = pd.read_csv ('df.csv')


# In[ ]:


df.head(1)


# In[ ]:


exp_df = df[['Income', 'CCAvg', 'Family', 'Education', 'CD Account', 'Personal Loan']].copy()


# In[ ]:


exp_df.head(2)


# Let's apply logistic regression on this subset

# In[ ]:


exp_df['intercept'] = 1


# In[ ]:


log_mod = sm.Logit(exp_df['Personal Loan'], exp_df[['intercept','Income', 'CCAvg', 'Family', 'Education', 'CD Account']]).fit()


# ##### Get P-Values for each variable

# In[ ]:


log_mod.pvalues[1:]


# All p-values are less than 0.05
# <br><br>

# ##### Get Odds for each variable

# In[ ]:


odds = np.exp(log_mod.params)


# In[ ]:


odds


# In[ ]:


odds_df = pd.DataFrame(odds[1:], columns = ["Odds"])


# In[ ]:


odds_df['odds_increment'] = odds_df.Odds


# <br><br>
# ##### Here is the data frame with Main Characteristics ...
# ... and their odds to increase the chance to sell Personal Loan with increase value of variable by one unit

# In[ ]:


odds_df.sort_values('Odds', ascending = False)


# ##### The chart demonstrating the proportion  of strength of association  between Personal Loan and values of Main Characteristics

# In[ ]:


sizes = odds_df.Odds.tolist()# list of sizes of slices
labels = odds_df.index.tolist() # list of labels 
explode = (0.15, 0.1, 0.2, 0.1, 0)  # "explode" the 2nd and 3rd slices  
fig = plt.figure(figsize=(10, 5))
plt.suptitle('The Proportion of Strength of Association  Between  \n Personal Loan and Main Characteristics',           fontsize = 14, y = 1.18)
plt.axis('equal'); # set aspect ration as equal to make sure the pie is drawn as a circle
plt.pie(sizes, labels = labels, explode = explode, radius = 1.5,         shadow = True, startangle = 90,autopct= '%1.1f%%')

plt.savefig('proportion_of_stregth_of_association.png', bbox_inches = 'tight');


# <br><br>
# 
# >>> ##  What the Segments of Main Characteristics, that has a higher strength of association with  Personal Loan?

# Lets get a closer look at each of Main Characteristics

# ### CD Account

# ##### Here is the distribution of "Personal Loan"  values among groups of "CD Account"  values

# In[ ]:


series_cd = exp_df[exp_df['Personal Loan'] == 1]['CD Account'].value_counts()
series_cd


# In[ ]:


series_cdd = exp_df[exp_df['Personal Loan'] == 0]['CD Account'].value_counts()
series_cdd


# In[ ]:


pd.DataFrame(dict( NO_PL= series_cdd, PL= series_cd,)).plot.bar(figsize = (8,6))
plt.ylabel('Frequency')
plt.xticks(np.arange(2),('No CD Account','CD Account'), rotation = 'horizontal')
plt.legend(('NO Personal Loan', 'Personal Loan'));
plt.title('Distribution of "Personal Loan" Values \n among Groups of "CD Account" Values', fontsize = 14, y = 1.05);
plt.savefig('distribution_of_PL_among_CDacc.png', bbox_inches = 'tight')


# We may say that **the proportion** of persons who has Personal Loan among them who has CD account with The Bank **is quit high**.<br>
# Let's see the exact number of proportion of "loanees" among "depositees"

# In[ ]:


series = exp_df[exp_df['CD Account'] == 1]['Personal Loan'].value_counts()


# In[ ]:


plt.axis('equal')
plt.title('Proportion of Customers Who Have Personal Loan and Who Don\'t,\n among CD Account Holders',           fontsize = 14, y = 1.2)
labels = ['NO Personal Loan','Personal Loan']
plt.pie(series, labels = labels,autopct= '%1.1f%%', shadow = True,explode = (0.1, 0), radius = 1.6, startangle = 90)
plt.savefig('Proportion_of_loanees_among_depositees.png', bbox_inches = 'tight');


# **Conclusion**
# 
# > - 46.4% of CD Account Holders have Perconal Loan. 
# > - For 'CD Account' characteristic - the main segment to sell Personal Loan is the people who already have a CD Account with the Bank.
# > - Target value of 'CD Account' variable = 1

# <br>
# 
# ### Education

# ##### Here is the distribution of "Personal Loan" values among groups of  "Education" values 

# In[ ]:


series_ed = exp_df[exp_df['Personal Loan'] == 1]['Education'].value_counts()
series_ed


# In[ ]:


series_edd = exp_df[exp_df['Personal Loan'] == 0]['Education'].value_counts()
series_edd


# In[ ]:


pd.DataFrame(dict(NO_PL= series_edd, PL= series_ed)).plot.bar(figsize = (8,6))
plt.ylabel('Frequency')
plt.xlabel('Education Level')
plt.xticks(np.arange(3),('1','2','3'), rotation = 'horizontal')
plt.legend(('NO Personal Loan', 'Personal Loan'))
plt.title('Distribution of "Personal Loan" Values \n among Groups of "Education" Values', fontsize = 14, y = 1.05);
plt.savefig('distribution_PL_among_Education.png', bbox_inches = 'tight')


# We may say that **the proportion** of persons who has Personal Loan among them who has Third and Second Level of Education **is higher** than proportion among people who has First level of Edication.
# 
# <br>
# Let's see the exact numbers of proportions.

# In[ ]:


series_edu_3 = exp_df[exp_df['Education'] == 3]['Personal Loan'].value_counts()


# In[ ]:


series_edu_2 = exp_df[exp_df['Education'] == 2]['Personal Loan'].value_counts()


# In[ ]:


series_edu_1 = exp_df[exp_df['Education'] == 1]['Personal Loan'].value_counts()


# In[ ]:


labels = ['NO Personal Loan','Personal Loan']
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (18,6),subplot_kw=dict(aspect="equal"))
plt.axis('equal')
ax1.pie(series_edu_3, labels = labels, autopct= '%1.1f%%', shadow = True,explode = (0, 0.1), radius = 1.25, startangle = 90)
ax1.set_title('Education Level 3',fontsize = 14, y = 1.1)

ax2.pie(series_edu_2, labels = labels, autopct= '%1.1f%%', shadow = True,explode = (0, 0.1), radius = 1.25, startangle = 90)
ax2.set_title('Education Level 2', fontsize = 14, y = 1.1)

ax3.pie(series_edu_1, labels = labels, autopct= '%1.1f%%', shadow = True,explode = (0, 0.1), radius = 1.25, startangle = 90);
ax3.set_title('Education Level 1',fontsize = 14, y = 1.1)

plt.suptitle('Proportion of Customers Who Have Personal Loan and Who Don\'t, among CD Account Holders',              fontsize = 16, y = 1.12);

plt.savefig('Proportion_of_PL_among edu_levels.png', bbox_inches = 'tight');


# In[ ]:


series_edu_4 = exp_df[exp_df['Personal Loan'] == 1]['Education'].value_counts()
series_edu_4


# In[ ]:


plt.axis('equal')
plt.title('Proportion of Customers With Different Levels of Education \n among Personal Loan Holders',           fontsize = 14, y = 1.3)
labels = ['Education Level  3',' Education Level 2','Education Level 1']
plt.pie(series_edu_4, labels = labels, autopct= '%1.2f%%', shadow = True,explode = (0.1, 0, 0), radius = 1.6, startangle = 90);
plt.savefig('Proportion_edu_levels_among_PL.png', bbox_inches = 'tight');


# **Conclusion**
# 
# > - 42.7%  and 37.9% of persons who have Personal Loan, have Education level 3 and Level 2 respectively. 
# > - For 'Education' characteristic - the main segments to sell Personal Loan is the people who have Second and Third levels of education
# > - Target values of 'Education' variable are 3 and 2 in descending order of priority

# # <br>
# 
# ## Family

# ##### Here is the distribution of "Personal Loan"  values among groups of "Family"  values

# In[ ]:


series_fam = exp_df[exp_df['Personal Loan'] == 1]['Family'].value_counts()
series_fam


# In[ ]:


series_famm = exp_df[exp_df['Personal Loan'] == 0]['Family'].value_counts()
series_famm


# In[ ]:


pd.DataFrame(dict( NO_PL = series_famm, PL= series_fam,)).plot.bar(figsize = (8,6))
plt.ylabel('Frequency')
plt.xlabel('Family Size')
plt.xticks(np.arange(4),('1', '2', '3', '4'), rotation = 'horizontal')
plt.legend(('NO Personal Loan', 'Personal Loan'));
plt.title('Distribution of "Personal Loan" Values \n among Groups of "Family" Values', fontsize = 14, y = 1.05);
plt.savefig('distribution_of_PL_among_family.png', bbox_inches = 'tight')


# **We may say** that the proportion of persons who has Personal Loan among them who has Family size 2 and 3 is highest proportion.
# Let's see the exact number of that proportions of "loanees" among "depositees"

# In[ ]:


series_fam_3 = exp_df[exp_df['Family'] == 3]['Personal Loan'].value_counts()


# In[ ]:


series_fam_4 = exp_df[exp_df['Family'] == 4]['Personal Loan'].value_counts()


# In[ ]:





# In[ ]:


labels = ['NO Personal Loan','Personal Loan']

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,6),subplot_kw=dict(aspect="equal"))
fig.suptitle('Proportion of Customers Who Have Personal Loan and Who Don\'t, among Different Family Sizes', fontsize = 16, y = 1.1, x = 0.51);

ax1.pie(series_fam_3, labels = labels, autopct= '%1.1f%%', shadow = True,explode = (0, 0.1), radius = 1.25, startangle = 90)
ax1.set_title('Family Size 3',fontsize = 14, y = 1.1)

ax2.pie(series_fam_4, labels = labels, autopct= '%1.1f%%', shadow = True,explode = (0, 0.1), radius = 1.25, startangle = 90)
ax2.set_title('Family Size 4', fontsize = 14, y = 1.1);

plt.savefig('Proportion_of_PL_among_family_levels.png', bbox_inches = 'tight');


# In[ ]:


series_fam


# In[ ]:


plt.axis('equal')
plt.title('Proportion of Customers With Different Family Sizes \n among Personal Loan Holders',           fontsize = 14, y = 1.3)
labels = ['Family 2',' Family 1','Family 3','Family 4']
plt.pie(series_fam.sort_values(ascending = True), labels = labels,         autopct= '%1.2f%%', shadow = True, explode = (0.1, 0.1, 0.1,0.15), radius = 1.6, startangle = 90);
plt.savefig('Proportion_family_size_among_PL.png', bbox_inches = 'tight');


# **Conclusion**
# 
# > - 27.9%  and 27.7% of persons who have Personal Loan, have Family size 4 and Level 3 respectively. 
# > - For 'Family' characteristic - the main segments to sell Personal Loan is the people who have Family Size 3 and 4.
# > - Target values of 'Family' variable are 3 and 4 in descending order of priority, since the proportion of people who has Personal Loan is the higthest with Family Size 3 - 13,2%.

# # <br>
# 
# ## CCAvg

# Here is the distribution of "CCAvg" values among Personal Loan holders and among whole population.

# In[ ]:


series_cca = exp_df[exp_df['Personal Loan'] == 1]['CCAvg'].value_counts()


# In[ ]:


series_cca.describe()


# In[ ]:


width = 1.5 #wdth of bins in histogram - play with it to find good point for groupping
series_cca.plot.hist(bins = np.arange(series_cca.min(), series_cca.max() + width, width ), figsize = (8,6))
plt.xlabel('CCAvg')
plt.axvline(x = series_cca.mean(), color = 'red')
plt.axvline(x = series_cca.min(), color = 'green')
plt.axvline(x = series_cca.mean() + series_cca.std(), color = 'green')
plt.title('Distribution of "CCAvg" values among "Personal Loan" holders', fontsize = 14, y = 1.05);
plt.savefig('Distrib_ccavg_among_PL.png', bbox_inches = 'tight')


# **We may say** that CCAvg characteristics values can be devided in three groups in descending order of priority consider its frequncy among Personal Loan holder:
# 
# 
# > - Group I:  1 < CCAvg < 2.5
# > - Group II:  4 < CCAvg < 5.5
# > - Group III:  7 < CCAvg < 8.5

# In[ ]:


series_ccaa = exp_df['CCAvg'].value_counts()
width = 8.5 #wdth of bins in histogram - play with it to find good point for groupping
series_ccaa.plot.hist(bins = np.arange(series_ccaa.min(), series_ccaa.max() + width, width ), figsize = (8,6))
plt.xlabel('CCAvg')
plt.title('Distribution of "CCAvg" values among whole population', fontsize = 14, y = 1.05);
plt.savefig('Distrib_ccavg_among_population.png', bbox_inches = 'tight')


# **We may say**, that all our groups of 'CCAvg' defined as priority groups to sell Personal Loan, lies inside segment with pretty high frequency among whole population. 

# <br><br>
# **Conclusion**
# 
# > **Target groups of 'CCAvg' characteristic is in descending order of priority:**
#  
# > - Group I:  1 < CCAvg < 2.5
# > - Group II:  4 < CCAvg < 5.5
# > - Group III:  7 < CCAvg < 8.5

# # <br>
# 
# ### Income

# Here is the distribution of "Income" values among Personal Loan holders and among whole population.

# In[ ]:


series_inc = exp_df[exp_df['Personal Loan'] == 1]['Income'].value_counts()


# In[ ]:


series_inc.describe()


# In[ ]:


width = 1.5 #wdth of bins in histogram - play with it to find good point for groupping
series_inc.plot.hist(bins = np.arange(series_inc.min(), series_inc.max() + width, width ), figsize = (8,6))
plt.xlabel('Income')
plt.axvline(x = series_inc.mean(), color = 'red')
plt.axvline(x = series_inc.min(), color = 'green')
plt.axvline(x = series_inc.mean() + series_inc.std(), color = 'green')
plt.title('Distribution of "Income" values among "Personal Loan" holders', fontsize = 14, y = 1.05);
plt.savefig('Distrib_income_among_PL.png', bbox_inches = 'tight')


# **We may say** that 'Income' characteristic values can be devided in three groups in descending order of priority consider its frequncy among Personal Loan holder:
# 
# 
# > - Group I:  1 < Income < 2.5
# > - Group II:  4 < Income < 5.5
# > - Group III:  7 < Income < 8.5

# In[ ]:


series_incc = exp_df['Income'].value_counts()
width = 8.5 #wdth of bins in histogram - play with it to find good point for groupping
series_incc.plot.hist(bins = np.arange(series_incc.min(), series_incc.max() + width, width ), figsize = (8,6))
plt.xlabel('Income')
plt.title('Distribution of "Income" values among whole population', fontsize = 14, y = 1.05);
plt.savefig('Distrib_income_among_population.png', bbox_inches = 'tight')


# **We may say**, that all our groups of 'Income' defined as priority groups to sell Personal Loan, lies inside segment with pretty high frequency among whole population. 

# <br><br>
# **Conclusion**
# 
# > **Target groups of 'Income' characteristic is:**
# 
# > - Group I:  1 < Income < 2.5
# > - Group II:  4 < Income < 5.5
# > - Group III:  7 < Income < 8.5

# # <br> 
# 
# >>> ## What is the sample of Data with persons from Main Segments.

# As we found above, there are some segments of main characteristics with higher strength of association with Personal Loan.
# 
# Let's look at the samples of the database with those segments...

# Here, in one place, are those characteristics segments we found earlier.
# Have a look one more time:
# 
# ### CD Account
# > - 46.4% of CD Account Holders have Perconal Loan. 
# > - For 'CD Account' characteristic - the main segment to sell Personal Loan is the people who already have a CD Account with the Bank.
# > - **Target value of 'CD Account' variable:**
# >> - **1**
# <br>
# ### Education
# > - 42.7%  and 37.9% of persons who have Personal Loan, have Education level 3 and Level 2 respectively. 
# > - For 'Education' characteristic - the main segments to sell Personal Loan is the people who have Second and Third levels of education
# > - **Target values of 'Education' variable in descending order of priority:**
# >> - **3**
# >> - **2**
# <br>
# ### Family
# > - 27.9%  and 27.7% of persons who have Personal Loan, have Family size 4 and Level 3 respectively. 
# > - For 'Family' characteristic - the main segments to sell Personal Loan is the people who have Family Size 3 and 4.
# > - **Target values of 'Family' variable in descending order of priority**, since the proportion of people who has Personal Loan is the higthest with Family Size 3 - 13,2%.
# >> - **3**
# >> - **4**
# <br>
# ### CCAvg
# > **Target groups of 'CCAvg' characteristic is in descending order of priority:**
#  <br>(in thousands usd)
# > - Group I:  1 < CCAvg < 2.5
# > - Group II:  4 < CCAvg < 5.5
# > - Group III:  7 < CCAvg < 8.5
# <br><br>
# ### Income
# > **Target groups of 'Income' characteristic is:**
# <br>(in thousands usd)
# > - Group I:  1 < Income < 2.5
# > - Group II:  4 < Income < 5.5
# > - Group III:  7 < Income < 8.5

# Now we may build subsets of customers step by step with the idea in mind that:
# <br>**the more segments we take into account, the higher probability to sell the product among this customers** 

# Here is the subset with the highest association with Personal Loan, that is, as we remember, mean the highest probability to sell the product.

# CD Account = 1<br>
# Education = 3<br>
# Family = 3<br>
# 1 < CCAvg < 2.5<br>
# 1 < Income < 2.5<br>

# In[ ]:


df[(df['Personal Loan'] == 0) &   (df['CD Account'] == 1) &   (df['Education'] == 3) &   (df['Family'] == 3) &   (df['CCAvg'] > 1) &   (df['CCAvg'] < 2.5) &   (df['Income'] > 1) &   (df['Income'] < 2.5)]


# <br><br>
# **We may say** that this ideal combination of characteristics does not exist among customers in our database.<br><br>
# Lets do step-by-step.

# In[ ]:


CD = df[(df['Personal Loan'] == 0) & (df['CD Account'] == 1)]
CD.shape[0]


# In[ ]:


CD_EDU_3 = df[(df['Personal Loan'] == 0) & (df['CD Account'] == 1)&(df['Education'] == 3)]
CD_EDU_3.shape[0]


# In[ ]:


CD_EDU_3_FM_3 = df[(df['Personal Loan'] == 0) & (df['CD Account'] == 1)&(df['Education'] == 3)&(df['Family'] == 3)]
CD_EDU_3_FM_3.shape[0]


# In[ ]:


CD_EDU_3_FM_4 = df[(df['Personal Loan'] == 0) & (df['CD Account'] == 1)&(df['Education'] == 3)&(df['Family'] == 4)]
CD_EDU_3_FM_4.shape[0]


# In[ ]:


CD_EDU_3_FM_12 = df[(df['Personal Loan'] == 0) & (df['CD Account'] == 1)&(df['Education'] == 3)&(df['Family'] < 3)]
CD_EDU_3_FM_12.shape[0]


# <br><br>
# **Here is the list customer's ID in descending order of probability to sell the product:**
# <br> Letter "A" is a label of higher priority and associated with value "1" in 'CD Account'

# In[ ]:


A_ID = CD_EDU_3_FM_3.ID.tolist()
len(A_ID)


# In[ ]:


A_ID.extend(CD_EDU_3_FM_4.ID.tolist())
len(A_ID)


# In[ ]:


A_ID.extend(CD_EDU_3_FM_12.ID.tolist())
len(A_ID)


# In[ ]:


#check
len(A_ID) == CD_EDU_3.shape[0]


# **"A_ID" is the list of customers ID** who has higher probability to bought the product.<br>
# Total 43 customers.<br><br>
# Use this list to subset data from dataset with customers contacts.
# 
# In case we need next subset with data of customers who has higher probability to bought the product - we may use the same approach.<br><br>

# This is subset of data with customers who has lower probability to bought the product than customers from first subset, but higher probability  among rest population

# In[ ]:


NOCD_EDU_3_FM_3_CCA_1 = df[(df['Personal Loan'] == 0) &   (df['CD Account'] == 0) &   (df['Education'] == 3) &   (df['Family'] == 3)&   (df['CCAvg'] > 1)&   (df['CCAvg'] < 2.5)]


# In[ ]:


NOCD_EDU_3_FM_3_CCA_1.head(3)


# **B_ID - the list of customers ID** to subset from dataset with customers contacts.

# In[ ]:


B_ID = NOCD_EDU_3_FM_3_CCA_1.ID.tolist()
len(B_ID)


# # <br>
# 
# ## Summary Conclusion
# 
# We made the simple step-by-step analysis of customer's characteristics to identify patterns to effectively choose the subset of customers who have a higher probability to buy new product "Personal Loan" from The Bank. We performed the following steps:
# > - We check all twelve characteristics whether or not each of them has an association with the fact the product been sold.
# > - We find FIVE main characteristics that have higher than moderate strength of association with the product.
# > - We analyze main characteristics and get segments in each with different strength of association with the product.
# > - We tried to make a subset of customers with ideal characteristics who has the highest probability to buy the product. Unfortunately, our dataset does not contain such information. So...
# > - We build a simple algorithm to make a subset of data to get the customers IDs who have a high probability to buy the product.
