#!/usr/bin/env python
# coding: utf-8

# # ANZ Customer Transaction Exploratory Data Analysis 

# In[ ]:


# importing the needed libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/anz-synthesised-transaction-dataset/anz.csv')
df.head()


# In[ ]:


# Checking the shape of the dataframe
df.shape


# In[ ]:


# Getting the info of the dataframe
df.info()


# In[ ]:


# Checking how many missing values are there
df.isna().sum()


# Merchant Code and Bpay biller code columns has almost 90% missing values either dropping the columns or ignoring is the best to do. 

# In[ ]:


# Converting the date column to pandas Timestap since this is an Time Series data 
df['date'] = pd.to_datetime(df['date'])


# In[ ]:


# Checking 
type(df['date'][0])


# In[ ]:


df['date'].head(7)


# Now the object are turned to pandas Timestamp object.

# In[ ]:


# By using the date we acquired converting them to days of that particular date
df['day_name'] = df['date'].dt.day_name()
df['day_name'].head()


# Successfully converted those dates to respective day.

# In[ ]:


# Creating Month by using the date gives so can be useful for EDA 
df['month_name'] = df['date'].dt.month_name()
df['month_name'].head()


# In[ ]:


# Checking out available months generated from the date give
df['month_name'].value_counts()


# In[ ]:


# Plotting the correlation heatmap 
cor_mat = df[['card_present_flag' , 'amount' , 'balance' ,'date' , 'status', 
             'bpay_biller_code' , 'account' , 'txn_description',
             'gender' , 'age' , 'extraction']].corr()
# Custom cmap pallete
cmap = sns.diverging_palette(220 , 10 , as_cmap=True)

# Building heatmap
sns.heatmap(cor_mat ,vmax=.3 , center=0 , cmap=cmap , square=True , linewidths=.5 , cbar_kws={'shrink': .5})


# In[ ]:


# Correlation matrix in Tabular form
cor_mat


# Things we can infer from the heatmap : 
# 
# 
# **Considerable Correlation**
# * The amount and balance have a pretty good correlation together
# * The amount and age have a decent correlation which we can consider.
# * The balance and age have a strong correlation.
# 
# **Non-Considerable Correlation**
# * The age and card_present_flag has negative correlation.
# * The amount and car-present_flag has negative correlation.
# 

# ### Filtering things out of Months and analyzing

# In[ ]:


# Checking amount transacted in October month
filt = (df['month_name'] == 'October')
df.loc[filt , 'amount']


# In[ ]:


# Average amount in october month
df.loc[filt , 'amount'].mean()


# In[ ]:


# Maximum Value transacted in October month 
df.loc[filt , 'amount'].max()


# In[ ]:


# Minimum Value transacted in October month 
df.loc[filt , 'amount'].min()


# In October Month : 
# * The average amount transacted was `196.42732321996542`
# * The maximum amount transacted was `8835.98`
# * The minimum amount transacted was `0.1`

# In[ ]:


# Checking amount transacted in September month
filt = (df['month_name'] == 'September')
df.loc[filt , 'amount']


# In[ ]:


# Average amount in september month
df.loc[filt , 'amount'].mean()


# In[ ]:


# Maximum amount in september month
df.loc[filt , 'amount'].max()


# In[ ]:


# Minimum Value transacted in september month 
df.loc[filt , 'amount'].min()


# In September Month : 
# * The average amount transacted was `182.04590331422853`
# * The maximum amount transacted was `8835.98`
# * The minimum amount transacted was `0.1`

# In[ ]:


# Checking amount transacted in August month
filt = (df['month_name'] == 'August')
df.loc[filt , 'amount']


# In[ ]:


# Average amount in august month
df.loc[filt , 'amount'].mean()


# In[ ]:


# Maximum amount in september month
df.loc[filt , 'amount'].max()


# In[ ]:


# Minimum amount in september month
df.loc[filt , 'amount'].min()


# In September Month : 
# * The average amount transacted was `185.12186659903654`
# * The maximum amount transacted was `8835.98`
# * The minimum amount transacted was `1.52`

# In[ ]:


print(df['gender'].value_counts())
plt.figure(figsize=(8,6))
sns.set(style="darkgrid")
sns.countplot(df['gender'])
plt.show()


# * There are more Male customers than Female customers

# In[ ]:


# Month where highest number of transaction took place
sns.countplot(x='month_name' , data=df)


# * October is the month where 4087 transaction took place by all those customers which was highest comparing rest of the months.
# * August has low transaction comparing other months of 3943.

# In[ ]:


# Month where highest number of transaction took place based on gender
plt.figure(figsize=(10,8))
sns.countplot(x='month_name' ,hue='gender', data=df)
plt.title('Month where highest number of\n'+'transaction took place based on gender',bbox={'facecolor':'0.9', 'pad':5})


# * We can clearly infer Male has made more transaction than Female on all three months

# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot(x='day_name' , data=df)


# In[ ]:


plt.figure(figsize=(10,7))
ax = sns.countplot(x="day_name", hue="gender", data=df) # for Seaborn version 0.7 and more
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.2%}'.format(height/total),
            ha="center") 

plt.title('Number of transaction made on each day of\n'+'a week with gender comparison',bbox={'facecolor':'0.9', 'pad':5})
plt.show()


# In[ ]:


plt.figure(figsize=(10,7))
print(df['card_present_flag'].value_counts())
ax = sns.countplot(x='card_present_flag' , data=df)
total = float(len(df['card_present_flag']))
plt.title('Number of customers made transaction\n'+'through a physical card while making purchase\n'+'1.0-Yes 0.0-No',bbox={'facecolor':'0.9', 'pad':5} )
plt.show()


# In[ ]:


print(df['merchant_state'].value_counts())
plt.figure(figsize=(10,7))
sns.countplot(df['merchant_state'])
plt.title('Number of transaction\n' 'done on each state',bbox={'facecolor':'0.9', 'pad':5})
plt.show()


# In[ ]:


print(df['txn_description'].value_counts())
sns.set(style="darkgrid")
plt.figure(figsize=(10,7))
ax = sns.countplot(df['txn_description'])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.2%}'.format(height/total),
            ha="center") 
plt.title('Percentage of Source by where transaction took place')
plt.ylabel('Number of Transaction')
plt.xlabel('Transaction Description')
plt.show()


# In[ ]:


# Distribution of Age of the customers.
plt.figure(figsize=(10,7))
sns.distplot(df['age']);
plt.title('Distribution of customers based on age group' , )


# In[ ]:


# Figuring out which age group has more balance.
plt.figure(figsize=(10,7))
sns.lineplot(x='age' , y='balance' , data=df)


# In[ ]:


# Figuring out which age group has transacted more
plt.figure(figsize=(10,7))
sns.lineplot(x='age' , y='amount' , data=df)


# In[ ]:


# Checking the mean for numerical data in dataframe
df.mean()


# **In the 3 months of transaction data:**
# * The average age group of customers who made the transactions are 30 years of age.
# * The average balance a customer have in their account is 14704.195 AUD.
# * The average transaction made by the customer is 187.93 AUD.

# In[ ]:


# making a group with merchant_state dataframe
mer_state_grp = df.groupby(['merchant_state'])


# In[ ]:


# Number of Male and Female made transaction in the particular merchant state's
print(mer_state_grp['gender'].value_counts(normalize=True))
gen_mer_state = mer_state_grp['gender'].value_counts()
fig,ax = plt.subplots(figsize=(10,10)) # (height,width)
gen_mer_state.plot.barh()
ax.set(xlabel="Number of transaction made",
      ylabel="State and Gender")
plt.title('Number of Male and Female\n'+'made transaction in particular state',bbox={'facecolor':'0.9', 'pad':5})



# ## Percentage of Male and Female made transaction in the particular merchant state
# ### Below are made taking account of Top 5 states in Australia by population
# * **At Tasmania `76.4%` Male made transaction and `23.5%` Female made transaction which shows Male contributed alot to Tasmania.**
# 
# * **At Western Australia Female have made `59.8%` of transaction and Male made `40.2%` of transaction which shows Female contribution is more in WA.**
# 
# * **At Queensland Female have made `51.14%` of transaction and Male made `48.96%` of transaction which shows Female contribution is more in QLD.**
# 
# * **At South Australia Female have made `59.03%` of transaction and Male made `40.96%` of transaction which shows Female contribution is more in SA.**
# 
# * **At New South Wales Male have made `54.82%` of transaction and Male made `45.18%` of transaction which shows Female contribution is more in NSW.**
# 
# * **At Victoria Male have made `56.92%` of transaction and Male made `43.08%` of transaction which shows Female contribution is more in Victoria**
# 

# wa_avg_age = mer_state_grp['age'].value_counts().loc['WA'].mean()
# vic_avg_age = mer_state_grp['age'].value_counts().loc['VIC'].mean()
# qs_avg_age = mer_state_grp['age'].value_counts().loc['QLD'].mean()
# sa_avg_age = mer_state_grp['age'].value_counts().loc['SA'].mean()
# nsw_avg_age = mer_state_grp['age'].value_counts().loc['NSW'].mean()

# In[ ]:


# Number of debit and credit transaction
plt.figure(figsize=(10,7))
print(df['movement'].value_counts())
sns.countplot(df['movement'])


# **We can infer that there were large number of Debit transaction made than Credit transaction**
# * Debit Transaction `11160`
# * Credit Transaction `883`
# 

# In[ ]:


# Which gender made most debit and credit transaction 
plt.figure(figsize=(10,7))
ax = sns.countplot(df['movement'] , hue=df['gender'])
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.2%}'.format(height/total),
            ha="center") 
plt.title('Percentage of Male and Female who made\n'+'Debit and Credit Transaction',bbox={'facecolor':'0.8', 'pad':5})


# ### Percentage of Male and Female who made Debit and Credit Transaction :
# * **Over `48.34%` Male and `44.33%` Female have made Debit Transaction.**
# * **Over `3.85%` Male and `3.48%` Female have made Credit Transaction.**
# 
# **To sum up approx 92% people have done debited transaction and 8% done credited transaction**
# 

# In[ ]:





# In[ ]:


# Percentage of contribution of months
pie_color = ['orange' , 'salmon', 'lightblue']
fig,ax = plt.subplots(figsize=(7,8)) # (height,width)

df['month_name'].value_counts(sort=True).plot.pie(labeldistance=0.2 ,
                                         colors=pie_color,
                                        autopct='%.2f', shadow=True, startangle=140,pctdistance=0.8 , radius=1)
plt.title("Percentage of contribution\n" + "of months", bbox={'facecolor':'0.8', 'pad':5})


# In[ ]:


# Percentage of contribution of gender 
plt.figure(figsize=(10,7))
df['gender'].value_counts(normalize=True).plot.pie(autopct='%.2f',labels=['Male',
                                                                         'Female'], labeldistance=0.5 ,
                                                   shadow=True, startangle=140,pctdistance=0.2 , radius=1)
plt.title('Percentage of contribution\n'+'of Male and Female' , bbox={'facecolor':'0.8', 'pad':5})


# ### Top 10 Customers who made transaction 

# In[ ]:


# Top 10 customers 
top_cust = df['first_name'].value_counts(sort=True).nlargest(10)
top_cust


# **Michael has made more transaction of 746**

# In[ ]:


fig,ax = plt.subplots(figsize=(10,10)) # (height,width)
top_cust.plot.barh(color=my_colors)
ax.set(title="Top 10 Customer",
      xlabel="Number of transaction made",
      ylabel="Name")


# ### Least 10 Customers who made transaction

# In[ ]:


tail_cust = df['first_name'].value_counts(sort=True).nsmallest(10)
tail_cust

#Colors for the bar of the graph
my_colors = ['r','b','k','y','m','c','#16A085','salmon' , '#32e0c4']


# In[ ]:


fig,ax = plt.subplots(figsize=(10,10)) # (height,width)
tail_cust.plot.barh(color=my_colors)
ax.set(title="Least 10 Customer",
      xlabel="Number of transaction made",
      ylabel="Name")


# **Jonathan has made fewer transaction of 31**

# In[ ]:


gender_grp = df.groupby(['gender'])


# In[ ]:


# Average transaction amount made by Male and Female 
gen_trans_amt = gender_grp['amount'].mean()
gen_trans_amt


# In[ ]:


fig,ax = plt.subplots(figsize=(10,8)) # (height,width)
gen_trans_amt.plot.barh(color=my_colors)
ax.set(title="Average amount transacted by Male and Female",
      xlabel="Average amount",
      ylabel="Gender")


# ### Getting the Maximum , Minimum and Average amount transacted in each merchant state

# In[ ]:


agg_amt_state = mer_state_grp['amount'].agg(['min' , 'mean' , 'max'])


# In[ ]:


agg_amt_state.columns


# In[ ]:


agg_amt_state


# In[ ]:


# Minimum ammount transacted in each state
fig,ax = plt.subplots(figsize=(10,8)) # (height,width)
print(agg_amt_state['min'])
agg_amt_state['min'].plot.barh(color=my_colors)
ax.set(title="Minimum amount transacted in each state",
      xlabel="Amount",
      ylabel="Merchant State")


# In[ ]:


# Maximum amount transacted in each state
fig,ax = plt.subplots(figsize=(10,8)) # (height,width)
print(agg_amt_state['max'])
agg_amt_state['max'].plot.barh(color=my_colors)
ax.set(title="Maximum amount transacted in each state",
      xlabel="Amount",
      ylabel="Merchant State")


# In[ ]:


trans_desc_grp = df.groupby(['txn_description'])


# In[ ]:


df['txn_description'].unique()


# In[ ]:


trans_desc_grp['first_name'].value_counts().loc['SALES-POS'].nlargest(10)


# ## Number of transaction made in each state by the Top 5 Customer's

# In[ ]:


# Printing out Top 5 Customer 
top_cust[:5]


# In[ ]:


michael_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Michael').sum())
diana_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Diana').sum())
jess_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Jessica').sum())
jose_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Joseph').sum())
jeff_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Jeffrey').sum())


# ### Number of transaction made by Michael in each state. 

# In[ ]:


fig,ax = plt.subplots(figsize=(10,8))
print(michael_tran_each_state);
michael_tran_each_state.plot.barh(color=my_colors)
ax.set(
    title='Number of transaction made by Michael in each state',
    xlabel='Number of transaction',
    ylabel='Merchant State'
)


# ### Number of transaction made by Diana  in each state

# In[ ]:


fig,ax = plt.subplots(figsize=(10,8))
print(diana_tran_each_state);
diana_tran_each_state.plot.barh(color=my_colors)
ax.set(
    title='Number of transaction made by Diana in each state',
    xlabel='Number of transaction',
    ylabel='Merchant State'
)


# ### Number of transaction made by Jessica in each state

# In[ ]:


fig,ax = plt.subplots(figsize=(10,8))
print(jess_tran_each_state);
jess_tran_each_state.plot.barh(color=my_colors)
ax.set(
    title='Number of transaction made by Jessica in each state',
    xlabel='Number of transaction',
    ylabel='Merchant State'
)


# ### Number of transaction made by Joseph in each state 

# In[ ]:


fig,ax = plt.subplots(figsize=(10,8))
print(jose_tran_each_state);
jose_tran_each_state.plot.barh(color=my_colors)
ax.set(
    title='Number of transaction made by Joseph in each state',
    xlabel='Number of transaction',
    ylabel='Merchant State'
)


# ### Number of transaction made by Jeffrey in each state 

# In[ ]:


fig,ax = plt.subplots(figsize=(10,8))
print(jeff_tran_each_state);
jeff_tran_each_state.plot.barh(color=my_colors)
ax.set(
    title='Number of transaction made by Jeffrey in each state',
    xlabel='Number of transaction',
    ylabel='Merchant State'
)


# ##  How many transactions do customers make each month, on average?

# In[ ]:


month_grp = df.groupby(['month_name'])


# In[ ]:


avg_amt_tran_month = month_grp['amount'].mean()
oct_amt_tran_month = month_grp['amount'].value_counts().loc['October']


# In[ ]:


fig,ax = plt.subplots(figsize=(10,8)) # (height,width)
print(avg_amt_tran_month);
avg_amt_tran_month.plot.barh(color=my_colors)
ax.set(
    title='Average transaction made my customer on average each month',
    xlabel='Average amount',
    ylabel='Month Name '
)


# In[ ]:


oct_amt_tran_month = month_grp['amount'].value_counts().loc['October']
oct_amt_tran_month


# In[ ]:


oct_date = month_grp['date'].value_counts().loc['October']


# ## Average amount transacted on particualr Days

# In[ ]:


day_name_grp = df.groupby(['day_name'])


# ### On Monday 

# In[ ]:


day_name_grp['amount'].mean().loc['Monday']


# ### On Tuesday

# In[ ]:


day_name_grp['amount'].mean().loc['Tuesday']


# ### On Wednesday 

# In[ ]:


day_name_grp['amount'].mean().loc['Wednesday']


# ### On Thursday 

# In[ ]:


day_name_grp['amount'].mean().loc['Thursday']


# ### On Friday

# In[ ]:


day_name_grp['amount'].mean().loc['Friday']


# ### On Satuday 

# In[ ]:


day_name_grp['amount'].mean().loc['Saturday']


# ### On Sunday

# In[ ]:


day_name_grp['amount'].mean().loc['Sunday']


# In[ ]:




