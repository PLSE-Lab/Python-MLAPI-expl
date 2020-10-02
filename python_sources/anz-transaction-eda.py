#!/usr/bin/env python
# coding: utf-8

# # EDA on the dataset of transaction<br>

# Working on improvement : More types of visualization<br>
# If you would like to collaborate [connect with me](https://www.linkedin.com/in/ashraf-ul/)
# <br><br><br>
# Notebook credit : [Ashik Shafi](https://www.kaggle.com/ashikshafi)<br>
# Dataset used : [ANZ Synthesized Transaction Dataset](https://www.kaggle.com/ashraf1997/anz-synthesised-transaction-dataset)
# 

# ### Content Navigation<a class="anchor" id="100"></a>
# 1. [Month wise transaction analysis](#1) 
#     * Month wise transaction count
#     * Percentage of contribution from each months
#     * Month wise transaction amount
#     * Average transaction amount each month
# 2. [Gender wise transaction analysis](#2)
#     * Gender wise transaction count
#     * Percentage of contribution from gender
#     * Gender wise transaction amount
#     * Average amount of transactions by gender
#     * Month with highest number of transaction based on gender
# 3. [Day wise transaction analysis](#3)
#     * Day wise transaction count
#     * Number of transaction made on each day of a week with gender comparison
# 4. [Age wise transaction analysis](#4)
#     * Distribution of age
#     * Age and balance
#     * Age and amount
#     * Age wise transaction count
#     * Age wise transaction amount
# 5. [Type of transaction](#5)
#     * Percentage of type of transaction
# 6. [State wise transaction analysis](#6)
#     * Number of transaction done on each state
#     * Number of transaction in a state
#     * Minimum Number of transaction in a state
#     * Maximum amount transacted in each state
# 7. [Transaction movement](#7)
#     * Movement type
#     * Transaction movement type by gender
# 8. [Customer analysis](#8)
#     * Top customers
#     * Transaction count by an individual customer in each state
# 9. [Card payment analysis](#9)
#     * Transaction count using physical card
#     * Percentage of card payment
# 10. [Transaction Status](#10)
#     * Percentage of transaction status

# In[ ]:


# importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates


# In[ ]:


# reading file

file_path="../input/anz-synthesised-transaction-dataset/anz.csv"
df=pd.read_csv(file_path)
df.head()


# In[ ]:


# shape of the dataframe

df.shape


# In[ ]:


# info of the dataframe

df.info()


# ### handling missing values

# In[ ]:


# total null values

df.isnull().sum() #or df.isna().sum() 


# In[ ]:


# classifying NA as categorical or numerical 

NA=df[['card_present_flag','bpay_biller_code','merchant_id','merchant_code','merchant_suburb','merchant_state','merchant_long_lat']]
NAcat=NA.select_dtypes(include='object')
NAnum=NA.select_dtypes(exclude='object')
print(NAcat.shape[1],'categorical features with missing values')
print(NAnum.shape[1],'numerical features with missing values')


# In[ ]:


# visulaizing missing values percentage

plt.figure(figsize=(10,5))
allna = (df.isnull().sum() / len(df))*100
allna = allna.drop(allna[allna == 0].index).sort_values()
allna.plot.barh(color=('red', 'black'), edgecolor='black')
plt.title('Missing values percentage per column',bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel('Percentage', fontsize=15)
plt.ylabel('Features with missing values',fontsize=15)
plt.yticks(weight='bold')
plt.show()


# 
# >merchant_code and bpay_biller_code has many null values so either drop or ignore them

# In[ ]:


# removing columns

df.drop(['bpay_biller_code','merchant_code'],axis=1,inplace=True)


# In[ ]:


df.info()


# ### checking correlation between some features

# In[ ]:


# vectorizing categorical values 

# 'status'
df.status = pd.Categorical(df.status)
df['cat_status']=df.status.cat.codes

# 'txn_description'
df.txn_description = pd.Categorical(df.txn_description)
#td_cat=df.txn_description.astype('category').cat.codes
df['cat_txn_description']=df.txn_description.cat.codes

# 'merchant_state'
df.merchant_state = pd.Categorical(df.merchant_state)
df['cat_merchant_state']=df.merchant_state.cat.codes


# In[ ]:


# correlation between some features

cor_mat = df[['card_present_flag' , 'amount' , 'balance' ,'age','cat_status','cat_txn_description','cat_merchant_state']].corr()
cor_mat


# In[ ]:


# visualizing the correlation heatmap 

plt.figure(figsize=(8,8))
# Custom cmap pallete
cmap = sns.diverging_palette(220 , 10 , as_cmap=True)

# Building heatmap
sns.heatmap(cor_mat ,vmax=.3 , center=0 , cmap=cmap , square=True , linewidths=.5 , cbar_kws={'shrink': .5})
plt.title("Correlation between features",bbox={'facecolor':'0.9', 'pad':5})


# In[ ]:


# average of some numerical data

df.mean()


# inference : In the 3 months of transaction data:<br>
# 
# * The average age group of customers who made the transactions are 30 years of age.
# * The average balance a customer have in their account is 14704.195 AUD.
# * The average transaction made by the customer is 187.93 AUD.

# In[ ]:


#Colors for the bar of the graph

my_colors = ['r','b','k','y','m','c','#16A085','salmon' , '#32e0c4']


# ### Transaction Trend

# In[ ]:


# visualize transaction trend

tt=df.groupby(['date'])
ttc=tt.date.count()

plt.figure(figsize=(20,8))
sns.lineplot(data=ttc)
plt.title("Transaction trend",bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel("Date")
plt.ylabel("Number of transactions")


# In[ ]:


# average number of transactions per day

ttc.mean()


# # 1. Month analysis
# <a class="anchor" id="1"></a>[Back to navigation](#100)

# ### day and month extraction from date column

# In[ ]:


# converting the date column to pandas Timestap

df['date'] = pd.to_datetime(df['date'])


# In[ ]:


# extracting day name 

df['day_name'] = df['date'].dt.day_name()
df['day_name'].head()


# In[ ]:


# extracting month name

df['month_name'] = df['date'].dt.month_name()
df['month_name'].head()


# In[ ]:


# months generated

df['month_name'].value_counts()


# In[ ]:


# visualize month wise transaction count

plt.figure(figsize=(15,5))
plt.title("Month wise transaction count",bbox={'facecolor':'0.9', 'pad':5})
sns.countplot(x='month_name' , data=df)
plt.ylabel("Count",fontsize=15)
plt.xlabel("Month",fontsize=15)


# >August, September and October has higher number of transactions

# In[ ]:


# visualize percentage of contribution from each month

pie_color = ['orange' , 'salmon', 'lightblue']
fig,ax = plt.subplots(figsize=(10,10)) # (height,width)

df['month_name'].value_counts(sort=True).plot.pie(labeldistance=0.2 ,
                                         colors=pie_color,
                                        autopct='%.2f', shadow=True, startangle=140,pctdistance=0.8 , radius=1)
plt.title("Percentage of contribution from each months", bbox={'facecolor':'0.8', 'pad':5})


# In[ ]:


# month wise transaction amount

month_amount=df.groupby(['month_name']).amount.agg([sum])
ma=month_amount.sort_values(by='sum',ascending=False)
ma


# In[ ]:


# visualize month wise transaction amount

plt.figure(figsize = (10,5))
df.groupby('month_name').amount.sum().plot(kind='bar')
plt.title("Month wise transaction amount",bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Amount",fontsize=15)
plt.xlabel("Month",fontsize=15)


# >August, September and October has higher amount of transaction

# In[ ]:


# visualize average transaction amount each month

month_grp = df.groupby(['month_name'])
avg_amt_tran_month = month_grp['amount'].mean()

fig,ax = plt.subplots(figsize=(10,5)) # (height,width)
print(avg_amt_tran_month);
avg_amt_tran_month.plot.barh(color=my_colors)
ax.set(xlabel="Average amount",
      ylabel="Month")
plt.title('Average transaction amount each month',bbox={'facecolor':'0.9', 'pad':5})


# In[ ]:


oct_amt_tran_month = month_grp['amount'].value_counts().loc['October']
oct_amt_tran_month


# In[ ]:


oct_date = month_grp['date'].value_counts().loc['October']


# In[ ]:


# amount transacted in October month

filt = (df['month_name'] == 'October')
df.loc[filt , 'amount']


# In[ ]:


# average amount in october month

df.loc[filt , 'amount'].mean()


# In[ ]:


# maximum value transacted in October month 

df.loc[filt , 'amount'].max()


# In[ ]:


# minimum value transacted in October month 

df.loc[filt , 'amount'].min()


# # 2. Gender wise transaction analysis
# <a class="anchor" id="2"></a>[Back to navigation](#100)

# In[ ]:


# gender wise transaction count

gencg=df.groupby('gender').gender.count()
gencs=gencg.sort_values(ascending=False)
gencs


# In[ ]:



# visualize gender wise transaction count

plt.figure(figsize = (10,5))
ax = sns.countplot(x = 'gender', data = df, palette = 'pastel')
ax.set_title(label = 'Gender wise transaction count',bbox={'facecolor':'0.9', 'pad':5})
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
ax.set_xlabel(xlabel = 'Gender', fontsize = 16)
plt.legend()


# >male customers with higher number of transactions

# In[ ]:


# visualize percentage of contribution from each gender 

plt.figure(figsize=(5,5))
df['gender'].value_counts(normalize=True).plot.pie(autopct='%.2f',labels=['Male',
                                                                         'Female'], labeldistance=0.5 ,
                                                   shadow=True, startangle=140,pctdistance=0.2 , radius=1)
plt.title('Percentage of contribution from gender' , bbox={'facecolor':'0.8', 'pad':5})


# In[ ]:


# gender wise transaction amount

genag=df.groupby(['gender']).amount.agg([sum])
genag


# In[ ]:


# average transaction amount by gender 

gender_grp = df.groupby(['gender'])
gen_trans_amt = gender_grp['amount'].mean()
gen_trans_amt


# In[ ]:


# total transaction amount by gender 

gender_total = df.groupby(['gender'])
gen_total_amt = gender_grp['amount'].sum()
gen_total_amt


# In[ ]:


# visualize gender wise transaction amount

fig,ax = plt.subplots(figsize=(10,5))
gen_total_amt.plot.barh(color=my_colors)
plt.title("Gender wise transaction amount",bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Gender",fontsize=15)
plt.xlabel("Amount",fontsize=15)


# In[ ]:


# visualize average amount transacted by gender

fig,ax = plt.subplots(figsize=(10,5))
gen_trans_amt.plot.barh(color=my_colors)
plt.title("Average amount of transactions by gender",bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Gender",fontsize=15)
plt.xlabel("Amount",fontsize=15)


# > male customers does more amount of transaction than female

# In[ ]:


# visualize month with highest number of transaction based on gender

plt.figure(figsize=(20,6))
sns.countplot(x='month_name' ,hue='gender', data=df)
plt.title('Month with highest number of transaction based on gender',bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel("Month",fontsize=15)
plt.ylabel("Count",fontsize=15)


# >Male has made more transaction count than Female on all three months

# # 3. Day wise transaction analysis
# <a class="anchor" id="3"></a>[Back to navigation](#100)

# In[ ]:


# visualize day wise transaction count

plt.figure(figsize=(10,5))
sns.countplot(x='day_name' , data=df)
plt.title("Day wise transaction count",bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Count",fontsize=15)
plt.xlabel("Day",fontsize=15)


# > Friday and wednesday most number of transactions

# In[ ]:


# average amount transacted on particular Day : Monday

day_name_grp = df.groupby(['day_name'])
day_name_grp['amount'].mean().loc['Monday']


# In[ ]:


# visualize day wise gender transaction count

plt.figure(figsize=(15,7))
ax = sns.countplot(x="day_name", hue="gender", data=df) # for Seaborn version 0.7 and more
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.2%}'.format(height/total),
            ha="center") 
plt.ylabel("Count",fontsize=15)
plt.xlabel("Day",fontsize=15)
plt.title('Number of transaction made on each day of a week with gender comparison',bbox={'facecolor':'0.9', 'pad':5})
plt.show()


# # 4. Age wise transaction analysis
# <a class="anchor" id="4"></a>[Back to navigation](#100)

# In[ ]:


# visualize distribution of age

plt.figure(figsize=(15,5))
sns.distplot(df['age']);
plt.title('Distribution of age',bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel('Age',fontsize=15)


# In[ ]:


# visualize age with balance

plt.figure(figsize=(15,5))
sns.lineplot(x='age' , y='balance' , data=df)
plt.title('Age and balance',bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel('Age',fontsize=15)
plt.ylabel('Balance',fontsize=15)


# In[ ]:


# visualize age with amount

plt.figure(figsize=(15,5))
sns.lineplot(x='age' , y='amount' , data=df)
plt.title('Age and amount',bbox={'facecolor':'0.9', 'pad':5})
plt.xlabel('Age',fontsize=15)
plt.ylabel('Amount',fontsize=15)


# In[ ]:


# age wise transaction count

agecg=df.groupby('age').age.count()
#agecs=agecg.sort_values(ascending=False)
agecg


# In[ ]:


# visualize age wise transaction count

plt.figure(figsize = (10, 5))
ax = sns.countplot(x = 'age', data = df, palette = 'pastel')
ax.set_title(label = 'Age wise transaction count',bbox={'facecolor':'0.9', 'pad':5})
ax.set_ylabel(ylabel = 'Count', fontsize = 15)
ax.set_xlabel(xlabel = 'Age', fontsize = 15)
plt.show()


# In[ ]:


# age wise transaction amount

ageag=df.groupby(['age']).amount.agg([sum])
ageag


# In[ ]:


# visualize age wise transaction amount

plt.figure(figsize = (10,5))
df.groupby('age').amount.sum().plot(kind='bar')
plt.title("Age wise transaction amount",bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Total amount",fontsize=15)
plt.xlabel("Age",fontsize=15)


# # 5. Type of transaction
# <a class="anchor" id="5"></a>[Back to navigation](#100)

# In[ ]:


# visualize type of transaction

print(df['txn_description'].value_counts())
sns.set(style="darkgrid")
plt.figure(figsize=(10,5))
ax = sns.countplot(df['txn_description'])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.2%}'.format(height/total),
            ha="center") 
plt.title('Percentage of type of transaction',bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel('Number of Transaction',fontsize=15)
plt.xlabel('Transaction Description',fontsize=15)
plt.show()


# > Most of the transactione are POS and SALES-POS

# # 6. State wise transaction analysis
# <a class="anchor" id="6"></a>[Back to navigation](#100)

# In[ ]:


# visualize state wise transaction count

print(df['merchant_state'].value_counts())
plt.figure(figsize=(10,5))
sns.countplot(df['merchant_state'])
plt.title('Number of transaction done on each state',bbox={'facecolor':'0.9', 'pad':5})
plt.ylabel("Count",fontsize=15)
plt.xlabel("State",fontsize=15)
plt.show()


# >Most number of transactions taking place in NSW and VLC and least number of transactions in ACT and TAS

# In[ ]:


# group using merchant_state 

mer_state_grp = df.groupby(['merchant_state'])


# In[ ]:


# visualize number of transaction in merchant state by gender

print(mer_state_grp['gender'].value_counts(normalize=True))
gen_mer_state = mer_state_grp['gender'].value_counts()
fig,ax = plt.subplots(figsize=(15,5))
gen_mer_state.plot.barh()
ax.set(xlabel="Number of transaction",
      ylabel="State and Gender")
plt.title('Number of transaction in a state',bbox={'facecolor':'0.9', 'pad':5})


# Percentage of Male and Female made transaction in the particular merchant state<br>
# Below are made taking account of Top 5 states in Australia by population<br>
# * At Tasmania 76.4% Male made transaction and 23.5% Female made transaction which shows Male contributed alot to Tasmania.
# 
# * At Western Australia Female have made 59.8% of transaction and Male made 40.2% of transaction which shows Female contribution is more in WA.
# 
# * At Queensland Female have made 51.14% of transaction and Male made 48.96% of transaction which shows Female contribution is more in QLD.
# 
# * At South Australia Female have made 59.03% of transaction and Male made 40.96% of transaction which shows Female contribution is more in SA.
# 
# * At New South Wales Male have made 54.82% of transaction and Male made 45.18% of transaction which shows Female contribution is more in NSW.
# 
# * At Victoria Male have made 56.92% of transaction and Male made 43.08% of transaction which shows Female contribution is more in Victoria

# In[ ]:


# maximum,minimum and average amount transacted in each merchant state

agg_amt_state = mer_state_grp['amount'].agg(['min' , 'mean' , 'max'])
agg_amt_state


# In[ ]:


# visualize minimum amount transacted in each state

fig,ax = plt.subplots(figsize=(15,5)) # (height,width)
print(agg_amt_state['min'])
agg_amt_state['min'].plot.barh(color=my_colors)
ax.set(xlabel="Number of transaction",
      ylabel="State")
plt.title('Minimum Number of transaction in a state',bbox={'facecolor':'0.9', 'pad':5})


# In[ ]:


# visualize maximum amount transacted in each state

fig,ax = plt.subplots(figsize=(15,5)) # (height,width)
print(agg_amt_state['max'])
agg_amt_state['max'].plot.barh(color=my_colors)
ax.set(xlabel="Amount",
      ylabel="State")
plt.title('Maximum amount transacted in each state',bbox={'facecolor':'0.9', 'pad':5})


# # 7. Transaction movement
# <a class="anchor" id="7"></a>[Back to navigation](#100)

# In[ ]:


# visualize movement type

plt.figure(figsize=(10,5))
print(df['movement'].value_counts())
sns.countplot(df['movement'])
ax.set(xlabel="Movement",
      ylabel="Count")
plt.title('Movement type',bbox={'facecolor':'0.9', 'pad':5})


# We can infer that there were large number of Debit transaction made than Credit transaction
# 
# * Debit Transaction 11160
# * Credit Transaction 883

# In[ ]:


# visualize transaction movement by gender

plt.figure(figsize=(10,5))
ax = sns.countplot(df['movement'] , hue=df['gender'])
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.2%}'.format(height/total),
            ha="center") 
plt.title('Transaction movement type by gender',bbox={'facecolor':'0.8', 'pad':5})


# Percentage of Male and Female who made Debit and Credit Transaction :<br>
# * Over 48.34% Male and 44.33% Female have made Debit Transaction.
# * Over 3.85% Male and 3.48% Female have made Credit Transaction.
# * To sum up approx 92% people have done debited transaction and 8% done credited transaction

# # 8. Customer analysis
# 
# <a class="anchor" id="8"></a>[Back to navigation](#100)

# In[ ]:


#employee dataframe
#df_emp = df[~df['txn_description'].isin(['POS','SALES-POS','PAYMENT','INTER BANK','PHONE BANK'])]
#df_emp.head(10)


# In[ ]:


# customer dataframe

df_cus = df[~df['txn_description'].isin(['PAY/SALARY'])]
df_cus.head()


# In[ ]:


# customers with highest transaction count 

top_customers = df_cus['first_name'].value_counts(sort=True).nlargest(20)
top_customers


# In[ ]:


# visualize top customers

fig,ax = plt.subplots(figsize=(20,6)) # (height,width)
top_customers.plot.barh(color=my_colors)
ax.set(xlabel="Number of transactions",
      ylabel="Name")
plt.title('Top customers',bbox={'facecolor':'0.9', 'pad':5})


# >These are the most valued customers

# ### individual customer analysis : Michael

# In[ ]:


michael_tran_each_state = mer_state_grp['first_name'].apply(lambda x: x.str.contains('Michael').sum())


# In[ ]:


# visualize transaction count by Michael in each state

fig,ax = plt.subplots(figsize=(20,6))
print(michael_tran_each_state);
michael_tran_each_state.plot.barh(color=my_colors)
ax.set(xlabel="Number of transaction",
      ylabel="Merchant State")
plt.title('Transaction count by an individual customer in each state',bbox={'facecolor':'0.9', 'pad':5})


# # 9. Card payment analysis
# <a class="anchor" id="9"></a>[Back to navigation](#100)

# In[ ]:


# visualize number of transaction by card

plt.figure(figsize=(10,5))
print(df['card_present_flag'].value_counts())
ax = sns.countplot(x='card_present_flag' , data=df)
total = float(len(df['card_present_flag']))
plt.xlabel("Card or No-card")
plt.ylabel("Count")
plt.title('Transaction count using physical card \n'+'0.0-No 1.0-Yes',bbox={'facecolor':'0.9', 'pad':5} )
plt.show()


# >Card is used for most of the transactions

# In[ ]:


# visualize transaction by card 

plt.figure(figsize=(10,7))
df['card_present_flag'].value_counts(normalize=True).plot.pie(autopct='%.2f',labels=['Card',
                                                                         'Non-card'], labeldistance=0.5 ,
                                                   shadow=True, startangle=140,pctdistance=0.2 , radius=1)
plt.title('Percentage of card payment' , bbox={'facecolor':'0.8', 'pad':5})


# >80% of the payment is done using physical card

# # 10.Transaction status 
# <a class="anchor" id="10"></a>[Back to navigation](#100)

# An authorized transaction is a debit or credit card purchase for which the merchant has received approval from the bank that issued the customer's payment card.<br>
# A posted transaction is a debit or credit that has been fully processed. Once a transaction is posted the account balance on the account is also updated.

# In[ ]:


# visualize transaction status

plt.figure(figsize=(10,7))
df['status'].value_counts(normalize=True).plot.pie(autopct='%.2f',labels=['authorized',
                                                                         'posted'], labeldistance=0.5 ,
                                                   shadow=True, startangle=140,pctdistance=0.2 , radius=1)
plt.title('Percentage of transaction status' , bbox={'facecolor':'0.8', 'pad':5})

