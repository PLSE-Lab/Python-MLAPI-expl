#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
#test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')


# In[ ]:


# Helper functions
# 1. For calculating % na values in  columns
def percent_na(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': percent_missing.index,
                                 'percent_missing': percent_missing.values})
    return missing_value_df
# 2. For plotting grouped histograms 
def sephist(col):
    yes = train_transaction[train_transaction['isFraud'] == 1][col]
    no = train_transaction[train_transaction['isFraud'] == 0][col]
    return yes, no


# In[ ]:


# Helper function for column value details

def column_value_freq(df,sel_col,cum_per):
    dfpercount = pd.DataFrame(columns=['col_name','num_values_99'])
    for col in sel_col:
        col_value = df[col].value_counts(normalize=True)
        colpercount = pd.DataFrame({'value' : col_value.index,'per_count' : col_value.values})
        colpercount['cum_per_count'] = colpercount['per_count'].cumsum()
        if len(colpercount.loc[colpercount['cum_per_count'] < cum_per,] ) < 2:
            num_col_99 = len(colpercount.loc[colpercount['per_count'] > (1- cum_per),])
        else:
            num_col_99 = len(colpercount.loc[colpercount['cum_per_count']< cum_per,] )
        dfpercount=dfpercount.append({'col_name': col,'num_values_99': num_col_99},ignore_index = True)
    dfpercount['unique_values'] = df[sel_col].nunique().values
    dfpercount['unique_value_to_num_values_99_ratio'] = 100 * (dfpercount.num_values_99/dfpercount.unique_values)
    dfpercount['percent_missing'] = percent_na(df[sel_col])['percent_missing'].round(3).values
    return dfpercount

def column_value_details(df,sel_col,cum_per):
    dfpercount = pd.DataFrame(columns=['col_name','values_'+str(round(cum_per,2)),'values_'+str(round(1-cum_per,2))])
    for col in sel_col:
        col_value = df[col].value_counts(normalize=True)
        colpercount = pd.DataFrame({'value' : col_value.index,'per_count' : col_value.values})
        colpercount['cum_per_count'] = colpercount['per_count'].cumsum()
        if len(colpercount.loc[colpercount['cum_per_count'] < cum_per,] ) < 2:
            values_freq = colpercount.loc[colpercount['per_count'] > (1- cum_per),'value'].tolist()
        else:
            values_freq = colpercount.loc[colpercount['cum_per_count']< cum_per,'value'].tolist() 
        values_less_freq =  [item for item in colpercount['value'] if item not in values_freq]
        dfpercount=dfpercount.append({'col_name': col,'values_'+str(round(cum_per,2)) : values_freq ,'values_'+str(round(1-cum_per,2)): values_less_freq},ignore_index = True)
    return dfpercount


# In[ ]:


def col_unique(df,cols):
    dat=df[cols].nunique()
    sns.set(rc={'figure.figsize':(8,4)})
    plot=sns.barplot(x=dat.index,y=dat.values)
    for p in plot.patches:
        plot.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='top', fontsize=12, color='black', xytext=(0, 20),
                 textcoords='offset points')
    plot=plot.set(xlabel='Column ',ylabel= 'Number of unique values')


# Of the 394 features/columns in the train_transaction data 15 columns begin in C .
# The officaila explanation of these columns is.
# 
# *C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.*
# 
# All C columns are of the numeric data type and summary is as below

# In[ ]:


Ccols= train_transaction.columns[train_transaction.columns.str.startswith('C')]
train_transaction[Ccols].describe()


# The graph below shows number of unique values in each of the C Columns as blue bars.Orange bar shows the number of  unique values in 96.5% of the data in each of the columns. The difference between the two bars is a measure of how distributed the data is across the range of unique values in the column. 
# 
# Red line is the percentage of missing values in  the columns.

# In[ ]:


col_freq = column_value_freq(train_transaction,Ccols,0.965)
sns.set(rc={'figure.figsize':(12,8)})
plot=col_freq.plot(x='col_name',y='percent_missing',color='r')
plot.set(ylabel='Percentage of missing values')
ax1=plot.twinx()
#Dcol_freq['percent_missing'].plot(secondary_y=True, color='k', marker='o')
#Dcol_freq['unique_value_to_num_values_99_ratio'].plot(secondary_y=True, color='r', marker='o')
plot1=col_freq.plot(x='col_name',y=['unique_values','num_values_99'],ax=ax1,kind='bar')
for p in plot1.patches[1:]:
    h = p.get_height()
    x = p.get_x()+p.get_width()/2.
    if h != 0:
        plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 
                   textcoords="offset points", ha="center", va="bottom")
plot1.set(ylabel='Count')
plot= plot.set(title='Data Details  in each C columns of train_transaction')


# **Interesting to  note that none of the C columns have missing values**.
# 
# Across the range of values in each of the column  few values make up 96.5% of data in each of the columns compared to total unique values.
# 

# Let's also look at test_transaction data set to verify whether the distribution of values are similiar

# In[ ]:


col_freq = column_value_freq(test_transaction,Ccols,0.965)
sns.set(rc={'figure.figsize':(12,8)})
plot=col_freq.plot(x='col_name',y='percent_missing',color='r')
plot.set(ylabel='Percentage of missing values')
ax1=plot.twinx()
#Dcol_freq['percent_missing'].plot(secondary_y=True, color='k', marker='o')
#Dcol_freq['unique_value_to_num_values_99_ratio'].plot(secondary_y=True, color='r', marker='o')
plot1=col_freq.plot(x='col_name',y=['unique_values','num_values_99'],ax=ax1,kind='bar')
for p in plot1.patches[1:]:
    h = p.get_height()
    x = p.get_x()+p.get_width()/2.
    if h != 0:
        plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 
                   textcoords="offset points", ha="center", va="bottom")
plot1.set(ylabel='Count')
plot= plot.set(title='Data Details  in each C columns of test_transaction')


# Interestingly C13 has more than 90% missing values and C14 has 20% missing values.  These features are potentially candidates to be dropped while building models.

# **Unique Card Identifier**

# It looks a combination of Card =features card1-card 6 and C columns will help us to identify the unique payment cards (cards with unique 15 or 16 digit card numbers).
# 
# After exploring various combiantion of features a combination  of ['card1' ,'card2','card3','card4','card5','card6', 'addr1','C1','C2' ,'C3', 'C4','C5','C6','C7','C8','C9','C10','C11']  shows some interesting patterns

# In[ ]:


# cards=['card1','card2','card3','card4','card5','card6']
# by = cards+['addr1']+Ccols.tolist()
# group1=train_transaction.groupby(by,as_index=False)['TransactionID'].count()
# group1.sort_values(by='TransactionID',ascending=False).head(30)


# In[ ]:


# pd.options.display.max_columns = None
# Dcols =train_transaction.columns[train_transaction.columns.str.startswith('D')]
# select=train_transaction.columns[1:55]
# group1_details=pd.merge(group1,train_transaction[select],on=by,how='right')
# #group1_details.sort_values(by=['TransactionID','TransactionDT'],ascending=False)
# #group1_details[(group1_details.card1==16075) & (group1_details.TransactionID==60)]
# group1_details[(group1_details[['D1','D2','D3']].notnull().all(1)) & (group1_details.TransactionID>30 )].head(5)
# group1_details[(group1_details.card1==1342) & (group1_details.TransactionID==39)]


# In[ ]:


pd.options.display.max_columns = None
by=['card1','card2','card3','card4','card5','card6','addr1','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11']
cards_addr1_Ccolsgroup_count=train_transaction.groupby(by,as_index=False)['TransactionID'].count()
cards_addr1_Ccolsgroup_count.rename(columns={"TransactionID": "Count"},inplace=True)
cards_addr1_Ccolsgroup_count.sort_values(by='Count',ascending=False).head(30)


# In[ ]:


print('Total number of groups: ',len(cards_addr1_Ccolsgroup_count))
print('Average number of transaction per group: ',len(train_transaction)/len(cards_addr1_Ccolsgroup_count))


# Taking as sample details of transactions with card1 =9885

# In[ ]:


cards_addr1_Ccolsgroup_count[cards_addr1_Ccolsgroup_count.card1==9885].sort_values(by='Count',ascending=False).head(30)


# In[ ]:


select=train_transaction.columns[0:55]
cards_addr1_Ccolsgroup_details=pd.merge(cards_addr1_Ccolsgroup_count,train_transaction[select],on=by,how='left')
#cards_addr1_Ccolsgroup_details.sort_values(by=['TransactionID','TransactionDT'],ascending=True)


# In[ ]:


pd.options.display.max_rows = None
cards_addr1_Ccolsgroup_details[(cards_addr1_Ccolsgroup_details.card1==9885) & (cards_addr1_Ccolsgroup_details.Count==64) ]


# From the above data for card  with  card1 = 9885 and number of transactions during the 6 month period = 64 it can easily be seen that D3 column is the difference in number of days for  succesive transaction values of D1 and D2.
# 
# D5 values are almost same as D3 . But where D4 is null D5 is also null which means D5 is the difference in days of successive D4 values
# 
# D1,D2,D4,D11,D15 are days from some card events as their values increase with time. D4 ,D11 and D15 appears to be the same value but D11 has some nulls. D10 values don't follow the time series and need further analysis.
# 
# The increase in the values of D1,D2 ,D4 ,D11 and D15 corresponds with increase in days of the TransactionDT values.
# 
# The difference between the value of D1 between the first and last transaction in this group is 174 days corresponding t a 6 month period.
# 
# from the above this set of values appear to be  transactions of a specific card during the 6 month period.
# 
# **Hence it's safe to assume that the combination of the features 'card1','card2','card3','card4','card5','card6', 'addr1', 'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10'and 'C11' can  uniquely identify the credit card.**

# There seems to be a few cases where the D3 value is noth edifference in days between succesive row. On case is where the d1 value is 236 and D2 value is 185 . D3 is 4 for this row . But the previous row D1& D2s value are 209 and 158. Hence instead of 4 the D3 value should have been 27 . So it looks like a row of values with D1=232 and D2=181 has missed out

# In[ ]:


train_transaction[(train_transaction.card1==9885) & (train_transaction.D2==181.0) ]


# The above row is the missing one and since it had null values for card2-card 6 it was not included in our grouping.
# 
# This throws up the possibility of imputing missing values in card2- card6 based on the feature groupings we identified as payment card identifier.
# 

# **Exploring D Columns further**

# The plot below shows the data distribution in D columns 

# In[ ]:


Dcols= train_transaction.columns[train_transaction.columns.str.startswith('D')]


# In[ ]:


col_freq = column_value_freq(train_transaction,Dcols,0.965)
sns.set(rc={'figure.figsize':(12,8)})
plot=col_freq.plot(x='col_name',y='percent_missing',color='r')
plot.set(ylabel='Percentage of missing values')
ax1=plot.twinx()
#Dcol_freq['percent_missing'].plot(secondary_y=True, color='k', marker='o')
#Dcol_freq['unique_value_to_num_values_99_ratio'].plot(secondary_y=True, color='r', marker='o')
plot1=col_freq.plot(x='col_name',y=['unique_values','num_values_99'],ax=ax1,kind='bar')
for p in plot1.patches[1:]:
    h = p.get_height()
    x = p.get_x()+p.get_width()/2.
    if h != 0:
        plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 
                   textcoords="offset points", ha="center", va="bottom")
plot1.set(ylabel='Count')
plot= plot.set(title='Data Details  in each D columns')


# From the above there is a fair uniform distribution of values in Dcolumns and these are truly numerical columns. histogram of data other than 0 and nulls are shown in the plot below.

# In[ ]:


np.warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize':(14,16)})
for num, alpha in enumerate(Dcols):
    plt.subplot(5,3,num+1)
    yes = train_transaction[(train_transaction['isFraud'] == 1)][alpha]
    no = train_transaction[(train_transaction['isFraud'] == 0) ][alpha]
    plt.hist(yes[yes>0], alpha=0.75, label='Fraud', color='r')
    plt.hist(no[no>0], alpha=0.25, label='Not Fraud', color='g')
    plt.legend(loc='upper right')
    plt.title('Histogram of values  in column ' + str(alpha) )
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# From the above histogram D6,D7,D8,D12,D13,D14 seems to be number of days from some card event date .In any case thes columns have close to 90% null values. D9 is the day fraction of D8. This is confirmed by sample of data below.

# In[ ]:


cards_addr1_Ccolsgroup_details[(cards_addr1_Ccolsgroup_details[['D6','D7']].notnull().all(1)) & (cards_addr1_Ccolsgroup_details.Count >10)].head(30)


# **Exploring C Columns further**

# The table below shows for each column the values that make 96.5% of data in each column(values_0.99) and values that make the remaining 1% data in (values_0.01)

# In[ ]:


pd.options.display.max_colwidth =300
Ccol_details=column_value_details(train_transaction,Ccols,0.965)
num_values_96 =[]
for i in range(len(Ccol_details)):
    num_values_96.append(len(Ccol_details['values_0.96'][i]))
Ccol_details['num_values_96'] = num_values_96
Ccol_details


# In[ ]:


C_cat = Ccol_details[Ccol_details['num_values_96'] <= 15].reset_index()


# ###### For Columns C3,C4 ,C7 ,C8 ,C10 & C12 15 or less values make 96.5% of the column values . These are like categorical values in a sense. 
# 
# The graph below shows a count plot of these categorical values which account for 96.5% values on the left and a histogram of remaining 3.5%  numeric values on the right.

# In[ ]:


sns.set(rc={'figure.figsize':(14,16)})
x=1
for num, alpha in enumerate(C_cat.col_name):
    plt.subplot(len(C_cat),2,x)
    sns.countplot(data=train_transaction[train_transaction[alpha].isin (C_cat['values_0.96'][num])],y=alpha,hue='isFraud')
    plt.legend(loc='lower right',title='is Fraud')
    plt.title('Count of unique values which make 96.5% of data in column ' + str(alpha) )
    plt.subplot(len(C_cat),2,x+1)
    yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (C_cat['values_0.04'][num]))][alpha]
    no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (C_cat['values_0.04'][num]))][alpha]
    plt.hist(yes, alpha=0.75, label='Fraud', color='r')
    plt.hist(no, alpha=0.25, label='Not Fraud', color='g')
    plt.legend(loc='upper right')
    plt.title('Histogram of values which make 3.5% of data in column ' + str(alpha) )
    x= x+2
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# **Curiously almost 30% of values that fall in the 1% of data in these columns except C3 are part of Fraud transactions**

# ###### Let's now examine the remaining columns which are numeric in nature.
# 
# The graph below shows a histogram of  96.5% of column values on the left and a histogram of remaining 3.5%  values on the right.
# 
# 
# 
# 

# In[ ]:


C_num = Ccol_details[Ccol_details['num_values_96'] > 15].reset_index()


# In[ ]:


sns.set(rc={'figure.figsize':(14,16)})
x=1
for num, alpha in enumerate(C_num.col_name):
    plt.subplot(len(C_num),2,x)
    yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (C_num['values_0.96'][num]))][alpha]
    no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (C_num['values_0.96'][num]))][alpha]
    plt.hist(yes, alpha=0.75, label='yes', color='r')
    plt.hist(no, alpha=0.25, label='no', color='b')
    plt.legend(loc='upper right')
    plt.title('Histogram of values which make 96.5% of data in column ' + str(alpha) )
    plt.subplot(len(C_num),2,x+1)
    yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (C_num['values_0.04'][num]))][alpha]
    no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (C_num['values_0.04'][num]))][alpha]
    plt.hist(yes, alpha=0.75, label='Fraud', color='r')
    plt.hist(no, alpha=0.25, label='Not Fraud', color='b')
    plt.title('Histogram of values which make 3.5% of data in column ' + str(alpha) )
    plt.legend(loc='upper right')
    x= x+2
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# There seem to be no major differnce in proportion of fraud transactions between the two types. However it's interesting to note that the majority of values in these columns are in a narrow range of 0-20  in most of the cases except C13, even though maximum value in these most of columns exceed 3000.
