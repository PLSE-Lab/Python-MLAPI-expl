#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)
# 
# **A Brief about LendingClub: **
# 
# **LendingClub** is the first US based peer to peer lending company, headquarter in SAN Francisco, California to register its offerings as securities and exchange commission. It offers loan trading on secondary market. LendingClub enables borrowers to create unsecured personal loans between 1000 and 40000 with standard loan period of 3 years. LendingClub acts like the "bridge" between borrowers and Investors.
# 
# 

# ![image.png](attachment:image.png)
# 
# **Why do they need this analysis?**
# 
# From above working model, it is clear that its very important for LendingClub to know if there is any chance of their borrowers defaulting.

# **Lets now understand the data:**
# 
# Our dataset contains information about all loans issued between 2007 and 2015 including current loan status(Current, late, Fully Paid etc.). It also contains credit scores, number of finance inqueries, zip code, state, collections etc. Lets get Started...  

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import missingno as msno

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# Added Population dataset from Kaggle which will help us perform EDA more comprehensively 

# In[ ]:


loan = pd.read_csv("../input/lending-club-loan-data/loan.csv")
pop = pd.read_csv("../input/population/population_us.csv")
loan.head()


# In[ ]:


pop.head(5)


# In[ ]:


loan.shape


# In[ ]:


loan.isnull().sum()


# In[ ]:


print(loan.isnull().any().value_counts(), "\n")
print(f"The columns that have missing values are total {loan.isnull().any().sum()}")


# **Lets start the treatment of missing values**
# 
# Since we have too many columns, lets find the percentage of missing data in each column and print columns which has more that 40 percent missing data

# In[ ]:


total_num = loan.isnull().sum().sort_values(ascending=False)
perc = loan.isnull().sum()/loan.isnull().count() *100
#perc1 = (round(perc,2).sort_values(ascending=False))

# Creating a data frame:
df_miss = pd.concat([total_num, perc], axis =1 , keys =["Total Missing Values", "Percentage %"]).sort_values(by ="Percentage %", ascending = False)

top_mis = df_miss[df_miss["Percentage %"]>40]
top_mis.reset_index(inplace=True)
top_mis


# Dropping these 45 columns from the dataset and will handle the remaining columns with missing data 

# In[ ]:


list_to_drop = top_mis['index']
loan_copy = loan
loan_copy = loan_copy.drop(list_to_drop,axis=1)


# In[ ]:


loan_copy.shape


# We are left with 99 columns now

# In[ ]:


print(loan_copy.isnull().any().value_counts(), "\n")
print(f"The columns that have missing values are total {loan_copy.isnull().any().sum()}")


# In[ ]:


total_num = loan_copy.isnull().sum().sort_values(ascending=False)
perc = loan_copy.isnull().sum()/loan_copy.isnull().count() *100
perc1 = (round(perc,2).sort_values(ascending=False))

# Creating a data frame:
df_miss_copy = pd.concat([total_num, perc], axis =1 , keys =["Total Missing Values", "Percentage %"]).sort_values(by ="Percentage %", ascending = False)

top_mis_copy = df_miss_copy[df_miss_copy["Percentage %"]>0]
top_mis_copy.reset_index(inplace=True)
top_mis_copy


# Deleting the rows from columns which have less than <2500 missing data points 

# In[ ]:


list_drop_rows = list(top_mis_copy['index'][49:])
list_drop_rows


# In[ ]:


loan_copy = loan_copy.dropna(axis=0,subset=list_drop_rows)


# In[ ]:


print("The dimension of the dataset is {}" .format(loan_copy.shape))
print(loan_copy.isnull().any().value_counts(), "\n")
print(f"The columns that have missing values are total {loan_copy.isnull().any().sum()}")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlist_object = []\nfor i in loan_copy.dtypes.index:  \n    if(loan_copy.dtypes[i] == "object"):\n        print(i + " ", loan_copy.dtypes[i]) \n        list_object.append(i)\n       \nloan_object = loan_copy[list_object]\n\n##Print the names of the columns with percentage with object datatype and having missing data\n\ntotal_num = loan_object.isnull().sum().sort_values(ascending=False)\nperc = loan_object.isnull().sum()/loan_object.isnull().count() *100\nperc1 = (round(perc,2).sort_values(ascending=False))\n\n# Creating a data frame:\ndf_miss_copy_object = pd.concat([total_num, perc1], axis =1 , keys =["Total Missing Values", "Percentage %"]).sort_values(by ="Percentage %", ascending = False)\n\ntop_mis_copy_object = df_miss_copy_object[df_miss_copy_object["Percentage %"]>0]\ntop_mis_copy_object.reset_index(inplace=True)\ntop_mis_copy_object')


# Reference : https://medium.com/ibm-data-science-experience/missing-data-conundrum-exploration-and-imputation-techniques-9f40abe0fd87
# 
# **Imputation Techniques**
# 
# Mean, Median and Mode Imputation
# 
# Imputation with Regression
# 
# k-Neareast Neighbor (kNN) Imputation
# 
# 
# After observing the various unique values in the missing columns, the kNN imputation seems to be the best. 
# The imputed values are obtained by using similarity-based methods that rely on distance metrics (Euclidean distance, Jaccard similarity, Minkowski norm etc). They can be used to predict both discrete and continuous attributes. The main disadvantage of using kNN imputation is that it becomes time-consuming when analyzing large datasets because it searches for similar instances through all the dataset 

# In[ ]:


loan_copy["emp_title"]= loan_copy["emp_title"].fillna(loan_copy["emp_title"].mode()[0])
loan_copy["emp_length"]= loan_copy["emp_length"].fillna(loan_copy["emp_length"].mode()[0])
loan_copy["title"]= loan_copy["title"].fillna(loan_copy["title"].mode()[0])


# In[ ]:


print("The dimension of the dataset is {}" .format(loan_copy.shape))
print(loan_copy.isnull().any().value_counts(), "\n")
print(f"The columns that have missing values are total {loan_copy.isnull().any().sum()}")


# In[ ]:


list_object = []
for i in loan_copy.dtypes.index:  
    if(loan_copy.dtypes[i] == "int64"):
        print(i + " ", loan_copy.dtypes[i]) 
        list_object.append(i)
list_object

print(loan_copy[list_object].isnull().sum())


# There are no missing values in these int type columns

# In[ ]:


list_object_float = []
for i in loan_copy.dtypes.index:  
    if(loan_copy.dtypes[i] == "float64"):
        print(i + " ", loan_copy.dtypes[i]) 
        list_object_float.append(i)
       
loan_object_float = loan_copy[list_object_float]
print(loan_object_float.isnull().sum())

##Print the names of the columns with percentage with object datatype and having missing data

total_num = loan_object_float.isnull().sum().sort_values(ascending=False)
perc = loan_object_float.isnull().sum()/loan_object_float.isnull().count() *100
#perc1 = (round(perc,2).sort_values(ascending=False))

# Creating a data frame:
df_miss_copy_float = pd.concat([total_num, perc], axis =1 , keys =["Total Missing Values", "Percentage %"]).sort_values(by ="Percentage %", ascending = False)

top_mis_copy_float = df_miss_copy_float[df_miss["Percentage %"]>0]
top_mis_copy_float.reset_index(inplace=True)
top_mis_copy_float


# Imputation has to be performed in different columns and the imputation will be performed in sets.
# 

# In[ ]:


list_fill_rows = list(top_mis_copy_float['index'][13:46])
list_fill_rows


# In[ ]:


for i in list_fill_rows:
    print("For column {0} mode is {1}".format(i,loan_copy[i].mode()[0]))
    print("The unique values in column {} are {}  ".format(i,loan_copy[i].value_counts()))


# In[ ]:


list_of_median_impute = ['mths_since_recent_bc','num_rev_accts','num_op_rev_tl','num_rev_tl_bal_gt_0','num_tl_op_past_12m','num_bc_tl','tot_hi_cred_lim','num_il_tl','total_rev_hi_lim','avg_cur_bal','num_actv_bc_tl','mo_sin_rcnt_tl','mo_sin_rcnt_rev_tl_op','mo_sin_old_rev_tl_op',
'num_actv_rev_tl','num_bc_sats','num_sats','acc_open_past_24mths','mort_acc','total_bal_ex_mort','total_bc_limit']


# In[ ]:


list_of_mode_impute = set(list_fill_rows) - set(list_of_median_impute)
list_of_mode_impute= list(list_of_mode_impute)
list_of_mode_impute


# In[ ]:


for i in list_of_mode_impute:
    loan_copy[i]= loan_copy[i].fillna(loan_copy[i].mode()[0])


# In[ ]:


for i in list_of_median_impute:
    loan_copy[i]= loan_copy[i].fillna(loan_copy[i].median())


# In[ ]:


print("The dimension of the dataset is {}" .format(loan_copy.shape))
print(loan_copy.isnull().any().value_counts(), "\n")
print(f"The columns that have missing values are total {loan_copy.isnull().any().sum()}")


# In[ ]:


total_num = loan_copy.isnull().sum().sort_values(ascending=False)
perc = loan_copy.isnull().sum()/loan_copy.isnull().count() *100
perc1 = (round(perc,2).sort_values(ascending=False))

# Creating a data frame:
df_miss_co = pd.concat([total_num, perc1], axis =1 , keys =["Total Missing Values", "Percentage %"]).sort_values(by ="Percentage %", ascending = False)

top_mis_co = df_miss_co[df_miss["Percentage %"]>0]
top_mis_co.reset_index(inplace=True)
top_mis_co


# In[ ]:


last_list_of_empty  = top_mis_co['index'][0:13]
last_list_of_empty


# Mode and Median Imputations have been performed on the data on basis of the "value_counts()" function output

# In[ ]:


for i in last_list_of_empty:
    print("For column {0} mode is {1}".format(i,loan_copy[i].mode()[0]))
    print("The unique values in column {} are {}  ".format(i,loan_copy[i].value_counts()))


# In[ ]:


last_list_of_median_impute = ['inq_fi','open_acc_6m','open_rv_24m','open_rv_12m','open_il_24m','open_act_il','total_cu_tl','open_il_12m',
                              'inq_last_12m','all_util','mths_since_recent_inq']


# In[ ]:


last_set_of_mode_impute = set(last_list_of_empty) - set(last_list_of_median_impute)
last_list_of_mode_impute = list(last_set_of_mode_impute)
last_list_of_mode_impute


# In[ ]:


for i in last_list_of_median_impute:
    loan_copy[i]= loan_copy[i].fillna(loan_copy[i].median())

for i in last_list_of_mode_impute:
    loan_copy[i]= loan_copy[i].fillna(loan_copy[i].mode()[0])


# In[ ]:


print("The dimension of the dataset is {}" .format(loan_copy.shape))
print(loan_copy.isnull().any().value_counts(), "\n")
print(f"The columns that have missing values are total {loan_copy.isnull().any().sum()}")


# In[ ]:


loan_copy.head(5)


# With this the data is clear and the imputations have been performed and the dataset is complete

# **Data Visualization**

# Seaborn 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16,5))

loan_amount = loan_copy["loan_amnt"].values
funded_amount = loan_copy["funded_amnt"].values
investor_funds = loan_copy["funded_amnt_inv"].values


sns.distplot(loan_amount, ax=ax[0], color="#F7522F")
ax[0].set_title("Loan Applied by the Borrower", fontsize=14)
sns.distplot(funded_amount, ax=ax[1], color="#2F8FF7")
ax[1].set_title("Amount Funded by the Lender", fontsize=14)
sns.distplot(investor_funds, ax=ax[2], color="#2EAD46")
ax[2].set_title("Total committed by Investors", fontsize=14)


# D

# In[ ]:


m = loan_copy['loan_status'].value_counts()
m = pd.DataFrame(m)
m.reset_index(level=0, inplace=True)

m['index'][6] = "DNMCP Fully Paid"
m['index'][7] = "DNMCP Charged Off"
m
m.columns = ['Loan Status','Count']
plt.subplots(figsize=(20,8))
sns.barplot(y='Count', x='Loan Status', data=m)
plt.xlabel("Length")
plt.ylabel("Count")
plt.title("Distribution of Loan Status in our Dataset")
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))

plt.subplot(121)
g = sns.distplot(loan_copy["loan_amnt"])
g.set_xlabel("", fontsize=12)
g.set_ylabel("Frequency Dist", fontsize=12)
g.set_title("Frequency Distribuition", fontsize=20)

plt.subplot(122)
g1 = sns.violinplot(y="loan_amnt", data=loan_copy, 
               inner="quartile", palette="hls")
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Amount Dist", fontsize=12)
g1.set_title("Amount Distribuition", fontsize=20)

plt.show()


# In[ ]:


loan_copy['int_round'] = loan_copy['int_rate'].round(0).astype(int)
plt.figure(figsize = (10,8))

#Exploring the Int_rate
plt.subplot(211)
g = sns.distplot(np.log(loan_copy["int_rate"]))
g.set_xlabel("", fontsize=12)
g.set_ylabel("Distribuition", fontsize=12)
g.set_title("Int Rate Log distribuition", fontsize=20)

plt.subplot(212)
g1 = sns.countplot(x="int_round",data=loan_copy, 
                   palette="Set1")
g1.set_xlabel("Int Rate", fontsize=12)
g1.set_ylabel("Count", fontsize=12)
g1.set_title("Int Rate Normal Distribuition", fontsize=20)

plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)

plt.show()


# In[ ]:


plt.figure(figsize = (14,6))
#Looking the count of defaults though the issue_d that is The month which the loan was funded
g = sns.countplot(x='issue_d', data=loan_copy[loan_copy['loan_status'] =='Default'])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Dates", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.legend(loc='upper left')
g.set_title("Analysing Defaults Count by Time", fontsize=20)
plt.show()


# **Charge off rate vs Verification status**
# 
# 
# My hypothesis is that Lending Club tends to take the effort to verify a borrower's income only when it is high. 
# To quickly examine this hypothesis, I look at the charge off rate across each grade depending on whether the income is Verified or not.
# 
# I define that a loan is considered charge-off when the value of loan_status is Charged Off or Default. I can expand it to 90 days behind dues, but it does not make that large of a difference.

# In[ ]:


loan_copy['verified'] = loan_copy['verification_status'] == 'Verified'
grade_yr_loanamnt = pd.pivot_table(loan_copy,index=["grade","verified"], values=['loan_amnt'], aggfunc=np.sum)

grade_yr_loanamnt_default = pd.pivot_table(loan_copy[(loan_copy.loan_status == 'Charged Off') | (loan_copy.loan_status == 'Default')],
                                           index=["grade","verified"], values=['loan_amnt'], aggfunc=np.sum)

grade_yr_loanamnt_default.columns = ['Charged_off']

loan_verified = pd.merge(grade_yr_loanamnt, grade_yr_loanamnt_default, left_index = True, right_index = True)
loan_verified['chargeoff_rate']  = loan_verified['Charged_off'] /  loan_verified['loan_amnt'] 

loan_verified_unstack = loan_verified.unstack("verified")
verified_chargedoff = loan_verified_unstack['chargeoff_rate']
verified_chargedoff.plot()


# Based on the graph, we can safely reject the hypothesis. Loans that have verified income actually has a high level of charge-off. It is reasonable to assume that these loans are often associated with lower level of income, hence higher charge-off.
# 
# The graph also show that the charge-off rate changes almost linearly from grade A (highest grade) through grade G. The charge-off rate for grade F-G is approximately 30%. To compare, this is almost equal to the historic default rate of non-investment grade bond, which is greater than 30% (Wikipedia Credit Rating.)

# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[ ]:



# create loan amount by state column
loan_amnt_by_state = loan_copy.groupby(["addr_state"]).sum()["loan_amnt"]
df_region = loan_amnt_by_state.to_frame()
df_region["loan_amnt"] = df_region["loan_amnt"].map("{:,.0f}".format)
temp = []
for x in df_region["loan_amnt"]:
    a = int(x.replace(',', ''))
    temp.append(a)
df_region["loan_amnt"] = temp

# create number of loan issued by state column
num_issued_loan = loan_copy.groupby(["addr_state"]).count()["loan_amnt"]
df_region["num_issued"] = num_issued_loan

# create average loan amount column
avg_loan_amnt_by_state = []
for a,b in zip(df_region["loan_amnt"], df_region["num_issued"]):
    temp = int(a/b)
    avg_loan_amnt_by_state.append(temp)
df_region["avg_loan_amnt_by_state"] = avg_loan_amnt_by_state


# In[ ]:


df_region_copy = df_region.copy()
addr_state = df_region_copy.index
df_region.index = list(range(1,52))
df_region["addr_state"] = addr_state


# In[ ]:


# population by states from http://worldpopulationreview.com/states/

pop.index = list(range(1,53))
df_region["population"] = pop['Pop']


# In[ ]:


dti = loan_copy.groupby("addr_state").agg([np.mean])["dti"]
dti.columns = ["dti"]
len(dti)


# In[ ]:


d = loan_copy[loan_copy["loan_status"].isin(["Late (16-30 days)","Late (31-120 days)","Default", "Charged Off", "Does not meet the credit policy. Status:Charged Off"])].groupby("addr_state").size()
d = d.to_frame()
e = pd.DataFrame([0],index=["ME"])
f = pd.concat([d,e])
f_copy = f.copy()
addr_state = f_copy.index
f.index = list(range(1,53))
f["addr_state"] = addr_state
f = f.sort_values(by="addr_state")
f.index = list(range(1,53))
df_region["num_default"] = f[0]

# create default_rate column
temp = []
for x, y in zip(df_region["num_default"], df_region["num_issued"].astype(int)):
    if x is not 0 and y is not 0:
        value = (x/y)
        value = "{0:.2f}".format(value)
        value = float(value)
        temp.append(value)
    else:
        temp.append(0)
df_region["default_rate"] = temp

# create average dti by the state
dti = loan_copy.groupby("addr_state").agg([np.mean])["dti"]
dti.columns = ["dti"]
dti.index = list(range(1,52))
df_region = df_region.join(dti)
# plotly color setting
for col in df_region.columns:
    df_region[col] = df_region[col].astype(str)
    scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


# create text column
df_region["text"] = df_region["addr_state"] + '<br>' + "Population: " + df_region["population"] + '<br>' + "Total loan amount ($ USD): " + df_region["loan_amnt"] + "<br>" + "Avg loan amount ($ USD): " + df_region["avg_loan_amnt_by_state"] + '<br>' + "Default rate: " + df_region["default_rate"] + "<br>" + "DTI: " + df_region["dti"]

# setting plotly and deploy the map
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_region['addr_state'],
        z = df_region['avg_loan_amnt_by_state'], 
        locationmode = 'USA-states',
        text = df_region['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "$s USD")
        ) ]

layout = dict(
        title = 'Lending Club Loan<br> Average Loan By State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )


# Due to lack of RAM in this kernel , the further details are available and continued with the dataset in the below link
# 
# Link : https://www.kaggle.com/shubhendra7/lending-club-analysis/
