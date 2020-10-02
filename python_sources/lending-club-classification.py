#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/lending-club-loan/loan.csv',encoding='latin-1',sep=',',names=['id','member_id','loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','installment','grade','sub_grade','emp_title',	'emp_length','home_ownership','annual_inc','verification_status','issue_d','loan_status','pymnt_plan','url','desc','purpose','title','zip_code','addr_state','dti','delinq_2yrs','earliest_cr_line','inq_last_6mths',	'mths_since_last_delinq',	'mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','initial_list_status','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee',	'last_pymnt_d','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d','collections_12_mths_ex_med',	'mths_since_last_major_derog','policy_code','application_type','annual_inc_joint','dti_joint','verification_status_joint','acc_now_delinq','tot_coll_amt','tot_cur_bal','open_acc_6m','open_act_il','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc','all_util','total_rev_hi_lim','inq_fi','total_cu_tl','inq_last_12m',	'acc_open_past_24mths',	'avg_cur_bal','bc_open_to_buy','bc_util','chargeoff_within_12_mths','delinq_amnt','mo_sin_old_il_acct','mo_sin_old_rev_tl_op',	'mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mort_acc','mths_since_recent_bc','mths_since_recent_bc_dlq','mths_since_recent_inq','mths_since_recent_revol_delinq','num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl','num_bc_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats',	'num_tl_120dpd_2m',	'num_tl_30dpd',	'num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq','percent_bc_gt_75','pub_rec_bankruptcies','tax_liens','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit','total_il_high_credit_limit','revol_bal_joint','sec_app_earliest_cr_line',	'sec_app_inq_last_6mths',	'sec_app_mort_acc','sec_app_open_acc','sec_app_revol_util','sec_app_open_act_il','sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med','sec_app_mths_since_last_major_derog','hardship_flag','hardship_type','hardship_reason','hardship_status','deferral_term','hardship_amount','hardship_start_date','hardship_end_date','payment_plan_start_date','hardship_length','hardship_dpd',	'hardship_loan_status','orig_projected_additional_accrued_interest','hardship_payoff_balance_amount','hardship_last_payment_amount','debt_settlement_flag','debt_settlement_flag_date','settlement_status','settlement_date','settlement_amount','settlement_percentage','settlement_term']) 


# In[ ]:


df.shape


# In[ ]:


pd.set_option('display.max_columns',150)
df.head()


# In[ ]:


df.tail()


# # Data Cleaning

# In[ ]:


df1=df.drop(['id','member_id'],axis=1)


# In[ ]:


df1.head()


# In[ ]:


df1.tail()


# In[ ]:


df1.shape


# In[ ]:


#Dropping rows with all NaN values
df1.dropna(how='all',inplace=True,axis=0)


# In[ ]:


df1.reset_index(drop=True,inplace=True)


# In[ ]:


df1.shape


# In[ ]:


df1.head()


# In[ ]:


df1.drop(index=[0],axis=0,inplace=True)


# In[ ]:


df1.reset_index(drop=True,inplace=True)


# In[ ]:


df1.shape


# In[ ]:


df1.head()


# In[ ]:


df1.tail()


# In[ ]:


#Dropping the columns with all NaN values
df1.dropna(how='all',axis=1,inplace=True)


# In[ ]:


df1.shape


# In[ ]:


#df1.to_csv('LendingClub.csv')


# In[ ]:


pd.set_option('display.max_rows',150)
df1.isnull().sum()


# In[ ]:


colnull=[]
for i in df1.columns:
    if df1[i].isnull().sum()>0:
        colnull.append(i)


# In[ ]:


#df1 columns list with null values
colnull


# In[ ]:


df1[df1.purpose=='moving']['title']


# In[ ]:


#Dropping columns with more than 70 percent null values
df2 = df1[[column for column in df1 if df1[column].count() / len(df1) >= 0.3]]
print("List of dropped columns:", end=" ")
for c in df1.columns:
    if c not in df2.columns:
        print(c, end=", ")


# In[ ]:


df2.shape


# In[ ]:


#df2=df1.drop(columns=['debt_settlement_flag_date','settlement_status', 'settlement_date', 'settlement_amount','settlement_percentage', 'settlement_term','next_pymnt_d','mths_since_last_record'],axis=1)


# In[ ]:


#df2.shape


# In[ ]:


df2.info()


# In[ ]:


df2.columns


# In[ ]:


#Checking for columns with only one unique value


# In[ ]:


for i in df2.columns:
    print('For {x}'.format(x=i))
    print(df2[i].nunique())
    print()


# In[ ]:


#Dropping columns with one unique value
df3 = df2[[column for column in df2 if df1[column].nunique()>1]]
print("List of dropped columns:", end=" ")
for c in df2.columns:
    if c not in df3.columns:
        print(c, end=", ")


# In[ ]:


for i in df3.columns:
    print('For {x}'.format(x=i))
    print(df3[i].unique())
    print()


# In[ ]:


#Deleting highly imbalanced columns
df3.tax_liens.value_counts()


# In[ ]:


df3.pub_rec_bankruptcies.value_counts()


# In[ ]:


df3.pub_rec_bankruptcies.replace({'0':0,'1':1,'2':2},inplace=True)


# In[ ]:


df3.pub_rec_bankruptcies.value_counts()


# In[ ]:


df3.delinq_amnt.value_counts()


# In[ ]:


df3.debt_settlement_flag.value_counts()


# In[ ]:


df3.acc_now_delinq.value_counts()


# In[ ]:


# Dropping future events and highly imbalanced columns
df3.drop(columns=['out_prncp','out_prncp_inv','collections_12_mths_ex_med','policy_code',
                  'chargeoff_within_12_mths','tax_liens','delinq_amnt','acc_now_delinq'],axis=1,inplace=True)


# In[ ]:


df3.shape


# In[ ]:


intr=[]
for i in df3['int_rate'].values:
    x=i[:-1]
    intr.append(x)


# In[ ]:


df3['int_rate']=np.array(intr)


# In[ ]:


ind=df3['revol_util'].dropna().index


# In[ ]:


rev=[]
for j in df3['revol_util'].dropna():
    y=j[:-1]
    rev.append(y)


# In[ ]:


rev1=pd.Series(rev,index=ind)


# In[ ]:


mask=df3['revol_util'].isnull()
rev2=df3[mask]['revol_util']


# In[ ]:


revol_ut=pd.concat([rev1,rev2],axis=0)


# In[ ]:


revol_ut.sort_index(inplace=True)


# In[ ]:


df3['revol_util']=revol_ut.values


# In[ ]:


df3.info()


# In[ ]:


# Variable Broadcasting
df4=df3.astype({'loan_amnt':float, 'funded_amnt':float, 'funded_amnt_inv':float,'int_rate':float,
       'installment':float,'annual_inc': float,'dti':float,'mths_since_last_delinq':float,
        'open_acc':float, 'revol_bal':float, 'revol_util':float, 'total_acc':float, 'total_pymnt':float,
                'total_pymnt_inv':float,'total_rec_prncp':float,
        'total_rec_int':float, 'total_rec_late_fee':float,'recoveries':float,
        'collection_recovery_fee':float,'last_pymnt_amnt':float})


# In[ ]:


df4.info()


# In[ ]:


df4.drop(columns=['desc','title','zip_code','emp_title'],axis=1,inplace=True)


# In[ ]:


dir(str)


# In[ ]:


help(str.split)


# In[ ]:


ter=[]
for i in df4['term'].values:
    x=i.split()[0]
    ter.append(x)


# In[ ]:


df4['term']=np.array(ter)


# In[ ]:


df4.emp_length.unique()


# In[ ]:


df4.replace({'10+ years':'10','< 1 year':'1','1 year':'1','3 years':'3', '8 years':'8', '9 years':'9',
       '4 years':'4', '5 years':'5', '6 years':'6', '2 years':'2', '7 years':'7'},inplace=True)


# In[ ]:


df4.head()


# In[ ]:


df4.shape


# In[ ]:


# Maximum and Minimum interest rate Grade Wise


# In[ ]:


df4['int_rate'][df4['grade']=='A'].min()


# In[ ]:


df4['int_rate'][df4['grade']=='A'].max()


# In[ ]:


df4['int_rate'][df4['grade']=='B'].min()


# In[ ]:


df4['int_rate'][df4['grade']=='B'].max()


# In[ ]:


df4['int_rate'][df4['grade']=='C'].min()


# In[ ]:


df4['int_rate'][df4['grade']=='C'].max()


# In[ ]:


df4['int_rate'][df4['grade']=='D'].min()


# In[ ]:


df4['int_rate'][df4['grade']=='D'].max()


# In[ ]:


df4['int_rate'][df4['grade']=='E'].min()


# In[ ]:


df4['int_rate'][df4['grade']=='E'].max()


# In[ ]:


df4['int_rate'][df4['grade']=='F'].min()


# In[ ]:


df4['int_rate'][df4['grade']=='F'].max()


# In[ ]:


df4['int_rate'][df4['grade']=='G'].min()


# In[ ]:


df4['int_rate'][df4['grade']=='G'].max()


# In[ ]:


df4.isnull().sum()


# In[ ]:


df4.loan_status.value_counts()


# In[ ]:


np.isnan(df4['pub_rec_bankruptcies'].values).sum()


# In[ ]:


# Box plot to check outliers
df4.boxplot(figsize=(20,12))
plt.xticks(rotation=90)
plt.show()


# In[ ]:


df4['annual_inc'].min()


# In[ ]:


Q1=df4['annual_inc'].quantile(0.25)
Q1


# In[ ]:


Q2=df4['annual_inc'].quantile(0.5)
Q2


# In[ ]:


Q3=df4['annual_inc'].quantile(0.75)
Q3


# In[ ]:


IQR=Q3-Q1
IQR


# In[ ]:


Imin=Q1-1.5*IQR
Imin


# In[ ]:


Imax=Q3+1.5*IQR
Imax


# In[ ]:


df4['annual_inc'].max()


# In[ ]:


df4['annual_inc'].mean()


# In[ ]:


"""np.set_printoptions(threshold=np.nan)
df4['annual_inc'].unique()
"""


# In[ ]:


mask=df4['annual_inc'].isnull()
df4[mask]


# In[ ]:


df4[df4['home_ownership']=='NONE']


# In[ ]:


df4['annual_inc'][(df4['emp_length']=='1') ].median()


# In[ ]:


df4['annual_inc'][(df4['emp_length']=='1') ].mean()


# In[ ]:


df4['home_ownership'].value_counts()


# In[ ]:


df4['annual_inc'][df4['home_ownership']=='NONE']


# In[ ]:


#Dropping annual income missing values as many columns contain NaN in that particular rows


# In[ ]:


df4.dropna(inplace=True,subset=['annual_inc'],axis=0)


# In[ ]:


df4.reset_index(drop=True,inplace=True)


# In[ ]:


df4.isnull().sum()


# In[ ]:


# Variable Creation
(df4['loan_status']=='Does not meet the credit policy. Status:Fully Paid') | (df4['loan_status']=='Does not meet the credit policy. Status:Charged Off')


# In[ ]:


df4['criteria']=np.where((df4['loan_status']=='Does not meet the credit policy. Status:Fully Paid')|(df4['loan_status']=='Does not meet the credit policy. Status:Charged Off'),'No','Yes')


# In[ ]:


df4[['criteria','loan_status']]


# In[ ]:


df4.replace({'Does not meet the credit policy. Status:Fully Paid':'Fully Paid','Does not meet the credit policy. Status:Charged Off':'Charged Off'},inplace=True)


# In[ ]:


df4['loan_status'].value_counts()


# In[ ]:


df4.to_csv('P2P1.csv')


# In[ ]:


df4.shape


# In[ ]:


df4.isnull().sum()


# In[ ]:


#mths_since_lst_delin missing value imputation


# In[ ]:


df4['mths_since_last_delinq'][(df4['loan_status']=='Fully Paid') & (df4['criteria']=='Yes')].median()


# In[ ]:


a=df4['mths_since_last_delinq'][(df4['loan_status']=='Fully Paid') & (df4['criteria']=='Yes')]
a.fillna(a.median(),inplace=True)


# In[ ]:


df4['mths_since_last_delinq'][(df4['loan_status']=='Fully Paid') & (df4['criteria']=='No')].median()


# In[ ]:


b=df4['mths_since_last_delinq'][(df4['loan_status']=='Fully Paid') & (df4['criteria']=='No')]
b.fillna(b.median(),inplace=True)


# In[ ]:


df4['mths_since_last_delinq'][(df4['loan_status']=='Charged Off')& (df4['criteria']=='Yes') ].median()


# In[ ]:


c=df4['mths_since_last_delinq'][(df4['loan_status']=='Charged Off') & (df4['criteria']=='Yes')]
c.fillna(c.median(),inplace=True)


# In[ ]:


df4['mths_since_last_delinq'][(df4['loan_status']=='Charged Off')& (df4['criteria']=='No') ].median()


# In[ ]:


d=df4['mths_since_last_delinq'][(df4['loan_status']=='Charged Off') & (df4['criteria']=='No')]
d.fillna(d.median(),inplace=True)


# In[ ]:


e=pd.concat([a,b,c,d])


# In[ ]:


e.sort_index(inplace=True)


# In[ ]:


df4['mths_since_last_delinq']=e


# In[ ]:


df4['mths_since_last_delinq']


# In[ ]:


df4.isnull().sum()


# In[ ]:


df4['emp_length']=pd.to_numeric(df4['emp_length'])


# In[ ]:


#Replacing NONE with OTHER in home_ownership


# In[ ]:


df4['home_ownership'].replace({'NONE':'OTHER'},inplace=True)


# In[ ]:


#Emp_length missing value imputation


# In[ ]:


df4['emp_length']=df4['emp_length'].fillna(df4['emp_length'].median())


# In[ ]:


df4.isnull().sum()


# In[ ]:


df4['delinq_2yrs']=df4['delinq_2yrs'].fillna(df4['delinq_2yrs'].median())


# In[ ]:


df4['inq_last_6mths']=df4['inq_last_6mths'].fillna(df4['inq_last_6mths'].median())


# In[ ]:


df4['open_acc']=df4['open_acc'].fillna(df4['open_acc'].median())


# In[ ]:


df4['pub_rec']=df4['pub_rec'].fillna(df4['pub_rec'].median())


# In[ ]:


df4['revol_util']=df4['revol_util'].fillna(df4['revol_util'].median())


# In[ ]:


df4['total_acc']=df4['total_acc'].fillna(df4['total_acc'].median())


# In[ ]:


df4['pub_rec_bankruptcies']=df4['pub_rec_bankruptcies'].fillna(df4['pub_rec_bankruptcies'].median())


# In[ ]:


df4.columns


# In[ ]:


df4.to_csv('LC.csv')


# # Univariate analysis

# In[ ]:


sns.distplot(df4.loan_amnt)
plt.show()


# In[ ]:


sns.distplot(df4.funded_amnt)
plt.show()


# In[ ]:


sns.distplot(df4.funded_amnt_inv)
plt.show()


# In[ ]:


sns.distplot(df4.int_rate)
plt.show()


# In[ ]:


sns.distplot(df4.installment)
plt.show()


# In[ ]:


sns.distplot(df4[df4['emp_length'].notnull()]['emp_length'])
plt.show()


# In[ ]:


sns.distplot(df4.annual_inc)
plt.show()


# In[ ]:


sns.distplot(df4.dti)
plt.show()


# In[ ]:


sns.distplot(df4.revol_bal)
plt.show()


# In[ ]:


sns.distplot(df4[df4['revol_util'].notnull()]['revol_util'])
plt.show()


# # Bivariate Analysis
# Categorical & Continuous

# In[ ]:


sns.boxplot('home_ownership','funded_amnt', data = df4)
plt.title('Home ownership Vs Funded_amnt')
plt.show()


# Type of home_ownership is not affecting the funded_amnt. Probably home ownership is independent of funded_amnt

# In[ ]:


plt.subplots(figsize=(12,10))
sns.boxplot(x='purpose', y= 'int_rate', data=df4)
plt.xticks(rotation=90)
plt.title(' Funded_amnt Vs Purpose')
plt.show()


# In[ ]:


m=df4.pivot_table(values = 'funded_amnt', index = 'emp_length', columns = 'loan_status', aggfunc = 'median')
m


# In[ ]:


m.plot(kind='bar',color=('r','g'))
plt.legend(loc='upper right')
plt.ylim((0,16000))
plt.show()


# On an average, 10+ yrs exp customer is borrowing more amount and charged off customers are borrowing more than fully paid customers. So, we are predicting the potential loan to reduce loss.

# In[ ]:


n=df4.pivot_table(values = 'int_rate', index = 'emp_length', columns = 'loan_status', aggfunc = 'median')
n


# In[ ]:


n.plot(kind='bar',color=('r','g'))
plt.legend(loc='upper right')
plt.ylim((0,18))
plt.show()


# On an average, charged off borrowers are paying high interest compared to Fully paid.
# Central limit theorem suggests, when number of observations are significantly large then more observations wont affect median

# In[ ]:


q=df4.pivot_table(values = 'annual_inc', index = 'emp_length', columns = 'loan_status', aggfunc = 'median')
q


# In[ ]:


q.plot(kind='bar',color=('r','g'))
plt.legend(loc='upper right')
plt.ylim((0,89000))
plt.show()


# In[ ]:


# Annual_inc of 10 yrs exp customer is highest


# In[ ]:


p=df4.pivot_table(values = 'funded_amnt', index = 'emp_length', columns = 'term', aggfunc = 'median')
p


# In[ ]:


p.plot(kind='bar')
plt.legend(loc='upper right')
plt.ylim((0,21000))
plt.show()


# In[ ]:


# Borrowers opting for 60 months are borrowing more amount. So, opting for high term


# In[ ]:


sns.countplot(df4['emp_length'].sort_values())


# # More no. of borrowers with 10 years experience followed by 1 year i.e., employees in early stage of career and at high stage of career are actively seeking loans

# In[ ]:


sns.countplot(df4['home_ownership'])


# In[ ]:


df4.describe()


# In[ ]:


df4.groupby('home_ownership').apply(lambda x: x[['funded_amnt','total_pymnt','annual_inc']].mean())


# In[ ]:


#Refer tableau annual_inc vs loan_amnt
sns.lmplot(x='funded_amnt',y='annual_inc',data=df4,hue='home_ownership',fit_reg=False)


# In[ ]:


df4.head()


# In[ ]:


sns.countplot(df4.term)


# # More borrowers are seeking 36 months than 60 months

# In[ ]:


b=df4['emp_length'][df4['term']=='60'].value_counts()
b.plot(kind='bar')
plt.title(' Count of Emp_length in 60 term ')
plt.show()


# # There is around 50% difference in number between 1 yr exp and 10 yr exp who opt for 60 months term.
# In 60 months term, 10 yrs exp are the highest. 

# In[ ]:


a=df4['emp_length'][df4['term']=='36'].value_counts()
a.plot(kind='bar')
plt.title(' Count of Emp_length in 36 term ')
plt.show()


# # In 36 months term, there isn't much difference between 1 yr exp and 10 yr exp but there is significant decrease between 1 yrand 2 yr. 
# Moreover, 1 yr exp are the highest

# i.e., More experience borrowers are seeking 60 term and early career stage(1 yr) borrowers are opting for 36 term.
# Early stage people have broad scope for growth and less family(dependent) expenses so opt for less term. 10 yr exp people have narrow scope for career growth but good salary and more family(dependent) expenses so opt for high term.

# In[ ]:


df4.columns


# In[ ]:


sns.FacetGrid(df4,hue='home_ownership',size=4).map(plt.scatter,'funded_amnt','annual_inc').add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(df4,hue='loan_status',size=4).map(plt.scatter,'annual_inc','funded_amnt').add_legend()
plt.show()


# In[ ]:


#import plotly.express as px


# In[ ]:


# fig = px.scatter_3d(df4, x='annual_inc', y='funded_amnt', z='revol_util',
#               color='loan_status')
# fig.show()


# In[ ]:


# fig = px.scatter_3d(df4, x='annual_inc', y='funded_amnt', z='emp_length',
#               color='loan_status')
# fig.show()


# # emp_length, dti, int_rate, revol_util is not affecting Charged Off

# In[ ]:


df4.columns


# In[ ]:


# df10=df5[['loan_amnt', 'funded_amnt', 'funded_amnt_inv','int_rate',
#        'installment','annual_inc','dti','revol_bal', 'revol_util', 'total_acc', 'total_pymnt',
#        'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
#        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee','loan_status']]


# In[ ]:


#sns.pairplot(df10)


# # Variable Transformation

# In[ ]:


df4.shape


# In[ ]:


import pylab
import scipy.stats as stats


# # QQ Plot for Graphical Normality test 

# In[ ]:


# N(0,1)
std_normal = np.random.normal(loc = 0, scale = 1, size=42531)

# 0 to 100th percentiles of std-normal
for i in range(0,101):
    print(i, np.percentile(std_normal,i))


# In[ ]:


stats.probplot(df4.funded_amnt, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


stats.probplot(np.log(df4.funded_amnt), dist="norm", plot=pylab)
pylab.show()


# In[ ]:


df4.funded_amnt.values.ndim


# In[ ]:


stats.probplot(df4.loan_amnt, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


stats.probplot(np.log(df4.loan_amnt), dist="norm", plot=pylab)
pylab.show()


# In[ ]:


stats.probplot(df4.funded_amnt_inv, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


stats.probplot(np.sqrt(df4.funded_amnt_inv), dist="norm", plot=pylab)
pylab.show()


# In[ ]:


stats.probplot(df4.annual_inc, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


stats.probplot(np.log(df4.annual_inc), dist="norm", plot=pylab)
pylab.show()


# In[ ]:


stats.probplot(df4.revol_bal, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


stats.probplot(np.sqrt(df4.revol_bal), dist="norm", plot=pylab) # Due to zero in column sqrt is applied
pylab.show()


# In[ ]:


stats.probplot(df4.revol_util, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


stats.probplot(np.sqrt(df4.revol_util), dist="norm", plot=pylab)
pylab.show()


# In[ ]:


# Boxcox transformation


# In[ ]:


xt=stats.boxcox(df4.funded_amnt.values,lmbda=0.26044195400343567)
xt


# In[ ]:


stats.probplot(xt, dist=stats.norm, plot=pylab)
pylab.show()


# In[ ]:


df4.funded_amnt.skew()


# In[ ]:


np.log(df4.funded_amnt).skew()


# In[ ]:


df4.annual_inc.skew()


# In[ ]:


(np.log(df4.annual_inc)).skew()


# In[ ]:


df4.revol_bal.skew()


# In[ ]:


np.sqrt(df4.revol_bal).skew()


# In[ ]:


df4.revol_util.skew()


# In[ ]:


np.sqrt(df4.revol_util).skew()


# # Statistical Normality Tests

# In[ ]:


stats.anderson(df4.annual_inc, dist='norm')


# In[ ]:


stats.anderson(np.log(df4.annual_inc), dist='norm')


# In[ ]:


stats.anderson(df4.funded_amnt, dist='norm')


# In[ ]:


stats.anderson(np.log(df4.funded_amnt), dist='norm')


# In[ ]:


stats.anderson(df4.loan_amnt, dist='norm')


# In[ ]:


stats.anderson(np.log(df4.loan_amnt), dist='norm')


# In[ ]:


stats.anderson(df4.funded_amnt_inv, dist='norm')


# In[ ]:


stats.anderson(np.sqrt(df4.funded_amnt_inv), dist='norm')


# In[ ]:


stats.anderson(df4.revol_bal, dist='norm')


# In[ ]:


stats.anderson(np.sqrt(df4.revol_bal), dist='norm')


# In[ ]:


stats.anderson(df4.revol_util, dist='norm')


# In[ ]:


stats.anderson(np.sqrt(df4.revol_util), dist='norm')


# Transformation is not applied on revol_util because skewness is increasing

# In[ ]:


df4['annual_inc']=np.log(df4.annual_inc)


# In[ ]:


df4['funded_amnt']=np.log(df4.funded_amnt)


# In[ ]:


df4['loan_amnt']=np.log(df4.loan_amnt) 


# In[ ]:


df4['funded_amnt_inv']=np.sqrt(df4.funded_amnt_inv) # due to zero in funded_amnt_inv we use sqrt


# In[ ]:


df4['revol_bal']=np.sqrt(df4.revol_bal) # Due to zero in revol_bal sqrt is applied


# # Statistical tests

# In[ ]:


A=df4['int_rate'][df4['grade']=='A'].values
A


# In[ ]:


B=df4['int_rate'][df4['grade']=='B'].values
B


# In[ ]:


C=df4['int_rate'][df4['grade']=='C'].values
C


# In[ ]:


D=df4['int_rate'][df4['grade']=='D'].values
D


# In[ ]:


E=df4['int_rate'][df4['grade']=='E'].values
E


# In[ ]:


F=df4['int_rate'][df4['grade']=='F'].values
F


# In[ ]:


G=df4['int_rate'][df4['grade']=='G'].values
G


# In[ ]:


intrate=pd.DataFrame()
d1=pd.DataFrame({'grade':'A','int_rate':A})
d2=pd.DataFrame({'grade':'B','int_rate':B})
d3=pd.DataFrame({'grade':'C','int_rate':C})
d4=pd.DataFrame({'grade':'D','int_rate':D})
d5=pd.DataFrame({'grade':'E','int_rate':E})
d6=pd.DataFrame({'grade':'F','int_rate':F})
d7=pd.DataFrame({'grade':'G','int_rate':G})


# In[ ]:


intrate=intrate.append(d1)
intrate=intrate.append(d2)
intrate=intrate.append(d3)
intrate=intrate.append(d4)
intrate=intrate.append(d5)
intrate=intrate.append(d6)
intrate=intrate.append(d7)


# In[ ]:


intrate.reset_index(drop=True,inplace=True)


# In[ ]:


print(intrate.head())


# In[ ]:


print(intrate.tail())


# In[ ]:


g=sns.boxplot(x='grade',y='int_rate',data=intrate)
g.set(xlabel='grade',ylabel='int_rate',title='Boxplot-Int_rate VS Grade')
plt.show()


# As the grade increases, median of int_ rate increases

# In[ ]:


import scipy.stats as stats
import statsmodels.formula.api as stm


# In[ ]:


stats.f.ppf(q=1-0.05,dfn=6,dfd=425)


# In[ ]:


model=stm.ols('int_rate~grade',data=intrate).fit()


# In[ ]:


model.summary()


# # Pvalue < 0.05 so reject the null hypo i.e. int_rate is dependent on grade. So, int_rate can be deleted

# In[ ]:


# Anova b/n funded_amnt and loan_status


# In[ ]:


from scipy.stats import f_oneway


# In[ ]:


fp=df4[df4['loan_status']=='Fully Paid']
co=df4[df4['loan_status']=='Charged Off']


# In[ ]:


f_oneway(fp['funded_amnt'],co['funded_amnt'])


# In[ ]:


# p<0.05 so reject the null hypo i.e. funded_amnt is dependent on loan_status. So, funded_amnt is present


# In[ ]:


df4[['delinq_2yrs','int_rate','grade','dti']]


# In[ ]:


# z test between loan_amnt and funded_amnt


# In[ ]:


std1=df4['loan_amnt'].std()


# In[ ]:


std1


# In[ ]:


std2=df4['funded_amnt'].std()


# In[ ]:


std2


# In[ ]:


df4['loan_amnt'].mean()-df4['funded_amnt'].mean()


# In[ ]:


import statsmodels.stats.weightstats


# In[ ]:


df4.columns


# In[ ]:


statsmodels.stats.weightstats.ztest(df4['loan_amnt'].values,df4['funded_amnt'].values,alternative='two-sided')


# In[ ]:


#p < 0.05 reject null hypo. So, loan_amnt and funded_amnt are dependent


# In[ ]:


# z test between funded_amnt_inv and funded_amnt


# In[ ]:


std3=df4['funded_amnt_inv'].std()
std3


# In[ ]:


df4['funded_amnt'].mean()-df4['funded_amnt_inv'].mean()


# In[ ]:


statsmodels.stats.weightstats.ztest(df4['funded_amnt_inv'].values,df4['funded_amnt'].values, alternative = 'two-sided',value=681.42)


# In[ ]:


#p < 0.05 so reject null hypo. So, loan_amnt and funded_amnt are dependent


# In[ ]:


stats.ttest_ind(df4['funded_amnt_inv'].values,df4['funded_amnt'].values)


# # Outlier Treatment using Winsorization

# In[ ]:


sns.boxplot(df4.annual_inc)


# In[ ]:


a = df4.annual_inc.values
stats.mstats.winsorize(a, limits=[0.025, 0.025],inplace=True)


# In[ ]:


b = df4.revol_bal
stats.mstats.winsorize(b, limits=[0.025, 0.025],inplace=True)


# In[ ]:


sns.boxplot(df4.annual_inc)


# In[ ]:


sns.boxplot(df4.revol_bal)


# In[ ]:


df5=df4.copy(deep=True)


# In[ ]:


df5.drop(columns=['loan_amnt','issue_d','funded_amnt_inv','int_rate','installment','sub_grade','addr_state','earliest_cr_line',
                  'last_pymnt_d','total_pymnt','total_pymnt_inv', 'total_rec_prncp', 'open_acc','total_rec_int',
                  'last_credit_pull_d','total_rec_late_fee','recoveries', 'collection_recovery_fee','last_pymnt_amnt'],axis=1,inplace=True)


# In[ ]:


df5.columns


# In[ ]:


plt.subplots(figsize=(15,8))
sns.heatmap(df5.corr(),annot=True,linewidth=0.2)


# In[ ]:


df5


# In[ ]:


df5.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le=LabelEncoder()


# In[ ]:


df5['grade']=le.fit_transform(df5.grade)


# In[ ]:


df5.head()


# In[ ]:


df5.info()


# In[ ]:


df5['delinq_2yrs']=pd.to_numeric(df5['delinq_2yrs'])


# In[ ]:


df5['inq_last_6mths']=pd.to_numeric(df5['inq_last_6mths'])
df5['pub_rec']=pd.to_numeric(df5['pub_rec'])


# In[ ]:


df6=pd.get_dummies(df5,columns=['term','home_ownership','verification_status','purpose','debt_settlement_flag','criteria'])


# In[ ]:


df6.columns


# In[ ]:


# Removing Dummy trap
df6.drop(columns=['criteria_No','debt_settlement_flag_Y','purpose_house','verification_status_Verified',
                  'home_ownership_RENT','term_60'],axis=1,inplace=True)


# In[ ]:


df6.shape


# In[ ]:


x=df6.drop(columns=['loan_status'])
y=df6['loan_status']


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc=StandardScaler()


# In[ ]:


xsc=sc.fit_transform(x)


# In[ ]:


# ! pip install imbalanced-learn
from imblearn.over_sampling import SMOTE


# In[ ]:


sm=SMOTE(random_state=2)


# In[ ]:


x1,y1=sm.fit_sample(xsc,y)


# In[ ]:


x1.shape


# In[ ]:


x1=pd.DataFrame(x1)


# In[ ]:


x1.rename(columns={0:'funded_amnt', 1:'grade', 2:'emp_length', 3:'annual_inc',
       4:'dti', 5:'delinq_2yrs', 6:'inq_last_6mths', 7:'mths_since_last_delinq',
       8:'pub_rec', 9:'revol_bal', 10:'revol_util', 11:'total_acc',
       12:'pub_rec_bankruptcies', 13:'term_36', 14:'home_ownership_MORTGAGE',
       15:'home_ownership_OTHER', 16:'home_ownership_OWN',
       17:'verification_status_Not Verified',
       18:'verification_status_Source Verified', 19:'purpose_car',
       20:'purpose_credit_card', 21:'purpose_debt_consolidation',
       22:'purpose_educational', 23:'purpose_home_improvement',
       24:'purpose_major_purchase', 25:'purpose_medical', 26:'purpose_moving',
       27:'purpose_other', 28:'purpose_renewable_energy', 29:'purpose_small_business',
       30:'purpose_vacation', 31:'purpose_wedding', 32:'debt_settlement_flag_N',
       33:'criteria_Yes'},inplace=True)


# In[ ]:


y1.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,stratify=y,random_state=2)


# In[ ]:


xtrain1,xtest1,ytrain1,ytest1=train_test_split(x1,y1,test_size=0.3,stratify=y1,random_state=2)


# In[ ]:


sc1=StandardScaler()
xtrainsc=sc1.fit_transform(xtrain)
xtestsc=sc1.transform(xtest)


# In[ ]:


xtrain1


# In[ ]:


dir(list)


# In[ ]:


help(list.remove)


# In[ ]:


v=df6.columns.tolist().remove('loan_status')
v
v


# In[ ]:


xtrain1.head()


# In[ ]:


xtrain1.head()


# In[ ]:


xtrain1.shape


# In[ ]:


ytrain1.shape


# # Logistic regresion

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr=LogisticRegression()


# In[ ]:


lr1=LogisticRegression()


# In[ ]:


modellr=lr.fit(xtrainsc,ytrain).predict(xtestsc)


# In[ ]:


modellr1=lr1.fit(xtrain1,ytrain1).predict(xtest1)


# In[ ]:


from sklearn import metrics


# In[ ]:


metric_imbal=pd.Series([metrics.accuracy_score(ytest,modellr),metrics.precision_score(ytest,modellr,average='weighted'),
              metrics.recall_score(ytest,modellr,average='weighted'),metrics.f1_score(ytest,modellr,average='weighted'),
             metrics.cohen_kappa_score(ytest,modellr)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])
print(metric_imbal)
print()
print('Confusion Matrix:')
print(metrics.confusion_matrix(ytest,modellr))


# In[ ]:


metric_bal=pd.Series([metrics.accuracy_score(ytest1,modellr1),metrics.precision_score(ytest1,modellr1,average='weighted'),
              metrics.recall_score(ytest1,modellr1,average='weighted'),metrics.f1_score(ytest1,modellr1,average='weighted'),metrics.cohen_kappa_score(ytest1,modellr1)],
                     index=['accuracy_score','precision_score','recall_score','f1_score','cohen_kappa_score'])
print(metric_bal)
print()
print(metrics.classification_report(ytest1,modellr1))
print()
print('Confusion Matrix:')
print(metrics.confusion_matrix(ytest1,modellr1))


# In[ ]:


# K-Fold validation for imbalanced 

le1=LabelEncoder()
z=le1.fit_transform(y)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
kf=KFold(n_splits=5,shuffle=True,random_state=2)
acc=[]
au=[]
for train,test in kf.split(x,z):
    M=LogisticRegression()
    Xtrain,Xtest=x.iloc[train,:],x.iloc[test,:]
    Ytrain,Ytest=z[train],z[test]
    M.fit(Xtrain,Ytrain)
    Y_predict=M.predict(Xtest)
    acc.append(metrics.accuracy_score(Ytest,Y_predict))
    fpr,tpr, _ = roc_curve(Ytest,Y_predict)
    au.append(auc(fpr, tpr))
print("Cross-validated Accuracy Mean Score:%.2f%% " % np.mean(acc))   
print("Cross-validated AUC Mean Score:%.2f%% " % np.mean(au))  
print("Cross-validated AUC Var Score:%.5f%% " % np.var(au,ddof=1)) 


# In[ ]:


# K-Fold Validation for balanced

le2=LabelEncoder()
z1=le2.fit_transform(y1)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
kf=KFold(n_splits=5,shuffle=True,random_state=2)
acc=[]
au=[]
for train,test in kf.split(x1,z1):
    M=LogisticRegression()
    Xtrain,Xtest=x1.iloc[train,:],x1.iloc[test,:]
    Ytrain,Ytest=z1[train],z1[test]
    M.fit(Xtrain,Ytrain)
    Y_predict=M.predict(Xtest)
    acc.append(metrics.accuracy_score(Ytest,Y_predict))
    fpr,tpr, _ = roc_curve(Ytest,Y_predict)
    au.append(auc(fpr, tpr))
print("Cross-validated Accuracy Mean Score:%.2f%% " % np.mean(acc))   
print("Cross-validated AUC Mean Score:%.2f%% " % np.mean(au))  
print("Cross-validated AUC Var Score:%.5f%% " % np.var(au,ddof=1)) 


# # Decision tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt=DecisionTreeClassifier(random_state=2)


# In[ ]:


modeldt=dt.fit(xtrain1,ytrain1).predict(xtest1)


# In[ ]:


metric_bal_dt=pd.Series([metrics.accuracy_score(ytest1,modeldt),metrics.precision_score(ytest1,modeldt,average='weighted'),
              metrics.recall_score(ytest1,modeldt,average='weighted'),metrics.f1_score(ytest1,modeldt,average='weighted'),
             metrics.cohen_kappa_score(ytest1,modeldt)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])
print(metric_bal_dt)
print()
print('Confusion Matrix:')
print(metrics.confusion_matrix(ytest1,modeldt))


# In[ ]:


print(metrics.classification_report(ytest1,modeldt))


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf=RandomForestClassifier(random_state=2)


# In[ ]:


modelrf=rf.fit(xtrain1,ytrain1).predict(xtest1)


# In[ ]:


metric_bal_rf=pd.Series([metrics.accuracy_score(ytest1,modelrf),metrics.precision_score(ytest1,modelrf,average='weighted'),
              metrics.recall_score(ytest1,modelrf,average='weighted'),metrics.f1_score(ytest1,modelrf,average='weighted'),
             metrics.cohen_kappa_score(ytest1,modelrf)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])
print(metric_bal_rf)
print()
print('Confusion Matrix:')
print(metrics.confusion_matrix(ytest1,modelrf))


# In[ ]:


print(metrics.classification_report(ytest1,modelrf))


# In[ ]:


Importance = pd.DataFrame({'Importance':rf.feature_importances_*100}, index=xtrain1.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r',figsize=(12,18) )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = {'n_estimators': np.arange(1,25)}
rf_cv = GridSearchCV(rf, grid, cv=5) # GridSearchCV
rf_cv.fit(xtrain1, ytrain1)

print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(rf_cv.score(xtrain1, ytrain1)))
print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(rf_cv.score(xtest1, ytest1)))
print("Tuned hyperparameter k: {}".format(rf_cv.best_params_)) 
print("Best score: {}".format(rf_cv.best_score_))


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = {'n_estimators': np.arange(25,40)}
rf_cv = GridSearchCV(rf, grid, cv=5) # GridSearchCV
rf_cv.fit(xtrain1, ytrain1)

print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(rf_cv.score(xtrain1, ytrain1)))
print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(rf_cv.score(xtest1, ytest1)))
print("Tuned hyperparameter k: {}".format(rf_cv.best_params_)) 
print("Best score: {}".format(rf_cv.best_score_))


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid = {'n_estimators': np.arange(40,60)}
rf_cv = GridSearchCV(rf, grid, cv=5) # GridSearchCV
rf_cv.fit(xtrain1, ytrain1)

print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(rf_cv.score(xtrain1, ytrain1)))
print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(rf_cv.score(xtest1, ytest1)))
print("Tuned hyperparameter k: {}".format(rf_cv.best_params_)) 
print("Best score: {}".format(rf_cv.best_score_))


# In[ ]:


rf1=RandomForestClassifier(n_estimators=29,random_state=2)
modelrf1=rf1.fit(xtrain1,ytrain1).predict(xtest1)


# In[ ]:


metric_bal_rf1=pd.Series([metrics.accuracy_score(ytest1,modelrf1),metrics.precision_score(ytest1,modelrf1,average='weighted'),
              metrics.recall_score(ytest1,modelrf1,average='weighted'),metrics.f1_score(ytest1,modelrf1,average='weighted'),
             metrics.cohen_kappa_score(ytest1,modelrf1)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])
print(metric_bal_rf1)
print()
print('Confusion Matrix:')
print(metrics.confusion_matrix(ytest1,modelrf1))


# In[ ]:


# K-Fold for all the models togeher with  


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


lr2=LogisticRegression()
knn=KNeighborsClassifier()
dt1=DecisionTreeClassifier(random_state=2)
rf1=RandomForestClassifier(random_state=2)
gnb = GaussianNB()


# In[ ]:



kf=KFold(n_splits=3,shuffle=True,random_state=2)
for model, name in zip([lr1,knn,dt1,rf1,gnb], ['Logistic','KNN','DecisionTree','RandomForest','NaiveBayes']):
    roc_auc=[]
    for train,test in kf.split(x1,z1):
        Xtrain,Xtest=x1.iloc[train,:],x1.iloc[test,:]
        Ytrain,Ytest=z1[train],z1[test]
        model.fit(Xtrain,Ytrain)
        Y_predict=model.predict(Xtest)
        #cm=metrics.confusion_matrix(Ytest,Y_predict)
        fpr,tpr, _ = roc_curve(Ytest,Y_predict)
        roc_auc.append(auc(fpr, tpr))
    print(roc_auc)
    print("AUC scores: %0.02f (+/- %0.5f) [%s]" % (np.mean(roc_auc), np.var(roc_auc,ddof=1), name ))
    print(metrics.confusion_matrix(Ytest,Y_predict))
    print(metrics.cohen_kappa_score(Ytest,Y_predict))
    print()


# In[ ]:





# In[ ]:





# In[ ]:


#t=df4.earliest_cr_line.dropna().index


# In[ ]:


#df4.earliest_cr_line.dropna().values


# In[ ]:


# tim=[]
# for i in df4.earliest_cr_line.dropna().values:
#     x=i[-4:]
#     tim.append(x)


# In[ ]:


# time=pd.Series(tim,index=t)


# In[ ]:


# time.shape


# In[ ]:


# mask=df4.earliest_cr_line.isnull()
# timen=df4[mask]['earliest_cr_line']


# In[ ]:


# timen.shape


# In[ ]:


# timen


# In[ ]:


# cr=pd.concat([time,timen],axis=0)


# In[ ]:


# cr.sort_index(inplace=True)


# In[ ]:


# df4['earliest_cr_line']=cr.values


# In[ ]:


#df4['earliest_cr_line']=pd.to_datetime(df4['earliest_cr_line'])


# In[ ]:


# df4['earliest_cr_line'].head()


# # bagging with logistic

# In[ ]:


#Bagging with logistic
from sklearn.ensemble import BaggingClassifier


# In[ ]:


bagg=BaggingClassifier(base_estimator=lr)


# In[ ]:


labels_bagg=bagg.fit(xtrain1,ytrain1).predict(xtest1)


# In[ ]:


metric_bal_bagg=pd.Series([metrics.accuracy_score(ytest1,labels_bagg),metrics.precision_score(ytest1,labels_bagg,average='weighted'),
              metrics.recall_score(ytest1,labels_bagg,average='weighted'),metrics.f1_score(ytest1,labels_bagg,average='weighted'),
             metrics.cohen_kappa_score(ytest1,labels_bagg)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])
print(metric_bal_bagg)
print()
print('Confusion Matrix:')
print(metrics.confusion_matrix(ytest1,labels_bagg))


# # bagging with decision tree

# In[ ]:


#Bagging with Decision Tree
bagg1=BaggingClassifier(base_estimator=dt)
labels_bagg1=bagg1.fit(xtrain1,ytrain1).predict(xtest1)


# In[ ]:


metric_bal_bagg1=pd.Series([metrics.accuracy_score(ytest1,labels_bagg1),metrics.precision_score(ytest1,labels_bagg1,average='weighted'),
              metrics.recall_score(ytest1,labels_bagg1,average='weighted'),metrics.f1_score(ytest1,labels_bagg1,average='weighted'),
             metrics.cohen_kappa_score(ytest1,labels_bagg1)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])
print(metric_bal_bagg1)
print()
print('Confusion Matrix:')
print(metrics.confusion_matrix(ytest1,labels_bagg1))


# # bagging with random forest

# In[ ]:


#bagging with rf
bagg2=BaggingClassifier(base_estimator=rf)
labels_bagg2=bagg2.fit(xtrain1,ytrain1).predict(xtest1)


# In[ ]:


metric_bal_bagg2=pd.Series([metrics.accuracy_score(ytest1,labels_bagg2),metrics.precision_score(ytest1,labels_bagg2,average='weighted'),
              metrics.recall_score(ytest1,labels_bagg2,average='weighted'),metrics.f1_score(ytest1,labels_bagg2,average='weighted'),
             metrics.cohen_kappa_score(ytest1,labels_bagg2)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])
print(metric_bal_bagg2)
print()
print('Confusion Matrix:')
print(metrics.confusion_matrix(ytest1,labels_bagg2))


# # boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


gbm=GradientBoostingClassifier(max_depth=4)


# In[ ]:


labels_gbm=gbm.fit(xtrain1,ytrain1).predict(xtest1)


# In[ ]:


metric_bal_gbm=pd.Series([metrics.accuracy_score(ytest1,labels_gbm),metrics.precision_score(ytest1,labels_gbm,average='weighted'),
              metrics.recall_score(ytest1,labels_gbm,average='weighted'),metrics.f1_score(ytest1,labels_gbm,average='weighted'),
             metrics.cohen_kappa_score(ytest1,labels_gbm)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])
print(metric_bal_gbm)
print()
print('Confusion Matrix:')
print(metrics.confusion_matrix(ytest1,labels_gbm))


# # voting classifier

# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


vc=VotingClassifier(estimators=[('Dt',dt),('Log reg',lr),('RF',rf),('Bagg',bagg),('Boost',gbm)])


# In[ ]:


labels_vc=vc.fit(xtrain1,ytrain1).predict(xtest1)


# In[ ]:


metric_bal_vc=pd.Series([metrics.accuracy_score(ytest1,labels_vc),metrics.precision_score(ytest1,labels_vc,average='weighted'),
              metrics.recall_score(ytest1,labels_vc,average='weighted'),metrics.f1_score(ytest1,labels_vc,average='weighted'),
             metrics.cohen_kappa_score(ytest1,labels_vc)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])
print(metric_bal_vc)
print()
print('Confusion Matrix:')
print(metrics.confusion_matrix(ytest1,labels_vc))


# # ada boost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


ada=AdaBoostClassifier(base_estimator=rf)


# In[ ]:


labels_ada=ada.fit(xtrain1,ytrain1).predict(xtest1)


# In[ ]:


metric_bal_ada=pd.Series([metrics.accuracy_score(ytest1,labels_ada),metrics.precision_score(ytest1,labels_ada,average='weighted'),
              metrics.recall_score(ytest1,labels_ada,average='weighted'),metrics.f1_score(ytest1,labels_ada,average='weighted'),
             metrics.cohen_kappa_score(ytest1,labels_ada)],index=['accuracy_score','precision_score','recall_score','f1_score','Cohen_kappa_score'])
print(metric_bal_ada)
print()
print('Confusion Matrix:')
print(metrics.confusion_matrix(ytest1,labels_ada))


# # xg boost

# In[ ]:


import xgboost as xgb


# In[ ]:


xg= xgb.XGBClassifier(max_depth=2, learning_rate=0.01) # 0.78947


# In[ ]:


# Fitting the Model
labels_xgb = xg.fit(xtrain1,ytrain1).predict(xtest1)


# In[ ]:


print(metrics.precision_score(ytest1,labels_xgb,average='weighted'))
print(metrics.recall_score(ytest1,labels_xgb,average='weighted'))
print(metrics.f1_score(ytest1,labels_xgb,average='weighted'))


# In[ ]:


metrics.cohen_kappa_score(ytest1,labels_xgb)

