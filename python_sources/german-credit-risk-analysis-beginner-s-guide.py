#!/usr/bin/env python
# coding: utf-8

# ## <a name="Introduction-Credit-Risk-Analysis">Introduction : Credit Risk Analysis</a>
# 
# #### <a name="credit_risk"> Credit risk</a>
# Credit Risk is the probable risk of loss resulting from a borrower's failure to repay a loan or meet contractual obligations. If a company offers credit to its client,then there is a risk that its clients may not pay their invoices.                                
# 
# #### <a name="credit_risk_types">Types of Credit Risk</a>           
#  __Good Risk__: An investment that one believes is likely to be profitable. The term most often refers to a loan made to a creditworthy person or company. Good risks are considered exceptionally likely to be repaid.               
#  __Bad Risk__: A loan that is unlikely to be repaid because of bad credit history, insufficient income, or some other reason. A bad risk increases the risk to the lender and the likelihood of default on the part of the borrower.
# 
# 
# ####  <a name="objective">Objective</a>: 
# Based on the attributes, classify a person as good or bad credit risk.
# #### <a name="dataset_description">Dataset Description</a>: 
# The dataset contains 1000 entries with 20 independent variables (7 numerical, 13 categorical) and 1 target variable prepared by Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes.The attributes are:                                  
# #### <a name="independent_variable">Independent Variables</a>
#    * Attribute 1:  (qualitative) __Status of existing checking account__                           
#       A11 :      ... <    0 DM                           
#       A12 : 0 <= ... <  200 DM                                   
#       A13 :      ... >= 200 DM / salary assignments for at least 1 year                           
#       A14 : no checking account    
#       
#    * Attribute 2:  (numerical) __Duration in month__
# 
#    * Attribute 3:  (qualitative) __Credit history__               
#       A30 : no credits taken/all credits paid back duly                   
#       A31 : all credits at this bank paid back duly                       
#       A32 : existing credits paid back duly till now                       
#       A33 : delay in paying off in the past                           
#       A34 : critical account/other credits existing (not at this bank)          
#       
#    * Attribute 4:  (qualitative)    __Purpose__              
#       A40 : car (new)                         
#       A41 : car (used)                              
#       A42 : furniture/equipment                              
#       A43 : radio/television                 
#       A44 : domestic appliances                     
#       A45 : repairs                        
#       A46 : education                      
#       A47 : vacation                    
#       A48 : retraining                     
#       A49 : business                         
#       A410 : others
#    * Attribute 5:  (numerical) __Credit amount__   
#    * Attibute 6:  (qualitative) __Savings account/bonds__           
#       A61 :          ... <  100 DM            
#       A62 :   100 <= ... <  500 DM           
#       A63 :   500 <= ... < 1000 DM            
#       A64 :          .. >= 1000 DM                   
#       A65 :   unknown/ no savings account         
#       
#    * Attribute 7:  (qualitative) __Present employment since__      
#         A71 : unemployed                       
#         A72 :       ... < 1 year           
#         A73 : 1  <= ... < 4 years         
#         A74 : 4  <= ... < 7 years            
#         A75 :       .. >= 7 years         
#    * Attribute 8:  (numerical) __Installment rate in percentage of disposable income__ 
#    * Attribute 9:  (qualitative) __Personal status and sex__                      
#         A91 : male   : divorced/separated               
#         A92 : female : divorced/separated/married              
#         A93 : male   : single               
#         A94 : male   : married/widowed              
#         A95 : female : single      
#    * Attribute 10: (qualitative) __Other debtors / guarantors__                   
#         A101 : none                  
#         A102 : co-applicant                   
#         A103 : guarantor             
#         
#    * Attribute 11: (numerical) __Present residence since__   
# 
#    * Attribute 12: (qualitative) __Property__
#         A121 : real estate                        
#         A122 : if not A121 : building society savings agreement/life insurance                           
#         A123 : if not A121/A122 : car or other, not in attribute 6           
#         A124 : unknown / no property    
#      
#    * Attribute 13: (numerical) __Age in years__
#    * Attribute 14: (qualitative) __Other installment plans__    
#         A141 : bank   
#         A142 : stores    
#         A143 : none  
#    * Attribute 15: (qualitative) __Housing__           
#         A151 : rent       
#         A152 : own          
#         A153 : for free      
#    * Attribute 16: (numerical) __Number of existing credits at this bank__
#    * Attribute 17: (qualitative) __Job__         
#         A171 : unemployed/ unskilled  - non-resident           
#         A172 : unskilled - resident                  
#         A173 : skilled employee / official                                
#         A174 : management/ self-employed/highly qualified employee/ officer 
#         
#        
#    
#   * Attribute 18: (numerical) __Number of people being liable to provide maintenance for__ 
#   * Attribute 19: (qualitative) __Telephone__             
#     A191 : none              
#     A192 : yes, registered under the customers name                     
#   * Attribute 20: (qualitative) __foreign worker__              
#     A201 : yes         
#     A202 : no   
# 
# 
# #### <a name="target_variable">Target Variable</a>                                        
# * __Cost Matrix__  
#     * 1 = __Good Risk__
#     * 2 = __Bad Risk__
# 

# ## <a name="data_collection">Data Collection</a>
# 
# #### <a name="import_libraries">Import Libraries</a>
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scikitplot as skplt
from math import floor,ceil
import statsmodels.api as sm
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
pd.set_option('display.max_columns', None)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve
from sklearn.model_selection import train_test_split, KFold, cross_val_score


import warnings
warnings.simplefilter('ignore', DeprecationWarning)


# In[ ]:


def style_specific_cell(x):

    color_thresh = 'background-color: lightpink'
    
    df_color = pd.DataFrame('', index=x.index, columns=x.columns)
    rows_number=len(x.index)
    column_number=len(x.columns)
    for r in range(0,rows_number): 
        for c in range(0,column_number):
            try:
                val=float(x.iloc[r, c])
                if x.iloc[r, 0]=="Percentage":
                    if val<10:
                        df_color.iloc[r, c]=color_thresh
            except:
                pass
            
    return df_color

def style_stats_specific_cell(x):

    color_thresh = 'background-color: lightpink'
    
    df_color = pd.DataFrame('', index=x.index, columns=x.columns)
    rows_number=len(x.index)
    for r in range(0,rows_number):
        try:
            val=(x.iloc[r, 1])
            if val>0.05:
                df_color.iloc[r, 1]=color_thresh
        except:
            pass
    return df_color


# ### <a name="import-data">Import Data</a> 
# Import data from __UCI machine learning repository__.                               
# link: http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data

# In[ ]:


#Reading Dataset from  http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
df=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",sep=" ",header=None)
headers=["Status of existing checking account","Duration in month","Credit history",         "Purpose","Credit amount","Savings account/bonds","Present employment since",         "Installment rate in percentage of disposable income","Personal status and sex",         "Other debtors / guarantors","Present residence since","Property","Age in years",        "Other installment plans","Housing","Number of existing credits at this bank",        "Job","Number of people being liable to provide maintenance for","Telephone","foreign worker","Cost Matrix(Risk)"]
df.columns=headers
df.to_csv("german_data_credit_cat.csv",index=False) #save as csv file

#for structuring only
Status_of_existing_checking_account={'A14':"no checking account",'A11':"<0 DM", 'A12': "0 <= <200 DM",'A13':">= 200 DM "}
df["Status of existing checking account"]=df["Status of existing checking account"].map(Status_of_existing_checking_account)

Credit_history={"A34":"critical account","A33":"delay in paying off","A32":"existing credits paid back duly till now","A31":"all credits at this bank paid back duly","A30":"no credits taken"}
df["Credit history"]=df["Credit history"].map(Credit_history)

Purpose={"A40" : "car (new)", "A41" : "car (used)", "A42" : "furniture/equipment", "A43" :"radio/television" , "A44" : "domestic appliances", "A45" : "repairs", "A46" : "education", 'A47' : 'vacation','A48' : 'retraining','A49' : 'business','A410' : 'others'}
df["Purpose"]=df["Purpose"].map(Purpose)

Saving_account={"A65" : "no savings account","A61" :"<100 DM","A62" : "100 <= <500 DM","A63" :"500 <= < 1000 DM", "A64" :">= 1000 DM"}
df["Savings account/bonds"]=df["Savings account/bonds"].map(Saving_account)

Present_employment={'A75':">=7 years", 'A74':"4<= <7 years",  'A73':"1<= < 4 years", 'A72':"<1 years",'A71':"unemployed"}
df["Present employment since"]=df["Present employment since"].map(Present_employment)



Personal_status_and_sex={ 'A95':"female:single",'A94':"male:married/widowed",'A93':"male:single", 'A92':"female:divorced/separated/married", 'A91':"male:divorced/separated"}
df["Personal status and sex"]=df["Personal status and sex"].map(Personal_status_and_sex)


Other_debtors_guarantors={'A101':"none", 'A102':"co-applicant", 'A103':"guarantor"}
df["Other debtors / guarantors"]=df["Other debtors / guarantors"].map(Other_debtors_guarantors)


Property={'A121':"real estate", 'A122':"savings agreement/life insurance", 'A123':"car or other", 'A124':"unknown / no property"}
df["Property"]=df["Property"].map(Property)




Other_installment_plans={'A143':"none", 'A142':"store", 'A141':"bank"}
df["Other installment plans"]=df["Other installment plans"].map(Other_installment_plans)

Housing={'A153':"for free", 'A152':"own", 'A151':"rent"}
df["Housing"]=df["Housing"].map(Housing)




Job={'A174':"management/ highly qualified employee", 'A173':"skilled employee / official", 'A172':"unskilled - resident", 'A171':"unemployed/ unskilled  - non-resident"}
df["Job"]=df["Job"].map(Job)

Telephone={'A192':"yes", 'A191':"none"}
df["Telephone"]=df["Telephone"].map(Telephone)



foreign_worker={'A201':"yes", 'A202':"no"}
df["foreign worker"]=df["foreign worker"].map(foreign_worker)



risk={1:"Good Risk", 2:"Bad Risk"}
df["Cost Matrix(Risk)"]=df["Cost Matrix(Risk)"].map(risk)


# ## <a name="data_overview">Data Overview</a>

# In[ ]:


df.head() #top 5 rows of the dataset


# ### <a name="data-types">Data Info</a>

# In[ ]:


df.info()


# ## <a name="summary_statistics">Table of Summary statistics</a>

# ### <a name="categorical-variable-stats">Categorical Variable</a>
# 
# #### Levels and Proportions

# In[ ]:


column_names=df.columns.tolist()
column_names.remove("Credit amount") #numerical variable
column_names.remove("Age in years") #numerical variable
column_names.remove("Duration in month") #numerical variable
#----------------------------------------------------------------------------------------

column_names_cat={}
for name in column_names:
    column_names_cat[name]=len(df[name].unique().tolist())

    marginal_report_cluster={}
for itr in range(0,np.asarray(list(column_names_cat.values())).max()+1):
    if [k for k,v in column_names_cat.items() if v == itr]:
        marginal_report_cluster[itr]=[k for k,v in column_names_cat.items() if v == itr]

#----------------------------------------------------------------------------------------
for key in marginal_report_cluster.keys():
    marginal_percentage_report=[]
    for name in sorted(marginal_report_cluster[key]):
        data=pd.crosstab(df[name],columns=["Percentage"]).apply(lambda r: (round((r/r.sum())*100,2)), axis=0).reset_index()
        data.columns=[name,"Percentage"]
        data=data.transpose().reset_index()
        [marginal_percentage_report.append(x) for x in data.values.tolist()]
        options=[]
    marginal_percentage_report=pd.DataFrame(marginal_percentage_report)
    [options.append("Category Option "+str(itr)) for itr in range(1,len(marginal_percentage_report.columns))]
    marginal_percentage_report.columns=["Attribute"]+options
    display(marginal_percentage_report.style.apply(style_specific_cell, axis=None))
    


# Most of the predictors are categorical with several levels, some of the levels have very few observations.
# Depending on the cell proportions given in the one-way table above,i have merged some of the levels for a particular attribute.
# 
# #### Other debtors / guarantors : 1) co-applicant/guarantor 2) None                            
# #### Other installment plans : 1) bank/store 2) None                                     
# #### Job : 1) Employed 2)Unemployed                 
# ####  Number of existing credits at this bank : 1 ) One  2) More than one	
# #### Personal status and sex : 1) Male  2)Female
# #### Status of existing checking account : 1) no checking account	2)< 0 DM 3) > 0 DM
# #### Credit history: 1) all credit / existing credits paid back duly till now  2)no credits taken 3)critical account/delay in paying off
# #### Savings account/bonds: 1) <100 DM 2) < 500 DM 3) > 500 DM 4)no savings account
# #### Purpose: 1) New car 2)Used car  3)Home Related 4)Other
# #### Present employment since: 1) < 1 years /unemployed 2) 1<= < 4 years 3) 4< = <7 years 4) >=7 years

# In[ ]:


df=pd.read_csv("german_data_credit_cat.csv")
number_of_credit={1:1,2:2,3:2,4:2}
df["Number of existing credits at this bank"]=df["Number of existing credits at this bank"].map(number_of_credit)

Status_of_existing_checking_account={'A14':"no checking account",'A11':"<0 DM", 'A12': ">0 DM",'A13':">0 DM"}
df["Status of existing checking account"]=df["Status of existing checking account"].map(Status_of_existing_checking_account)

Credit_history={"A34":"critical account/delay in paying off","A33":"critical account/delay in paying off","A32":"all credit / existing credits paid back duly till now","A31":"all credit / existing credits paid back duly till now","A30":"no credits taken"}
df["Credit history"]=df["Credit history"].map(Credit_history)




Purpose={"A40" : "car (new)", "A41" : "car (used)", "A42" : "Home Related", "A43" :"Home Related" , "A44" : "Home Related", "A45" : "Home Related", "A46" : "others", 'A47' : 'others','A48' : 'others','A49' : 'others','A410' : 'others'}
df["Purpose"]=df["Purpose"].map(Purpose)

Saving_account={"A65" : "no savings account","A61" :"<100 DM","A62" : "<500 DM","A63" :">500 DM", "A64" :">500 DM"}
df["Savings account/bonds"]=df["Savings account/bonds"].map(Saving_account)



           
Present_employment={'A75':">=7 years", 'A74':"4<= <7 years",  'A73':"1<= < 4 years", 'A72':"<1 years",'A71':"<1 years"}
df["Present employment since"]=df["Present employment since"].map(Present_employment)




Personal_status_and_sex={ 'A95':"female",'A94':"male",'A93':"male", 'A92':"female", 'A91':"male"}
df["Personal status and sex"]=df["Personal status and sex"].map(Personal_status_and_sex)


Other_debtors_guarantors={'A101':"none", 'A102':"co-applicant/guarantor", 'A103':"co-applicant/guarantor"}
df["Other debtors / guarantors"]=df["Other debtors / guarantors"].map(Other_debtors_guarantors)


Property={'A121':"real estate", 'A122':"savings agreement/life insurance", 'A123':"car or other", 'A124':"unknown / no property"}
df["Property"]=df["Property"].map(Property)




Other_installment_plans={'A143':"none", 'A142':"bank/store", 'A141':"bank/store"}
df["Other installment plans"]=df["Other installment plans"].map(Other_installment_plans)

Housing={'A153':"for free", 'A152':"own", 'A151':"rent"}
df["Housing"]=df["Housing"].map(Housing)




Job={'A174':"employed", 'A173':"employed", 'A172':"unemployed", 'A171':"unemployed"}
df["Job"]=df["Job"].map(Job)

Telephone={'A192':"yes", 'A191':"none"}
df["Telephone"]=df["Telephone"].map(Telephone)



foreign_worker={'A201':"yes", 'A202':"no"}
df["foreign worker"]=df["foreign worker"].map(foreign_worker)



risk={1:"Good Risk", 2:"Bad Risk"}
df["Cost Matrix(Risk)"]=df["Cost Matrix(Risk)"].map(risk)


# After, merging following are the Levels and Proportions for the data
# 

# In[ ]:


column_names=df.columns.tolist()
column_names.remove("Credit amount") #numerical variable
column_names.remove("Age in years") #numerical variable
column_names.remove("Duration in month") #numerical variable
#----------------------------------------------------------------------------------------

column_names_cat={}
for name in column_names:
    column_names_cat[name]=len(df[name].unique().tolist())

    marginal_report_cluster={}
for itr in range(0,np.asarray(list(column_names_cat.values())).max()+1):
    if [k for k,v in column_names_cat.items() if v == itr]:
        marginal_report_cluster[itr]=[k for k,v in column_names_cat.items() if v == itr]

#----------------------------------------------------------------------------------------
for key in marginal_report_cluster.keys():
    marginal_percentage_report=[]
    for name in sorted(marginal_report_cluster[key]):
        data=pd.crosstab(df[name],columns=["Percentage"]).apply(lambda r: (round((r/r.sum())*100,2)), axis=0).reset_index()
        data.columns=[name,"Percentage"]
        data=data.transpose().reset_index()
        [marginal_percentage_report.append(x) for x in data.values.tolist()]
        options=[]
    marginal_percentage_report=pd.DataFrame(marginal_percentage_report)
    [options.append("Category Option "+str(itr)) for itr in range(1,len(marginal_percentage_report.columns))]
    marginal_percentage_report.columns=["Attribute"]+options
    display(marginal_percentage_report.style.apply(style_specific_cell, axis=None))
    


# ### <a name="numerical-variable-stats">Numerical Variable</a>

# In[ ]:


df[["Credit amount","Age in years","Duration in month"]].describe()


# ## <a name="eda">Exploratory Data Analysis (EDA)</a>

# ### <a name="categorical-eda">Categorical Variable</a>

# In[ ]:


def visualize_distribution(attr):
    good_risk_df = df[df["Cost Matrix(Risk)"]=="Good Risk"]
    bad_risk_df = df[df["Cost Matrix(Risk)"]=="Bad Risk"]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    attr_good_risk_df = good_risk_df[[attr, 'Cost Matrix(Risk)']].groupby(attr).count()
    attr_bad_risk_df = bad_risk_df[[attr, 'Cost Matrix(Risk)']].groupby(attr).count()
    ax[0].barh( attr_good_risk_df['Cost Matrix(Risk)'].index.tolist(), attr_good_risk_df['Cost Matrix(Risk)'].tolist(), align='center', color="#5975A4")
    ax[1].barh( attr_bad_risk_df['Cost Matrix(Risk)'].index.tolist(), attr_bad_risk_df['Cost Matrix(Risk)'].tolist(), align='center', color="#B55D60")
    ax[0].set_title('Good Risk')
    ax[1].set_title('Bad Risk')
    ax[0].invert_xaxis()
    ax[1].yaxis.tick_right()
    
    num_para_change=["Present residence since","Number of existing credits at this bank","Installment rate in percentage of disposable income","Number of people being liable to provide maintenance for"]
    if attr in num_para_change:
        for i, v in enumerate(attr_good_risk_df['Cost Matrix(Risk)'].tolist()):
            ax[0].text(v+15, i+1, str(v), color='black')
        for i, v in enumerate(attr_bad_risk_df['Cost Matrix(Risk)'].tolist()):
            ax[1].text(v+2, i+1, str(v), color='black')
    else:
        for i, v in enumerate(attr_good_risk_df['Cost Matrix(Risk)'].tolist()):
            ax[0].text(v+25, i + .05, str(v), color='black')
        for i, v in enumerate(attr_bad_risk_df['Cost Matrix(Risk)'].tolist()):
            ax[1].text(v+1, i + .05, str(v), color='black')
    plt.suptitle(attr)
    plt.tight_layout()
    plt.show()


# #### Status of existing checking account
# A checking account is a deposit account held at a financial institution that allows withdrawals and deposits.

# In[ ]:


visualize_distribution("Status of existing checking account")


# #### Credit history
# A credit history is a record of a borrower's responsible repayment of debts.

# In[ ]:


visualize_distribution("Credit history")


# #### Purpose
# The underlying reason an applicant is seeking a loan

# In[ ]:


visualize_distribution("Purpose")


# #### Savings account/bonds
# A savings account is an interest-bearing deposit account held at a bank or other financial institution that provides a modest interest rate.m

# In[ ]:


visualize_distribution("Savings account/bonds")


# #### Present employment since
# 

# In[ ]:


visualize_distribution("Present employment since")


# #### Installment rate in percentage of disposable income

# In[ ]:


visualize_distribution("Installment rate in percentage of disposable income")


# #### Personal status and sex

# In[ ]:


visualize_distribution("Personal status and sex")


# #### Other debtors / guarantors
# A guarantor is a person who guarantees to pay for someone else's debt if he or she should default on a loan obligation. A guarantor acts as a co-signer of sorts, in that they pledge their own assets or services if a situation arises in which the original debtor cannot perform their obligations

# In[ ]:


visualize_distribution("Other debtors / guarantors")


# #### Present residence since

# In[ ]:


visualize_distribution("Present residence since")


# #### Property
# Borrower's property

# In[ ]:


visualize_distribution("Property")


# #### Other installment plans
# Installment plan enable people to buy goods over an extended period of time, without having to put down very much money at the time of purchase

# In[ ]:


visualize_distribution("Other installment plans")


# #### Housing

# In[ ]:


visualize_distribution("Housing")


# #### Number of existing credits at this bank

# In[ ]:


visualize_distribution("Number of existing credits at this bank")


# #### Number of people being liable to provide maintenance for

# In[ ]:


visualize_distribution("Number of people being liable to provide maintenance for")


# #### Foreign worker

# In[ ]:


visualize_distribution("foreign worker")


# #### Duration in Year
# Duration in month converted to years.
# 

# In[ ]:


df["Duration in year"]=df["Duration in month"].apply(lambda x: (floor(x/12)))
visualize_distribution("Duration in year")


# #### Age category
# age converted to categorica variable based on range

# In[ ]:


age_interval = [18, 24, 35, 55, 120]
age_category = ['Student', 'Young-Adult', 'Middle-Aged Adult', 'Senior']
df["Age_Category"] = pd.cut(df["Age in years"], age_interval, labels=age_category)
visualize_distribution("Age_Category")


# ### <a name="numerical-eda">Numerical Variable</a>

# In[ ]:


sns.set()
f, axes = plt.subplots(1, 3,figsize=(15,5))
sns.boxplot(y=df["Credit amount"],x=df["Cost Matrix(Risk)"],orient='v' , ax=axes[0],palette=["#5975A4","#B55D60"]) #box plot
sns.boxplot(y=df["Duration in month"],x=df["Cost Matrix(Risk)"], orient='v' , ax=axes[1],palette=["#5975A4","#B55D60"]) #box plot
sns.boxplot(y=df["Age in years"],x=df["Cost Matrix(Risk)"], orient='v' , ax=axes[2],palette=["#5975A4","#B55D60"]) #box plot
plt.show()


# ### <a name="feature-selection">Feature Selection</a>

# In[ ]:


column_names_cat_stats=["Status of existing checking account","Credit history","Purpose","Savings account/bonds","Present employment since","Installment rate in percentage of disposable income","Personal status and sex","Other debtors / guarantors","Present residence since","Property","Other installment plans","Housing","Number of existing credits at this bank","Job","Number of people being liable to provide maintenance for","Telephone","foreign worker"]

statistical_significance=[]
for attr in column_names_cat_stats:
    data_count=pd.crosstab(df[attr],df["Cost Matrix(Risk)"]).reset_index()
    obs=np.asarray(data_count[["Bad Risk","Good Risk"]])
    chi2, p, dof, expected = stats.chi2_contingency(obs)
    statistical_significance.append([attr,round(p,6)])
statistical_significance=pd.DataFrame(statistical_significance)
statistical_significance.columns=["Attribute","P-value"]
display(statistical_significance.style.apply(style_stats_specific_cell, axis=None))


statistical_significance=[]
column_names_cont_stats=["Credit amount","Age in years","Duration in month"]
good_risk_df = df[df["Cost Matrix(Risk)"]=="Good Risk"]
bad_risk_df = df[df["Cost Matrix(Risk)"]=="Bad Risk"]
for attr in column_names_cont_stats:
    statistic, p=stats.f_oneway(good_risk_df[attr].values,bad_risk_df[attr].values)
    statistical_significance.append([attr,round(p,6)])
statistical_significance=pd.DataFrame(statistical_significance)
statistical_significance.columns=["Attribute","P-value"]
display(statistical_significance.style.apply(style_stats_specific_cell, axis=None))


# <a name="selected-feature">__Selected_Features__</a>: Status of existing checking account, Credit history, Purpose,Savings account/bonds, Present employment since, Personal status and sex, Property, Other installment plans, Housing, foreign worker, Credit amount, Age in years, Duration in month

# In[ ]:


attr_significant=["Status of existing checking account","Credit history","Purpose","Savings account/bonds","Present employment since","Personal status and sex","Property","Other installment plans","Housing","foreign worker","Credit amount","Age in years","Duration in month"]
target_variable=["Cost Matrix(Risk)"]
df=df[attr_significant+target_variable]


# <a name="data-pre">__Creating Dummy Variable from Categorical Variables__</a>

# In[ ]:


col_cat_names=["Status of existing checking account","Credit history","Purpose","Savings account/bonds","Present employment since","Personal status and sex","Property","Other installment plans","Housing","foreign worker"]
for attr in col_cat_names:
    df = df.merge(pd.get_dummies(df[attr], prefix=attr), left_index=True, right_index=True)
    df.drop(attr,axis=1,inplace=True)
 
#converting target variable into numeric
risk={"Good Risk":1, "Bad Risk":0}
df["Cost Matrix(Risk)"]=df["Cost Matrix(Risk)"].map(risk)


# In[ ]:


#view of the dataset for modelling
df.head()


# ####  <a name="pca">Principal Component Analysis : Dimensionality Reduction</a>

# In[ ]:


X = df.drop('Cost Matrix(Risk)', 1).values #independent variables
y = df["Cost Matrix(Risk)"].values #target variables

pca = PCA(n_components=16)
X = pca.fit_transform(X)


# ### <a name="model">Modelling</a>

# ### <a name="train-test">Train/Test Split </a>

# In[ ]:


# Spliting dataset into train and test version
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,random_state=0)


# ### <a name="xgb">Model: XGBClassifier</a>

# In[ ]:


model=XGBClassifier()


# ### <a name="accuracy">Accuracy Score</a>

# In[ ]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy: ")
print(round(accuracy_score(y_test,y_pred)*100,2))


# ### <a name="roc"> ROC curve </a>

# In[ ]:


y_pred_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

