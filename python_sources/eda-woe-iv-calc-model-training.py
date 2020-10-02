#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import math
import warnings
warnings.filterwarnings("ignore")


# ## Application record EDA 

# In[ ]:


application_record = pd.read_csv("/kaggle/input/credit-card-approval-prediction/application_record.csv")
application_record.head()


# In[ ]:


application_record.shape


# In[ ]:


application_record.isnull().sum()


# In[ ]:


#Check for duplicate records
application_record[application_record.duplicated()]


# In[ ]:


#Gender proportion in applicants
gender_val = application_record.CODE_GENDER.value_counts(normalize = True)
gender_val


# In[ ]:


gender_val.plot.pie()
plt.show()


# Around 67.14% of the applicants are female

# In[ ]:


housing_val = application_record.NAME_HOUSING_TYPE.value_counts(normalize = True)
housing_val


# In[ ]:


housing_val.plot.bar()
plt.show()


# In[ ]:


#House Ownership percentage
housing_ownership_count = application_record.groupby(['CODE_GENDER','NAME_HOUSING_TYPE']).agg({'ID': 'count'})
housing_ownership_count


# In[ ]:


housing_ownership_percent = housing_ownership_count.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
housing_ownership_percent


# In[ ]:


housing_ownership_percent.sort_values(by = 'ID').plot.barh()
plt.show()


# Housing ownership percentage for females is 90.89% while for males it is 87.58%. Females applicants have higher house ownership percentage.

# In[ ]:


#Education level and income relation
application_record.groupby(["NAME_EDUCATION_TYPE"]).AMT_INCOME_TOTAL.mean().sort_values(ascending=False).plot.barh()
plt.show()


# The average income increases with the education level.

# In[ ]:


#Mean & Median of amount income 
print(application_record.AMT_INCOME_TOTAL.mean())
print(application_record.AMT_INCOME_TOTAL.median())


# Mean of income = 87524.29 Median of income = 160780.50

# ## Credit record EDA 

# In[ ]:


credit_record = pd.read_csv("/kaggle/input/credit-card-approval-prediction/credit_record.csv")
credit_record.head()


# In[ ]:


credit_record.shape


# In[ ]:


credit_record.info()


# In[ ]:


credit_record.describe()


# In[ ]:


credit_record.STATUS.value_counts().plot.bar()
plt.show()


# In[ ]:


credit_record.MONTHS_BALANCE.value_counts().plot.hist()
plt.show()


# In[ ]:


#Find out for how long each customer has had a card
credit_record['MONTHS_BALANCE'] = credit_record.MONTHS_BALANCE.apply(lambda x : x*(-1))
cardholder_tenure = pd.DataFrame(credit_record.groupby('ID').agg({'MONTHS_BALANCE' : max}))
cardholder_tenure.rename(columns = {'MONTHS_BALANCE':'CUST_FOR_MONTHS'},inplace = True)
cardholder_tenure.head()


# In[ ]:


#Merging application_records & credit_record to get the number of months for which the customer has had a card.
cust_data = pd.merge(application_record,cardholder_tenure,on = 'ID',how = 'inner')
cust_data.head()


# In[ ]:


cust_data.shape


# In[ ]:


credit_record['STATUS'][credit_record["STATUS"] == 'C'] = -1
credit_record['STATUS'][credit_record["STATUS"] == 'X'] = -1
credit_record.head()


# In[ ]:


credit_record['STATUS'] = credit_record.STATUS.apply(lambda x : int(x))


# In[ ]:


credit_record.sort_values(by = 'STATUS',ascending = False,inplace = True)
credit_record.drop_duplicates(subset = ['ID'],inplace = True)
credit_record.shape


# #### Assuming that a person is consider a defaulter to bank if he has a payment withstanding for more than 60 days. So all the customers having STATUS >= 2 will be considered as defaulters or bad customers 

# In[ ]:


credit_record['target'] = credit_record.STATUS.apply(lambda x : 0 if x>=2 else 1)
credit_record.drop(['STATUS','MONTHS_BALANCE'],axis = 1,inplace = True)
credit_record.head()


# In[ ]:


credit_record.target.value_counts()


# ### Merging Credit & Application Data 

# In[ ]:


#Merge DF to get final dataframe with all the columns
cust_data = pd.merge(cust_data, credit_record, on = "ID", how = "inner")
cust_data.shape


# In[ ]:


cust_data.head()


# In[ ]:


cust_data.columns


# In[ ]:


#Drop duplicated values
cust_data.drop_duplicates(subset = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
       'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
       'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'CUST_FOR_MONTHS',
       'target'],inplace = True)
cust_data.shape


# In[ ]:


pvt_tbl = pd.pivot_table(data = cust_data, index = ['OCCUPATION_TYPE'], columns = ['NAME_FAMILY_STATUS'], values = 'target', aggfunc = sum,  fill_value = 0)
plt.figure(figsize=[10,10])
hm = sns.heatmap(data = pvt_tbl, annot = True, fmt='.0f', linewidths=.2, center = 1600)
bottom, top = hm.get_ylim()
hm.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[ ]:


pvt_tbl = pd.pivot_table(data = cust_data, index = ['NAME_INCOME_TYPE'], columns = ['NAME_HOUSING_TYPE'], values = 'target', aggfunc = sum,  fill_value = 0)
plt.figure(figsize=[10,6])
hm = sns.heatmap(data = pvt_tbl, annot = True, fmt='.0f', linewidths=.2, center = 1600)
bottom, top = hm.get_ylim()
hm.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[ ]:


cust_data.info()


# In[ ]:


cust_data.isnull().sum()


# In[ ]:


#Convert binary categorical columns to have integer value
cust_data['CODE_GENDER'] = cust_data.CODE_GENDER.apply(lambda x : 0 if x == 'M' else 1)
cust_data['FLAG_OWN_CAR'] = cust_data.FLAG_OWN_CAR.apply(lambda x : 0 if x == 'N' else 1)
cust_data['FLAG_OWN_REALTY'] = cust_data.FLAG_OWN_REALTY.apply(lambda x : 0 if x == 'N' else 1)
cust_data.head()


# In[ ]:


cust_data.describe()


# In[ ]:


cust_data.FLAG_MOBIL.value_counts()


# Since all the rows have FLAG_MOBIL as 1 so there is no variation available for this column and hence it would add any value to the model. Hence it can be dropped

# In[ ]:


cust_data.drop('FLAG_MOBIL', axis = 1, inplace = True)
cust_data.head()


# In[ ]:


#CNT_FAM_MEMBERS can not be float. Convert the column to int type
cust_data['CNT_FAM_MEMBERS'] = cust_data['CNT_FAM_MEMBERS'].astype('int')
cust_data.head()


# In[ ]:


def box_plot(df,col) :
    sns.boxplot(data = df, y = col)
    plt.show()


# In[ ]:


boxplt_col = ["DAYS_BIRTH","DAYS_EMPLOYED","AMT_INCOME_TOTAL","CNT_FAM_MEMBERS","CNT_CHILDREN"]
for col in boxplt_col :
    box_plot(cust_data,col)


# In[ ]:


#Checking outliers for DAYS_EMPLOYED column
cust_data[cust_data['DAYS_EMPLOYED'] > 0]


# In[ ]:


cust_data[(cust_data['DAYS_EMPLOYED'] > 0) & (cust_data.NAME_INCOME_TYPE == 'Pensioner')].shape


# All the customers with a positive 'DAYS_EMPLOYED' are pensioners and represent a valid data.

# In[ ]:


#Convert the Integer columns to positive integers
def convert_to_positive(df,column) :
    df[column] = df[column].apply(lambda x : x*-1)
    return df


# In[ ]:


continuous_variable = ['DAYS_BIRTH','DAYS_EMPLOYED']
for c in continuous_variable :
    cust_data = convert_to_positive(cust_data,c)
cust_data.head()


# In[ ]:


#Convert DAYS to YEARS
cust_data['EMP_YEARS'] = cust_data.DAYS_EMPLOYED/365
cust_data['AGE'] = cust_data.DAYS_BIRTH/365
cust_data.drop(["DAYS_BIRTH","DAYS_EMPLOYED"],axis = 1,inplace = True)
cust_data.head()


# Replacing all the EMP_YEARS for all pensioners to be -1.

# In[ ]:


cust_data['EMP_YEARS'] = cust_data.EMP_YEARS.apply(lambda x : -1 if x<0 else x)


# In[ ]:


def bad_cust_proportion(col) :
    bad_prop_cnt = cust_data.groupby([col,'target']).agg({'ID': 'count'})
    bad_prop_percent = bad_prop_cnt.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
    return bad_prop_percent


# In[ ]:


bad_cust_proportion('FLAG_OWN_CAR')


# Proportion of bad customers for people who own a car is less than those who don't own a car

# In[ ]:


bad_cust_proportion('NAME_HOUSING_TYPE')


# People living on rent don't have the highest proportion of bad customers. People having 'Office apartment' have the highest proportion of bad customers

# In[ ]:


bad_cust_proportion('NAME_FAMILY_STATUS')


# Single customers have a higher proportion of bad customers as compared to married customers

# In[ ]:


cust_data['OCCUPATION_TYPE'] = cust_data.OCCUPATION_TYPE.apply(lambda x : 'UNEMPLOYED' if pd.isnull(x) else x)
cust_data.OCCUPATION_TYPE.value_counts()


# ## Calculating WOE & IV 

# In[ ]:


def calc_woe_iv(col) :
    df = pd.DataFrame(columns = ['values','total','good','bad','event_rate','non_event_rate','per_total_events','per_total_non_events','WOE','IV'])
    df['values'] = cust_data[col].unique()
    df.set_index('values',inplace = True)
    
    values = cust_data[col].unique()
    total_dict = dict(cust_data.groupby(col).size())
    col_target_dict = dict(cust_data.groupby([col,'target']).size())
    target_count = dict(cust_data.groupby(['target']).size())
    
    for value in values :
        df.loc[value]['total'] = total_dict[value]
        if (value,1) in col_target_dict:
            df.loc[value]['good'] = col_target_dict[(value,1)]
        else :
            df.loc[value]['good'] = 0
        
        if (value,0) in col_target_dict:
            df.loc[value]['bad'] = col_target_dict[(value,0)]
        else :
            df.loc[value]['bad'] = 0
            
        if df.loc[value]['bad'] == 0 :
            df = df.drop([value])
        
    df['event_rate'] = df['good']/df['total']
    df['non_event_rate'] = df['bad']/df['total']
    
    df['per_total_events'] = df['good']/target_count[1]
    df['per_total_non_events'] = df['bad']/target_count[0]
    
    df['WOE'] = np.log(df.per_total_events.astype('float64')/df.per_total_non_events.astype('float64'))
    df['IV'] = (df['per_total_events'] - df['per_total_non_events']) * df['WOE']
    
    return df    


# In[ ]:


iv_values = pd.DataFrame(columns = ['col_name','iv_value'])


# In[ ]:


iv_values['col_name'] = cust_data.columns
iv_values.set_index(['col_name'],inplace = True)
iv_values.drop(['ID','target'],inplace = True)
iv_values


# ### 1. Binary Variable 

# #### a. CODE_GENDER

# In[ ]:


CODE_GENDER_df = calc_woe_iv('CODE_GENDER')
iv_values.loc['CODE_GENDER'] = CODE_GENDER_df.IV.sum()
print(iv_values.loc['CODE_GENDER'])
CODE_GENDER_df


# #### b.FLAG_OWN_CAR

# In[ ]:


FLAG_OWN_CAR_df = calc_woe_iv('FLAG_OWN_CAR')
iv_values.loc['FLAG_OWN_CAR'] = FLAG_OWN_CAR_df.IV.sum()
print(iv_values.loc['FLAG_OWN_CAR'])
FLAG_OWN_CAR_df


# #### c. FLAG_OWN_REALTY

# In[ ]:


FLAG_OWN_REALTY_df = calc_woe_iv('FLAG_OWN_REALTY')
iv_values.loc['FLAG_OWN_REALTY'] = FLAG_OWN_REALTY_df.IV.sum()
print(iv_values.loc['FLAG_OWN_REALTY'])
FLAG_OWN_REALTY_df


# #### d. FLAG_WORK_PHONE

# In[ ]:


FLAG_WORK_PHONE_df = calc_woe_iv('FLAG_WORK_PHONE')
iv_values.loc['FLAG_WORK_PHONE'] = FLAG_WORK_PHONE_df.IV.sum()
print(iv_values.loc['FLAG_WORK_PHONE'])
FLAG_WORK_PHONE_df


# #### e. FLAG_PHONE

# In[ ]:


FLAG_PHONE_df = calc_woe_iv('FLAG_PHONE')
iv_values.loc['FLAG_PHONE'] = FLAG_PHONE_df.IV.sum()
print(iv_values.loc['FLAG_PHONE'])
FLAG_PHONE_df


# #### f. FLAG_EMAIL

# In[ ]:


FLAG_EMAIL_df = calc_woe_iv('FLAG_EMAIL')
iv_values.loc['FLAG_EMAIL'] = FLAG_EMAIL_df.IV.sum()
print(iv_values.loc['FLAG_EMAIL'])
FLAG_EMAIL_df


# ### 2. Categorical Variables 

# #### a. NAME_INCOME_TYPE 

# In[ ]:


NAME_INCOME_TYPE_df = calc_woe_iv('NAME_INCOME_TYPE')
iv_values.loc['NAME_INCOME_TYPE'] = NAME_INCOME_TYPE_df.IV.sum()
print(iv_values.loc['NAME_INCOME_TYPE'])
NAME_INCOME_TYPE_df


# #### b. NAME_EDUCATION_TYPE

# In[ ]:


NAME_EDUCATION_TYPE_df = calc_woe_iv('NAME_EDUCATION_TYPE')
iv_values.loc['NAME_EDUCATION_TYPE'] = NAME_EDUCATION_TYPE_df.IV.sum()
print(iv_values.loc['NAME_EDUCATION_TYPE'])
NAME_EDUCATION_TYPE_df


# #### c. NAME_FAMILY_STATUS

# In[ ]:


NAME_FAMILY_STATUS_df = calc_woe_iv('NAME_FAMILY_STATUS')
iv_values.loc['NAME_FAMILY_STATUS'] = NAME_FAMILY_STATUS_df.IV.sum()
print(iv_values.loc['NAME_FAMILY_STATUS'])
NAME_FAMILY_STATUS_df


# #### d. NAME_HOUSING_TYPE

# In[ ]:


NAME_HOUSING_TYPE_df = calc_woe_iv('NAME_HOUSING_TYPE')
iv_values.loc['NAME_HOUSING_TYPE'] = NAME_HOUSING_TYPE_df.IV.sum()
print(iv_values.loc['NAME_HOUSING_TYPE'])
NAME_HOUSING_TYPE_df


# #### e. OCCUPATION_TYPE

# In[ ]:


OCCUPATION_TYPE_df = calc_woe_iv('OCCUPATION_TYPE')
iv_values.loc['OCCUPATION_TYPE'] = OCCUPATION_TYPE_df.IV.sum()
print(iv_values.loc['OCCUPATION_TYPE'])
OCCUPATION_TYPE_df


# ### 3. Continuous Variables

# In[ ]:


cust_data.describe()


# #### a. CNT_CHILDREN

# Since all the bins should have atleast 5% of the total observations, therefore dividing 'CNT_CHILDREN' into [0,1,1+] bins

# In[ ]:


cust_data['cnt_child_category'] = cust_data.CNT_CHILDREN.apply(lambda x : '1+' if x>= 2 else str(x))


# In[ ]:


CNT_CHILDREN_df = calc_woe_iv('cnt_child_category')
iv_values.loc['CNT_CHILDREN'] = CNT_CHILDREN_df.IV.sum()
print(iv_values.loc['CNT_CHILDREN'])
CNT_CHILDREN_df


# #### b. AMT_INCOME_TOTAL

# In[ ]:


bins = [0, 70000, 100000, 150000, 200000, 250000, 300000, 350000, 1600000]
labels = ['70000', '100000', '150000', '200000', '250000', '300000', '350000', '1600000']
cust_data['income_bin'] = pd.cut(cust_data['AMT_INCOME_TOTAL'], bins = bins, labels = labels)
cust_data.head()


# In[ ]:


cust_data.income_bin.value_counts()


# In[ ]:


AMT_INCOME_TOTAL_df = calc_woe_iv('income_bin')
iv_values.loc['AMT_INCOME_TOTAL'] = AMT_INCOME_TOTAL_df.IV.sum()
print(iv_values.loc['AMT_INCOME_TOTAL'])
AMT_INCOME_TOTAL_df.sort_values(by = 'WOE',inplace = True)
AMT_INCOME_TOTAL_df


# #### c. CNT_FAM_MEMBERS

# In[ ]:


cust_data.CNT_FAM_MEMBERS.value_counts()


# In[ ]:


cust_data['cnt_family_bin'] = cust_data.CNT_FAM_MEMBERS.apply(lambda x : '3+' if x>= 4 else str(x))


# In[ ]:


CNT_FAM_MEMBERS_df = calc_woe_iv('cnt_family_bin')
iv_values.loc['CNT_FAM_MEMBERS'] = CNT_FAM_MEMBERS_df.IV.sum()
print(iv_values.loc['CNT_FAM_MEMBERS'])
CNT_FAM_MEMBERS_df.sort_values(by = 'WOE',inplace = True)
CNT_FAM_MEMBERS_df


# #### d. CUST_FOR_MONTHS

# In[ ]:


bins = [-1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
labels = ['0.5','1','1.5','2','2.5','3','3.5','4','4.5','5']
cust_data['months_bin'] = pd.cut(cust_data['CUST_FOR_MONTHS'], bins = bins,labels = labels)
cust_data.head()


# In[ ]:


cust_data.months_bin.value_counts()


# In[ ]:


CUST_FOR_MONTHS_df = calc_woe_iv('months_bin')
iv_values.loc['CUST_FOR_MONTHS'] = CUST_FOR_MONTHS_df.IV.sum()
print(iv_values.loc['CUST_FOR_MONTHS'])
CUST_FOR_MONTHS_df.sort_values(by = 'WOE',inplace = True)
CUST_FOR_MONTHS_df


# #### e. EMP_YEARS

# In[ ]:


bins = [-2, -1,  5, 10, 15,  20, 50]
labels = ['retired','5','10','15','20','20+']
cust_data['emp_years_bin'] = pd.cut(cust_data['EMP_YEARS'], bins = bins, labels = labels)
cust_data.head()


# In[ ]:


cust_data.emp_years_bin.value_counts()


# In[ ]:


EMP_YEARS_df = calc_woe_iv('emp_years_bin')
iv_values.loc['EMP_YEARS'] = EMP_YEARS_df.IV.sum()
print(iv_values.loc['EMP_YEARS'])
EMP_YEARS_df.sort_values(by = 'WOE',inplace = True)
EMP_YEARS_df


# #### f. AGE

# In[ ]:


bins = [19, 27, 30, 35, 40, 45, 50, 55, 62, 70]
labels = ['27','30','35','40','45','50','55','62','70']
cust_data['age_bin'] = pd.cut(cust_data['AGE'], bins = bins,labels = labels)
cust_data.head()


# In[ ]:


cust_data.age_bin.value_counts()


# In[ ]:


AGE_df = calc_woe_iv('age_bin')
iv_values.loc['AGE'] = AGE_df.IV.sum()
print(iv_values.loc['AGE'])
AGE_df.sort_values(by = 'WOE',inplace = True)
AGE_df


# In[ ]:


iv_values.sort_values(by = 'iv_value',ascending=False,inplace = True)
iv_values


# In[ ]:


cust_data.columns


# ## Logistic Regression

# ### Data preparation for Model training 

# In[ ]:


cust_data_train = cust_data[[ 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY','NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                             'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
                             'OCCUPATION_TYPE', 'CUST_FOR_MONTHS', 'cnt_child_category', 'income_bin', 'cnt_family_bin',
                             'months_bin', 'emp_years_bin', 'age_bin', 'target']]


# #### Dummy features for categorical values 

# In[ ]:


def creatingDummyVariables(df, columns) :
    # Creating a dummy variable for some of the categorical variables and dropping the first one.
    dummy1 = pd.get_dummies(df[columns], drop_first=True)
    
    # Adding the results to the master dataframe
    df1 = pd.concat([df, dummy1], axis=1)
    
    #Dropping the initial column
    df1.drop(columns, axis = 1, inplace = True)
    
    return df1


# In[ ]:


cust_data_train = creatingDummyVariables(cust_data_train, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',
                      'OCCUPATION_TYPE', 'cnt_child_category','income_bin', 'cnt_family_bin', 'months_bin',
                      'emp_years_bin', 'age_bin'])


# In[ ]:


cust_data_train.head()


# ### Model building

# In[ ]:


from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics


# In[ ]:


# Logistic regression model 
def logisticReg (df) :
    # Putting feature variable to X
    X = df.drop(['target'], axis=1)
    y = df['target']
    
    # Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)
    
    logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
    res = logm1.fit()
    return res


# In[ ]:


res = logisticReg(cust_data_train)
res.summary()


# In[ ]:


iv_values


# ### Doing feature selection based on IV values

# #### a. Removing all the columns with IV_value < 0.005

# In[ ]:


cust_data_train = cust_data[[ 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY','NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                             'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_WORK_PHONE','OCCUPATION_TYPE',
                             'CUST_FOR_MONTHS', 'income_bin', 'cnt_family_bin', 'months_bin', 'emp_years_bin', 'age_bin',
                             'target']]


# In[ ]:


cust_data_train = creatingDummyVariables(cust_data_train, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                            'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'income_bin', 'cnt_family_bin',
                            'months_bin', 'emp_years_bin', 'age_bin'])


# In[ ]:


res = logisticReg(cust_data_train)
res.summary()


# #### b. Removing all columns with IV_value < 0.015

# In[ ]:


cust_data_train = cust_data[['FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                             'OCCUPATION_TYPE','CUST_FOR_MONTHS', 'months_bin', 'emp_years_bin', 'age_bin','target']]


# In[ ]:


cust_data_train = creatingDummyVariables(cust_data_train, ['NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
                            'OCCUPATION_TYPE', 'months_bin', 'emp_years_bin', 'age_bin'])


# In[ ]:


res = logisticReg(cust_data_train)
res.summary()


# #### c. Removing all columns with IV_value < 0.02

# In[ ]:


cust_data_train = cust_data[['FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                             'OCCUPATION_TYPE','CUST_FOR_MONTHS', 'months_bin', 'emp_years_bin', 'age_bin','target']]


# In[ ]:


cust_data_train = creatingDummyVariables(cust_data_train, ['NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
                            'OCCUPATION_TYPE', 'months_bin', 'emp_years_bin', 'age_bin'])


# In[ ]:


# Putting feature variable to X
X = cust_data_train.drop(['target'], axis=1)
y = cust_data_train['target']
    
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[ ]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)

y_train_pred_final = pd.DataFrame({'target':y_train.values, 'target_Prob':y_train_pred})
y_train_pred_final['CustID'] = y_train.index

y_train_pred_final.head()


# ### Plotting ROC 

# In[ ]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.target, y_train_pred_final.target_Prob, drop_intermediate = False )


# In[ ]:


draw_roc(y_train_pred_final.target, y_train_pred_final.target_Prob)


# ### Run model on test data

# In[ ]:


X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)


# In[ ]:


y_pred_1 = pd.DataFrame(y_test_pred)
y_test_df = pd.DataFrame(y_test)
y_test_df['ID'] = y_test_df.index
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[ ]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Predicted_prob'})
# Rearranging the columns
y_pred_final = y_pred_final.reindex(['ID','target','Predicted_prob'], axis=1)
y_pred_final.head()


# In[ ]:


y_pred_final['final_predicted'] = y_pred_final.Predicted_prob.map(lambda x: 1 if x > 0.8 else 0)


# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.target, y_pred_final.final_predicted)


# In[ ]:


confusionMatrix = metrics.confusion_matrix(y_pred_final.target, y_pred_final.final_predicted )
confusionMatrix


# In[ ]:


TP = confusionMatrix[1,1] # true positive 
TN = confusionMatrix[0,0] # true negatives
FP = confusionMatrix[0,1] # false positives
FN = confusionMatrix[1,0] # false negatives


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let us calculate specificity
TN / float(TN+FP)


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# In[ ]:


cust_data_train = cust_data[['ID', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_WORK_PHONE',
       'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'CUST_FOR_MONTHS', 'EMP_YEARS', 'AGE', 'target']]


# In[ ]:


cust_data_train = creatingDummyVariables(cust_data_train, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                            'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'])


# In[ ]:


x = cust_data_train.drop('target',1)
y = cust_data_train['target']


# In[ ]:


# Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


# In[ ]:


# Create a Decision Tree
dt_basic = DecisionTreeClassifier(max_depth=10)


# In[ ]:


# Fit the training data
dt_basic.fit(x_train,y_train)


# In[ ]:


# Predict based on test data
y_preds = dt_basic.predict(x_test)


# In[ ]:


# Calculate Accuracy
accuracy_value = metrics.accuracy_score(y_test,y_preds)
accuracy_value


# In[ ]:


# Create and print confusion matrix
confusion_matrix(y_test,y_preds)


# In[ ]:


print(classification_report(y_test,y_preds))


# In[ ]:


# Calculate the number of nodes in the tree
dt_basic.tree_.node_count


# ### Hyperparameter tuning for Decision Trees

# In[ ]:


# Create a Parameter grid
param_grid = {
    'max_depth' : range(5,20,5),
    'min_samples_leaf' : range(50,210,50),
    'min_samples_split' : range(50,210,50),
    'criterion' : ['gini','entropy'] 
}


# In[ ]:


n_folds = 5


# In[ ]:


dtree = DecisionTreeClassifier()
grid = GridSearchCV(dtree, param_grid, cv = n_folds, n_jobs = -1,return_train_score=True)


# In[ ]:


grid.fit(x_train,y_train)


# In[ ]:


cv_result = pd.DataFrame(grid.cv_results_)
cv_result.head()


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


best_grid = grid.best_estimator_
best_grid


# In[ ]:


best_grid.fit(x_train,y_train)


# In[ ]:


best_grid.score(x_test,y_test)

