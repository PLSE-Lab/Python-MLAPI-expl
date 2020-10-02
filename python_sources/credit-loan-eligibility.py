#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Loan Eligibility
# <hr>
# <h4> Chintan Chitroda </h4>
# 

# ## Table Of Content:  <a id="index"></a>
# * [Problem Statement](#problemstatement)
# * [Data & Libraries Import & Analysis](#dataimport)
# * [Data Wrangling & Cleaning](#dwdc)
# * [Exporatary Data Analysis](#eda)
# * [Data Preprocessing](#pp)
# * [Machine Learning](#machinelearning)
#     * [Random Forest Classifier](#rfc)
#     * [Reccursive Feature Elimination (RFE)](#rfe)
#     * [XGB Classifier](#xgb)
#     * [Light GBM Classifier](#Lgb)
# * [Result Analysis](#resultviz)

# ### Problem Statement <a id="problemstatement"></a>
# The task is to Preidcit wheather the customer is Elgible for loan/Credit or not on the Basis of Given data columns.
# <Br>
# [Dataset can be found here](https://www.kaggle.com/shadabhussain/credit-risk-loan-eliginility)

# <hr>

# ### Data Import & Important Libraries Import <a id="dataimport"></a>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('/kaggle/input/credit-risk-loan-eliginility/train_split.csv')
print('Shape of Dataframe:',df.shape)


# In[ ]:


df.columns


# In[ ]:


df.head(10)


# In[ ]:


print(df.info())
print('\n\nNo of columns:',len(df.columns))


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.heatmap(df.isnull())


# ## Data Wrangling & Data Cleaning <a id="dwdc"></a>

# In[ ]:


dfcopy = df.copy()


# In[ ]:


toomanynull = ['mths_since_last_delinq','mths_since_last_record',
               'mths_since_last_major_derog','pymnt_plan','desc',
               'verification_status_joint']
df.drop(toomanynull,inplace=True,axis=1)


# In[ ]:


## getting numeric columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num = df.select_dtypes(include=numerics).columns


# In[ ]:


num


# In[ ]:


## getting categorical columns
cat = df.drop(num,axis=1)
cat = cat.columns


# In[ ]:


cat


# In[ ]:


df[cat].head(3)


# In[ ]:


df[num].head(3)


# ### numeric columns Handling

# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(df[num].corr(),annot=True,square=True,cmap='Set2')


# In[ ]:


df[num].isnull().sum()


# In[ ]:


## revol_util


# In[ ]:


plt.figure(figsize = (14,10)) 
plt.subplot(221)
sns.boxplot(df['revol_util'])
plt.subplot(222)
sns.violinplot(df['revol_util'])
plt.subplot(223)
df['revol_util'].plot.hist()
plt.suptitle('revol_util columns',size=20)


# #### The Distribution of the data forms bell curve & data has some outlier so we impute median in nulls

# In[ ]:


## checking mean nd median and imputing median
print(df['revol_util'].mean())
print(df['revol_util'].median())
df['revol_util'].fillna(value=df['revol_util'].median(),inplace=True)


# In[ ]:


sns.distplot(df['revol_util'])


# In[ ]:


# tot_coll_amt  (total collected amount)


# In[ ]:


plt.figure(figsize = (14,10)) 
plt.subplot(221)
sns.boxplot(df['tot_coll_amt'])
plt.subplot(222)
sns.violinplot(df['tot_coll_amt'])
plt.subplot(223)
df['tot_coll_amt'].plot.hist()
plt.suptitle('tot_coll_amt columns',size=20)

print('Mean :',df['tot_coll_amt'].mean())
print('Median :',df['tot_coll_amt'].median())


# In[ ]:


df.tot_coll_amt.value_counts()


# #### Data is totally biased and too many nulls like 0 se we drop it

# In[ ]:


df.drop('tot_coll_amt',axis=1,inplace=True)


# In[ ]:


## tot_cur_bal 


# In[ ]:


plt.figure(figsize = (14,10)) 
plt.subplot(221)
sns.boxplot(df['tot_cur_bal'])
plt.subplot(222)
sns.violinplot(df['tot_cur_bal'])
plt.subplot(223)
df['tot_cur_bal'].plot.hist()
plt.suptitle('tot_cur_bal (Total Current Balance of user) columns',size=20)

print('Mean :',df['tot_cur_bal'].mean())
print('Median :',df['tot_cur_bal'].median())


# In[ ]:


#### Data is totally biased and has too many outliers so imputing Median


# In[ ]:


df['tot_cur_bal'].fillna(value=df['tot_cur_bal'].median(),inplace=True) 


# In[ ]:


## total_rev_hi_lim


# In[ ]:


plt.figure(figsize = (14,10)) 
plt.subplot(221)
sns.boxplot(df['total_rev_hi_lim'])
plt.subplot(222)
sns.violinplot(df['total_rev_hi_lim'])
plt.subplot(223)
df['total_rev_hi_lim'].plot.hist()
plt.suptitle('total_rev_hi_lim columns',size=20)

print('Mean :',df['total_rev_hi_lim'].mean())
print('Median :',df['total_rev_hi_lim'].median())


# In[ ]:


df['total_rev_hi_lim'].fillna(value=df['total_rev_hi_lim'].median(),inplace=True) 


# In[ ]:


df['collections_12_mths_ex_med'].value_counts()


# In[ ]:


df['collections_12_mths_ex_med'].plot.hist()


# In[ ]:


## dropping column as its just 0
df.drop('collections_12_mths_ex_med',axis=1,inplace=True)


# ### Categorical columns

# In[ ]:


df[cat]


# In[ ]:


## Dropping Useless columns
print(df['batch_enrolled'].head(5))


# In[ ]:


df['title']


# In[ ]:


# 1. batch_enrolled >> it doesn't concern which batch the user was from
# 2. desc >> too many null values 
# 5. zip_code >> not a significant column
## the columns is no significance so we drop it
temp = ['batch_enrolled','zip_code']
df.drop(temp,axis=1,inplace=True)


# ####  Employment Title (emp_title)

# In[ ]:


df['emp_title'].value_counts()


# In[ ]:


df.purpose


# In[ ]:


df.title


# In[ ]:


#we drop 'title' as its serves  same pupose as 'purpose'
df.drop('title',axis=1,inplace=True)


# In[ ]:


## Replaceing Nan Employment Type with 'Unknown' as we cannot mode it and guess it(impute)
df['emp_title'].fillna(value="Unknown",inplace=True)


# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num = df.select_dtypes(include=numerics).columns
cat = df.drop(num,axis=1)
cat = cat.columns


# In[ ]:


## emp_length
df['emp_length'].head()


# In[ ]:


## extracts the number form emp_length
df['emp_length'] = df['emp_length'].astype(str)
df['emp_length'].replace("[^0-9]","",regex=True,inplace=True)
df['emp_length'].replace("","-1",regex=True,inplace=True)
df['emp_length'] = df['emp_length'].apply(lambda x: x.strip())


# In[ ]:


df.emp_length = df.emp_length.astype(int)


# In[ ]:


## here -1 stands for unknown
df['emp_length'].fillna(value='-1',inplace=True)


# In[ ]:


### remoing moths tag from term
df.term = df.term.apply(lambda x: x.split(' ')[0])
df.term = df.term.astype(int)


# In[ ]:


df[cat].isnull().sum()


# In[ ]:


df.verification_status.value_counts()


# In[ ]:


df[cat]


# In[ ]:


## serves no relevance
df.drop('addr_state',inplace=True,axis=1)


# In[ ]:


## extracts the number form 'last_week_pay'
df['last_week_pay'] = df['last_week_pay'].astype(str)
df['last_week_pay'].replace("[^0-9]","",regex=True,inplace=True)
df['last_week_pay'].replace("","-1",regex=True,inplace=True)
df['last_week_pay'] = df['last_week_pay'].apply(lambda x: x.strip())
df.last_week_pay = df.last_week_pay.astype(int)


# In[ ]:


df['last_week_pay']


# ## EDA <a id="eda"></a>

# In[ ]:


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[ ]:


## Making seprate df for Visualization
df1 = df.copy()
df1.drop('member_id',inplace=True,axis=1)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num = df1.select_dtypes(include=numerics).columns
cat = df1.drop(num,axis=1)
cat = cat.columns


# In[ ]:


df1.loan_status.value_counts().values


# In[ ]:


### getting ratio of target variable to check balance between values
labels = ['Loan Granted','Loan Not Granted']
fig = px.pie(names=labels,values = df1.loan_status.value_counts().values,title='Percentage Loan Granted of Total Application')
fig.update_traces(hole=.4, hoverinfo="label+percent+name")
fig.show()


# ## we see there are many only 1/4 people were granted loan

# In[ ]:


# data prepararion
from wordcloud import WordCloud 
x2011 = df1.emp_title
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(x2011))
plt.title('Employment Types Word Cloud',size=25)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

This shows us the most frequent word in emp_title which tells us that Director,Manager,Superviseor,Teacher are apply more for loan in compare to other profession
# In[ ]:


px.histogram(df1,x='loan_amnt',color='loan_status',title='Loam amount W.R.t Loan Status',
             labels = labels)

This shows us distribution of Loan amnt appplied for and count and frequency the loan was passed for in red and blue respectively
# In[ ]:


## lets see successs rate for loan pass for each profession people
temp = pd.DataFrame()
temp['emp'] = df1.emp_title
temp['loan_status'] = df1.loan_status

list1 = temp.emp.value_counts().head(25).index

labels = ['Loan Granted','Loan Not Granted']
for i in list1:
    temp1 = temp[temp.emp == i].loan_status.value_counts()
    fig1 = make_subplots(rows=1, cols=2)
    fig1.add_trace(go.Pie(labels=labels,values=temp1))
    fig1.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig1.update_layout(
        title_text="Percentage Loan Pass Success according to Employment -- "+ i,
        # Add annotations in the center of the donut pies.
        annotations=[dict(text=i, x=0.50,y=0.5, font_size=20, showarrow=False)])
    fig1.show()


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(df1.purpose,df1.loan_amnt)
plt.title('loan amount passed for each purpose',size=20)

This shows the HIgh Loan amount were granted for purpose like small business,credit card, debt_consolation etc,
# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(df1.emp_length,df1.loan_amnt)
plt.title('Loan amnt wrt Experience',size=20)
plt.xlabel('NO of Year Experience')


# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(df1.term,df1.loan_amnt)
plt.title('Term period wrt loan amount',size=20)


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(df1.grade,df1.loan_amnt)
plt.title('Grade wrt Loan amount',size=20)


# In[ ]:


temp = ['loan_amnt','funded_amnt','funded_amnt_inv']
sns.heatmap(df[temp].corr(),annot=True)

We see there three columns are exactly same so only take loan_amnt and drop rest
# ### dropping 

# In[ ]:


df.drop('funded_amnt',axis=1,inplace=True)
df.drop('funded_amnt_inv',axis=1,inplace=True)


# In[ ]:


temp = ['total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee']


# In[ ]:


for i in temp:
    sns.distplot(df[i])
    plt.show()


# In[ ]:


for i in temp:
    print(df[i].value_counts())


# we see 'total_rec_late_fee','recoveries','collection_recovery_fee' has only 0 in it so we drop them

# In[ ]:


temp = ['total_rec_late_fee','recoveries','collection_recovery_fee']
df.drop(temp,axis=1,inplace=True)


# In[ ]:


df.acc_now_delinq.value_counts()


# In[ ]:


## has only 0 in it mostly so we drop it
df.drop('acc_now_delinq',axis=1,inplace=True)


# In[ ]:


df.delinq_2yrs.value_counts()


# In[ ]:


sns.distplot(df.delinq_2yrs)


# In[ ]:


df.drop('delinq_2yrs',axis=1,inplace=True)


# In[ ]:


df.pub_rec.value_counts()


# In[ ]:


sns.distplot(df.pub_rec)


# In[ ]:


df.drop('pub_rec',axis=1,inplace=True)


# In[ ]:


sns.countplot(df.application_type)


# In[ ]:


## its one sided data so we drop column
df.drop('application_type',axis=1,inplace=True)


# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num = df.select_dtypes(include=numerics).columns
cat = df.drop(num,axis=1)
cat = cat.columns


# In[ ]:


df[cat]


# In[ ]:


df[num]


# <hr>

# ## Data PreProcessing <a id="pp"></a>

# In[ ]:


for i in df[cat].columns:
    print(i,":\n\n",df[i].value_counts())


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


df[cat] = df[cat].apply(LabelEncoder().fit_transform)


# In[ ]:


df.home_ownership.value_counts()


# In[ ]:


dfcopy.home_ownership.value_counts()


# In[ ]:


df[cat].head(10)


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,square=True,cmap='ocean')


# In[ ]:


toohightcorr = ['grade','sub_grade','total_rev_hi_lim','total_acc']


# In[ ]:


df.drop(toohightcorr,axis=1,inplace=True)


# In[ ]:


## Storing member id 
ids = df['member_id']
df.drop('member_id',axis=1,inplace=True)


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,square=True,cmap='ocean')
plt.title('After Removing Highly Co-related Columns')


# ## Machine learning Algorithm <a id="machinelearning"></a>

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score


# In[ ]:


X = df.drop('loan_status',axis=1)
y = df['loan_status']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=101)


# ## Random Forest Classifier <a id="rfc"></a>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=100,max_depth=8, random_state=101,class_weight='balanced')
rfc.fit(X_train,y_train)


# In[ ]:


y_pred = rfc.predict(X_test)


# In[ ]:


print('AUC-ROC Score :',roc_auc_score(y_test, y_pred))
print('Report:\n',classification_report(y_test, y_pred))
print('confusion Matrix:\n',confusion_matrix(y_pred,y_test))
print('cross validation:',cross_val_score(rfc, X, y, cv=3).mean())


# In[ ]:


importances=rfc.feature_importances_
feature_importances=pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10,7))
sns.barplot(x=feature_importances[0:10], y=feature_importances.index[0:10])
plt.title('Feature Importance',size=20)
plt.ylabel("Features")
plt.show()


# ## Using RFE (Reccursive Feature Elimination) <a id="rfe"></a>

# In[ ]:


from sklearn.feature_selection import RFE


# In[ ]:


rfe = RFE(rfc, 8) 
rfe.fit(X_train,y_train)


# In[ ]:


rfecols = X_train.columns[rfe.support_]


# In[ ]:


rfecols


# Using Rfe selected Featrues

# In[ ]:


rfc = RandomForestClassifier(n_estimators=200,random_state=101,class_weight='balanced')
rfc.fit(X_train[rfecols],y_train)
y_pred = rfc.predict(X_test[rfecols])
print('AUC-ROC Score :',roc_auc_score(y_test, y_pred))
print('Report:\n',classification_report(y_test, y_pred))
print('confusion Matrix:\n',confusion_matrix(y_test,y_pred))
#print('cross validation:',cross_val_score(rfc, X, y, cv=3).mean())


# #### Under Sampling X_train

# In[ ]:


X_train['laon_status'] = y_train
X_train.laon_status.value_counts()
temp = X_train[X_train.laon_status == 0].sample(12000)
X_train = X_train[X_train.laon_status==1]
X_train = X_train.append(temp)
X_train.laon_status.value_counts()
X_train = X_train.sample(frac=1)


# In[ ]:


y_train = X_train.laon_status
X_train.drop('laon_status',axis=1,inplace=True)


# ## Oversampling

# In[ ]:


X_train['laon_status'] = y_train
temp = X_train[X_train.laon_status==1]
X_train = X_train.append(temp)
X_train = X_train.append(temp)


# In[ ]:


X_train.laon_status.value_counts()


# In[ ]:


X_train = X_train.sample(frac=1)


# In[ ]:


y_train = X_train.laon_status
X_train.drop('laon_status',axis=1,inplace=True)


# ### after oversampling result with rfe and without rfe cols

# with Rfe

# In[ ]:


rfc = RandomForestClassifier(n_estimators=200,random_state=101,class_weight='balanced')
rfc.fit(X_train[rfecols],y_train)
y_pred = rfc.predict(X_test[rfecols])
print('AUC-ROC Score :',roc_auc_score(y_test, y_pred))
print('Report:\n',classification_report(y_test, y_pred))
print('confusion Matrix:\n',confusion_matrix(y_test,y_pred))
#print('cross validation:',cross_val_score(rfc, X, y, cv=3).mean())


# ## XGB Classifier <a id="xgb"></a>

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb = XGBClassifier(n_estimator=100,max_depth=12,class_weight='balanced',refit='AUC')


# In[ ]:


xgb.fit(X_train[rfecols],y_train)


# In[ ]:


y_pred = xgb.predict(X_test[rfecols])


# In[ ]:


print('AUC-ROC Score :',roc_auc_score(y_test, y_pred))
print('Report:\n',classification_report(y_test, y_pred))
print('confusion Matrix:\n',confusion_matrix(y_pred,y_test))
#print('cross validation:',cross_val_score(xgb, X, y, cv=3).mean())


# ## LigthGBM <a id="lgb"></a>

# In[ ]:


import lightgbm as lgb


# In[ ]:


#y_train = y_train.values


# In[ ]:


model = lgb.LGBMClassifier(n_estimators=600,random_state=101,max_depth=8,class_weight='balanced')
model.fit(X_train[rfecols], y_train)


# In[ ]:


y_pred = model.predict(X_test[rfecols])


# In[ ]:


print('AUC-ROC Score :',roc_auc_score(y_test, y_pred))
print('Report:\n',classification_report(y_test, y_pred))
print('confusion Matrix:\n',confusion_matrix(y_pred,y_test))
print('cross validation:',cross_val_score(model, X, y, cv=5).mean())


# ## Analyzing & Visualizing Results <a id="resultviz"></a>

# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
lgb.plot_importance(model, max_num_features=10, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# ## ROC - AUC curve
# - AUC stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1). AUC provides an aggregate measure of performance across all possible classification thresholds.
# 

# In[ ]:


fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d')
plt.title('Confusion Matrix',size=20)


# In[ ]:


tempval = pd.Series(y_test).value_counts()
tempvalpred = pd.Series(y_pred).value_counts()


# In[ ]:


labels = ['Loan Granted','Loan Not Granted']

fig1 = make_subplots(rows=1, cols=2)
fig1.add_trace(go.Pie(labels=labels, values=tempval))
fig2 = make_subplots(rows=1, cols=2)
fig2.add_trace(go.Pie(labels=labels,values=tempvalpred))

# Use `hole` to create a donut-like pie chart
fig1.update_traces(hole=.4, hoverinfo="label+percent+name")
fig2.update_traces(hole=.4, hoverinfo="label+percent+name")

fig1.update_layout(
    title_text="Predicted Vs Actual Loan Granted Ratio Comparision",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Actual Loan Status Ratio', x=0.25, y=0.5, font_size=20, showarrow=False)])
fig2.update_layout(
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Predicted Loan Status Ratio', x=0.25,y=0.5, font_size=20, showarrow=False)])


fig1.show()
fig2.show()


# ## Key Points Conclusion

# * After analysis the Loan Granting status of Banks is 25%.
# * Means only 25% of applicants are granted loan of total application.
# * The Accuracy in cosideration with AUC is 81 %
# * Interest Rate & Last week Pay have higher Significance in Predicting Loan status.
# * The Ratio of Acutal and Predicted has 1% error.

# ### Thank you 
