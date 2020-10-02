#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
from sklearn import cluster
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix
import math


# # Lending Club
# ## Lending Club dataset is a database of about 2.2 Million Loans with 145 features in them.
# ## Each Loan has it's Status which can be classified as "Bad" Loan or "Good" Loan; by grouping "Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)" as "BAD LOAN" and considering "Fully Paid" as "GOOD LOAN"
# ## There are a significant recent loans which aren't classified as either "GOOD LOAN" or "BAD LOAN", in this notebook, I am trying to train a model on the classified "GOOD" and "BAD" Loans to try and predict which one of the "Current" Loans will become "GOOD" Loans and which would become  "BAD"
# 
# <a id='table_of_contents'></a>
# 
# 
# 
# ## Table of Contents
# 
# ### <a href=#loan_condition>Loan Condition Analysis</a>
# ### <a href=#numerical_features>Numerical Features Analysis across Loan Condition</a>
# ### <a href=#categorical_features>Catogorical Features Analysis across Loan Condition</a>
# ### <a href=#feature_engineering>Feature Engineering</a>
# ### <a href=#predictive_model>Predictive Model</a>
# ### <a href=#results>Results</a>
# ### <a href=#future_scope_for_improvement>Future scope for Improvement</a>
# 

# In[2]:


load_data = pd.read_csv('../input/lending-club-loan-data/loan.csv', low_memory=False)


# In[3]:


load_data.head()


# In[4]:


load_data.shape


# In[5]:


load_data.isnull().sum()[load_data.columns[load_data.isnull().mean()<0.8]]


# ### Removing all columns with more than 80% missing Values
# ### Removing a few rows with a lot of missing columns
# ### Fill Missing Values with 0
# ### Remove Outlier for annual income further than 3 std dev
# #### Iterate back on these with more time

# In[6]:


load_data = load_data[load_data.columns[load_data.isnull().mean()<0.8]]
load_data = load_data[~load_data.earliest_cr_line.isnull()]
load_data = load_data.fillna(0)
# Remove outliers fro annual_income


# # Get Issue Year and Issue Month
# 

# In[7]:


date_issued = pd.to_datetime(load_data['issue_d'])
date_last_payment = pd.to_datetime(load_data['last_pymnt_d'])
load_data['issue_year']=date_issued.dt.year
load_data['issue_month']=date_issued.dt.month




# 
# 
# 
# <a id='loan_condition'></a>
# 
# 
# # Loan Condition
# ## Let's Group the Different Loan Status into "Bad Loan", "Good Loan" and "Current"
# 
# 
# ### <a href=#table_of_contents>Back to Table of Contents</a>

# In[8]:


bad_loan = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period", 
            "Late (16-30 days)", "Late (31-120 days)"]


load_data['loan_condition'] = np.nan

def loan_condition(status):
    if status in bad_loan:
        return 'Bad Loan'
    elif status == "Current":
        return 'Current'
    else:
        return 'Good Loan'
    
    
loan_status = load_data['loan_status'].apply(loan_condition)
load_data['loan_condition']=loan_status


# In[9]:


f, ax = plt.subplots(1,2, figsize=(16,8))

colors = ["#0B6623","#E1AD01", "#D72626" ]
labels ="Good Loans","Current", "Bad Loans"

plt.suptitle('Information on Loan Conditions', fontsize=20)

load_data["loan_condition"].value_counts().plot.pie(explode=[0,0.1, 0.2], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=70)


# ax[0].set_title('State of Loan', fontsize=16)
ax[0].set_ylabel('% of Condition of Loans', fontsize=14)


palette = ["#E1AD01","#0B6623", "#D72626" ]

sns.countplot(x="issue_year", hue="loan_condition", data=load_data, palette=palette)


# In[10]:


def countplot_category_against_loan_condition(column):
    palette = ["#E1AD01","#0B6623", "#D72626" ]
    order = sorted(load_data[column].unique())
    sns.countplot(x=column, hue="loan_condition", data=load_data, palette=palette, order=order)
    
countplot_category_against_loan_condition('grade')    
    


# In[11]:


color = ["#D72626" ,"#E1AD01","#0B6623" ]
ax = load_data.groupby(['issue_year','loan_condition']).int_rate.mean().unstack().plot(title="Interest Rate by Load Status", color=color)
ax.set_ylabel('Interest Rate (%)', fontsize=12)
ax.set_xlabel('Year', fontsize=12)


# <a id='feature_engineering'></a>
# 
# 
# 
# 
# 
# 
# # Feature Engineering
# 
# ### <a href=#relative_income_index>Relative Income Index</a>
# ### <a href=#payment_index>Payment Index</a>
# ### <a href=#employment_length>Employment Length</a>
# ### <a href=#employment_title>Employment Title</a>
# 
# 
# 
# 
# <a id='relative_income_index'></a>
# 
# 
# ## Relative Income Index
# ### I wanted to try stitching some data Using the applicants Income, State, Loan Issue Date, Purchasing Power Index per State per Year, Median Household Income per year per state to get a single value that can represent the person relative income index normalized across time and state
# 
# 
# ### The reason behind this thinking was that the US economy could be affected by various factors through time. For Example, the economic crisis in 2008 could've hindered the economy which could've reduced a person's income reported at that time of loan application. When I am modelling the classification of loans, it made sense to normalize his income across time and geography using different data sources
# 
# ### I found 2 datasets on which gave me break down of Purchasing Power Per State per year and median :
# https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_income
# ### I have attached hte CSV files in the email for reference
# 
# 
# ### <a href=#table_of_contents>Back to Table of Contents</a>

# In[12]:


house_hold_income = pd.read_csv("../input/purchase-power-index/HouseHold_Income_by_State_by_year.csv")
house_hold_income= house_hold_income.drop(["Unnamed: 0","State"], axis=1)
house_hold_income.head()


# In[13]:


house_hold_income = house_hold_income.set_index('State_abbr').stack().reset_index().rename(columns ={'level_1':'year',0:'median_household_income'})
house_hold_income['year'] = house_hold_income['year'].astype(int)
house_hold_income.head()


# In[14]:


load_data = pd.merge(house_hold_income,load_data,right_on=['addr_state','issue_year'],left_on=['State_abbr','year'],how='right')
load_data = load_data.drop(["State_abbr","year"], axis=1)


# In[15]:


ppi = pd.read_csv("../input/purchase-power-index/ppi.csv")
ppi.head()


# In[16]:



ppi = ppi.drop("GeoName",axis=1)
ppi = ppi.set_index('State_abbr').stack().reset_index().rename(columns = {'level_1':'year',0:'ppi'})
ppi['year'] = ppi['year'].astype(int)
ppi.head()


# In[17]:



load_data = pd.merge(ppi,load_data,right_on=['addr_state','issue_year'],left_on=['State_abbr','year'],how='right')
load_data = load_data.drop(["State_abbr","year"], axis=1)
load_data['relative_income_index'] = (load_data['annual_inc']/load_data['ppi']*100) - load_data['median_household_income']
load_data = load_data.drop(["ppi","median_household_income"], axis=1)


# <a id='payment_index'></a>
# 
# 
# 
# 
# ## Payment Index
# ### Using features like "total amount of principle paid" might make your model learn something like, "low amount of principle paid" is a Bad Loan and "high amount of principle paid" is a Good Loan. This could be dangerous since most of the "Current" Loans are recent loans, which might have "low amount of principle paid" as they haven't gotten the time to repay them yet. Hence I am trying to come up with a "Payment Index" which would be a ration of months_into_loan / ratio_of_principle_paid
# 
# ### This will ensure newer loans are not penalised and even might reward loans which are ahead of it's target
# 
# 
# ### <a href=#feature_engineering>Back to Feature Engineering</a>

# In[18]:


def sigmoid(x):
    if x < -709:
        return 0.0
    else:
        return 1.0 / (1.0 + math.exp(-x))


months_into_loan = 12*(date_last_payment.dt.year-load_data['issue_year']) + (date_last_payment.dt.month-load_data['issue_month'])
payment_index = (months_into_loan / pd.to_numeric(load_data['term'].str.extract('(\d+)')[0]))/(load_data['out_prncp']/load_data['funded_amnt'])
load_data['payment_index'] = payment_index.fillna(0)
load_data['payment_index'] = load_data['payment_index'].apply(lambda x: sigmoid(x))


# <a id='employment_length'></a>
# 
# 
# 
# ## Employment Lenght
# 
# 
# ### <a href=#feature_engineering>Back to Feature Engineering</a>
# 

# In[19]:


load_data['emp_length_integer'] = load_data.emp_length.replace({'< 1 year':0,     
                                                      '1 year':1,
                                                      '2 years':2,
                                                      '3 years':3,
                                                      '4 years':4,
                                                      '5 years':5,
                                                      '6 years':6,
                                                      '7 years':7,
                                                      '8 years':8,
                                                      '9 years':9,
                                                      '10+ years':10,
                                                      None:-1})
palette = ["#E1AD01","#0B6623", "#D72626" ]
countplot_category_against_loan_condition('emp_length_integer')
"""
NUM_CLUSTERS = range(3, 20)
model = [cluster.KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=1) for i in NUM_CLUSTERS]
score = [model[i].fit(X).score(X) for i in range(len(model))]
plt.plot(NUM_CLUSTERS,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
"""


# <a id='employment_title'></a>
# 
# 
# ## Employment Title
# ### There are a lot of titles provided by the applicants. I used TFIDF Vectorizer followed by K Means to try and cluster the titles together. I would have also like to get a word2vec embedding and validate the cluster formation with more time. I also would have like to determine the ideal number of clusters using Elbow Method. Didn't get into the details for this feature, as it needs a more work
# 
# 
# ### <a href=#feature_engineering>Back to Feature Engineering</a>

# In[20]:


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(load_data.emp_title.replace({0:"Missing"}))
NUM_CLUSTERS = 10
model = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
load_data['emp_title_tfidf'] = model.predict(X)
sns.countplot(x="emp_title_tfidf",data=load_data)


# In[21]:


load_data['emp_title_tfidf'] = load_data['emp_title_tfidf'].astype('object')


# In[22]:


def distplot_numberical_feature_across_loan_condition(column):
    plt.figure()
    remove_outliers = load_data[~((load_data[column]-load_data[column].mean()).abs() > 3*load_data[column].std())]
    palette = ["#0B6623", "#D72626" , "#E1AD01"]
    loan_condition = ["Good Loan","Bad Loan","Current"]
    for i in range(len(loan_condition)):
        # Subset to the airline

        subset = remove_outliers[remove_outliers['loan_condition'] == loan_condition[i]]
        # Draw the density plot
        sns.distplot(subset[column], hist = False, kde = True,
                     kde_kws = {'linewidth': 1},
                     label = loan_condition[i],
                     color = palette[i])

    # Plot formatting
    plt.legend(prop={'size': 16}, title = 'Loan Condition')
    plt.title('Density Plot of {}'.format(column))
    plt.ylabel('Density')

    


# <a id='numerical_features'></a>
# 
# 
# # Numerical Features
# 
# ## What to read from these plots?
# ### Ideally, I would like to pick Numerical Features that bring out a variation in the distribution of the features across Good Loans and Bad Loans. Since most of the current loans are recent loans, as compared to the older loans which already have a result of "Bad Loan" or "Good Loan",  some of the features might be time bound. For example, "tot_cur_bal" is a feature which represents the Current Balance in the Loan. When training my model on just the "Good Loan" and "Bad Loan" Category, it is very easy for my model to learn things like "having a low current balance" implies "Good Loan", and "having a high current balance" implies "Bad Loan". But this is a bad feature to learn since all of my recent loans would also have a "low current balance", hence these features can't be used in their raw format.  
# 
# 
# ## loan_amnt, funded_amnt, funded_amnt_inv
# ### These seem like they are similar features and might have a very high correlation to each other
# 
# ## int_rate
# ### Interest Rate would be a feature which could potential indicate a good loan from a bad loan, but I would like to exclude this feature from the model as I would like to use int_rate as a multiplying factor with the predicted probability of the loan being good to determine if the loan is worth investing in. An example of how this would add value is :
# 
# ### Let's try to rank these loans in the order of preference to invest in
# ### LOAN A -  Predicted Probablity of being a good loan = 0.4 , interest rate = 15%
# ### LOAN B -  Predicted Probablity of being a good loan = 0.45 , interest rate = 12%
# ### Without using "int_rate" as a multiplying factor, we would choose Loan B over Loan A ; But if we use "int_rate" as a multiplying factor, we would be choosing Loan A(0.4 x 15/100 = 0.06) over Loan B(0.45 x 12/100 = 0.054)
# 
# 
# ### <a href=#table_of_contents>Back to Table of Contents</a>

# In[ ]:


for numeric_features in load_data.columns[load_data.dtypes!='object'].tolist():
    distplot_numberical_feature_across_loan_condition(numeric_features)


# <a id='categorical_features'></a>
# 
# 
# 
# 
# # Categorical Features
# 
# ## What to read from these plots?
# ### Ideally, I would like to pick Categorical Features that bring out a variation of Good Loans/Bad Loan ratio across the different categorical values. I would also remove those features which do not have adequate number of representation of all 3 loan conditions.
# 
# 
# ## pymnt_plan, hardship_flag, debt_settlement_flag, disbursement_method
# ### Although these variables seems really important, they would be adding little value or probably harmful to the model since they don't have enough representation of the 3 loan_conditions across their classes
# 
# 
# 
# ### <a href=#table_of_contents>Back to Table of Contents</a>

# In[ ]:


cat_columns = ["term",
                "grade",
                "home_ownership",
                "verification_status",
                "purpose",
                "application_type",
                "hardship_flag",
                "debt_settlement_flag",
                "pymnt_plan",
                "disbursement_method"]
for categorical_features in cat_columns:
    plt.figure()
    palette = ["#0B6623", "#D72626","#E1AD01" ]
    sns.countplot(x=categorical_features, hue="loan_condition", data=load_data, palette=palette)


# 

# In[ ]:





# In[ ]:


date_columns = ["issue_d",
                "issue_year",
                "issue_month",
               "earliest_cr_line",
               "last_pymnt_d",
               "next_pymnt_d",
               "last_credit_pull_d"]
onehot_columns = ["term",
                    "grade",
                    "home_ownership",
                    "verification_status",
                    "purpose",
                    "application_type"]
string_columns =["emp_title",
                    "emp_length",
                    "title",
                    "zip_code",
                    "addr_state",
                    "loan_condition"]

engineered_features =["emp_length_integer",
                     "emp_title_tfidf"]

remove_columns = ["title",
                  "zip_code",
                  "loan_status",
                 "policy_code",
                 "acc_now_delinq"]
remove_columns_time_skewed = [
                        "total_rec_prncp",
                        "total_pymnt",
                        "total_rec_int",
                        "total_pymnt_inv",
                        "max_bal_bc",
                        "last_pymnt_amnt",
                        "out_prncp",
                        "out_prncp_inv",
                        "recoveries",
                        "collection_recovery_fee",
                        "total_rec_late_fee",
                        "inq_last_6mths",
                        "mths_since_rcnt_il",
                        "mths_since_recent_bc",
                        "mths_since_recent_bc_dlq",
                        "mths_since_recent_inq",
                        "mths_since_recent_revol_delinq"
                 ]


remove_one_hot_column = ["hardship_flag",
                        "debt_settlement_flag",
                        "pymnt_plan",
                        "disbursement_method"]




# <a id='predictive_model'></a>
# 
# 
# # Predictive Model
# 
# ## <a href=#prepare_test_train>Pick Features and Prepare Train Test Dataset</a>
# ## <a href=#feature_selection>Feature Selection</a>
# ## <a href=#gradient_boost>Gradient Boost</a>
# ## <a href=#logistic_regression>Logistic Regression</a>
# ## <a href=#xgboost>XGBoost Classification</a>
# ## <a href=#feature_importance>Feature Importance</a>
# 
# <a id='prepare_test_train'></a>
# 
# 
# 
# 
# 
# ### <a href=#table_of_contents>Back to Table of Contents</a>

# In[ ]:


onehot_loan_df = pd.get_dummies(load_data[list(set(onehot_columns)-set(remove_one_hot_column))])
print(onehot_loan_df.shape)
features_df = onehot_loan_df.join(load_data.drop((date_columns+
                                                  onehot_columns+
                                                  string_columns+
                                                  remove_columns+
                                                 remove_columns_time_skewed),axis=1))
features_df = features_df.drop(features_df.columns[features_df.dtypes=='object'].tolist(), axis=1)


# In[ ]:


features_df['loan_status']=load_data['loan_condition']
features_df.head()


# In[ ]:


current_loans = features_df[features_df['loan_status']=="Current"]
current_loan_status = features_df[features_df['loan_status']=="Current"]['loan_status']


# In[ ]:


past_loans = features_df[features_df['loan_status']!="Current"]
past_loan_status = features_df[features_df['loan_status']!="Current"]['loan_status']


# In[ ]:


past_loan_status = past_loan_status.replace({'Good Loan':1,
                         'Bad Loan':0})
past_loans = past_loans.drop("loan_status", axis=1)
current_loans = current_loans.drop("loan_status", axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(past_loans, 
                                                    past_loan_status, 
                                                    test_size=0.20, 
                                                    random_state=42,
                                                    stratify=past_loan_status)


# <a id='feature_selection'></a>
# 
# 
# 
# 
# ## Feature Selection
# ### Nice to have, but it would in the current setup encode the variable names hence losing the interpretability of the feature importance in the XGBoost Step. Commenting this for now
# ## <a href=#predictive_model>Back to Predictive Model</a>

# In[ ]:


"""
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X_train, y_train)
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)
"""


# <a id='gradient_boost'></a>
# 
# 
# 
# ## Gradient Boost
# 
# 
# 
# ## <a href=#predictive_model>Back to Predictive Model</a>

# In[ ]:


gb = GradientBoostingClassifier(n_estimators=50,learning_rate=0.1)
gb.fit(X_train,y_train)
y_gb_pred = gb.predict(X_test)
print(classification_report(y_test,y_gb_pred, target_names=["Bad Loan","Good Loan"]))


# In[ ]:


"""
gb_params = {"n_estimators":np.arange(40,80,10),"learning_rate":np.arange(0.05,0.3,0.05)}
grid_gb = GridSearchCV(GradientBoostingClassifier(),gb_params)
grid_gb.fit(X_train,y_train)
print(classification_report(y_test, grid_gb.predict(X_test), target_names=["Lost Sales","Funded Loans"]))
print(grid_gb.best_params_)

"""


# In[ ]:





# In[ ]:


labels = ["Good Loan","Bad Loan"]
cm = confusion_matrix(y_true = np.array(y_test),
                      y_pred = pd.DataFrame(gb.predict(X_test)))
print(cm)


# <a id='logistic_regression'></a>
# 
# 
# 
# ## Logistic Regression
# 
# 
# 
# ## <a href=#predictive_model>Back to Predictive Model</a>

# In[ ]:


scaler = StandardScaler()
X_train_preprocessed_std = scaler.fit_transform(X_train)
X_test_preprocessed_std = scaler.fit_transform(X_test)
lr_std = LogisticRegression(class_weight="balanced")
lr_std.fit(X_train_preprocessed_std,y_train)
y_lr_pred = lr_std.predict(X_test_preprocessed_std)
print(classification_report(y_test,y_lr_pred, target_names=["Bad Loan","Good Loan"]))


# In[ ]:


lr = LogisticRegression(max_iter=400, class_weight="balanced")
lr.fit(X_train,y_train)
y_lr_pred = lr.predict(X_test)
print(classification_report(y_test,y_lr_pred, target_names=["Bad Loan","Good Loan"]))


# In[ ]:


labels = ["Good Loan","Bad Loan"]
cm = confusion_matrix(y_true = np.array(y_test),
                      y_pred = pd.DataFrame(lr.predict(X_test)))
print(cm)


# <a id='xgboost'></a>
# 
# 
# 
# 
# 
# 
# ## XGBoost
# 
# ## <a href=#predictive_model>Back to Predictive Model</a>

# In[ ]:


XGB_model = XGBClassifier()
XGB_model.fit(X_train,y_train)


# In[ ]:



y_xgb_pred = XGB_model.predict(X_test)
print(classification_report(y_test,y_xgb_pred, target_names=["Bad Loan", "Good Loan"]))


# <a id='feature_importance'></a>
# 
# 
# 
# ## Feature Importance
# 
# ## <a href=#predictive_model>Back to Predictive Model</a>
# 

# In[ ]:


plot_importance(XGB_model, max_num_features= 15)


# <a id='results'></a>
# 
# 
# 
# 
# # Results
# 
# ## <a href=#auc_plot>AUC PLOT</a>
# ## <a href=#Confusion_Matrix>Confusion Matrix</a>
# ## <a href=#predict_current_loans>Predict Current Loans</a>
# 
# 
# ### <a href=#table_of_contents>Back to Table of Contents</a>

# In[ ]:



def add_roc_curve(y_pred, model_name, y_test=y_test, plt=plt):
    fpr, tpr, thres = roc_curve(y_test, y_pred)
    auc = round(roc_auc_score(y_test, y_pred),2)
    plt.plot(1-fpr,tpr,label="{model_name}, auc={auc}".format(model_name=model_name,auc=auc))
    plt.legend(loc=0)
    return(plt)


# <a id='auc_plot'></a>
# 
# 
# 
# 
# # AUC PLOT
# 
# ### <a href=#results>Back to results</a>

# In[ ]:


gb_plt = add_roc_curve(pd.DataFrame(gb.predict_proba(X_test))[1], y_test= y_test, model_name="Gradient Boost")
xgb_plt = add_roc_curve(pd.DataFrame(XGB_model.predict_proba(X_test))[1], y_test= y_test, model_name="XGBoost")
lr_plt = add_roc_curve(pd.DataFrame(lr.predict_proba(X_test))[1], y_test= y_test, model_name="Logistic")
lr_std_plt = add_roc_curve(pd.DataFrame(lr_std.predict_proba(X_test))[1], y_test= y_test, model_name="Logistic_std")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


# <a id='Confusion_Matrix'></a>
# 
# 
# 
# 
# # Confusion_Matrix
# 
# ### <a href=#results>Back to results</a>

# In[ ]:


labels = ["Good Loan","Bad Loan"]
cm = confusion_matrix(y_true = np.array(y_test),
                      y_pred = pd.DataFrame(XGB_model.predict(X_test)))
print(cm)


# In[ ]:


cm = confusion_matrix(y_true = np.array(y_train),
                      y_pred = pd.DataFrame(XGB_model.predict(X_train)))
print(cm)


# In[ ]:


y_train.value_counts()


# In[ ]:


pd.Series(XGB_model.predict(current_loans)).value_counts()


# In[ ]:


palette = ["#0B6623", "#D72626" ]

sns.countplot(x="issue_year", hue="loan_condition", data=load_data[load_data['loan_condition']!="Current"], palette=palette)


# <a id='predict_current_loans'></a>
# 
# 
# 
# 
# # Predict Current Loans
# 
# ### <a href=#results>Back to results</a>

# In[ ]:


palette = ["#D72626" , "#0B6623"]
current_loans_raw_data = load_data[load_data['loan_condition']=="Current"]
#keeping threshold low; as model needs more tweeks
current_loans_raw_data['predicted_loan_condition'] = (pd.DataFrame(XGB_model.predict_proba(current_loans))[1]>0.12)
sns.countplot(x="issue_year", hue="predicted_loan_condition", data=current_loans_raw_data, palette=palette)


# <a id='future_scope_for_improvement'></a>
# 
# 
# 
# # Future Scope for Improvement
# ## More Meaningful Feature Engineering
# ### Need Domain knowledge to interpret the meaning of features like revolving trades and open trades and how the opening of such accounts would factor in the prediction of the applicant's loan
# ### Hyperparameter Tuning, For now, I have gone with a Vanila XGBoost Model without paying attention to the hyperparameter tuning. I would be very interested to find out if the performance would vary with parameters like "scale_pos_weight" to balance the classes; max_depth, learning rate and even try out a custom loss function
# 
# 
# ### <a href=#table_of_contents>Back to Table of Contents</a>
