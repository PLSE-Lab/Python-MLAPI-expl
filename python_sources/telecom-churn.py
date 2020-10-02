#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn [](http://) Project

# ### Problem Statement
# In the telecom industry, customers switch their operators when they come across better deals. Since it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention is more important than customer acquisition. In this project, we predict the churn for pre-paid customers who are predominant in the Indian and Southeast Asian market. The business objective is to predict the churn in the ninth month using the data  from the first three months.

# ### Solution Abstract
# We start with exploratory data analysis, impute null values and drop columns that do not add additional information. We then filter for high value customers who have recharged with an amount more than or equal the 70th percentile of the average recharge amount in the first two months. We also derived 6 columns. We then prepared the data and applied multiple algorithms to generate a model by hyperparameter tuning. 
# 
# We first applied four machine learning algorithms and measured the effectiveness of the models by using ROC AUC score and stratified accuracy scores for Churn and Non Churn Customers.
# * Basic Logistic Regression 
# * Logistic Regression with PCA 
# * Random Forest with Hyperparameter tuning 
# * SVM with Hyperparameter tuning 
# 
# Since the data set is highly imbalanced with the churn customers being 9.6% of the dataset, the above algorithms did not produce production quality results.
# 
# Hence we applied the following two techniques for Handling this highly imbalanced dataset.
# 
# * XGBoost 
# * SMOTE followed by Logistic Regression, Random Forest, SVM and PCA 
# 
# Here is the summary of the Model Performance of all the models that we tried.
# 
# No | Algorithm | Accuracy Score |  AUC Score | ROC_AUC Score | Non-Churn Accuracy Rate(Specificity) | Churn Accuracy Rate (Sensitivity) |
# -|----------|----------------|------------|---------------|-------------------------|---------------------|
# 1|Basic Logistic Regression | 0.92 | 0.73 | 0.91 | 0.97 | 0.48 |
# 2|Logistic Regression with PCA | 0.91 | 0.60 | 0.88 | 0.99 | 0.22 |
# 3|Random Forest with Hyperparameter tuning | 0.93 | 0.70 | 0.89 | 0.98 | 0.41 | 
# 4|Linear SVM with Hyperparameter tuning | 0.89 | 0.5 | 0.5 | 1 | 0 | 
# 5|Using XGBoost | 0.93 | 0.75 | 0.94 | 0.97 | 0.52 | 
# 6|Using SMOTE and Logistic Regression | 0.87 | 0.83 | 0.9 | 0.88 | 0.78 |
# 7|<b><font color='#0000FF'>Using SMOTE and Random Forest </font></b> | 0.89 | 0.84 | <b><font color='#00FF00'>0.93 </font></b>| 0.91 | <b><font color='#FF0000'>0.77 </font></b>|
# 8|Using SMOTE and Linear SVM | 0.89 | 0.5 | 0.5 | 1 | 0 | 
# 9|<b><font color='#0000FF'>Using SMOTE and PCA </font></b> | 0.81 | 0.81 | <b><font color='#00FF00'>0.88</font></b> | 0.80 | <b><font color='#00FF00'>0.83 </font></b> | 
# 
# With SMOTE Data Sampling all the algorithms perform better than the unsampled data. Among these with SMOTE followed by PCA, we achieved the best churn prediction model with AUC 81% and Churn Predicton Accuracy more than 82.5%. This is the best model among all the six models that we applied. Linear SVM fared as the worst model.
# We summarize the results alongwith metrics at the end of this project.
# 
# 

# ### Table of Contents
# This notebook is organized as follows. Please click on the following links to go to the respective section.
# 
# ### <a href="#common_functions">Common Functions </a></b>
# 
# ### <a href = "#eda">Basic Data Analysis and Null Value imputation </a></b>
#    * #### <a href ="#monthwise"> Get Columns Monthwise and Basic Understanding of Columns </a>
#    * #### <a href = "#total_recharge"> Derive Columns to get Total Recharge Amount in Month 6 and 7th </a>
#    * #### <a href="#high_value_customer">Filter High Value Customer </a>
#    * #### <a href="#drop_nulls"> Null Value Check and Drop Columns High % NULL Values </a>
# 
#    * #### <a href="#single_categorical">Check Categorical Variables with Single Value columns and Drop them.</a>
#    * #### <a href="#null_less_than_50"> Analyze NULL Value with less than 50% NULL data. </a>
#    * #### <a href="#distribution"> Check for columns with less than 50% NULL value and distribution in all months. </a>
#    * #### <a href="#impute"> Impute other NULL fields with 0. </a>
#    * #### <a href="#derived_columns"> Get a new derived columns </a>
# 
# ### <a href="#exploratory_data_analysis">EDA </a>
#    * #### <a href="#churn_nonchurn">Get Churn, Non-Churn ratio in complete dataset </a>
#    * #### <a href="#correlation_trend">Monthwise trend of each features </a>.
#    * #### <a href="#age_on_network"> Age on Network Correlation trend </a>
#    * #### <a href="#eda_insights"> Insights from EDA </a>
# 
# ### <a href="#outlier_handling"> Outlier Handling </a>
#     
# ### <a href="#data_modelling"> Applying Machine Learning Algorithms </a>
#    * #### <a href="#data_standardization">Data standardization and preparation.</a>
#    * #### <a href = "#basic_logistic_regression"> Basic Logistic Regression </a>
#    * #### <a href="#pca"> Logistic Regression with PCA </a>
#    * #### <a href="#random_forest"> Random Forest with Hyperparameter tuning </a>
#    * #### <a href="#svm"> SVM with Hyperparameter tuning </a>
#    * #### <a href="#data_imbalance">Imbalance Dataset Handling </a>
#        * #### <a href="#xgboost"> Using XGBoost </a>
#        * #### <a href ="#smote_logistic_regression"> Using SMOTE and Logistic Regression </a>
#        * #### <a href ="#smote_random_forest"> Using SMOTE and Random Forest </a>
#        * #### <a href ="#smote_svm"> Using SMOTE and Linear SVM </a>
#        * #### <a href ="#smote"> Using SMOTE and PCA </a>
#    
# ### <a href="#lasso"> Lasso Regression to find the major Determining Parameters </a>
# 
# ### <a href= "#summary" >Summary </a></b>
#    * #### <a href = "#smote_pca">SMOTE with PCA as Best Balanced Model </a>
#    * #### <a href="#driving_features"> Churn Indicators and Business Recommendation </a>    

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import time
import datetime
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/telecom_churn_data.csv')


# <a id="common_functions"></a>
# # Common Functions

# In[ ]:


# Function to Return Monthwise ColumnsList. Returns arrays of columns belonging to 6,7,8,9 month separately.
# Also returns an array of columns that are not month specific as common columns.
def returnColumnsByMonth(df):
    column_Month_6 = []
    column_Month_7 = []
    column_Month_8 = []
    column_Month_9 = []
    column_Common = []
    for eachColumns in df.columns:
        if((eachColumns.find("_6") >=0) | (eachColumns.find("jun_") >=0)):
            column_Month_6.append(eachColumns)
        elif((eachColumns.find("_7") >=0) | (eachColumns.find("jul_") >=0)):
            column_Month_7.append(eachColumns)
        elif((eachColumns.find("_8") >= 0) | (eachColumns.find("aug_") >=0)):
            column_Month_8.append(eachColumns)
        elif((eachColumns.find("_9") >=0) | (eachColumns.find("sep_") >=0)):
            column_Month_9.append(eachColumns)
        else:
            column_Common.append(eachColumns)
    return column_Month_6, column_Month_7, column_Month_8, column_Month_9, column_Common

# Function to Get Columns Based on Null %. 
#Returns columns that have % of null values higher or lower than nullPercentLimit
def getColumnsBasedOnNullPercent(df, nullPercentLimit, limitType = 'Upper'):
    col2NullPercent_df = pd.DataFrame(round((df.isnull().sum()/len(df.index))* 100, 2), columns=['NullPercent'])
    col2NullPercent_df = pd.DataFrame(round((df.isnull().sum()/len(df.index))* 100, 2), columns=['NullPercent'])
    if(limitType == 'Upper'):
        columnsList = np.array(col2NullPercent_df.apply(lambda x: x['NullPercent'] > nullPercentLimit , axis=1))
    if(limitType == 'Lower'):
        columnsList = np.array(col2NullPercent_df.apply(lambda x: ((x['NullPercent'] < nullPercentLimit) & (x['NullPercent'] > 0)) , axis=1))
    return np.array(df.loc[:, columnsList].columns)

# Function to get Days Since Last Recharge for 6/7/8/9 months
def daysSinceLastRechargeMonthwise(df, month):
    if(month == 6):
        return pd.to_datetime(df['last_date_of_month_6']) - pd.to_datetime(df['date_of_last_rech_6'])
    elif(month == 7):
        return pd.to_datetime(df['last_date_of_month_7']) - pd.to_datetime(df['date_of_last_rech_7'])
    elif(month == 8):
        return pd.to_datetime(df['last_date_of_month_8']) - pd.to_datetime(df['date_of_last_rech_8'])
    elif(month == 9):
        return pd.to_datetime(df['last_date_of_month_9']) - pd.to_datetime(df['date_of_last_rech_9'])

def plotCategoricalChurn_NotChurn(df, columnsList, flag = 0):
    for eachMonth in columnsList:
    #flag = 1        
    #eachMonth = "days_from_LastRechage_6"
        col = eachMonth
        X1 = df.groupby('churn')[col].agg(['mean']).reset_index()
        X1.rename(columns={'mean':col}, inplace=True)
        if(flag == 1):
            seventhMonth = eachMonth[:-1] + "7"
            X2 = df.groupby('churn')[seventhMonth].agg(['mean']).reset_index()    
            X2.rename(columns={'mean':seventhMonth}, inplace=True)
            X2 = pd.merge(X1,X2, on = ['churn'])
            newCol = eachMonth[:-1] + "goodPeriod_Avg"
            print(newCol)
            X2[newCol] = (X2[eachMonth] + X2[seventhMonth])/2
            p = sns.barplot(x='churn', y=newCol, data=X2)
            p.set_xticklabels(['Not-Churn','churn'], fontsize= 12)
            plt.ylabel(newCol,fontsize = 12)
            plt.xlabel('churn', fontsize = 12)
            plt.show()
            X2.head()

        else:
            print(eachMonth)
            p = sns.barplot(x='churn', y=col, data=X1)
            p.set_xticklabels(['Not-Churn','churn'], fontsize= 12)
            plt.ylabel(col,fontsize = 12)
            plt.xlabel('churn', fontsize = 12)
            plt.show()
            X1.head()

#Function to Show Howmuch % usage done for Churn Subscriber with respect to total usage on that month, for tht particular feature
# e.g: arpu (Averag Revenue Per User) ==> How much is the average arpu for churn and non-churn subscriber in month 6 7 and 8.
# Then check for churn subscriber how much % usage on total mean Usage in each month
def churnSubscriberUsageChangePercentage():
    for count, eachFeature in enumerate(column_Month_6):
        col = eachFeature
        X1 = df.groupby(['churn'])[col].agg(['mean']).reset_index()
        X1.rename(columns={'mean': "mean_"+col}, inplace=True)
        if(col == 'jun_vbc_3g'):
            col = 'jul_vbc_3g'
        else:
            col = col[:-1] + "7"
        X2 = df.groupby(['churn'])[col].agg(['mean']).reset_index()
        X2.rename(columns={'mean': "mean_"+col}, inplace=True)
        if(col == 'jul_vbc_3g'):
            col = 'aug_vbc_3g'
        else:
            col = col[:-1] + "8"
        X3 = df.groupby(['churn'])[col].agg(['mean']).reset_index()
        X3.rename(columns={'mean': "mean_"+col}, inplace=True)

        X1 = pd.merge(X1, X2, on = ['churn'])
        X1 = pd.merge(X1,X3, on = ['churn'])
        X1.head()
        X1 = X1.transpose().reset_index()
        X1 = X1.loc[1:]
        X1.columns = ['Feature', 'Not-Churn', 'Churn']
        #X1.head()
        X1['Usage%_During_Churn'] = round((X1['Churn']/(X1['Not-Churn'] + X1['Churn']))*100,2)
        print(X1.head())
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
        p = sns.barplot(x='Feature', y='Usage%_During_Churn', data=X1)
        p.set_xticklabels(p.get_xticklabels(),rotation=45)
        plt.title('Churn subscriber usage to Total Usage % for {}'.format(col[:-2]), fontsize = 12)
        X1.rename(columns={'Usage%_During_Churn':'Churn_Subscriber_Usage_Trend'}, inplace=True)
        plt.plot(X1['Churn_Subscriber_Usage_Trend'], 'r-')
        #plt.title(title, fontsize = 12)
        ax.legend(loc='upper center', bbox_to_anchor=(0.8, 1.00), shadow=True, ncol=2, fontsize = 10)
        plt.grid(True)
        plt.show()
        
# Method to draw AUC curve
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 6))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return fpr, tpr, thresholds

# Common Method for Hyperparameter Tuning using Random Forest
def randomforestHyperparameterTuning(parameters, X_train, y_train, n_folds = 5, n_jobs = 4):
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestClassifier

    # instantiate the model
    param = list(parameters.keys())[0]
    if((param == 'max_features') | (param == 'n_estimators')):
        rfc = RandomForestClassifier(max_depth=4)
    else:
        rfc = RandomForestClassifier()
    # fit tree on training data
    rfc = GridSearchCV(rfc, parameters, 
                        cv=n_folds, 
                       scoring="accuracy",
                      return_train_score=True,
                      n_jobs = n_jobs)
    rfc.fit(X_train, y_train)
    #print("Best Parameter ==> {}".format(rfc.best_params_))
    # printing the optimal accuracy score and hyperparameters
    print('We can get accuracy of',rfc.best_score_,'using Best Parameter',rfc.best_params_)
    
    if(len(list(parameters.keys())) == 1):
        scores = rfc.cv_results_
        scoreParam = "param_" + list(parameters.keys())[0]

        plt.figure()
        plt.plot(scores[scoreParam], 
                 scores["mean_train_score"], 
                 label="training accuracy")
        plt.plot(scores[scoreParam], 
                 scores["mean_test_score"], 
                 label="test accuracy")
        plt.xlabel(param)
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

def convertCategorical(contVal, threshold):
    if(contVal > threshold):
        return 1
    else:
        return 0
convertCategorical = np.vectorize(convertCategorical)

# Function to Check the outlier in each Feature for Non-Churn and Churn Subscribers
def featurewiseOutlierBetweenChurnAndNonChurn():
    columnList = (list(df.columns[3:]))
    columnList.remove('churn')
    for count, eachColumn in enumerate(columnList):
        if(count%2 == 0):
            plt.figure(count, figsize=(18,6))
            p = plt.subplot(121)
            sns.boxplot(y = df[eachColumn], x = df['churn'])
            p.set_xticklabels(['Not-Churn','churn'], fontsize= 12)
            p.grid(True)
        else:
            q = plt.subplot(122)
            sns.boxplot(y = df[eachColumn], x = df['churn'])
            q.set_xticklabels(['Not-Churn','churn'], fontsize= 12)
            q.grid(True)

# Common Function to Do the Model Evalution
def modelEvaluation(y_test, y_pred, model, flag = 0):
    print(confusion_matrix(y_test,y_pred))
    print("Accuracy Score ==> {}".format(accuracy_score(y_test,y_pred)))
    print("AUC Score ==> {}".format(roc_auc_score(y_test,y_pred)))
    if flag == 1: #For PCA
        pred_probs_test = model.predict_proba(df_test_pca)[:,1]
    elif flag == 2: #For XGBoost For Imbalance Data
        pred_probs_test = model.predict_proba(np.array(X_test))[:,1]
    elif flag == 3: #For Lasso
        pred_probs_test = lasso.predict(X_test)
    else:
        pred_probs_test = model.predict_proba(X_test)[:,1]
    print("ROC_AUC Score ==> {:2.2}".format(metrics.roc_auc_score(y_test, pred_probs_test)))
    TP = (confusion_matrix(y_test,y_pred))[0][0]
    FP = (confusion_matrix(y_test,y_pred))[0][1]
    FN = (confusion_matrix(y_test,y_pred))[1][0]
    TN = (confusion_matrix(y_test,y_pred))[1][1]
    print("Not-Churn Accuracy Rate:(Specificity) ==> {}".format(TP/(TP+FP)))
    print("Churn Accuracy Rate:(Sensitivity) ==> {}".format(TN/(TN+FN)))
    draw_roc(y_test, y_pred)
    


# <a id="eda"></a>
# # Basic Data Analysis and Null Value Imputation

# In[ ]:


df.head()


# In[ ]:


df.columns.values


# <a id="monthwise"></a>
# ####  * Get Columns Monthwise & Basic Understanding of Columns

# In[ ]:


column_Month_6, column_Month_7, column_Month_8, column_Month_9, column_Common = returnColumnsByMonth(df)

print("Month 6 Columns Count ==> {}".format(len(column_Month_6)))
print("Month 7 Columns Count ==> {}".format(len(column_Month_7)))
print("Month 8 Columns Count ==> {}".format(len(column_Month_8)))
print("Month 9 Columns Count ==> {}".format(len(column_Month_9)))
print("Common Columns Count ==> {}".format(len(column_Common)))


# In[ ]:


# All Months are having same type of columns So lets see the columns in general
print ("\nMonth based Columns:\n \t\t==> {}".format(np.array(column_Month_6)))
print ("\nCommon Columns:\n \t\t==> {}".format(np.array(column_Common)))


# <a id="total_recharge"></a>
# #### * Derive Columns Total_Recharge_Amount from 6th and 7th Month total_rech_amt

# In[ ]:


df['Total_Recharge_Amount'] = df['total_rech_amt_6'] + df['total_rech_amt_7']

# Get 70% of "Total Recharge Amount" to identify the recharge Amount Range for High value customer
print(df['Total_Recharge_Amount'].describe(percentiles = [0.7]))
print("\n70% of Total Recharge Amount of first 2 months are {}".format(df['Total_Recharge_Amount'].describe(percentiles = [0.7])[5]))


# <a id="high_value_customer"> </a>
# #### * Filter High Value Customer from main data frame

# In[ ]:


df = df[df['Total_Recharge_Amount'] > 737].reset_index(drop=True)
print("\nTotal High Value Customer Count ==> {}".format(df.shape[0]))
df.drop(columns=['Total_Recharge_Amount'], inplace=True)


# <a id="drop_nulls"></a>
# #### * Null Value Checking and Drop High Null Value Columns

# In[ ]:


#Get Null Percentage in dataFrame and Filter
nullPercentageLimit = 50
columns_More_Than_50_PercentNull = getColumnsBasedOnNullPercent(df,nullPercentageLimit)
#Drop Columns with More than 50% NUll
df = df.loc[:, ~df.columns.isin(columns_More_Than_50_PercentNull)]

print("\nColumn List Dropped with More than 50% of Null Value:==>\n {}\n".format(columns_More_Than_50_PercentNull))


# <a id="single_categorical"></a>
# #### * Check Categorical Variables and Single Record Variables. Then drop the columns which have a single value.

# In[ ]:


# Get Columns which have only one value for all the rows.
singleCategoryColumns = df.loc[:, np.array(df.apply(lambda x: x.nunique() == 1))].columns

#Print these single value column names, and the value that they contain.
for eachSingleCatgory in singleCategoryColumns:
    print("{}: {}".format(eachSingleCatgory, df[eachSingleCatgory].unique()))
print("\n<=== Drop Single Category Columns, Other than last_date_of_month_6/7/8/9, as it will be used for Derive Columns ===>\n")
singleCategoryColumns = [x for x in singleCategoryColumns if x not in list(['last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8', 'last_date_of_month_9'])]
singleCategoryColumns = np.array(singleCategoryColumns)
df = df.loc[:, ~df.columns.isin(singleCategoryColumns)]

# Fill the NA value of the last_date_of_month column
df['last_date_of_month_7'] = df['last_date_of_month_7'].fillna('7/31/2014')
df['last_date_of_month_8'] = df['last_date_of_month_8'].fillna('8/31/2014')
df['last_date_of_month_9'] = df['last_date_of_month_9'].fillna('9/30/2014')


# <a id="null_less_than_50"></a>
# #### * Analyze Null Value for Less than 50%

# In[ ]:


#Get the columns where the number of null valued rows are less than 50%
columns_Less_Than_50_PercentNull = getColumnsBasedOnNullPercent(df,nullPercentageLimit, limitType='Lower')
df_temp = df.loc[:, columns_Less_Than_50_PercentNull]

#Get the % of rows that are having null values in each of these columns
round(df_temp.isnull().sum()/len(df_temp.index) * 100,2)


# <a id="impute"></a>
# #### * As the Null % is very less, lets see if Null Value Can be imputed with some value

# In[ ]:


column_Month_6, column_Month_7, column_Month_8, column_Month_9, column_Common = returnColumnsByMonth(df_temp)

print("Month 6 Columns Count ==> {}".format(len(column_Month_6)))
print("Month 7 Columns Count ==> {}".format(len(column_Month_7)))
print("Month 8 Columns Count ==> {}".format(len(column_Month_8)))
print("Month 9 Columns Count ==> {}".format(len(column_Month_9)))
print("Common Columns Count ==> {}".format(len(column_Common)))
print("==> All Months are having same columns with less% of Null Value")
print(np.array(column_Month_7))
df_temp.loc[:, column_Month_7].head()


# #### * One derive column for each month from the date columns and drop the date column, then impute NULL for derive column with 30 days

# In[ ]:


# 4 Derive Columns for each month, which will tell before how many days from month end, 
#recharge happened by subscriber.
df['days_from_LastRechage_6'] = daysSinceLastRechargeMonthwise(df, 6).apply(lambda x: x.days)
df['days_from_LastRechage_7'] = daysSinceLastRechargeMonthwise(df, 7).apply(lambda x: x.days)
df['days_from_LastRechage_8'] = daysSinceLastRechargeMonthwise(df, 8).apply(lambda x: x.days)
df['days_from_LastRechage_9'] = daysSinceLastRechargeMonthwise(df, 9).apply(lambda x: x.days)
df['days_from_LastRechage_6'] = df['days_from_LastRechage_6'].fillna(30)
df['days_from_LastRechage_7'] = df['days_from_LastRechage_7'].fillna(30)
df['days_from_LastRechage_8'] = df['days_from_LastRechage_8'].fillna(30)
df['days_from_LastRechage_9'] = df['days_from_LastRechage_9'].fillna(30)


# In[ ]:


#Drop the last Recharge Date and End date of the month columns for all the months
dateColumns = ['last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8','last_date_of_month_9',
              'date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8','date_of_last_rech_9']
df = df.loc[:, ~df.columns.isin(dateColumns)]


# #### * Fill Other NULL field with value 0

# * Following columns with less % of NULL value can be imputed with 0, as all these columns represent some kind of special MOU (Minutes of usage)
# * It is possible that one susbcriber may not have done a special number call in one month, or not done a isd call and so on..
# * With this above logic, following columns can be imputed with 0 for NULL values.
# 
# Columns List Left with NULL Value (Null % less than 50):
#     
# 'onnet_mou_6/7/8' 'offnet_mou_6/7/8' 'roam_ic_mou_6/7/8' 'roam_og_mou_6/7/8'
#  'loc_og_t2t_mou_6/7/8' 'loc_og_t2m_mou_6/7/8' 'loc_og_t2f_mou_6/7/8'
#  'loc_og_t2c_mou_6/7/8' 'loc_og_mou_6/7/8' 'std_og_t2t_mou_6/7/8' 'std_og_t2m_mou_6/7/8'
#  'std_og_t2f_mou_6/7/8' 'std_og_mou_6/7/8' 'isd_og_mou_6/7/8' 'spl_og_mou_6/7/8'
#  'og_others_6/7/8' 'loc_ic_t2t_mou_6/7/8' 'loc_ic_t2m_mou_6/7/8' 'loc_ic_t2f_mou_6/7/8'
#  'loc_ic_mou_6/7/8' 'std_ic_t2t_mou_6/7/8' 'std_ic_t2m_mou_6/7/8' 'std_ic_t2f_mou_6/7/8'
#  'std_ic_mou_6/7/8' 'spl_ic_mou_6/7/8' 'isd_ic_mou_6/7/8' 'ic_others_6/7/8'
#     

# In[ ]:



df = df.fillna(0).reset_index()


# <a id="churn_nonchurn"></a>
# #### * Derive a Column which will tell if subscriber is churn or not churn

# In[ ]:


# Label churn and non-churn customers
df['churn'] = np.where(
            (
                (df['total_ic_mou_9'] == 0.0) | 
                (df['total_og_mou_9'] == 0.0)
            ) & 
            (
                (df['vol_2g_mb_9'] == 0.0) & 
                (df['vol_3g_mb_9'] == 0.0)
            ),1,0
        )


# #### * Drop 9th Month Columns

# In[ ]:


# Remove columns with '9'
df = df.drop(df.filter(like = '9').columns, axis=1)


# <a id="exploratory_data_analysis"></a>
# # EDA and Derive Columns

# #### Get the distribution of churn vs non churn customers. Churn is 10.6% which indicates an unbalanced datasets.

# In[ ]:


df.groupby(['churn'])['churn'].count()


# <a id="derived_columns"></a>
# #### * Derive Columns

# * Column Analysis for Derive Columns
#       - From the data provided following are the overall calculation and relation between below 5 set of columns.
#       - Total Outgoing MOU is sum of all kind of local, std, special and other outgoing MOU.
#           i.e:
#               total_og_mou_6/7/8 = loc_og_mou_6/7/8 +  std_og_mou_6/7/8 + spl_og_mou_6/7/8 + og_others_6/7/8
#       - Total Incoming MOU is sum of local, Std, special and others incoming MOU.
#            i.e:
#               total_ic_mou_6/7/8 = loc_ic_mou_6/7/8 + std_ic_mou_6/7/8 + spl_ic_mou_6/7/8 + ic_others_6/7/8
# * Derive Columns
#   - In place of using direct columns, as we have total value and its distribution, lets get the % of usage in each distribution.
#     
#     e.g: How much local outgoing MOU with respect to total outgoing and so on for all month data
#     
#              loc_og_mou_Percent_6/7/8, std_og_mou_Percent_6/7/8, spl_og_mou_Percent_6/7/8, og_others_Percent_6/7/8
#              loc_ic_mou_Percent_6/7/8, std_ic_mou_Percent_6/7/8, spl_ic_mou_Percent_6/7/8, ic_others_Percent_6/7/8

# In[ ]:


df['loc_og_mou_Percent_6'] = round((df['loc_og_mou_6']/df['total_og_mou_6']) * 100,2)
df['std_og_mou_Percent_6'] = round((df['std_og_mou_6']/df['total_og_mou_6']) * 100,2)
df['spl_og_mou_Percent_6'] = round((df['spl_og_mou_6']/df['total_og_mou_6']) * 100,2)
df['og_others_Percent_6'] = round((df['og_others_6']/df['total_og_mou_6']) * 100,2)
df['loc_ic_mou_Percent_6'] = round((df['loc_ic_mou_6']/df['total_ic_mou_6']) * 100,2)
df['std_ic_mou_Percent_6'] = round((df['std_ic_mou_6']/df['total_ic_mou_6']) * 100,2)
df['spl_ic_mou_Percent_6'] = round((df['spl_ic_mou_6']/df['total_ic_mou_6']) * 100,2)
df['ic_others_Percent_6'] = round((df['ic_others_6']/df['total_ic_mou_6']) * 100,2)

df['loc_og_mou_Percent_7'] = round((df['loc_og_mou_7']/df['total_og_mou_7']) * 100,2)
df['std_og_mou_Percent_7'] = round((df['std_og_mou_7']/df['total_og_mou_7']) * 100,2)
df['spl_og_mou_Percent_7'] = round((df['spl_og_mou_7']/df['total_og_mou_7']) * 100,2)
df['og_others_Percent_7'] = round((df['og_others_7']/df['total_og_mou_7']) * 100,2)
df['loc_ic_mou_Percent_7'] = round((df['loc_ic_mou_7']/df['total_ic_mou_7']) * 100,2)
df['std_ic_mou_Percent_7'] = round((df['std_ic_mou_7']/df['total_ic_mou_7']) * 100,2)
df['spl_ic_mou_Percent_7'] = round((df['spl_ic_mou_7']/df['total_ic_mou_7']) * 100,2)
df['ic_others_Percent_7'] = round((df['ic_others_7']/df['total_ic_mou_7']) * 100,2)

df['loc_og_mou_Percent_8'] = round((df['loc_og_mou_8']/df['total_og_mou_8']) * 100,2)
df['std_og_mou_Percent_8'] = round((df['std_og_mou_8']/df['total_og_mou_8']) * 100,2)
df['spl_og_mou_Percent_8'] = round((df['spl_og_mou_8']/df['total_og_mou_8']) * 100,2)
df['og_others_Percent_8'] = round((df['og_others_8']/df['total_og_mou_8']) * 100,2)
df['loc_ic_mou_Percent_8'] = round((df['loc_ic_mou_8']/df['total_ic_mou_8']) * 100,2)
df['std_ic_mou_Percent_8'] = round((df['std_ic_mou_8']/df['total_ic_mou_8']) * 100,2)
df['spl_ic_mou_Percent_8'] = round((df['spl_ic_mou_8']/df['total_ic_mou_8']) * 100,2)
df['ic_others_Percent_8'] = round((df['ic_others_8']/df['total_ic_mou_8']) * 100,2)

# Fill All Nan Value because of 0 division set to 0.
df = df.fillna(0).reset_index()


# In[ ]:


column_Month_6, column_Month_7, column_Month_8, column_Month_9, column_Common = returnColumnsByMonth(df)

print("Month 6 Columns Count ==> {}".format(len(column_Month_6)))
print("Month 7 Columns Count ==> {}".format(len(column_Month_7)))
print("Month 8 Columns Count ==> {}".format(len(column_Month_8)))
print("Month 9 Columns Count ==> {}".format(len(column_Month_9)))
print("Common Columns Count ==> {}".format(len(column_Common)))
print("==> All Months are having same columns with less% of Null Value")
print(np.array(column_Month_7))
df.loc[:, column_Month_7].head()


# <a id="correlation_trend"></a>
# #### * EDA For all the features (Monthly Featues) to find how much % usage for each feature monthwise for Churn Subscriber on total usage

# In[ ]:


churnSubscriberUsageChangePercentage()


# <a id="eda_insights"></a>
# ### Insights from EDA

# * By Considering 8th Month as Churn Decision month and 6th and 7th as Good Month, following are the insight for all the features
#     * Average Revenue per user (ARPU) is less in 8th month than 6th and 7th month. 
#     * Onnet-Monthly-Usage (Onnet_MOU) is less in 8th Month than 6th and 7th Month.
#     * Offnet-Montly-Usage (Offnet_MOU) is less in 8th Month than 6th and 7th Month.
#     * ROAM_Incoming_MontlyUsage (Roam_IC_MOU) is not having any changes in good and decision period.
#     * Roam_Outgoing_MonthlyUsage (Roam_OG_MOU) is not having any changes in good and decision perod.
#     * Local Outgoing t2t Monthly Usage is less in 8th Month than 6th and 7th Month.
#     * Local Outgoing t2m Monthly Usage is less in 8th month than 6th and 7th month.
#     * Local Outgoing t2f Monthly Usage is less in 8th month than 6th and 7th month.
#     * Local Outgoing t2c Monthly Usage is less in 8th month than 6th and 7th month.
#     * Local Outgoing Monthly Usage is less in 8th month than 6th and 7th month.
#     * STD outgoing t2t Monthly Usage is less in 8th month than 6th and 7th month.
#     * STD outgoing t2m Monthly usage is less in 8th month than 6th and 7th month.
#     * STD Outgoinh t2f Monthly Usage is less in 8th month than 6th and 7th month.
#     * STD outgoing Monthly Usage is less in 8th month than 6th and 7th month.
#     * ISD Outgoing Monthly Usage is less in 8th month than 6th and 7th month.
#     * Special Outgoing Monthly Usage is less in 8th month than 6th and 7th month.
#     * Outgoing Others is not having any changes in good and decision period.
#     * Total Outgoing Monthly Usage is less in 8th Month than 6th and 7th month.
#     * Local Incoming t2t Monthly Usage is less in 8th month than 6th and 7th month.
#     * Local incoming t2m Monthly Usage is less in 8th month than 6th and 7th month.
#     * Local incoming t2f Monthly Usage is less in 8th month than 6th and 7th month.
#     * Local incoming usage Monthly Usage is less in 8th month than 6th and 7th month.
#     * STD incoming t2t Monthly Usage is less in 8th month than 6th and 7th month.
#     * STD incoming t2m monthly usage is less in 8th month than 6th and 7th month.
#     * STD incoming t2f monthly usage is less in 8th month than 6th and 7th month.
#     * STD incoming monthly usage is less in 8th month than 6th and 7th month.
#     * Total Incoming Monthly Usage is less in 8th month than 6th and 7th month.
#     * spl_ic_mou is less in 8th month than 6th and 7th month.
#     * isd_ic_mou is less in 8th month than 6th and 7th month.
#     * ic_others is less in 8th month than 6th and 7th month.
#     * total_rech_num is less in 8th month than 6th and 7th month.
#     * toal_rech_amt is less in 8th month than 6th and 7th month.
#     * max_rech_amt is less in 8th month than 6th and 7th month.
#     * last_rech_amt is less in 8th month than 6th and 7th month.
#     * vol_2g_mb is less in 8th month than 6th and 7th month.
#     * vol_3g_mb is less in 8th month than 6th and 7th month.
#     * monthly_2g is less in 8th month than 6th and 7th month.
#     * sachet_2g is less in 8th month than 6th and 7th month.
#     * monhtly_3g is less in 8th month than 6th and 7th month.
#     * sachet_3g is less in 8th month than 6th and 7th month.
#     * vbc_3g is less in 8th month than 6th and 7th month.

# <a id="age_on_network"></a>
# #### * AON feature trend in all the 3 months for churn and non-churn subscribers

# In[ ]:


X1 = df.groupby('churn')['aon'].agg(['mean']).reset_index()
p = sns.barplot(x='churn', y='mean', data=X1)
p.set_xticklabels(['Not-Churn', 'Churn'],rotation=30)
p.set_ylabel('Average Age in Network')
plt.title('Average Age in Network between Churn and Not-Churn subscriber')
plt.show()


# ### Insight

# * Churn subscriber is having less average AON than Non-Churn Subscriber. Hence subsriber is having high AON, then chances of Churn is less

# #### * Identify Outlier for each Feature By comparing data of Churn and Non-Churn

# * As Non-Churn Count is more, need to check outlier for each feature by comparing Churn and Non-Churn subscriber 

# <a id="outlier_handling"> </a>
# ### Outlier Handling
# 

# In[ ]:


featurewiseOutlierBetweenChurnAndNonChurn()


# ### Insights and Actions to be taken on outliers

# * As Non-Churn subscriber % is more, some as per the feature value spread, some of the records can delete in following condition for Non-Churn subscriber.
#     1. Drop Record with arpu_6 more than 15000
#     2. Drop arpu_7 more than 20000
#     3. Drop arpu_8 more than 20000
#     4. onnet_mou_8 more than 8000
#     5. offnet_mou_7 more than 9000
#     6. offnet_mou_8 more than 10000
#     7. loc_og_t2t_mou_6 more tan 6000
#     8. loc_og_t2t_mou_7 more than 5000
#     9. loc_og_t2t_mou_8 more than 6000
#     10. loc_og_t2m_mou_6 more than 4000
#     11. loc_og_t2f_mou_7 more than 600
#     12. loc_og_t2f_mou_8  more than 600
#     13. loc_og_t2c_mou_8 more than 250
#     14. loc_og_mou_6 more than 8000
#     15. loc_og_mou_7 more than 6000
#     16. loc_og_mou_8  more than 6000
#     17. std_og_t2m_mou_8 more than 8000
#     18. std_og_t2f_mou_6 more than 400
#     19. std_og_t2f_mou_7  more than 400
#     20. std_og_t2f_mou_8 more than 400
#     21. std_og_mou_8 more than 10000
#     22. spl_og_mou_7 more than 800
#     23. spl_og_mou_8 more than 600
#     24. total_og_mou_8 more than 8000
#     25. loc_ic_t2m_mou_8 more than 3000
#     26. loc_ic_t2f_mou_6 more than 1000
#     27. loc_ic_t2f_mou_8 more than 1000
#     28. loc_ic_mou_8 more than 4000
#     29. std_ic_t2m_mou_8 more than 3000
#     30. std_ic_t2f_mou_8 more than 800
#     31. std_ic_mou_8 more than 3000
#     32. total_ic_mou_8 more than 4000
#     33. isd_ic_mou_7 more than 3000
#     34. isd_ic_mou_8 more than 2000
#     35. ic_others_8 more than 400
#     36. sachet_2g_8 more than 30 
#     37. sachet_3g_8 more than 30

# #### * Drop Outlier for Non-Churn Subscriber

# In[ ]:


df = df.drop(df.loc[(df['churn'] == 0) & (
    (df['arpu_6'] > 15000) | (df['arpu_7'] > 20000) | (df['arpu_8'] > 20000) | (df['onnet_mou_8'] > 8000) | 
    (df['offnet_mou_7'] > 9000) | (df['offnet_mou_8'] > 10000) | (df['loc_og_t2t_mou_6'] > 6000) | (df['loc_og_t2t_mou_7'] > 5000) |
    (df['loc_og_t2t_mou_8'] > 6000) | (df['loc_og_t2m_mou_6'] > 4000) | (df['loc_og_t2f_mou_7'] > 600) | (df['loc_og_t2f_mou_8'] > 600) |
    (df['loc_og_t2c_mou_8'] > 250) | (df['loc_og_mou_6'] > 8000) | (df['loc_og_mou_7'] > 6000) | (df['loc_og_mou_8'] > 6000) |
    (df['std_og_t2m_mou_8'] > 8000) | (df['std_og_t2f_mou_6'] > 400) | (df['std_og_t2f_mou_7'] > 400) | (df['std_og_t2f_mou_8'] > 400) |
    (df['std_og_mou_8'] > 10000) | (df['spl_og_mou_7'] > 800) | (df['spl_og_mou_8'] > 600) |(df['total_og_mou_8'] > 8000) |
    (df['loc_ic_t2m_mou_8'] > 3000) | (df['loc_ic_t2f_mou_6'] > 1000) |(df['loc_ic_t2f_mou_8'] > 1000) |(df['loc_ic_mou_8'] > 4000) |
    (df['std_ic_t2m_mou_8'] > 3000) | (df['std_ic_t2f_mou_8'] > 800) | (df['std_ic_mou_8'] > 3000) | (df['total_ic_mou_8'] > 4000) |
    (df['isd_ic_mou_7'] > 3000) | (df['isd_ic_mou_8'] > 2000) | (df['ic_others_8'] > 400) | (df['sachet_2g_8'] > 30) | (df['sachet_3g_8'] > 30)
)].index)


# #### Correlation matrix

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize = (40,20))        # Size of the figure
sns.heatmap(df.corr(),annot = True)


# Due to a large number of variables, we cannot visualize the corellation matrix properly. We will address this after PCA.

# <a id="data_modelling"></a>
# # Data Modelling

# <a id="data_standardization"></a>
# ### <font color='#0000FF'> * Data standardization and preparation </font>

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn import metrics


# In[ ]:


X = (df.iloc[:,3:])
X = X.loc[:,X.columns != 'churn']
y = df.loc[:, 'churn']

#Standardization of Data
scaler = StandardScaler()
scaler.fit(X)
#Using a Train : Test Split of 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# <a id="basic_logistic_regression"></a>
# ### <font color='#0000FF'> * Basic Logistic Regression </font>

# In[ ]:


print("Number of Features ==> {}".format(len(X.columns)))


# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, y_train) #Use Balanced Data for Logistic Regression
y_pred = lr.predict(X_test)


# In[ ]:


modelEvaluation(y_test, y_pred, lr)


# <a id="pca"></a>
# ### <font color='#0000FF'> * Logistic Regression Using PCA </font>

# In[ ]:


#Improting the PCA module
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)
#Doing the PCA on the train data
pca.fit(X_train)

#Making the screeplot - plotting the cumulative variance against the number of components
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("Scree Plot")
plt.show()


# In[ ]:


pca.components_


# ### Insight

#  * As per PCA 25 features can give 95% accuracy.

# #### * Applying 25 features in final PCA

# In[ ]:


#Using incremental PCA for efficiency
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=25)
df_train_pca = pca_final.fit_transform(X_train)
#creating correlation matrix for the principal components
corrmat = np.corrcoef(df_train_pca.transpose())
#plotting the correlation matrix
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (20,10))
sns.heatmap(corrmat,annot = True)
corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:==> {}".format(corrmat_nodiag.max()), "\n min corr: ==> {}".format(corrmat_nodiag.min()))


# #### * Fit test data with the same PCA model and execute Logistic Regression

# In[ ]:


df_test_pca = pca_final.transform(X_test)
lr = LogisticRegression()
model_pca = lr.fit(df_train_pca, y_train)

# Making prediction on the test data
y_pred = model_pca.predict(df_test_pca)


# In[ ]:


# draw_roc(y_test, y_pred)
modelEvaluation(y_test, y_pred, model_pca, 1)


# #### * PCA Again

# In[ ]:


pca_again = PCA(0.95)
df_train_pca = pca_again.fit_transform(X_train)
df_train_pca.shape


# In[ ]:


learner_pca = LogisticRegression()
model_pca = learner_pca.fit(df_train_pca,y_train)

df_test_pca = pca_again.transform(X_test)
df_test_pca.shape
# #Making prediction on the test data
y_pred = model_pca.predict(df_test_pca)


# In[ ]:


# draw_roc(y_test, y_pred)
modelEvaluation(y_test, y_pred, model_pca, 1)


# <a id="random_forest"></a>
# ### <font color='#0000FF'> * Using Random Forest </font>

# #### * Basic Random Forest with default parameters

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[ ]:


# Making predictions
y_pred = rfc.predict(X_test)
modelEvaluation(y_test, y_pred, rfc)


# #### * Random Forest with Grid Search and Hyperparameter Tuning

# #### `` 1. max_depth `` 

# In[ ]:


parameters = {'max_depth': range(2,30, 5)}
randomforestHyperparameterTuning(parameters, X_train, y_train)


# #### `` 2. n_estimators `` 

# In[ ]:


# Number of Trees
parameters = {'n_estimators': range(100, 1500, 400)}
#randomforestHyperparameterTuning(parameters)
randomforestHyperparameterTuning(parameters, X_train, y_train)


# #### `` 3. max_feature ``

# In[ ]:


# Maximum Features to split in a node
parameters = {'max_features': [4, 8, 14, 20, 24, 28, 32]}
#randomforestHyperparameterTuning(parameters)
randomforestHyperparameterTuning(parameters, X_train, y_train)


# #### `` 4. min_samples_leaf ``

# In[ ]:


parameters = {'min_samples_leaf': range(50, 400, 40)}
#randomforestHyperparameterTuning(parameters)
randomforestHyperparameterTuning(parameters, X_train, y_train)


# #### `` 5. min_samples_split ``

# In[ ]:


parameters = {'min_samples_split': range(200, 500, 50)}
#randomforestHyperparameterTuning(parameters)
randomforestHyperparameterTuning(parameters, X_train, y_train)


# #### * Grid Search to find Optimal Hyperparameter for Random Forest

# In[ ]:


# param_grid = {
#     'max_depth': [4,8,10],
#     'min_samples_leaf': [30, 50 , 70],
#     'min_samples_split': [150, 170, 200],
#     'n_estimators': [100,200, 300], 
#     'max_features': [15,20,25]
# }
# # Create a based model
# rf = RandomForestClassifier()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1,verbose = 1)


# In[ ]:


# grid_search.fit(X_train, y_train)


# In[ ]:


# # printing the optimal accuracy score and hyperparameters
# print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)


# #### * Final Random Forest Model with all optimal parameter

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=8,
                             min_samples_leaf=50, 
                             min_samples_split=150,
                             max_features=20,
                             n_estimators=100)
# fit
rfc.fit(X_train,y_train)

# Making predictions
y_pred = rfc.predict(X_test)


# In[ ]:


modelEvaluation(y_test, y_pred, rfc)


# <a id="svm"></a>
# ### <font color='#0000FF'> * Using SVM </font>

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


# #SVM
# from sklearn.svm import SVC

# folds = KFold(n_splits = 5, shuffle = True, random_state = 4)
# # specify range of parameters (C) as a list
# params = {"C": [0.1, 1, 10]}

# model = SVC()

# # set up grid search scheme
# # note that we are still using the 5 fold CV scheme we set up earlier
# model_cv = GridSearchCV(estimator = model, param_grid = params, 
#                         scoring= 'accuracy', 
#                         cv = folds, 
#                         verbose = 1,
#                        return_train_score=True,
#                        n_jobs = -1)   
# # fit the model - it will fit 5 folds across all values of C
# model_cv.fit(X_train, y_train)  

# best_score = model_cv.best_score_
# best_C = model_cv.best_params_['C']

# print(" The highest test accuracy is {0} at C = {1}".format(best_score, best_C))


# In[ ]:


# model_svm = SVC(C = 0.1, probability=True)
# model_svm.fit(X_train, y_train)
# y_pred = model_svm.predict((X_test))


# In[ ]:


# modelEvaluation(y_test, y_pred, model_svm)


# <a id="data_imbalance"></a>
# ### <font color='#0000FF'> * Imbalance dataset Handling</font>

# * As data is imbalanced, none of the algorithms is providing effective result for Churn Subscriber % in above steps.
# * Lets apply 2 methods for imbalanced data set handling
#      - Use XGBoost directly
#      - Use SMOTE (Synthetic Minority Over Sampling Technique) and apply various algorithms
#          - SMOTE with Logistic Regression
#          - SMOTE with Random Forest
#          - SMOTE with SVM
#          - SMOTE with PCA

# In[ ]:


print("Non-Churn Percentage ==> {}".format(df[df['churn'] == 0].shape[0]/df.shape[0]))
print("Churn Percentage ==> {}".format(df[df['churn'] == 1].shape[0]/df.shape[0]))


# <a id="xgboost"></a>
# #### * GridSearch with different Hyperparameter for XGBoost

# In[ ]:


import xgboost as xgb
from xgboost import XGBClassifier

# hyperparameter tuning with XGBoost

# creating a KFold object 
folds = 3

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


# specify model
xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True,
                        n_jobs = -1)    
model_cv.fit(X_train, y_train)


# In[ ]:


model_cv.best_params_


# #### * Run XGBoost with Best Parameter got from Hyperparameter Tuning

# In[ ]:


#Execute XGBoost using the best parameter value got from GridSearch Cross  Validation
params = {'learning_rate': 0.2,
          'max_depth': 2, 
          'n_estimators':200,
          'subsample':0.9,
         'objective':'binary:logistic'}

# fit model on training data
model = XGBClassifier(params = params)
model.fit(X_train, y_train)

# # Making predictions
y_pred = model.predict((X_test))


# In[ ]:


modelEvaluation(y_test, y_pred, model)


# #### Handle Imbalance training dataset using SMOTE

# In[ ]:


from imblearn.over_sampling import SMOTE

print('Before OverSampling, the shape of train_X: {}'.format(X_train.shape))
print('Before OverSampling, the shape of train_y: {} \n'.format(y_train.shape))
print("Before OverSampling, y_train count: '1': Churn ==> {}".format(sum(y_train == 1)))
print("Before OverSampling, y_train count: '0': Not-Churn ==> {}".format(sum(y_train == 0)))


sm = SMOTE()
X_train_sam, y_train_sam = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_sam.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_sam.shape))

print("After OverSampling, count: '1': Churn ==> {}".format(sum(y_train_sam==1)))
print("After OverSampling, count: '0': Not-Churn ==>{}".format(sum(y_train_sam==0)))


# <a id="smote_logistic_regression"> </a>
# #### * Applying Logistic Regression with SMOTE

# In[ ]:


lr = LogisticRegression()
lr.fit(X_train_sam, y_train_sam) #Use Balanced Data for Logistic Regression
y_pred = lr.predict(X_test)
modelEvaluation(y_test, y_pred,lr)


# <a id="smote_random_forest"></a>
# #### * Applying RandomForest With SMOTE

# In[ ]:


#Use the Same Hyper parameter got in last execution
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=8,
                             min_samples_leaf=50, 
                             min_samples_split=150,
                             max_features=20,
                             n_estimators=100)
# fit
rfc.fit(X_train_sam,y_train_sam)

# Making predictions
y_pred = rfc.predict(X_test)
modelEvaluation(y_test, y_pred,rfc)


# <a id="smote_svm"></a>
# #### *Applying SVM with SMOTE

# In[ ]:


# model_svm = SVC(C = 0.1, probability=True)
# model_svm.fit(X_train_sam, y_train_sam)
# y_pred = model_svm.predict((X_test))
# modelEvaluation(y_test, y_pred,model_svm)


# <a id="smote"></a>
# #### * Applying PCA with SMOTE

# In[ ]:


pca_again = PCA(0.99)
df_train_pca = pca_again.fit_transform(X_train_sam)
df_train_pca.shape


# In[ ]:


learner_pca = LogisticRegression()
model_pca = learner_pca.fit(df_train_pca,y_train_sam)

df_test_pca = pca_again.transform(X_test)
df_test_pca.shape
# #Making prediction on the test data
y_pred = model_pca.predict(df_test_pca)


# In[ ]:


modelEvaluation(y_test, y_pred, model_pca, 1)


# <a id="lasso"></a>
# ### <font color='#0000FF'> * Lasso Regression to find the major Determining Parameters </font>

# #### * GridSearch with different value of Alpha for Lasso regression

# In[ ]:


from sklearn.linear_model import Lasso
# hide warnings
import warnings
warnings.filterwarnings('ignore')

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = 5, 
                        return_train_score=True,
                        verbose = 1,
                        n_jobs = -1)            

model_cv.fit(X_train, y_train)


# In[ ]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()


# In[ ]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# In[ ]:


print("Best Alpha Value ==> {} ".format(model_cv.best_params_))


# #### * Run Lasso with best Alpha Parameter

# In[ ]:


alpha =0.01
lasso = Lasso(alpha=alpha)
lasso.fit(X_train, y_train)


y_pred = lasso.predict(X_test)
y_pred = convertCategorical(y_pred, 0.5)


# In[ ]:


modelEvaluation(y_test, y_pred, lasso, 3)


# #### * Driving Parameter Selection

# In[ ]:


#Get Features list and co-efficient values from the Lasso Regression and make a single dataframe
s1 = pd.DataFrame(np.insert(np.array(X.columns),0,"constant"), columns=['feature'])
s1.reset_index(drop = True, inplace = True)

s2 = pd.DataFrame(np.insert(np.array(lasso.coef_), 0, lasso.intercept_), columns=['Values'])
s2.reset_index(drop = True, inplace= True)

s2['Values'] = s2['Values'].apply(lambda x: round(x,3))
drivingFeaturedf = pd.concat([s1,s2], axis=1)
drivingFeaturedf = drivingFeaturedf.iloc[1:]
drivingFeaturedf.reset_index(drop= True, inplace = True)


# In[ ]:


# Draw the complete date frame of features and coefficient.
import matplotlib.pyplot as plt
plt.figure(figsize=(26,10))
plt.subplot(111)
ax1 = sns.barplot(x = drivingFeaturedf['feature'], y = drivingFeaturedf['Values'])
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90, fontsize= 9)
ax1.set_yticklabels(ax1.get_yticklabels(), fontsize = 15)
plt.ylabel('Co-efficient', fontsize = 10)
plt.show()


# In[ ]:


#Drop features with 0 coefficient.
drivingFeaturedf = drivingFeaturedf.loc[drivingFeaturedf['Values'] != 0]


# In[ ]:


#Draw the plot for the features with non-zero co-efficient
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
plt.subplot(111)
ax1 = sns.barplot(x = drivingFeaturedf['feature'], y = drivingFeaturedf['Values'])
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90, fontsize=10)
plt.ylabel('Co-efficient')
plt.show()


# In[ ]:


print("Driving Feature List: ==>\n \t\t\t{}".format(np.array(drivingFeaturedf['feature'])))


# <a id="summary"></a>
# # Summary

# <a id="smote_pca"></a>
# * With SMOTE (Synthetic Minority Over Sampling Technique) and PCA, we are getting best balance prediction with ROC AUC 88% and Churn Predicton Accuracy more than 82.5%. A close competitor to this model is the Random Forest model with a ROC AUC score of 93%. However the Churn accuracy rate for this model is lower at 77%. Since we are more concerned with the accuracy of prededicting churn, SMOTE with PCA  is the best model among all the six models that we applied. Linear SVM fared as the worst model.

# <a id="driving_features"></a>
# ** Based on the Lasso Regression results above The major factors to that determine if a subscriber is going to churn are **
# 
#     1) Minutes of Usage on 8th Month for outgoing and incoming calls (Mostly Local/STD/Special/Others) and 
#     2) Recharge count on 8th and 7th months. 
#     
# ** Below are the detailed features that have a bearing on whether the customer will churn**
# 
# ** Recommendation: If the total usage as measured by the total minutes of usage and the recharge amount in 7th and 8th month is declining as compared to 6th month, then it is likely that such a customer will churn. If the Total Outgoing Minutes of Usage falls below 228 minutes in the 8th  73 We recommend the telecom provider to reach out to such customers, and provide them with lockin offers that will prevent their churn. **
# 
# * Driving Feature List As per Model ( 9 features)
#       - total_Rech_Num_7 (Total Number of recharge on 7th Month)
#       - total_Rech_Num_8 (Total Number of Reccharge on 8th Month)
#       Derived Columns:
#       - Days_From_LastRecharge_8 (When the last recharge done on 8th Month)
#       - loc_og_mou_Percent_8 (Local Outgoing Minutes of Usage % in 8th month wrt to Total Outgoing usage on 8th Month)
#       - std_og_mou_Percent_8 (STD outgoing Minutes of usage % in 8th month wrt to Total Outgoing usage on 8th Month)
#       - spl_og_mou_Percent_8 (Special Outgoing Minutes of usage % in 8th Month wrt to total Outgoing usage on 8th Month)
#       - loc_ic_mou_Percent_8 (Local Incoming Minutes of Usage % in 8th Month wrt to Total incoming usage on 8th month)
#       - std_ic_mou_Percent_8 (STD Incoming Minutes of Usage % in 8th Month wrt to total incoming Usage on 8th Month)
#       - ic_others_Percent_8 (Others incoming Minutes of Usage % in 8th Month wrt to Toal incoming Usage on 8th Month)
#       
# * Driving Feature List As per Original Features given in Dataset (12 Features)
# 
#       - total_Rech_Num_7 (Total Number of Recharge on 7th Month)
#       - total_Rech_Num_8 (Total number of Recharge on 8th Month)
#       - last_date_of_month_8 (Last Date of 8th Month)
#       - date_of_last_rech_8 (Last recharge date on 8th Month)
#       - loc_og_mou_8 (Local Outgoing Minutes of Usage on 8th Month)
#       - std_og_mou_8 (STD Outgoing Minutes of Usage on 8th Month)
#       - spl_og_mou_8 (Special Outgoing Minutes of usage on 8th Month)
#       - loc_ic_mou_8 (Local Incoming Minutes of Usage on 8th Month)
#       - std_ic_mou_8 (STD Incoming Minutes of Usage on 8th Month)
#       - ic_others_8 (Others Incoming Minutes of Usage on 8th Month)
#       - total_ic_mou_8 (Total Incoming Minutes of Usage on 8th Month)
#       - total_og_mou_8 (Toal Outgoing Minutes of Usage on 8th Month)
# 

# In[ ]:




