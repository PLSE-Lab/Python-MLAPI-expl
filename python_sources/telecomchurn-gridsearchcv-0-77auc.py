#!/usr/bin/env python
# coding: utf-8

# ## Customer Churn
# 
# Customer Churn is one of the important metrics to evaluate the growth potential of a business enterprise. It is measured as the percentage of customers that stopped using a company's product or services at the end of a time period by the number of customers it had at the beginning of the time period.
# Customer churn analysis is used widely by telecommunication services, internet service providers, online streaming services, insuarance firms, etc. as the cost of retaining an existing customer is cheaper than acquiring a new one. Now, churn can be volunatry or  involuntary. Involutary churn can be the decision of a customer to switch to an alternative company or service providers. Involuntary churn includes relocation to other location, death, etc. In majority of the application, analysis is based on involuntary churn which can be primarily due to the customer's current product or service experience of a company or due to a better alternative options provided by a company's business competitors.
# In this study, we use the data from a telecommunication company to perform churn analysis. Such an analysis will help in understanding the customer retention policies that are required to reduce the churn rate. The structure of this study are as follows:
# - **I**: Exploratory data analysis
# 
# - **II**: Data pre-processing
# 
# - **III**: Predictive modeling
# 
# - **IV**: Conclusion

# In[ ]:


from datetime import datetime, timedelta,date
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import Image
from IPython.core.display import HTML 

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error

import sklearn.metrics as metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans


# > ## I. Exploratory data analysis:

# In[ ]:


TelecomChurn = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
#first few rows
TelecomChurn.head()


# In[ ]:


# data summary
print("Data dimension:",TelecomChurn.shape)
TelecomChurn.info()


# The customer churn data mainly consists of customer's social characteristics, type of service packages used by them and the method of payment. There are 7043 observations with 21 variables.

# In[ ]:


# encoding the churn variable into 0 and 1
TelecomChurn['Churn'] = TelecomChurn['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
TelecomChurn.head()


# The variable 'Total Charges' is converted into a numeric format. Then check for any missing observations. Some observations for this variable is found to be missing. I'll be deleting the missing observations from the dataset.

# In[ ]:


# changing TotalCharges column from object to float
TelecomChurn['TotalCharges'] = TelecomChurn['TotalCharges'].apply(pd.to_numeric, downcast='float', errors='coerce')
print(TelecomChurn.dtypes)


# In[ ]:


# check for null and total observations related to it
null_columns=TelecomChurn.columns[TelecomChurn.isnull().any()]
TelecomChurn[null_columns].isnull().sum()


# There were 11 missing observations under the variable 'Total Charges' which will be removed from the dataset.

# In[ ]:


# drop na variables
TelecomChurn = TelecomChurn.dropna()
TelecomChurn.shape


# In[ ]:


# summary description of the numeric variables of the dataset
TelecomChurn[['tenure', 'MonthlyCharges', 'TotalCharges']].describe()


# There is no presence of negative or missing observations in these three variables. Next, I check the dimension of the categorical variables by looking at their unique values. The unique values vary between 2 and 4 for the categorical variables.

# In[ ]:


# to check the number of unique values in each of the columns
for col in list(TelecomChurn.columns):
    print(col, TelecomChurn[col].nunique())


# In[ ]:


# calculate the proportion of churn vs non-churn
TelecomChurn['Churn'].mean()


# The churn rate is around 27% in this dataset. Now, I'll be looking at few variables which can give us more information about the factors that can effect customer churn.

# In[ ]:


# calculate the proportion of churn by gender
churn_by_gender = TelecomChurn.groupby(by='gender')['Churn'].sum() / TelecomChurn.groupby(by='gender')['Churn'].count() * 100.0
print('Churn by gender:',churn_by_gender)


# There is no such significant difference in churn rate by gender.

# In[ ]:


# calculate the proportion of churn by contract
churn_by_contract = TelecomChurn.groupby(by='Contract')['Churn'].sum() / TelecomChurn.groupby(by='Contract')['Churn'].count() * 100.0
print('Churn by contract:',churn_by_contract)


# Churn rate is highest for the short-term contract compared to the long-term contract. For a month-to-month contract, the churn rate is 42% which is higher compared to the 'one-year' or the 'two-year' contract. The churn rates for the 'one-year' or the 'two-year' contract are 11% and 2% respectively.

# In[ ]:


# calculate the proportion of churn by payment method
churn_by_payment = TelecomChurn.groupby(by='PaymentMethod')['Churn'].sum() / TelecomChurn.groupby(by='PaymentMethod')['Churn'].count() * 100.0
print('Churn by payment method:',churn_by_payment)
pd.DataFrame(churn_by_payment)


# The churn rate is the highest when the method of payment is through an electronic check. The other segments of the payment methods have a churn rate between 15% - 20%. The graph for this analysis is presented below.

# In[ ]:


# figure
ax = churn_by_payment.plot(
    kind='bar',
    color='skyblue',
    grid=False,
    figsize=(10, 7),
    title='Churn Rates by Payment Methods'
)

ax.set_xlabel('Payment Methods')
ax.set_ylabel('Churn rate (%)')

plt.show()


# In[ ]:


# proportion of churn by gender and contract
churn_gendercontract = TelecomChurn.groupby(['gender', 'Contract'])['Churn'].sum()/TelecomChurn.groupby(['gender', 'Contract'])['Churn'].count()*100
churn_gendercontract


# In[ ]:


# keep gender in row and contract by column
churn_gendercontract1 = churn_gendercontract.unstack('Contract').fillna(0)
churn_gendercontract1 


# In[ ]:


# figure
ax = churn_gendercontract1.plot(
    kind='bar', 
    grid= False,
    figsize=(10,7)
)

ax.set_title('Churn rates by Gender & Contract Status')
ax.set_xlabel('Gender')
ax.set_ylabel('Churn rate (%)')

plt.show()


# The above figure analyzed the churn rate when it is grouped by gender and contract type. Irrespective of the gender, we find that the churn rate is the highest for the month-to-month contract type. The conclusion of this analysis is similar to the conclusion derived from their individual analysis presented above.

# In[ ]:


# observations by citizen type
TelecomChurn['SeniorCitizen'].value_counts()


# In[ ]:


# Total observations by citizen type, contract and tech support 
TelecomChurn.groupby(['SeniorCitizen','Contract','TechSupport'])['Churn'].count()


# In[ ]:


# proportion of churn by gender and contract
churn_citizentechcontract = TelecomChurn.groupby(['SeniorCitizen','Contract','TechSupport'])['Churn'].sum()/TelecomChurn.groupby(['SeniorCitizen','Contract','TechSupport'])['Churn'].count()*100
churn_citizentechcontract


# In[ ]:


# keep gender and payment method in row and contract by column
churn_citizentechcontract1 = churn_citizentechcontract.unstack(['TechSupport']).fillna(0)
churn_citizentechcontract1


# In[ ]:


# figure
ax = churn_citizentechcontract1.plot(
    kind='bar', 
    grid= False,
    figsize=(10,7)
)

ax.set_title('Churn rates by Citizen Type, Tech Support & Contract Status')
ax.set_xlabel('Non-Senior:"0"   Senior:"1"')
ax.set_ylabel('Churn rate (%)')

plt.xticks()
plt.show()


# Few interesting conclusions can be derived from the above exploratory data analysis. The above analysis is based on the citizen type, contract type and technical support. The senior citizen is denoted as 1 or 0 otherwise. Non-senior citizens are the majority in this dataset by around 5:1 ratio. For the senior citizen, there is no churn rate for the no internet service for the one-year and two-year contracts.
# 
# The availability of tech support helps in reducing the churn rate compared to those with no tech support. Though churn rate is even lower for those with no internet service compared to those with technical support. It is not clear if they have any other means of customer service available to them without an internet service or not. Nevertheless, improvement in the technical support can be useful in reducing the churn rate.

# In[ ]:


# summary of tenure, monthly charges and total charges
TelecomChurn[['tenure','MonthlyCharges','TotalCharges']].describe()


# In[ ]:


# plot a histogram
plt.hist(TelecomChurn['tenure'], bins= 100, alpha=0.5,)
plt.title('Frequency Distribution by Tenure')
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.show()


plt.hist(TelecomChurn['tenure'], cumulative=1, density =True, bins= 100)
plt.title('Cumulative Frequency Distribution by Tenure')
plt.xlabel('Tenure')
plt.ylabel('Cumulative Frequency Distribution')
plt.show()


# From the above analysis, I found that a significant share of the customers are either new with a tenure of less than 10 months or old customers with more than 70 months of tenure. From the cumulative frequency distribution plot, we find that around 20% of the customers have a tenure of less than 10 months. Around 70% of the customers have a tenure between 10 months and 60 months. And the remaining 20% have a tenure more than 60 months.
# 
# From the graph presented below, the churn rate decreases as the tenure length increases. This indicates customer loyalty in terms of tenure length and its effect on the churn rate. I can conclude that the telecom service producer should focus on the new customers so that they avail its telecom services for a longer period of time.

# In[ ]:


# proportion of churn by tenure
churn_monthlycharges = TelecomChurn.groupby(by = 'tenure')['Churn'].mean().reset_index()
churn_monthlycharges

plt.figure(figsize=(10,7))
plt.scatter(churn_monthlycharges.tenure, churn_monthlycharges.Churn)
plt.title('Churn Rate by Tenure')
plt.xlabel('Tenure')
plt.ylabel('Churn Rate')
plt.show()


# In[ ]:


# proportion of churn by MonthlyCharges
churn_monthlycharges = TelecomChurn.groupby(by = 'MonthlyCharges')['Churn'].mean().reset_index()
plt.figure(figsize=(6,5))
plt.scatter(churn_monthlycharges.MonthlyCharges, churn_monthlycharges.Churn)
plt.title('Churn Rate by Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Churn Rate')
plt.show()

# proportion of churn by TotalCharges
churn_totalcharges = TelecomChurn.groupby(by = 'TotalCharges')['Churn'].mean().reset_index()
plt.figure(figsize=(6,5))
plt.scatter(churn_totalcharges.TotalCharges, churn_totalcharges.Churn)
plt.title('Churn Rate by Total Charges')
plt.xlabel('Total Charges')
plt.ylabel('Churn Rate')
plt.show()


# The pattern for churn rate for monthly charges varies quite a lot between 0 and 1. Whereas, the churn rate is mostly either 0 or 1 for the total charges. It is hard to graphically derive additional insights about churn vs no-churn from these two variables.

# In[ ]:


# finding correlations
corrdata = TelecomChurn[['Churn','tenure','MonthlyCharges','TotalCharges']]
corr = corrdata.corr()
# plot the heatmap
sns.heatmap(corr,cmap="coolwarm",
        xticklabels=corrdata.columns,
        yticklabels=corrdata.columns,annot=True)


# From the above, correlation plot I find some interesting conclusions. Total Charges are highly correlated with tenure and monthly charges. Whereas, correlation between tenure and monthly charges is low.
# 
# There is a negative relationship between churn and tenure. There is a positive relationship between churn and tenure. Both these relationships make sense. But, I see a negative relationship between total charges and churn. This means that as the total charges increases, the churn rate goes down. This conclusion is absurd! Why a customer will be willing to pay a higher cost for the services? I'll be droping 'Total Charges' from the final dataset due to high collinearity with tenure and monthly charges, as well as a negative relationship with churn.

# > ## II. Data pre-processing:
# 
# In this section, I'm going to process the data which will be used for predictive modeling. I'll scale the 'tenure' and 'Monthly Charges', and pre-process the categorical variables.

# In[ ]:


# segmenting based on data type and pre-processing
#customer id col
Id_col     = ['customerID']
#Target column. y should be an array
target_col = ["Churn"]
y = (TelecomChurn[target_col]).values.ravel()
# cluster column 
cluster_col = ["tenure"]
#categorical columns with categories less than 6
cat_cols   = TelecomChurn.nunique()[TelecomChurn.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
print(cat_cols)
#Binary columns with 2 values
bin_cols   = TelecomChurn.nunique()[TelecomChurn.nunique() == 2].keys().tolist()
print(bin_cols)
#Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]
print(multi_cols)
# continuous column
cont_col = ["tenure","MonthlyCharges"]
print(cont_col)
print(y)


# In[ ]:


#Label encoding Binary columns
le = LabelEncoder()
binary = TelecomChurn[bin_cols]
print(binary.shape) 
print(binary.info())
binary.head()
for i in bin_cols :
    binary[i] = le.fit_transform(binary[i])


# In[ ]:


# multi-label categorical columns
dummy_vars = pd.get_dummies(TelecomChurn[multi_cols])
print(dummy_vars.shape)
print(dummy_vars.info())


# In[ ]:


#Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(TelecomChurn[cont_col])
scaled = pd.DataFrame(TelecomChurn,columns= cont_col)
scaled.shape
print(scaled.info())


# In[ ]:


# creating a dataset to combine pre-processed variables
X = pd.concat([binary,scaled,dummy_vars], axis = 1)
# drop churn variable from the X dataset
X = X.drop(['Churn'],axis=1)
print(X.shape)
print(X.info())


# > ## III. Predictive Modeling    
# 
# The objective of this study is to predict the churn vs no-Churn situation from the telecom dataset.Inorder to predict the outcome, we need to predict whether a customer will churn or not i.e.(Churn/NoChurn as (1/0)).
# 
# Now for each of the observations, four different events can occur when we try to predict:
# - case 1: predicted as 1 which are actually 1. Also known as True Positives (TP).
# - case 2: predicted as 1 which are actually 0. Also known as False Positives (FP).Also known as Type I error.
# - case 3: predicted as 0 which are actually 0. Also known as True Negatives (TN).
# - case 4: predicted as 0 which are actually 1. Also known as False Negatives (FN).Also known as Type II error.
# 
# In this study, predicting churn accurately is very important as it has a far-reaching impact on the future prospect of the business. This means a prediction analysis will be able to identify case 1 with greater accuracy. This means the objective of the study is to minimize case 4 or Type II error. There are certain criterias based on which prediction models will be judged. Let's look at those criteria.
# 
# Accuracy,precision, recall, F1 and specificity are the different types of precision metrics used in evaluating the performance of the prediction models. These are also used to compare the performances between different alternative models. Accuracy measures the ratio of TP and TN given all the observations. Precision measures the ratio of the relevant class (i.e.Churn(1)) correctly predicted given the total number of predictions made for the relevant class. Recall also known as sensitivity measures the ratio of the relevant class given the actual number of observations of the relevant class. F1 score is the weighted average of precision and recall. It is particularly useful when we want to strike a balance between precision and recall. Sensitivity is the ratio of TN given the actual number of observations that belongs to the negative class. Another important metrics are Receiver Operating Curve (ROC) and Area under the Curve (AUC). A good discussion on these metrics are available at: https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5.
# 
# As mentioned earlier, one of the important objectives of the prediction model here is to reduce False Negatives (FN). This means that we have to consider improving the recall. But reducing the FN sometimes leads to an increase in FP which means precision may decrease. In this case, I'll look at the F1 scores along with the AUC values to decide on the best model for this study.

# In[ ]:


# import machine learning libraries
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import xgboost as xgb

# creating the function
# XGBoost and SVC functions are used while modeling and are thus not presented here
logreg = LogisticRegression(solver='lbfgs', max_iter = 10000)
DT = DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()


# Recursive Feature Elimination (RFE) is a method of selecting a subset of independent variables (features) that are relevant for predicting the target variable i.e. churn in this study. It uses the model accuracy to identify the attributes that contribute the most towards explaining the target variable. One can use logistic regression and tree-based models for feature extraction. I'm using random forest for feature extraction here. More information on RFE is provided in https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

# In[ ]:


# recursive feature extraction for the top 15 features
rfe = RFE(rfc, 10)
rfe = rfe.fit(X, y)
print(rfe.support_)
print(rfe.ranking_)

#identifying columns for RFE
rfe_data = pd.DataFrame({"rfe_support" :rfe.support_,
                       "columns" : [i for i in X.columns if i not in Id_col + target_col],
                       "ranking" : rfe.ranking_,
                      })

# extract columns as a list
rfe_var = rfe_data[rfe_data["rfe_support"] == True]["columns"].tolist()

rfe_data


# 
# 
# 

# In[ ]:


# select a subset of variables for the dataframe based on RFE method
X1 = X[rfe_var]


# #### Logistic Regression

# In[ ]:


# running a logistic regression
# copy the dataset 
X2 = X1
# manually add intercept
X2['intercept'] = 1.0;
#X2.head()
logit_model=sm.Logit(y,X2)
result=logit_model.fit()
print(result.summary2())


# I ran a logistic regression model to understand the effect of the independent variables on the target variable. Given the p-values, most of the variables are insignificant at 5% level of significance. Let's interpret the above result for some selected variables. For gender, male is denoted as 1 and female is denoted 0. The negative coefficient estimate before the 'gender' variable suggest that the log-odd of churn for male relative to female is lower. This is consistent with the data analysis done with gender earlier.
# 
# If the tenure length increases then the log-odds of churning i.e. the possibility of churn decreases. This statistical result confirms the result obtained earlier from the exploratory data analysis. In case of the tech support availability, if the tech support is not available then the possibility of churn increases. If there is a month-to-month contract then the chances of churn increases. An increase in monthly charges increases the churn rate. These statistical results are similar to the conclusions drawn from the exploratory data analysis.

# In[ ]:


# create a train and test set with the new selected variables
Xtrain, Xtest, ytrain,ytest = train_test_split(X1,y,test_size = 0.2,random_state = 111)


# In[ ]:


print('Ratio of churn in the training sample:',ytrain.mean())
print('Ratio of churn in the training sample:',ytest.mean())


# In the preceding section, I have split the dataset into train and test set. I'll be using 10-fold cross validation which is an useful method for model training when you have an unbalanced dataset. The data here is unbalanced because the ratio of churn to non-churn is 1:3. I'll also use grid search to derive the optimal model. To know more about GridSearchCV and its importance check this link: https://www.datacamp.com/community/tutorials/parameter-optimization-machine-learning-models 
# 
# First objective of prediction modeling is to create a base model. In this case, a simple logistic regression is my base model. Then explore alternative machine learning models to find better models for improving the prediction capability. For comparison across different models, I'll look at precision, recall, f1-score and auc score.

# #### Logistic Regression (Base)

# In[ ]:


parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(estimator=logreg, param_grid = parameters,cv = 10,scoring = 'accuracy')
grid.fit(Xtrain,ytrain)

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters:", grid.best_params_)


# In[ ]:


def result(X_test,y_test):

    y_test_pred = grid.predict(X_test)
    print('Accuracy score:{:.2f}'.format(accuracy_score(y_test, y_test_pred)))
    print(                                                                                )

    confusionmat_data = pd.DataFrame({'y_Predicted': y_test_pred,'y_Actual': y_test},columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(confusionmat_data['y_Actual'], confusionmat_data['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    print('Confusion Matrix:\n {}\n'.format(confusion_matrix))
    print(                                                                               )

    class_report = classification_report(y_test, y_test_pred)
    print('Classification report:\n {}\n'.format(class_report))
    print(                                                                               )

    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    print('Mean-squared error:\n {}\n'.format(rmse))

    # predict probabilities
    #probs = grid.predict_proba(X_test)
    #probs = grid.predict(X_test)

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_test_pred)
    roc_auc = metrics.auc(fpr, tpr)
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


# result
result(Xtest,ytest)


# #### Decision Tree Classification Model

# In[ ]:


parameters = {'min_samples_split': [10,100,1000,10000],
              'max_depth':[2,5,10,100,150,200,250]}

grid = GridSearchCV(estimator= DT, param_grid = parameters,cv = 10,scoring = 'accuracy')
grid.fit(Xtrain,ytrain)

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters:", grid.best_params_)


# In[ ]:


grid.best_estimator_.feature_importances_
for name, importance in sorted(zip(grid.best_estimator_.feature_importances_,Xtrain.columns),reverse= True)[:5]:
    print(name, importance)
    
featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = Xtrain.columns)
featureimp_plot.nlargest(5).plot(kind='barh')  


# In[ ]:


# result
result(Xtest,ytest)


# #### Random Forest Classification Model

# In[ ]:


parameters = {'n_estimators': [1,5,10,100,200],'min_samples_split': [10,100,1000,10000],'max_depth':[2,5,10,100,150,200,250]}

grid = GridSearchCV(estimator= rfc, param_grid = parameters,cv = 10,scoring = 'accuracy')
grid.fit(Xtrain,ytrain)

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters:", grid.best_params_)


# In[ ]:


grid.best_estimator_.feature_importances_
for name, importance in sorted(zip(grid.best_estimator_.feature_importances_,Xtrain.columns),reverse= True)[:5]:
    print(name, importance)
    
featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = Xtrain.columns)
featureimp_plot.nlargest(5).plot(kind='barh')   


# In[ ]:


# result
result(Xtest,ytest)


# #### XGBoost Classification Model

# In[ ]:


#building the model & printing the score
parameter = {
'max_depth': [1,5,10,15],
'n_estimators': [50,100,150,300],
'learning_rate': [0.01, 0.1, 0.3],
}

grid = GridSearchCV(xgb.XGBClassifier(objective='binary:logistic'), param_grid = parameter, cv= 5, scoring='balanced_accuracy')
grid.fit(Xtrain,ytrain)

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters:", grid.best_params_)


# In[ ]:


featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = Xtrain.columns)
featureimp_plot.nlargest(5).plot(kind='barh') 


# In[ ]:


# result
result(Xtest,ytest)


# #### SVC Classification Model

# In[ ]:


parameter = {'C': [5,10, 100]}
grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid = parameter, cv= 4)

grid.fit(Xtrain,ytrain)

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters:", grid.best_params_)


# In[ ]:


# result
result(Xtest,ytest)


# We can summarize the result as follows:
# 
# **1.**: Variables such as contract type, status of online security, method of internet service provided, availability status of technical support and payment method are among the top 5 variables in the predictive analysis.
# 
# **2.**: Precision is greater than recall. The f1-score ranges between 0.55 and 0.62 for the 'churn' class.It is the highest for decision tree model.
# 
# **3.**: The AUC score ranges between 0.70 and 0.74. In terms of the AUC score, an optimal Decision Tree is the best model followed by SVM and XGBoost.
# 
# As I'm working with an unbalanced dataset, machine learning models can be bias in favor of the majority classi.e. no churn as opposed to the minority class i.e. churn. Given the objective of this study, understanding and predicting 'churn' becomes very important. But unbalanced dataset effect the prediction analysis for 'churn' with a lower recall value. In order to overcome this problem, I'm going to use SMOTE which is a type of an over-sampling method.
# 
# SMOTE balances the class distribution between 'churn' and 'no churn'. Under this method new observations for 'churn' are created between the existing 'no churn' observations. It generates the training sample by the linear transformation of the existing observations of the 'churn' class.

# #### SMOTE (Synthetic Minority Oversampling Technique)
# 
# As I'm working with an unbalanced dataset, machine learning models can be biased in favor of the majority class i.e. no churn as opposed to the minority class i.e. churn. Given the objective of this study, predicting 'churn' becomes very important. But unbalanced dataset effect the prediction analysis for 'churn' with a lower recall value. In order to overcome this problem, I'm going to use SMOTE which is a type of an over-sampling method.
# 
# SMOTE balances the class distribution between 'churn' and 'no churn'. Under this method new observations for 'churn' are created between the existing 'no churn' observations. It generates the training sample by the linear transformation of the existing observations of the 'churn' class.

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


#Split train and test data
smote_train_X,smote_test_X,smote_train_Y,smote_test_Y = train_test_split(X,y,
                                                                         test_size = .25 ,
                                                                         random_state = 111)

#oversampling minority class using smote
os = SMOTE(random_state = 0)
os_smote_X,os_smote_Y = os.fit_sample(smote_train_X,smote_train_Y)
os_smote_X = pd.DataFrame(data = os_smote_X,columns= X.columns)
os_smote_Y = pd.DataFrame(data = os_smote_Y,columns= ['Churn'])

print(os_smote_X.shape)
print(os_smote_Y.shape)


# In[ ]:


rfe = RFE(rfc, 10)
rfe = rfe.fit(X, y)
print(rfe.support_)
print(rfe.ranking_)

#identified columns Recursive Feature Elimination
rfe_data = pd.DataFrame({"rfe_support" :rfe.support_,
                       "columns" : [i for i in X.columns if i not in Id_col + target_col],
                       "ranking" : rfe.ranking_,
                      })
selected_cols = rfe_data[rfe_data["rfe_support"] == True]["columns"].tolist()

rfe_data
print(selected_cols)


# In[ ]:


# calculate the proportion of churn is now equal to the no churn
os_smote_Y.mean()


# In[ ]:


#train and test data under SMOTE
train_smoterfe_X = os_smote_X[selected_cols]
train_smoterfe_Y = os_smote_Y.values.ravel()
test_smoterfe_X  = smote_test_X[selected_cols]
test_smoterfe_Y  = smote_test_Y


# #### Logistic Regression with SMOTE

# In[ ]:


parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(estimator=logreg, param_grid = parameters,cv = 10,scoring = 'accuracy')
grid.fit(train_smoterfe_X,train_smoterfe_Y)

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters:", grid.best_params_)


# In[ ]:


# result
result(test_smoterfe_X,test_smoterfe_Y)


# #### Decision Tree with Smote    

# In[ ]:


parameters = {'min_samples_split': [10,100,1000,10000],
              'max_depth':[2,5,10,100,150,200,250]}

grid = GridSearchCV(estimator= DT, param_grid = parameters,cv = 10,scoring = 'accuracy')
grid.fit(train_smoterfe_X,train_smoterfe_Y)

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters:", grid.best_params_) 

grid.best_estimator_.feature_importances_
for name, importance in sorted(zip(grid.best_estimator_.feature_importances_,train_smoterfe_X.columns),reverse= True)[:5]:
    print(name, importance)
    
featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = train_smoterfe_X.columns)
featureimp_plot.nlargest(5).plot(kind='barh')   


# In[ ]:


# result
result(test_smoterfe_X,test_smoterfe_Y)


# #### Random Forest with SMOTE

# In[ ]:


parameters = {'n_estimators': [1,5,10,100,200],'min_samples_split': [10,100,1000,10000],'max_depth':[2,5,10,100,150,200,250]}

grid = GridSearchCV(estimator= rfc, param_grid = parameters,cv = 10,scoring = 'accuracy')
grid.fit(train_smoterfe_X,train_smoterfe_Y)

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters:", grid.best_params_)


# In[ ]:


grid.best_estimator_.feature_importances_
for name, importance in sorted(zip(grid.best_estimator_.feature_importances_,train_smoterfe_X.columns),reverse= True)[:5]:
    print(name, importance)
    
featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = train_smoterfe_X.columns)
featureimp_plot.nlargest(5).plot(kind='barh')   


# In[ ]:


# result
result(test_smoterfe_X,test_smoterfe_Y)


# #### XGBoost with SMOTE

# In[ ]:


#building the model & printing the score
parameter = {
'max_depth': [1,5,10,15],
'n_estimators': [50,100,150,300],
'learning_rate': [0.01, 0.1, 0.3],
}

grid = GridSearchCV(xgb.XGBClassifier(objective='binary:logistic'), param_grid = parameter, cv= 5, scoring='balanced_accuracy')
grid.fit(train_smoterfe_X,train_smoterfe_Y)

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters:", grid.best_params_)


# In[ ]:


grid.best_estimator_.feature_importances_
for name, importance in sorted(zip(grid.best_estimator_.feature_importances_,train_smoterfe_X.columns),reverse= True)[:5]:
    print(name, importance)
    
featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = train_smoterfe_X.columns)
featureimp_plot.nlargest(5).plot(kind='barh')  


# In[ ]:


# result
result(test_smoterfe_X,test_smoterfe_Y)


# #### SVM with SMOTE

# In[ ]:


parameter = {'C': [1,5,10, 100]}
grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid = parameter, cv= 4)
grid.fit(train_smoterfe_X,train_smoterfe_Y)

print("Best score: %0.3f" % grid.best_score_)
print("Best parameters:", grid.best_params_)


# In[ ]:


# result
result(test_smoterfe_X,test_smoterfe_Y)


# > ## IV. Conclusion:
# 
# **1.**: SMOTE has reduced the gap between precision and recall for the 'churn' class. Precision still remains higher than recall for the 'churn class' except for the Logistic Regression and linear SVM models. For these two models, recall becomes higher than precision. The recall value improved in majority of the models due to a lower False Negatives. The f1-score is between 0.60 and 0.65 for the 'churn' class. 
# 
# **2.**: The AUC score is now between 0.72 and 0.77. The AUC score decresed for the Decision Tree. Logistic Regression and linear SVM model has the highest AUC score of 0.77. In terms of AUC score, both these models are best suited for this study. I will prefer Logistic Regression because it is simple, intuitive and interpretable.
