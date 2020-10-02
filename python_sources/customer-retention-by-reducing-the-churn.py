#!/usr/bin/env python
# coding: utf-8

# # **Customer Retention**
#  *A Data Science Project By Aparna Shasty*

# ## Overview
# 
# + Recognizing/defining the business problem
# + Data Wrangling
# + Exploratory Data Analysis (EDA) and Visualizations
# + Data Storytelling
# + Training and Testing Machine Learning models: Expect to see a few rarely discussed concepts like Recursive Feature Selection
# + Recommendations to retain the customers
# + Scope for future work
# + A Final Note
# + References
# 
# ### 1) Business Problem
# 
# A telecom company has been affected by the increasing number of customers subscribing to the services of a competitor. It is much more expensive to attract new customer than retaining old customer. At the same time, spending too much on or spending on the wrong factor for retaining customer who has no intention to leave (or who was not leaving for that factor which was addressed) could be a waste of  money. Therefore it is important to identify the customer who has high probability of leaving and zero down on the reason for it. An analysis of the past records of the customers can give great insights on who might leave and what is the cause. The telecom company already has this data available and data scientist need not collect the data in this case. The data can be found in [IBM page](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/) : [Telecom Dataset]('https://community.watsonanalytics.com/wp-content/uploads/2015/03/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# 
# ### Step by Step approach to Predicting and Preventing Customer Churn Rate
# #### Quick Examination of the Dataset

# In[ ]:


import numpy as np
import scipy as sp
import pandas as pd

import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pylab

import time
from scipy.stats import pearsonr


# In[ ]:


ch = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
ch.info()
ch.head()


# Note that,
# 
# Target (Dependent) Variable: **Churn**
# 
# Feature (Independent) Variables: 19 of them out of 21 shown above. Churn is target, Customer ID is just unique identity.
# 
# **We can divide predictor variables into,**
# + Service specific : 
#   - Phone: PhoneService, MultipleLines
#   - Internet: InternetService, Online Security, Online backup, Streaming TV, Streaming Movies, Tech support, Device protection
# + Person specific : gender, SeniorCitizen,Partner, Dependents, tenure (loyalty)
# + Money specific: Monthly Charges, TotalCharges, Contract, Paperlessbilling, Payment Method
# 
# In a real scenario, we would get additional information from the business owner on relationship between charges and remaining factors. Still we will have to verify them from the data, because their assumptions on how they run business and the reality can differ. Hence,
# 
# **The Questions to which we seek answers:**
# - Are monthly charges different for different contract types?
# - Are monthly charges solely dependent on the number/type of the services?
# - Are there any discounts given to loyal customers?
# - Is there a correlation between monthly charges and churn?
# - Is there a correlation between tenure and churn?
# - Is there a correlation between certain type of services and churn?
# - Is there any person specific trends in churn?
# - Can we make predictions on likelihood of a customer churn given the predictor variables listed above
# 
# and, Try quantifying these correlations into actionable items.
# 
# First check what percentage of the given dataset has data for customers who switch and what percentage is loyal?

# In[ ]:


_ = (ch.groupby('Churn')['customerID'].count()/ch['customerID'].count()).plot.bar()
_ = plt.title('Proportion of Customers')
_ = plt.ylabel('Proportion')
_ = plt.xlabel('Left (Yes) or Remained (No)')
print('Overall Customer Churn percentage in the given dataset is {} %'.format(round(ch.Churn.replace({'No':0,'Yes':1}).mean()*100,2)))


# 26.54% or 1869 of the 7043 records in the dataset belong to customers who switched to a competitor. This is a binary classification problem with moderately  imbalanced dataset. 

# ### 2) Data Wrangling
# The output of the cell number 2 above gives all the data types and counts of non-null entries. Non-null does not mean valid entries. We need to ensure all of them have meaningful datatypes and valid entries. The TotalCharges is of type object, which means there is some non-numeric entry. It is expected to be float. Look at non-churn and churn group separately, as their dynamics might differ. Compute the I choose to impute the invalid entries with median of each group.
# 
# Senior Citizen is a category variable, however it is given as int. It need not be converted, as it will be eventually converted back to numeric.

# In[ ]:


# Examine the rows with total charges blank
ch[ch['TotalCharges'] == ' ']


# tenure for all these 11 rows is 0. Churn is "No". One can interpret this as belonging to all new customers, if indeed these are the only rows with tenure = 0 too. This is verified to be true below. Hence we can set TotalCharges to 0, whenever tenure is 0.

# In[ ]:


ch[ch['tenure'] == 0]


# In[ ]:


ch.TotalCharges = ch.TotalCharges.apply(lambda x: 0  if x == ' ' else float(x))


# In[ ]:


# Change No internet service to NoInt for brevity
ch[['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']] = ch[['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']].astype(str).replace({'No internet service': 'NoInt'})
ch['MultipleLines'] = ch['MultipleLines'].replace({'No phone service':'NoPh'})


# All other variables are fine. Converting object data types to categories and doing one hot encoding are explained along the way. 
# ### **3) Exploratory Data Analysis, Data Story Telling**
# 
# Now that we have clean data, let us quantify what impact this project could make, if all the customers who go away can be convinced to staying. We will also group tenure into 4 categories for better comprehension.

# In[ ]:


# Split the customers into 4 groups of tenures and see their Churn Rate
_ = plt.figure(figsize=(12,4))
_ = plt.subplot(1,2,1)
ch['Tenure Group'] = 'Between 2 to 5 Years'
ch.loc[ch['tenure'] >59,'Tenure Group'] = 'More than 5 Years' 
ch.loc[ch['tenure'] <= 24,'Tenure Group'] = '1 Year -> 2 Year'
ch.loc[ch['tenure'] <= 12,'Tenure Group'] = 'Less Than 1 Year'
ch['Ch10'] = ch['Churn'].replace({'Yes':1,'No':0})
ch_rate = ch.groupby('Tenure Group')['Ch10'].mean().sort_values(ascending=False)
(round(100*ch_rate,2)).plot.bar(color='pink')

# Evaluate the Revenue Loss per month
ch['revloss'] = ch['MonthlyCharges']*ch['Ch10']
_ = plt.ylabel('Churn Percentage')
_ = plt.title('Percentage Churn Vs Tenures')
_ = plt.subplot(1,2,2)
revenue_group = ch.groupby('Tenure Group')['revloss'].sum().sort_values(ascending=False)
(round(100*revenue_group/revenue_group.sum(),2)).plot.bar(color='g')
_ = plt.ylabel('Loss Percentage')
_ = plt.title('Percentage Revenue loss/Month Vs Tenure group')
print('Total Revenue Lost/Month due to Churn: $',int(revenue_group.sum()))


# + The business is losing approx $140k every month as per the current data!
# + Customers less than 1 year have the highest churn and cause highest loss.
# + The second and third place are interchanged in the two plots. 
# 
# A business would be interested in retaining the category that causes higher loss with immediate priority, as that will be more return on investment. 

# In[ ]:


# Check the stats for numeic types within Churn and No Churn group
numvar = ['tenure','MonthlyCharges','TotalCharges']
round(ch.groupby('Churn')[numvar].describe().T,2)


# + Mean tenure for Churn group is lower than the other group, which is as expected.
# + Mean Monthly charges are higher for Churn group, this has to be analyzed further.
# + Mean Total Charges is higher for loyal group than the churn group.

# In[ ]:


# Plot the histogram of the tenure and see if it tells any story!
fig, ax = plt.subplots(figsize=(14,5))
_ = plt.subplot(1,2,1)
sns.distplot(ch.loc[ch.Churn=='No','tenure'],hist=True,color='g',kde=False)
_ = plt.title('Histogram of tenure values for loyal customers')
_ = plt.xlabel('Tenure in Months')
_ = plt.ylabel('People count')
_ = plt.subplot(1,2,2)
sns.distplot(ch.loc[ch.Churn=='Yes','tenure'],hist=True,kde=False)
_ = plt.title('Histogram of tenure values for customers who left')
_ = plt.xlabel('Tenure in Months')
_ = plt.ylabel('People count')
print("Mean Tenure of Two groups\n",round(ch.groupby('Churn').tenure.mean(),2))
_ = plt.figure()
_ = ch[['Churn','tenure']].boxplot(by='Churn')


# **Remarks: ** There is a clear distinction in the shape of the two histograms above. The percentiles are visible in the box plot. 
# + Customers who cut the contract are highly concentrated towards lesser tenure (<= 6 months). The number of customers in each higher bin progressively reduces.!
# + The first peak in the first plot tells that there are many more new customers than those in each of the other bins except the last one.
# + First bin to second bin, there is huge reduction, as some have Churned. There is slight reduction until bin centered at 40.
# + Customers who have crossed 20 months are likely to remain loyal and hence the counts are almost same until bin centered 60.
# + The last peak is a fact about the current dataset. It seems like there are many old customers (count > 850) who have stayed with the company's connection for more than 66 months. Suddenly there was a drop in number of customers and that is why there is less count in the bin previous to that. This clearly means, **Many customers switched to other companies about 5.5 years ago.** There could be a number of reasons for this:
#    - This business entity might not have been able to upgrade its services to state of the art technology. 
#    - There was monopoly but suddenly a competitor company popped up and attracted its customers with inauguration offers.
#    - The progressive decrease from bin 65 to bin 60 to bin 55 and so on support the above points.
#    - We can not do further analysis on this currently, as there is no data for customers who left 5.5 years ago. We only know the number of such customers.

# In[ ]:


# Find the correlation between tenure*Monthly Vs TotalCharges
print("Correlation between Monthly*tenure Vs. Total Charges:",pearsonr(ch.tenure*ch.MonthlyCharges,ch.TotalCharges))


# In[ ]:


ch['Temp'] = ch.tenure*ch.MonthlyCharges
lm = ols('TotalCharges ~ Temp',ch).fit()
lm.summary()


# TotalCharges should be removed, as it is just monthly charges accumulated till date, as shown by the regression model above.

# In[ ]:


ch.drop(['Temp'],axis=1,inplace=True)


# In[ ]:


_ = pd.crosstab(ch.Contract,ch.Churn).plot.bar()
_ = plt.title('Churn Count for Contract')
_ = plt.ylabel('Churn/No Churn Counts')
print('Mean Churn Across',ch.groupby('Contract')['Ch10'].mean())
_ = pd.crosstab(ch.PhoneService,ch.Churn).plot.bar(color='cb')
_ = plt.title('Churn Count for Phone Service')
_ = plt.ylabel('Churn/No Churn Counts')
_ = pd.crosstab(ch.InternetService,ch.Churn).plot.bar(color='mr')
_ = plt.title('Churn Count for Internet Service')
_ = plt.ylabel('Churn/No Churn Counts')
print('Mean Churn Across',ch.groupby('PhoneService')['Ch10'].mean())
print('Mean Churn Across',ch.groupby('InternetService')['Ch10'].mean())


# **Remark**: The Mean Rates printed above the plots only give some idea on the churn rate and that can give skewed perception on importance of set of people in a category.
# 
# The plots serve two purposes. They give counts and the relative counts of Churn Vs No churn. Across the categories, they tell us, which category has more members. This is helpful in deciding a few things. Let me explain with an example of Contract.
# 
# The month to month contract has high number of people in general and higher proportion of people Churning. The Contract with 2 years has the least members across categories and also within the category, relatively smaller proportion of the people Churn.
# 
# Similar Statements can be made about the other two plots. However, they do not give us an integrated picture of which combinations lead to certain customers' dissatisfaction. Hence we need a model to derive the relationship across them.

# In[ ]:


# Phone service is redundant.
print("Multiple Lines category counts:\n",ch.MultipleLines.value_counts())
print("Phone Lines category counts:\n",ch.PhoneService.value_counts())


# **Remark:**From the above, Phone service is a subset of multiple lines. It does not contain any new information. Hence we choose to omit it. 
# 
# We can not rule out the possibility that the digits in customerID has some kind of information encoded. It can be taken up as future work. At this time, for ease of analysis, let us drop it.

# In[ ]:


y = ch.Ch10
X = ch.drop(['customerID','Churn','Ch10','TotalCharges','PhoneService','Tenure Group','revloss'],axis=1,inplace=False).copy()
temp = ch[['tenure','MonthlyCharges','SeniorCitizen']]
X.drop(['tenure','MonthlyCharges','SeniorCitizen'],axis=1,inplace=True)
X = X.apply(lambda x: x.astype('category')).apply(lambda x: x.cat.codes)
X[['tenure','MonthlyCharges','SeniorCitizen']] = temp
X1 = X.copy() # Saving a copy


# In[ ]:


# We will reduce all features to 2D by PCA.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  # Essential to see the effect of all
X = sc.fit_transform(X)
fig = plt.figure(figsize=(12,6))
pca = PCA()
xx = pca.fit_transform(X)
xs = xx[:,0]
ys = xx[:,1]
fig.add_subplot(1,2,1)
_ = plt.scatter(xs,ys,c=y)
_ = plt.title('PCA analysis result by removing Total Charges')
_ = plt.xlabel("PCA x component")
_ = plt.ylabel("PCA y component")
fig.add_subplot(1,2,2)
_ = plt.bar(np.arange(pca.n_components_),100*np.round(pca.explained_variance_ratio_,4),color='m')
_ = plt.xlabel("PCA Feature number")
_ = plt.ylabel("PCA Variance % ")
_ = plt.title('Variance using PCA')
print("Percentage Variance by removing TotalCharges:",100*np.round(pca.explained_variance_ratio_,4))


# **Remark**: The PCA indicates that there are two dominant components which explain all the variance! My guess would be tenure and MonthlyCharges. But Monthly charges may be highly correlated with services, hence we will remove that and try again.

# In[ ]:


X1.drop(['MonthlyCharges'],axis=1,inplace=True)
fig = plt.figure(figsize=(12,6))
sc = StandardScaler()
X1 = sc.fit_transform(X1)
xx = pca.fit_transform(X1)
xs = xx[:,0]
ys = xx[:,1]
fig.add_subplot(1,2,1)
_ = plt.scatter(xs,ys,c=y)
_ = plt.title('PCA analysis result by dropping monthly charges')
_ = plt.xlabel("PCA x component")
_ = plt.ylabel("PCA y component")
fig.add_subplot(1,2,2)
_ = plt.bar(np.arange(pca.n_components_),100*np.round(pca.explained_variance_ratio_,4),color='m')
_ = plt.xlabel("PCA Feature number")
_ = plt.ylabel("PCA Variance %")
_ = plt.title('Variance using PCA')
print("Percentage Variance by tenure and monthly charges:",100*np.round(pca.explained_variance_ratio_,6))


# **Remarks: ** 
# Separability seems to have reduced, when monthly charges is removed.

# ### K-Means Clustering:
# 
# Clustering is done on tenure and Monthly Charges to get intuitive feel for the customer groups behavior. Churn population and non-Churn population are separately clustered into 3 and compared.

# In[ ]:


# Import KMeans Model
from sklearn.cluster import KMeans

# Graph and create 3 clusters of Customer Churn
kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(ch[ch.Churn=='Yes'][["tenure","MonthlyCharges"]])

kmeans_colors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_]
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplot(2,1,1) #figsize=(10, 6))
plt.scatter(x="tenure",y="MonthlyCharges", data=ch[ch.Churn=='Yes'],
            alpha=0.25,color = kmeans_colors)
plt.xlabel("Tenure in months ")
plt.ylabel("Monthly Charges in Dollars")
plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)
plt.title("Clusters of Customers who switch")
print("Cluster Centers for loyal customers are at:")
print("Month, Dollars, Numbers")
print(np.round(kmeans.cluster_centers_[0,:],2),(kmeans.labels_==0).sum())
print(np.round(kmeans.cluster_centers_[2,:],2),(kmeans.labels_==2).sum())
print(np.round(kmeans.cluster_centers_[1,:],2),(kmeans.labels_==1).sum())

plt.subplot(2,1,2)

# Graph and create 3 clusters of Customer Churn
kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(ch[ch.Churn=='No'][["tenure","MonthlyCharges"]])

kmeans_colors = ['darkgreen' if c == 0 else 'orange' if c == 2 else 'purple' for c in kmeans.labels_]

plt.scatter(x="tenure",y="MonthlyCharges", data=ch[ch.Churn=='No'],
            alpha=0.25,color = kmeans_colors)
plt.xlabel("Tenure in months ")
plt.ylabel("Monthly Charges in Dollars")
plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)
plt.title("Clusters of Loyal customers")
print("Cluster Centers for loyal customers are at:")
print("Month  Dollars  Numbers")
print(np.round(kmeans.cluster_centers_[1,:],2),(kmeans.labels_==1).sum())
print(np.round(kmeans.cluster_centers_[0,:],2),(kmeans.labels_==0).sum())
print(np.round(kmeans.cluster_centers_[2,:],2),(kmeans.labels_==2).sum())
_ = plt.tight_layout()


# + In the churn group (first plot), blue cluster has the maximum count. The number is more than sum of other two clusters. Indicates that relatively new customers but those who have subscribed to more services are the ones who are more likely to leave.

# #### **Preparing the data for Predictive Analysis, by one-hot encoding**
# Creating one hot encoding to suit Linear/Logistic regression is important for the correct interpretation of the features by the model. We do that below, and then drop original columns and the last column of each variable, as it is correlated to the others.

# In[ ]:


X = ch.drop(['customerID','Churn','Ch10','TotalCharges','PhoneService','Tenure Group','revloss'],axis=1,inplace=False).copy()


# In[ ]:


cat_vars=['gender','Partner','Dependents','PaperlessBilling','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(ch[var], prefix=var)
    X1=X.join(cat_list)
    X=X1
X.drop(cat_vars,axis=1,inplace=True) # Originals need to be dropped


# In[ ]:


X.columns


# In[ ]:


X.drop(['MultipleLines_NoPh','InternetService_No','OnlineSecurity_No','OnlineBackup_No',
        'DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No',        
        'gender_Male','Partner_No','Dependents_No','PaperlessBilling_No',
        'Contract_Month-to-month','PaymentMethod_Credit card (automatic)'],axis=1,inplace=True)
X.drop(['StreamingMovies_NoInt','StreamingTV_NoInt','TechSupport_NoInt','DeviceProtection_NoInt','OnlineBackup_NoInt','OnlineSecurity_NoInt'],axis=1,inplace=True)
XLin = X[[ 'MultipleLines_No', 'MultipleLines_Yes','InternetService_Fiber optic', 'InternetService_DSL',
         'OnlineSecurity_Yes','OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes','StreamingTV_Yes', 'StreamingMovies_Yes']]


# In the Real Scenario, Data Scientist can ask questions to the business entity on how the customers are charged. Here we need to find out the relation between services and monthly Charges. We could try fitting a Linear Regression Curve. The customers are assumed to be independent. There are enough number of samples. We can plot the residuals later and check for normality. If it is found to be normal, with number of outliers(points beyond 2 stds) < 5% and if it explains 99% of the variance (R2 measure), we will accept it.
# 
# Our objective at this point is not prediction. It is to evaluate the coefficients of Linear regression, which tell us how much each service costs. This can later be used to quantify the gain in monthly revenue if some actions are taken which would result in retaining a customer.
# 
# Since it is not a predictive Linear Regression, but an inference one, we need not split it into Test and train set. There is no need to standardize any variable too.
# When the first model was run, intercept came out to be < 10 cents. Hence ran this with intercept False

# In[ ]:


# Fit Linear Regression for Monthly Charges using services
from sklearn.linear_model import LinearRegression
LinReg = LinearRegression(fit_intercept=False)
yLin = ch.MonthlyCharges
LinReg.fit(XLin,yLin)
pred = LinReg.predict(XLin)
print("R^2 of the fit:",np.round(LinReg.score(XLin,yLin),3))
print("MSE of the model {:.2f}".format(np.mean((pred - yLin) ** 2)))
lincoeff = pd.DataFrame(np.round(LinReg.coef_,3),index=XLin.columns,columns=['$ Per month'])
lincoeff.sort_values('$ Per month',ascending=False).plot.bar(color='orange')
lincoeff.sort_values('$ Per month',ascending=False)


# **Remarks:**
# + The Model is a Good fit as shown by R^2 and MSE
# + The monthly charges are proportional to services taken, no fixed charges.  
# + Fiber optic internet is the most expensive service at \$50, the double of DSL. 
# + Streaming Sevices are priced around \$9.94, Other internet related services are around \$
# + A single phone line costs \$20, an additional line costs \$5
# + We will next examine residuals just to make sure the assumptions hold true and hence results are reliable

# In[ ]:


# Checking for verification of normality
resid = pred-yLin
_ = sm.qqplot(resid,line='r')
_ = plt.title('Quantile Plot')
_ = plt.figure()
_ = sns.jointplot(pred,resid,color='r')
_ = plt.title('Residual Plot')
print("Percentage of outliers:{:.2f}".format(100*((abs(resid) > 2.25).sum())*resid.std()/XLin.shape[0]))
#(abs(resid) > resid.std()unt()
#print("Indices of outlier points:",list(np.argsort(abs(pred-ydev)).tail(10)))


# + The quantile Plot is nearly linear implying normality.
# + Normal distribution is further confirmed by the histogram to the right of the second plot. (The residue Vs Monthly Charges)
# + Correlation of monthly charges with residual is very small. Indicates good fit
# + As a result, we include all other variables and drop MonthlyCharges from next predictive model

# In[ ]:


all_cust = round((np.sum(XLin,axis=0)*LinReg.coef_),2)
index1 = y > 0
churn_cust = round((np.sum(XLin.loc[index1,:],axis=0)*LinReg.coef_),2)
joined = pd.concat([all_cust,churn_cust],axis=1)
joined.columns = ['All Customers','Churn Customers']
joined.plot.bar(width = 0.9)
_ = plt.title('Monthly Income Bar Chart across services')
_ = plt.ylabel('Monthly Income in Dollars')


# In[ ]:


# We need to drop a few dummies to prevent correlations, in nonservice specific ones.
# Plotting correlation for top 10 features
# Ref : https://matplotlib.org/examples/color/colormaps_reference.html for colormap
X.drop('MonthlyCharges',axis=1,inplace=True)
_ = plt.figure(figsize=(16,12))
mask = np.zeros_like(X.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(X.corr(),mask=mask,cmap='bwr')


# Wow! That is good, except for a couple of strong squares. These will be okay.

# ### 4) Training and Testing Predictive Models
#  
# The goals of Predictive Model fitting are,
# + To identify the customers with high probability of switching to competition
# + To identify the major causes behind the tendency to leave (feature importance)
# + To make predictions and assess the capacity of the model for future data (Predictive power)
# + Use all the results and make recommendations to the telecom company.
# 
# #### Metrics to assess the model:
# We are interested in customers who have tendency to go away (i.e. label 1). We may not mind a few false alarms especially if the measures taken to retain them is relatively inexpensive compared to the loss due to missing the true alarms. The measures taken on the customers who had no intention to leave can result in improved customer satisfaction and hence long term benefits.
# 
# We want high recall on class 1. 

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


# In[ ]:


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# clf - original classifier
# parameters - grid to search over
# X - usually your training X matrix
# y - usually your training y 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
from sklearn.model_selection import GridSearchCV

def cv_optimize(clf, parameters, X, y, n_jobs=2, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
    gs.fit(X, y)
    print("BEST", gs.best_params_, gs.best_score_)
    #print(gs.grid_scores_)
    best = gs.best_estimator_
    return best
#------------------------------------------------------------------------------#
# Function to plot ROC and find area under ROC                                 #
#------------------------------------------------------------------------------#
def find_auc_score(clf,Xin,yin,color='b',name='LogReg',label=1,prob=1) :
    '''Function to plot Receiver characteristics and find AUC'''
    if prob == 1:
        yscore = clf.predict_proba(Xin)
    else :
        yscore = clf.decision_function(Xin)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(yin, yscore[:,label],pos_label=label)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate,color ,label='AUC '+name+' = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return roc_auc


def pre_process_Xy(Xarray,yarray,test_tr_split_size=0.4) :
    '''Function to split given data into test and (train, dev) set'''
    Xtr,Xdev,ytr,ydev = train_test_split(Xarray,yarray,test_size=test_tr_split_size,random_state=42,stratify=yarray)
    return Xtr,Xdev,ytr,ydev
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Important parameters
# indf - Input dataframe
# featurenames - vector of names of predictors
# targetname - name of column you want to predict (e.g. 0 or 1, 'M' or 'F', 
#              'yes' or 'no')
# target1val - particular value you want to have as a 1 in the target
# mask - boolean vector indicating test set (~mask is training set)
# reuse_split - dictionary that contains traning and testing dataframes 
#              (we'll use this to test different classifiers on the same 
#              test-train splits)
# score_func - we've used the accuracy as a way of scoring algorithms but 
#              this can be more general later on
# n_folds - Number of folds for cross validation ()
# n_jobs - used for parallelization
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def plot_train_test_error(clf,X,y,N=50):
    '''This function plots Train and Test Accuracy for different lengths'''

    training_error = np.empty([N,1])
    dev_error = np.empty([N,1])
    len_tr = int(X.shape[0]/N)
    re_ind = np.random.permutation(X.index)
    X = X.reindex(re_ind)
    y = y.reindex(re_ind)
    for i in range(N) :
        X1 = X[:(i+1)*len_tr]
        y1 = y[:(i+1)*len_tr]
        Xtr,Xte,ytr,yte = train_test_split(X1,y1,test_size=0.5,random_state=42,stratify=y1)
        clf = clf.fit(Xtr, ytr)
        training_error[i,0] = 1 - clf.score(Xtr, ytr)
        dev_error[i,0] = 1 - clf.score(Xte, yte)
    
    plt.plot(np.arange(N)*len_tr,training_error.reshape(np.arange(N).shape),label='train error')
    plt.plot(np.arange(N)*len_tr,dev_error.reshape(np.arange(N).shape),color='m',label='test error')
    plt.title('Train Error and Test Error Vs Number of Samples used (train: test 1:1 ratio)')
    plt.ylabel('Error rate')
    plt.xlabel('Number of samples')
    plt.legend(loc='best')
    return
    
def do_classify(clf, parameters, Xtr,ytr,Xdev,ydev, score_func=None, n_folds=5, n_jobs=2,model_name='LogReg',label=1,prob_dec=1):

    if parameters:
        clf = cv_optimize(clf, parameters, Xtr, ytr, n_jobs=n_jobs, n_folds=n_folds, score_func=score_func)
    clf=clf.fit(Xtr, ytr)
    training_accuracy = clf.score(Xtr, ytr)
    test_accuracy = clf.score(Xdev, ydev)
    print("############# based on standard predict ################")
    print("Accuracy on training data: %0.2f" % (100*training_accuracy)+'%')
    print("Accuracy on test data:     %0.2f" % (100*test_accuracy)+'%')
    print("confusion_matrix on dev data")
    ypred =  clf.predict(Xdev)
    print(confusion_matrix(ydev,ypred))
    print("classification report on dev data")
    print(classification_report(ydev,ypred))
    print("########################################################")
  #  multi_auc_roc(clf,Xdev,ydev,prob=1)
    auc_tr = find_auc_score(clf,Xtr,ytr,color='g',name=model_name+'_tr',label=label,prob=prob_dec) 
    auc_dev = find_auc_score(clf,Xdev,ydev,color='orange',name=model_name+'_dev',label=label,prob=prob_dec) 
    return clf,auc_tr,auc_dev


# ### Logistc Regression Classifier
# 
# There are two numerical variables, and they don't need standardization. Result in terms of performance is found to be same with and without. Hence not doing.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Keep a copy to access columns\nXcpy = X.copy()\nX[\'tenure\'] = X[\'tenure\'].transform(lambda x: (x - x.mean()) / x.std())\nXtrain, Xdev, ytrain,ydev = train_test_split(X,y,test_size=0.4,stratify=y)\n# This is commented because hyperparameter tuning is not done currently.\n#Xdev, Xtest, ydev,ytest = train_test_split(Xt,yt,test_size=0.5,random_state=42,stratify=yt)\nparameters = {"C": [0.1,1,10,100,10000],"class_weight":[\'balanced\',None]}\nlogreg,aucrf1,aucrf2 = do_classify(LogisticRegression(), parameters, Xtrain,ytrain,Xdev,ydev, score_func=\'recall\', n_folds=5, n_jobs=2,label=1,prob_dec=1)')


# **Remarks:** The model is having similar accuracy on dev and train set. Since the class weight is set to balanced, the model has adjusted for imbalance by sampling methods. In logistic regression model, one can not do better than this, once the training and dev set accuracy is equal. Ref: The book, Introduction to Statistical Learning, Chapter 2

# In[ ]:


coeff=logreg.coef_
intercept = logreg.intercept_
coeffs_b= logreg.coef_[0,np.argsort(abs(logreg.coef_[0,:]))[::-1]]
names_b = list(Xcpy.columns[np.argsort(abs(logreg.coef_[0,:]))[::-1]])
logfimp = pd.DataFrame(np.round(coeffs_b,3),index=names_b,columns=['Coeff value'])
_ = logfimp.head(10).plot.bar(color='purple')
_ = plt.title('Feature Importance (Log Reg)')
_ = plt.ylabel('Coefficient value')
_ = plt.xlabel('Features')
logfimp


# **Interpretation:** The negative coefficients indicate that customers with higher magnitude for negative coefficient tend to remain loyal and higher for positive coefficients indicate the opposite. 

# **Scikit-learn's RFE**: Scikit Learn has a nice package called Recursive feature elimination (RFE). Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of RFE is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

# In[ ]:


# Trying Feature Selection by limiting to 10
from sklearn.feature_selection import RFE
model = LogisticRegression(class_weight='balanced')
rfe = RFE(model, 10)
rfe = rfe.fit(Xtrain, ytrain)
# After RFE has chosen, now do a prediction using that
print("Chosen Predictors:",Xcpy.columns[rfe.support_])
Xp = Xcpy.loc[:,Xcpy.columns[rfe.support_]]
Xp = sc.fit_transform(Xp)
Xtrain1, Xt, ytrain1,yt = train_test_split(Xp,y,test_size=0.4,stratify=y)
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(Xtrain1,ytrain1)
yp = logreg.predict(Xt)
print("Report:\n",classification_report(yt,yp))
print("Dev Set Accuracy %",np.round(accuracy_score(yt,yp)*100,2))
print("Train set Accuracy %",np.round(accuracy_score(ytrain1,logreg.predict(Xtrain1))*100,2))
yprob = logreg.predict_proba(Xt)
false_positive_rate, true_positive_rate, thresholds = roc_curve(yt, yprob[:,1],pos_label=1)
roc_auc = auc(false_positive_rate, true_positive_rate)
_ = plt.title('Receiver Operating Characteristic')
_ = plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
_ = plt.legend(loc='lower right')
_ = plt.plot([0,1],[0,1],'r--')
_ = plt.xlim([-0.01,1.01])
_ = plt.ylim([-0.01,1.01])
_ = plt.ylabel('True Positive Rate')
_ = plt.xlabel('False Positive Rate')


# **Remarks**: Performance with RFE is slightly degraded. That is expected as only 10 predictors are used as opposed to double of that earlier.

# In[ ]:


# Coefficients with LogReg
Xp = Xcpy.loc[:,Xcpy.columns[rfe.support_]]
coeffs = logreg.coef_[0,np.argsort(abs(logreg.coef_[0,:]))[::-1]]
names = list(Xp.columns[np.argsort(abs(logreg.coef_[0,:]))[::-1]])
print("Coefficients and their values in decreasing importance")
pd.DataFrame(np.round(coeffs,2),index=names,columns=['Coeff value'])


# **Remarks**: The feature importance is not really matching with and without RFE. RFE is an overkill for problems with small number of features.

# In[ ]:


# To evaluate the extent of relation between churn and tenure, trying Logistic with tenure alone.
# Recognizing that tenure and churn are just correlated, it may not be causation
# We could verify if the coefficient with this is close to that obtained by the first 
# logistic regression that includes all variables
logreg_red = LogisticRegression(class_weight='balanced')
Xtrain, Xdev, ytrain,ydev = train_test_split(np.array(ch['tenure']),y,test_size=0.4,random_state=42,stratify=y)
logreg_red.fit(Xtrain.reshape(-1,1),ytrain)
ypred_red = logreg_red.predict_proba(Xdev.reshape(-1,1))
_ = plt.plot(np.sort(ypred_red[:,1]),label = 'Probability values')
ypred = logreg_red.predict(Xdev.reshape(-1,1))
ypred_s = ypred[np.argsort(ypred_red[:,1])]
spred = np.sort(ypred_red[:,1])
vline = spred[ypred_s.argmax()]
print("Threshold Chosen for classification:",round(vline,2))
print("Threshold Tenure:{} months".format(round(0.997/0.037)))
print("Max and Min Prob values:{} and {}".format(round(ypred_red.max(),2),round(ypred_red.min(),2)))
_ = plt.axhline(vline,color='k',linestyle='--',label = 'Threshold')
_ = plt.scatter(np.arange(len(ypred)),ypred_s,color='m',marker='.',label = 'Predictions')
_ = plt.legend(loc='best')
_ = plt.xlabel('Test sample index')
_ = plt.ylabel('Probability values')
_ = plt.title('Probability Plot of Churn')
print("Train Set Accuracy :{:.2f}%".format(100*accuracy_score(ytrain,logreg_red.predict(Xtrain.reshape(-1,1)))))
print("Dev Set Accuracy {:.2f}%".format(100*accuracy_score(ydev,ypred)))
print("Report:\n",classification_report(ydev,ypred))
print("Coefficient:{}, Intercept:{}".format(round(logreg_red.coef_[0,0],3),round(logreg_red.intercept_[0],3)))


# Even with tenure alone it gives about 64% overall accuracy, a mere 11% reduction compared to all features taken for the fitting.
# + The coefficient value -0.037 is close to -0.030 obtained with the regression with all other variables included. This gives additional confidence.
# + Churn is closely negatively correled with tenure.
# + The curve ranges between 0.16 to 0.84, not between 0 to 1
# + The threshold of 0.5 corresponds to tenure = 27 months => contract of two year is preferred.
# + It should be noted that the sigmoid curve is almost linear between 0.66 and 0.23, corresponding to tenures 9 months and 60 months respectively. 
# + Before 9 months the curve is flat, implying, it takes lot of effort to make them subscribed for longer than a few months.
# + After 60 months it is again flat at the other end, implying these customers have stabilized
# 

# #### Data Preparation
# The remaining part does not need one hot encoding, and also categories need to be handled again.

# In[ ]:


X_rf = X = ch.drop(['customerID','Churn','Ch10','TotalCharges','PhoneService','Tenure Group','revloss'],axis=1,inplace=False).copy()
temp = X_rf[['tenure','MonthlyCharges','SeniorCitizen']]
X_rf = X_rf.drop(['tenure','MonthlyCharges','SeniorCitizen'],axis=1)
X_rf = X_rf.apply(lambda x: x.astype('category')).apply(lambda x: x.cat.codes)
X_rf[['tenure','MonthlyCharges','SeniorCitizen']] = temp
Xtrain, Xdev, ytrain,ydev = train_test_split(X_rf,y,test_size=0.4,stratify=y)


# ### Decision Tree Classifier
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {"max_depth": [3,4,6,8,12], \'min_samples_leaf\': [1,2,4,8],"class_weight":[\'balanced\',None]}\ntr,aucrf1,aucrf2 = do_classify(DecisionTreeClassifier(), parameters, Xtrain,ytrain,Xdev,ydev, score_func=\'recall\', n_folds=5, n_jobs=2,model_name=\'DecTree\',label=1,prob_dec=1)')


# In[ ]:


import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(tr, out_file='dtree.dot', 
                         feature_names=X_rf.columns,  
                         class_names=['N','Y'],  
                         filled=True, rounded=True,  
                         special_characters=True)
#graph = graphviz.Source(dot_data)


# ### Random Forest Classifier
# There is actually no need to drop correlated features for Random Forest. It selects best features at every node of every tree by itself. But from a business perspective, TotalCharges is related to monthly and tenure, we intentionally drop it to get the correct picture (better interpretation) on feature importance. Phone service is also dropped.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'parameters = {"max_depth": [3,4,6,8,12,None], \'min_samples_leaf\': [1,2,4,6],"n_estimators":[10,50,100,200],"class_weight":[\'balanced\',None]}\nrf,aucrf1,aucrf2 = do_classify(RandomForestClassifier(), parameters, Xtrain,ytrain,Xdev,ydev, score_func=\'recall\', n_folds=5, n_jobs=2,model_name=\'RandomForest\',label=1,prob_dec=1)')


# In[ ]:


feature_labels = np.array(list(X_rf.columns))
(pd.Series(rf.feature_importances_,index=feature_labels).sort_values(ascending=True)/np.max(rf.feature_importances_)).plot.barh(color='purple',width=0.9)
_ = plt.title('Normalized Feature Importance From Random Forest Classifier')
_ = plt.axvline(0.05,linestyle='--',color='olive')
_ = plt.text(0.05,7,'5% of the max',rotation=87,fontsize=16)
pd.DataFrame(rf.feature_importances_,index=feature_labels,columns=['Feature importance']).sort_values('Feature importance',ascending=False)


# ### 5) Comparing The Results, Feature Importance: 
# Logistic Regression is more interpretable, faster, gives better overall accuracy than Random Forest, with slight degradation in recall of Churn group.

# In[ ]:


# Predictive Model to use
# Assumes clean csv file
def cust_churn_prob_finder(coeff,intercept,csvfile):
    df = pd.read_csv(csvfile,usecols=range(20))
    df_monthly = df.MonthlyCharges
    df.drop(['TotalCharges','customerID','PhoneService','MonthlyCharges'],axis=1,inplace=True)
    cat_vars=['gender','Partner','Dependents','PaperlessBilling','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(ch[var], prefix=var)
        X1=df.join(cat_list)
        df=X1
    df.drop(cat_vars,axis=1,inplace=True) # Originals need to be dropped
    df.drop(['MultipleLines_NoPh','InternetService_No','OnlineSecurity_No','OnlineBackup_No',
        'DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No',        
        'gender_Male','Partner_No','Dependents_No','PaperlessBilling_No',
        'Contract_Month-to-month','PaymentMethod_Credit card (automatic)'],axis=1,inplace=True)
    df.drop(['StreamingMovies_NoInt','StreamingTV_NoInt','TechSupport_NoInt','DeviceProtection_NoInt','OnlineBackup_NoInt','OnlineSecurity_NoInt'],axis=1,inplace=True)
    df['tenure'] = df['tenure'].transform(lambda x: (x- x.mean())/x.std())
    prob_test = 1/ (1+ np.exp(-np.dot(np.array(df),coeff[0,:])-intercept))
    df['churn_prob'] = prob_test 
    df['charge*prob'] = df_monthly* prob_test 
    return df


# In[ ]:


df = cust_churn_prob_finder(coeff,intercept,'../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Report",classification_report(y,(df['churn_prob'] >= 0.5)))
print("Confusions\n",confusion_matrix(y,(df['churn_prob'] >= 0.5)))


# In[ ]:


df_churn = df[df['churn_prob'] >= 0.5]
index1 = df_churn.sort_values('charge*prob',ascending=False).index
temp = y*(df['churn_prob'] >= 0.5)
index2 = ch.loc[(temp > 0),'MonthlyCharges'].index
print("Potential revenue savings:${}".format(round(ch.loc[index2,'MonthlyCharges'].head(1485).sum())))


# ### 6) Recommendations to retain the customers:
# A predictive model is given that ranks customers based on their probability of churn and the revenue that they bring.
# + Use this model to prioritize whose concerns to be addressed first. Sometimes it might be case by case basis.
# + Take the following actions immediately: 
#   - Try striking a longer contract with new customers: two year or one year in that order of preference.
#   - Leverage the time to improve the quality of services, on the high cost ones like Fiber optic.
#   - Improve on the Technical support on all services like streaming, phone connection and internet. 
# Be up-to-date with current technology.
#   - Collect customer feedback and act on it immediately to prevent new customer churn
# + Next: It will be helpful to understand why churn started 5.5 years ago. Give more historical data to the data scientist for analysis.
# 

# ### 7) Scope for future work:
# + More predictive models could be tried, however, there is no guarantee of better accuracy, as we have seen similar accuracy witn logistic regression and random forest. This actually means most of the variance in the data is explained.
# + One could collect more data through surveys, analyze them using NLP techniques and take more measures.
# + There is a scope to collect historical data on company customers over a few decades, and fight out clear reason for customer drop happened 70 years ago.

# ### 8) A Final Note:
# 
# It was fun doing this small project. Made me understand what more I need to work on. In the process I learnt,
# + Data Science Process
# + To Apply Machine Learning basics studied in courses and text books. 
# + Metrics to measure the goodness of the model. Learnt that accuracy is definitely not a good enough measure in case of imbalanced classes. class_weight argument works magically by penalizing the mistakes in minority class more!
# 
# Thanks for taking time to go through this project. Feedback is highly appreciated. Please email [me](mailto:aparnack@gmail.com) to discuss your views

# ### 9) References:
# 
# + Introduction to Statistical Learning Book by Gareth James et. al
# + Machine Learning Course by Andrew Ng in Coursera
# + How to plot ROC is [here](https://datamize.wordpress.com/2015/01/24/how-to-plot-a-roc-curve-in-scikit-learn/)
# + Colormap codes for heatmap [here](https://matplotlib.org/examples/color/colormaps_reference.html for colormap)
# + [Here](https://towardsdatascience.com/predict-customer-churn-with-r-9e62357d47b4) is another implementation of this problem by Susan Li in Towards Data Science blog

# In[ ]:




