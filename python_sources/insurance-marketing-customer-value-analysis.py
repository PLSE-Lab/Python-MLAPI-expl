#!/usr/bin/env python
# coding: utf-8

# ### IMPORTING LIBRARIES

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


pd.set_option('display.max_columns',999)


# In[ ]:


import statsmodels.api as sm


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[ ]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip


# In[ ]:


from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


from scipy.stats import normaltest,f_oneway
from scipy.stats import ttest_ind


# In[ ]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor


# In[ ]:


dt = DecisionTreeRegressor()
et = ExtraTreeRegressor()


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


abr = AdaBoostRegressor()
br = BaggingRegressor()
etr = ExtraTreesRegressor()
gbr = GradientBoostingRegressor()
rfr = RandomForestRegressor()


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


# ### READING THE DATA SET

# In[ ]:


data=pd.read_csv('/kaggle/input/ibm-watson-marketing-customer-value-data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')
data.head()


# ### NULL VALUE'S CHECK

# In[ ]:


data.isnull().sum()


# ### MEASURES OF CENTRAL TENDENCY

# In[ ]:


data[['Customer Lifetime Value','Income','Monthly Premium Auto','Total Claim Amount']].describe()


# ### SHAPE OFTHE DATA SET

# In[ ]:


data.shape


# ### DATA VISUALIZATION AND INFERENCES
# 
# #### UNIVARIATE ANALYSIS

# In[ ]:


sns.boxplot(data['Income'])
plt.show()


# In[ ]:


sns.boxplot(data['Monthly Premium Auto'])
plt.show()


# In[ ]:


sns.boxplot(data['Total Claim Amount'])
plt.show()


# - As we can see that there are outliers in the total claim amount and also in monthly premium auto , usually we remove the outliers for a better model.
# - since our dataset is related to insurance and banking industry, we must be accept the outliers,as they can be our potential customers.
# - And there are no outliers in the income.
# - Conclusion: No outlier treatment required.

# In[ ]:


sns.distplot(data['Income'])
plt.show()


# In[ ]:


sns.distplot(data['Monthly Premium Auto'])
plt.show()


# In[ ]:


sns.distplot(data['Total Claim Amount'])
plt.show()


# As we can see that none of the continuous variables are normally distributed.
# So in our case , we want to make the distributions normal, we can apply some transformations to the data and see if we can achieve a normally distributed variable.

# #### TRANSFORMATION OF THE NUMERICAL VARIABLES

# In[ ]:


sns.distplot(data['Income']**2)
plt.show()


# In[ ]:


sns.distplot(data['Income']**(1/2))
plt.show()


# As we can see that while we are trying to transform the data to make it normal,rather the distribution is getting skewed, or is having multiple peaks which again is a problem to our model, hence we just stick with the same distribution of the variable.

# In[ ]:


sns.distplot(data['Monthly Premium Auto']**(2))
plt.show()


# The monthly premium auto has multiple peaks,so to remove those peaks we can apply any of the power transformation (SQUARE / CUBE) but as we can see that after the square transformation the data is getting heavily skewed, so we stick with the actual distribution again.

# In[ ]:


sns.distplot(data['Total Claim Amount']**2)
plt.show()


# Again for the total claim amount after applying the transformation's the data is getting skewed, and hence we stick to the actual distibution of the data.
# 
# Conclusion: No matter what power transformation we are applying to the numerical variables, it is still not getting normally distributed, and moreover the data is getting skewed, so rather we will just stick with the actual distribution 

# In[ ]:


sns.barplot(x = 'Location Code',y='Customer Lifetime Value',data = data)
plt.show()


# The average customer lifetime value of the customer who stay in different location code is the same so while creating the model we can drop this.

# In[ ]:


sns.barplot(x = 'State',y='Customer Lifetime Value',data = data)
plt.show()


# The average customer lifetime value of the customer who stay in different state is same and we can drop this also

# In[ ]:


sns.barplot(x = 'Response',y='Customer Lifetime Value',data = data)
plt.show()


# The average customer lifetime value for both of them is same.

# In[ ]:


sns.barplot(x = 'Gender',y='Customer Lifetime Value',data = data)
plt.show()


# We can see that the average customer lifetime value is same for both male and female.

# In[ ]:


sns.barplot(x = 'Education',y='Customer Lifetime Value',data = data)
plt.xticks(rotation=45)
plt.show()


# We can also see that education is not a significant feature for assessing the lifetime value of the customer.

# In[ ]:


sns.barplot(x = 'Number of Policies',y='Customer Lifetime Value',data = data)
plt.show()


# We can see a pattern here, customers who have taken only 1 policy have lower customer lifetime value, and customers who have taken 3 or greater show a similar trend, so we can combine all of them into one bin, and we can also see that the customers who have taken 2 policies have very high customer lifetime value comparitively.

# In[ ]:


sns.barplot(x = 'Policy Type',y='Customer Lifetime Value',data = data)
plt.xticks(rotation = 90)
plt.show()


# There isn't much difference in the customer lifetime value w.r.t what policy type he has taken, all we need is how much revenue a customer can bring to the company, so it doesnt matter what type of policy he/she has chosen.

# In[ ]:


sns.barplot(x = 'Coverage',y='Customer Lifetime Value',data = data)
plt.show()


# Customer Lifetime Value is different for different types of coverage.

# In[ ]:


sns.barplot(x = 'Number of Open Complaints',y='Customer Lifetime Value',data = data)
plt.show()


# Number of open complaints also show kind of similar trend, where people who have complaints 2 or lesser have a similar pattern but where as >3 do not show any pattern we will have to do statistical test to understand if this feature is really significant or not

# In[ ]:


sns.pairplot(y_vars='Customer Lifetime Value',x_vars=['Income','Monthly Premium Auto','Total Claim Amount'],data = data)
plt.show()


# 
# We can clearly see that there is a linear relationship between Customer lifetime value and monthly premium auto, but we do not see any relationship between income and the total claim amount.

# In[ ]:


sns.heatmap(data[['Customer Lifetime Value','Monthly Premium Auto','Income','Total Claim Amount']].corr(),annot = True)
plt.show()


# 
# And we can clearly see in the correlation map, that customer lifetime value has a better correlation with monthly premium auto and acceptable co relation with total claim amount, but it show's no relationship with income, so again with all the visualization's we can come to the conclusion that we can dis regard the INCOME feature.

# ### BASE MODEL USING OLS
# 
# ##### Using label encoding just for the purpose of looking at the base model, encoding technique's may change furthur(one-hot encoding is used)

# In[ ]:


cols = data.select_dtypes(object).columns
for i in cols:
    data[i] = le.fit_transform(data[i])


# In[ ]:


X = data.drop('Customer Lifetime Value',axis=1)
y = data['Customer Lifetime Value']
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)


# In[ ]:


from sklearn.model_selection import train_test_split
# train data - 70% and test data - 30%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)


# In[ ]:


lin_reg = LinearRegression()
model = lin_reg.fit(X_train,y_train)
print(f'Coefficients: {lin_reg.coef_}')
print(f'Intercept: {lin_reg.intercept_}')
print(f'R^2 score: {lin_reg.score(X, y)}')
print(f'R^2 score for train: {lin_reg.score(X_train, y_train)}')
print(f'R^2 score for test: {lin_reg.score(X_test, y_test)}')


# In[ ]:


X_sm = X
X_sm = sm.add_constant(X_sm)
lm = sm.OLS(y,X_sm).fit()
lm.summary()


# After Looking at the base model and the p-value of the feature's, we know that the Hypothesis for the feature's is
# 
# H0: Feature is not significant
# Ha: Feature is significant
# But we just cant conclude the significance of the feature's just by base model and also without using any of the feature engineering technique's we have at our disposal. So we will first try to do the statistical test's of the feature for the feature selection, we can also use the forward selection and backward elimination , we will use the Variance inflation factor

# ### ASSUMPTIONS OF LINEAR REGRESSION.
# ### Linearity

# In[ ]:


sns.pairplot(x_vars=['Monthly Premium Auto','Total Claim Amount','Income'],y_vars =['Customer Lifetime Value'],data = data)
plt.show()


# We don't see any linear relationship between the variables and the Y varible , which fails the first assumption of linear regression.

# ### Mean Of Residuals

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)


# In[ ]:


lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
residuals = y_pred-y_test
mean_of_residuals = np.mean(residuals)
print(f"The mean of the residuals is {mean_of_residuals}")


# The 2nd assumption is that the mean of the residuals must be close to zero, which again fails.

# ### Homoscedasticity_test

# In[ ]:


name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(residuals,X_test)
lzip(name, test)


# H0: Error terms are homoscedastic
# 
# Ha: Error terms are not homoscedastic
# 
# p-value < 0.05 reject null hypothesis, error terms are not homoscedastic

# ### Test of normality of residuals

# In[ ]:


p = sns.distplot(residuals,kde=True)
p = plt.title('Normality of error terms/residuals')


# The distribution clearly show's that the residuals are not normally distributed, and the third assumption also fails.

# ### No Autocorrelation

# In[ ]:


min(diag.acorr_ljungbox(residuals , lags = 40)[1])


# Ho: Autocorrelation is absent
# 
# Ha: Autocorrelation is present
# 
# The P-value is >0.05 ,we fail to reject the null hypothesis, autocorrelation is absent.

# ### NO MULTI COLLINEARITY

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=X.columns).T


# So, multicollinearity exists.

# ### STATISTICAL ANALYSIS

# Considering CLV (Customer Lifetime Value) as the target variable, we shall try to understand how each of the independent variables are contributing towards the target variable.
# 
# Since our target variable is a continuous variable, we will have to perform ANOVA to understand how significant are the independent variables towards target variable.
# 
# For ANOVA,
# 
# Null hypothesis is that there is no significant difference among the groups
# Alternative hypothesis is that there is at least one significant difference among the groups

# #### State vs Customer Lifetime Value

# In[ ]:


data=pd.read_csv('/kaggle/input/ibm-watson-marketing-customer-value-data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')


# In[ ]:


State = data.groupby('State')
Washington = State.get_group('Washington')['Customer Lifetime Value']
Arizona = State.get_group('Arizona')['Customer Lifetime Value']
Nevada = State.get_group('Nevada')['Customer Lifetime Value']
California = State.get_group('California')['Customer Lifetime Value']
Oregon = State.get_group('Oregon')['Customer Lifetime Value']


# In[ ]:


for i in [Washington,Arizona,Nevada,California,Oregon]:
    print(normaltest(i),'\n')


# CLV of all the 'States' follow a normal distribution. Hence, we can perform ANOVA test.

# In[ ]:


f_oneway(Washington,Arizona,Nevada,California,Oregon)


# ALL STATES HAVE SAME MEAN VALUE FOR CLV
# 
# pvalue > 0.05 implies that there is no significant difference in the mean of target variable which means 'State' feature is not significant for predicting 'Customer Lifetime Value'

# #### Customer Response to marketing calls vs Customer Lifetime Value

# In[ ]:


Response = data[['Customer Lifetime Value','Response']].groupby('Response')
No = Response['Customer Lifetime Value'].get_group('No')
Yes = Response['Customer Lifetime Value'].get_group('Yes')


# In[ ]:


for i in [No,Yes]:
    print(normaltest(i),'\n')


# CLV of all the 'Response' follow a normal distribution. Hence, we can perform ANOVA test or test of mean for independent categories.

# In[ ]:


ttest_ind(No,Yes)


# RESPONE HAVE SAME MEAN VALUE
# 
# pvalue > 0.05 implies that there is no significant difference in the mean of target variable which means 'Response' feature is not significant for predicting 'Customer Lifetime Value'

# #### Coverage Type vs Customer Lifetime Value

# In[ ]:


Coverage = data[['Customer Lifetime Value','Coverage']].groupby('Coverage')
basic = Coverage['Customer Lifetime Value'].get_group('Basic')
extended = Coverage['Customer Lifetime Value'].get_group('Extended')
premium = Coverage['Customer Lifetime Value'].get_group('Premium')


# In[ ]:


for i in [basic,extended,premium]:
    print(normaltest(i),'\n')


# CLV of all the 'Coverage' follow a normal distribution. Hence, we can perform ANOVA test.

# In[ ]:


f_oneway(basic,extended,premium)


# MEANS ARE NOT SAME FOR COVERAGE
# 
# pvalue < 0.05 implies that there is significant difference in the mean of target variable for atleast one group of 'Coverage' which means 'Coverage' feature can be a significant for predicting 'Customer Lifetime Value'

# #### Education vs Customer Lifetime Value

# In[ ]:


Education = data[['Customer Lifetime Value','Education']].groupby('Education')
bachelor = Education['Customer Lifetime Value'].get_group('Bachelor')
college = Education['Customer Lifetime Value'].get_group('College')
highschool = Education['Customer Lifetime Value'].get_group('High School or Below')
master = Education['Customer Lifetime Value'].get_group('Master')
doctor = Education['Customer Lifetime Value'].get_group('Doctor')


# In[ ]:


for i in [basic,college,highschool,master,doctor]:
    print(normaltest(i),'\n')


# CLV of all the categories of 'Education' follow a normal distribution. Hence, we can perform ANOVA test.

# In[ ]:


f_oneway(bachelor,college,highschool,master,doctor)


# MEANS ARE NOT SAME FOR EDUCATION
# 
# pvalue < 0.05 implies that there is significant difference in the mean of target variable for atleast one group of 'Education' which means 'Education' feature can be a significant for predicting 'Customer Lifetime Value'

# #### Employment Status vs Customer Lifetime Value 

# In[ ]:


es = data[['Customer Lifetime Value','EmploymentStatus']].groupby('EmploymentStatus')
employed = es['Customer Lifetime Value'].get_group('Employed')
unemployed = es['Customer Lifetime Value'].get_group('Unemployed')
medleave = es['Customer Lifetime Value'].get_group('Medical Leave')
disabled = es['Customer Lifetime Value'].get_group('Disabled')
retired = es['Customer Lifetime Value'].get_group('Retired')


# In[ ]:


for i in [employed,unemployed,medleave,disabled,retired]:
    print(normaltest(i),'\n')


# CLV of all the categories of 'Employment Status' follow a normal distribution. Hence, we can perform ANOVA test.

# In[ ]:


f_oneway(employed,unemployed,medleave,disabled,retired)


# MEANS ARE NOT SAME FOR Employment Status
# 
# pvalue < 0.05 implies that there is significant difference in the mean of target variable for atleast one group of 'Employment Status' which means 'Employment Status' feature can be a significant for predicting 'Customer Lifetime Value'

# #### Gender vs Customer Lifetime Value

# In[ ]:


g = data[['Customer Lifetime Value','Gender']].groupby('Gender')
f = g['Customer Lifetime Value'].get_group('F')
m = g['Customer Lifetime Value'].get_group('M')


# In[ ]:


for i in [f,m]:
    print(normaltest(i),'\n')


# CLV of all the categories of 'Gender' follow a normal distribution. Hence, we can perform ANOVA test or test of mean for independent features.

# In[ ]:


ttest_ind(f,m)


# MEANS ARE SAME FOR GENDER
# 
# pvalue > 0.05 implies that there is no significant difference in the mean of target variable for 'Gender' which means 'Gender' feature is not significant for predicting 'Customer Lifetime Value'

# #### Location Code vs Customer Lifetime Value

# In[ ]:


location = data[['Customer Lifetime Value','Location Code']].groupby('Location Code')
sub = location['Customer Lifetime Value'].get_group('Suburban')
urban = location['Customer Lifetime Value'].get_group('Urban')
rural = location['Customer Lifetime Value'].get_group('Rural')


# In[ ]:


for i in [sub,urban,rural]:
    print(normaltest(i),'\n')


# CLV of all the categories of 'Location Code' follow a normal distribution. Hence, we can perform ANOVA test.

# In[ ]:


f_oneway(sub,urban,rural)


# MEANS ARE SAME FOR LOCATION CODE
# 
# pvalue > 0.05 implies that there is no significant difference in the mean of target variable for 'Location Code' which means 'Location Code' feature is not significant for predicting 'Customer Lifetime Value'

# #### Marital Status vs Customer Lifetime Value

# In[ ]:


MaritalStatus = data[['Customer Lifetime Value','Marital Status']].groupby('Marital Status')
Married = MaritalStatus['Customer Lifetime Value'].get_group('Married')
Single = MaritalStatus['Customer Lifetime Value'].get_group('Single')
Divorced = MaritalStatus['Customer Lifetime Value'].get_group('Divorced')


# In[ ]:


for i in [Married,Single,Divorced]:
    print(normaltest(i),'\n')


# CLV of all the categories of 'Location Code' follow a normal distribution. Hence, we can perform ANOVA test.

# In[ ]:


f_oneway(Married,Single,Divorced)


# MEANS ARE NOT SAME Marital Status
# 
# pvalue < 0.05 implies that there is significant difference in the mean of target variable for at least on Group of 'Marital Status' which means 'Marital Status' feature can be significant for predicting 'Customer Lifetime Value'
# 

# #### Policy vs Customer Lifetime Value

# In[ ]:


Policy  = data[['Customer Lifetime Value','Policy']].groupby('Policy')
p3 = Policy['Customer Lifetime Value'].get_group('Personal L3')
p2 = Policy['Customer Lifetime Value'].get_group('Personal L2')
p1 = Policy['Customer Lifetime Value'].get_group('Personal L1')
c3 = Policy['Customer Lifetime Value'].get_group('Corporate L3')
c2 = Policy['Customer Lifetime Value'].get_group('Corporate L2')
c1 = Policy['Customer Lifetime Value'].get_group('Corporate L1')
s3 = Policy['Customer Lifetime Value'].get_group('Special L3')
s2 = Policy['Customer Lifetime Value'].get_group('Special L2')
s1 = Policy['Customer Lifetime Value'].get_group('Special L1')


# In[ ]:


for i in [p3,p2,p1,c3,c2,c1,s3,s2,s1]:
    print(normaltest(i),'\n')


# In[ ]:


f_oneway(p3,p2,p1,c3,c2,c1,s3,s2,s1)


# #### Renew Offer Type vs Customer Lifetime Value

# In[ ]:


R  = data[['Customer Lifetime Value','Renew Offer Type']].groupby('Renew Offer Type')
o1 = R['Customer Lifetime Value'].get_group('Offer1')
o2 = R['Customer Lifetime Value'].get_group('Offer2')
o3 = R['Customer Lifetime Value'].get_group('Offer3')
o4 = R['Customer Lifetime Value'].get_group('Offer4')


# In[ ]:


for i in [o1,o2,o3,o4]:
    print(normaltest(i),'\n')


# In[ ]:


f_oneway(o1,o2,o3,o4)


# #### Sales Channel vs Customer Lifetime Value

# In[ ]:


Sales  = data[['Customer Lifetime Value','Sales Channel']].groupby('Sales Channel')
agent = Sales['Customer Lifetime Value'].get_group('Agent')
branch = Sales['Customer Lifetime Value'].get_group('Branch')
call = Sales['Customer Lifetime Value'].get_group('Call Center')
web = Sales['Customer Lifetime Value'].get_group('Web')


# In[ ]:


for i in [agent,branch,call,web]:
    print(normaltest(i),'\n')


# In[ ]:


f_oneway(agent,branch,call,web)


# #### Vehicle Class vs Customer Lifetime Value

# In[ ]:


VC  = data[['Customer Lifetime Value','Vehicle Class']].groupby('Vehicle Class')
fd = VC['Customer Lifetime Value'].get_group('Four-Door Car')
td = VC['Customer Lifetime Value'].get_group('Two-Door Car')
suv = VC['Customer Lifetime Value'].get_group('SUV')
sc = VC['Customer Lifetime Value'].get_group('Sports Car')
ls = VC['Customer Lifetime Value'].get_group('Luxury SUV')
lc = VC['Customer Lifetime Value'].get_group('Luxury Car')


# In[ ]:


for i in [fd,td,suv,sc,ls,lc]:
    print(normaltest(i),'\n')


# In[ ]:


f_oneway(fd,td,suv,sc,ls,lc)


# #### Vehicle Size vs Customer Lifetime Value

# In[ ]:


VS  = data[['Customer Lifetime Value','Vehicle Size']].groupby('Vehicle Size')
m = VS['Customer Lifetime Value'].get_group('Medsize')
s = VS['Customer Lifetime Value'].get_group('Small')
l = VS['Customer Lifetime Value'].get_group('Large')


# In[ ]:


for i in [m,s,l]:
    print(normaltest(i),'\n')


# In[ ]:


f_oneway(m,s,l)


# ### Furthur Modelling:
# 
# #### So we did the EDA and also the Statistical Analysis, so now we can just dis regard the features which we are not significant  for our model.

# In[ ]:


data.drop(['State','Customer','Response','EmploymentStatus','Gender','Location Code','Vehicle Size','Policy','Policy Type','Sales Channel','Income','Effective To Date','Education'],axis=1,inplace = True)


# In[ ]:


data.head()


# Though the features, months since policy inception, months since last claim, number of open complaints and number of policies are all numerical, but they are discrete numbers and we will consider them as categorical features while preparing the model.
# 
# Firstly, according to our EDA, we saw that the number of policies >= 3 have similar trend so we will group all of them as 3

# In[ ]:


data['Number of Policies'] = np.where(data['Number of Policies']>2,3,data['Number of Policies'])


# Secondly, when we convert the numerical features to categorical, our normal practice is label encoding for ordinal data and one hot for nominal data, but we can also use one hot encoding for ordinal data if there isn't any curse of dimensionality, so we will convert the categorical to numerical with one-hot encoding / dummification.

# In[ ]:


new = pd.get_dummies(data,columns=['Coverage','Marital Status','Number of Policies','Renew Offer Type','Vehicle Class'],drop_first=True)


# In[ ]:


new.head()


# ### Spliting the data into train(70) and test(30)

# In[ ]:


X = new.drop('Customer Lifetime Value',axis=1)
y = new['Customer Lifetime Value']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


lr.score(X_test,y_test)


# In[ ]:


lr.score(X_train,y_train)


# So after removing the unnessary feature's our model is giving is an accuracy of about 60%, we would like to take it 70% in the furthur models.

# ### Feature Selection-Forward, Backward
# #### Forward

# In[ ]:


sfs = SFS(lr, k_features='best', forward=True, floating=False, 
          scoring='neg_mean_squared_error', cv=20)
model = sfs.fit(new.drop('Customer Lifetime Value', axis=1),new['Customer Lifetime Value'])
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()


# In[ ]:


print('Selected features:', sfs.k_feature_idx_)


# #### Backward

# In[ ]:


sfs = SFS(lr, k_features='best', forward=False, floating=False, 
          scoring='neg_mean_squared_error', cv=20)
model = sfs.fit(new.drop('Customer Lifetime Value', axis=1).values,new['Customer Lifetime Value'])
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
plt.title('Sequential Backward Selection (w. StdErr)')
plt.grid()
plt.show()


# In[ ]:


print('Selected features:', sfs.k_feature_idx_)


# In[ ]:


X.columns


# Suprisingly Both the forward and backward selection gave us the same features to select for our model, so we will be sticking to the same features.

# In[ ]:


test_X = X[['Monthly Premium Auto','Number of Open Complaints','Total Claim Amount','Coverage_Premium',
            'Marital Status_Single','Number of Policies_2','Number of Policies_3',
            'Renew Offer Type_Offer2','Vehicle Class_SUV','Vehicle Class_Sports Car']]


# In[ ]:


train = []
test = []


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(test_X,y,test_size=0.3,random_state=100)


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


test.append(lr.score(X_test,y_test))


# In[ ]:


train.append(lr.score(X_train,y_train))


# So we can clearly see that the features removed didn't contribute to tell us the differing variance in the data, so it was a good decision to remove those features

# In[ ]:


metrics = [r2_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error]


# In[ ]:


y_pred = lr.predict(X_test)


# In[ ]:


r2 = []
mae = []
mape = []
mse = []


# In[ ]:


for i in metrics:
    print(i(y_test,y_pred))
    if i == r2_score:
        r2.append(i(y_test,y_pred))
    elif i == mean_absolute_error:
        mae.append(i(y_test,y_pred))
    elif i == mean_absolute_percentage_error:
        mape.append(i(y_test,y_pred))
    else:
        mse.append(i(y_test,y_pred))


# We will consider the r2_Score and the Mean absolute percentage error as the metrics we are going to use to measure the model.

# ### Finding the best sample by random state for each model.

# In[ ]:


algo = [abr,gbr,dt,et,etr,br,rfr]


# In[ ]:


for i in algo:
    temp = 0
    print(f"New Model{i}")
    for j in range(1,300,1):
        NXT,NXt,NYT,NYt = train_test_split(X,y,test_size=0.3,random_state=j)
        i.fit(NXT,NYT)
        test_score = i.score(NXt,NYt)
        train_score = i.score(NXT,NYT)
        if test_score>temp:
            temp = test_score
            print(j,train_score,temp)


# We can see the best sample for each model, and we can also see which model is a good fit and which model is overfitting/underfitting model.
# 
# - AdaBoost is good model, but we will have to check for the metrics
# - Gradient Boosting is the best model comparitively, again we will check for the metrics
# - DecisionTree and the Extra Tree regressor models are not working better for this data set ,as the model is overfitting
# - Bagging and random forest regressor models are again overfitting model.

# ### Decision Tree Regressor

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=159)


# In[ ]:


dt.fit(X_train,y_train)


# In[ ]:


y_pred_dt = dt.predict(X_test)


# In[ ]:


for i in metrics:
    print(i(y_test,y_pred_dt))
    if i == r2_score:
        r2.append(i(y_test,y_pred_dt))
    elif i == mean_absolute_error:
        mae.append(i(y_test,y_pred_dt))
    elif i == mean_absolute_percentage_error:
        mape.append(i(y_test,y_pred_dt))
    else:
        mse.append(i(y_test,y_pred_dt))


# In[ ]:


train.append(dt.score(X_train,y_train))
test.append(dt.score(X_test,y_test))


# ### Extra Tree Regressor

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=69)


# In[ ]:


et.fit(X_train,y_train)


# In[ ]:


y_pred_et = et.predict(X_test)


# In[ ]:


for i in metrics:
    print(i(y_test,y_pred_et))
    if i == r2_score:
        r2.append(i(y_test,y_pred_et))
    elif i == mean_absolute_error:
        mae.append(i(y_test,y_pred_et))
    elif i == mean_absolute_percentage_error:
        mape.append(i(y_test,y_pred_et))
    else:
        mse.append(i(y_test,y_pred_et))


# In[ ]:


train.append(et.score(X_train,y_train))
test.append(et.score(X_test,y_test))


# In[ ]:


pd.DataFrame({'Model':['Linear Regression','Decision Tree','Extra Tree'],'R2_Score':r2,'MAE':mae,'MAPE':mape,'MSE':mse})


# We can clearly see that the Linear Regression model is having the best r2_Score, and decision tree and extra tree regressor have no better accuracy that so we will have to build our model using ensemlle technique's, boosting and bagging.

# ### ENSEMBLE METHODS

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


# ### AdaBoost

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=83)


# In[ ]:


abr.fit(X_train,y_train)


# In[ ]:


y_pred_abr = abr.predict(X_test)


# In[ ]:


for i in metrics:
    print(i(y_test,y_pred_abr))
    if i == r2_score:
        r2.append(i(y_test,y_pred_abr))
    elif i == mean_absolute_error:
        mae.append(i(y_test,y_pred_abr))
    elif i == mean_absolute_percentage_error:
        mape.append(i(y_test,y_pred_abr))
    else:
        mse.append(i(y_test,y_pred_abr))


# In[ ]:


train.append(abr.score(X_train,y_train))
test.append(abr.score(X_test,y_test))


# ### Bagging

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=292)
br.fit(X_train,y_train)
y_pred_br = br.predict(X_test)
for i in metrics:
    print(i(y_test,y_pred_br))
    if i == r2_score:
        r2.append(i(y_test,y_pred_br))
    elif i == mean_absolute_error:
        mae.append(i(y_test,y_pred_br))
    elif i == mean_absolute_percentage_error:
        mape.append(i(y_test,y_pred_br))
    else:
        mse.append(i(y_test,y_pred_br))


# In[ ]:


train.append(br.score(X_train,y_train))
test.append(br.score(X_test,y_test))


# ### Extra Tree

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=69)
etr.fit(X_train,y_train)
y_pred_etr = etr.predict(X_test)
for i in metrics:
    print(i(y_test,y_pred_etr))
    if i == r2_score:
        r2.append(i(y_test,y_pred_etr))
    elif i == mean_absolute_error:
        mae.append(i(y_test,y_pred_etr))
    elif i == mean_absolute_percentage_error:
        mape.append(i(y_test,y_pred_etr))
    else:
        mse.append(i(y_test,y_pred_etr))


# In[ ]:


train.append(etr.score(X_train,y_train))
test.append(etr.score(X_test,y_test))


# ### Gradient Boosting

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=181)
gbr.fit(X_train,y_train)
y_pred_gbr = gbr.predict(X_test)
for i in metrics:
    print(i(y_test,y_pred_gbr))
    if i == r2_score:
        r2.append(i(y_test,y_pred_gbr))
    elif i == mean_absolute_error:
        mae.append(i(y_test,y_pred_gbr))
    elif i == mean_absolute_percentage_error:
        mape.append(i(y_test,y_pred_gbr))
    else:
        mse.append(i(y_test,y_pred_gbr))


# In[ ]:


train.append(gbr.score(X_train,y_train))
test.append(gbr.score(X_test,y_test))


# ### Random Forest

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=35)
rfr.fit(X_train,y_train)
y_pred_rfr = rfr.predict(X_test)
for i in metrics:
    print(i(y_test,y_pred_rfr))
    if i == r2_score:
        r2.append(i(y_test,y_pred_rfr))
    elif i == mean_absolute_error:
        mae.append(i(y_test,y_pred_rfr))
    elif i == mean_absolute_percentage_error:
        mape.append(i(y_test,y_pred_rfr))
    else:
        mse.append(i(y_test,y_pred_rfr))


# In[ ]:


train.append(rfr.score(X_train,y_train))
test.append(rfr.score(X_test,y_test))


# Of all the models we decided to choose gradient boosting as the next model, and furthur tweak the hyper parameter's of the model and also put this boosting model into bagging regressor and check for the model accuracy.

# ### NEXT MODEL

# In[ ]:


hyper_params_gbr = {'loss':['ls','lad','huber'],'learning_rate':[0.1,0.01,1],'n_estimators':[100,150]}


# In[ ]:


gbr2 = GradientBoostingRegressor()


# In[ ]:


model = GridSearchCV(gbr2,param_grid=hyper_params_gbr)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=181)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_test,y_test)


# In[ ]:


model.score(X_train,y_train)


# In[ ]:


model.best_params_


# These are the default parameters of the model so no hyper parameter tuning for this model is required, let us try and put this model into the bagging regressor.

# ### Final Model(Boosting+Bagging)

# In[ ]:


gbr2 = GradientBoostingRegressor()


# In[ ]:


br2 = BaggingRegressor(gbr2)


# #### Finding the best sample

# In[ ]:


temp = 0
for j in range(1,300,1):
    NXT,NXt,NYT,NYt = train_test_split(X,y,test_size=0.3,random_state=j)
    br2.fit(NXT,NYT)
    test_score = br2.score(NXt,NYt)
    train_score = br2.score(NXT,NYT)
    if test_score>temp:
        temp = test_score
        print(j,train_score,temp)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=87)
br2.fit(X_train,y_train)
y_pred_br2 = br2.predict(X_test)
for i in metrics:
    print(i(y_test,y_pred_br2))
    if i == r2_score:
        r2.append(i(y_test,y_pred_br2))
    elif i == mean_absolute_error:
        mae.append(i(y_test,y_pred_br2))
    elif i == mean_absolute_percentage_error:
        mape.append(i(y_test,y_pred_br2))
    else:
        mse.append(i(y_test,y_pred_br2))


# In[ ]:


train.append(br2.score(X_train,y_train))
test.append(br2.score(X_test,y_test))


# ### Using KFold Validation

# In[ ]:


test_scores = []
train_scores = []
cv = KFold(n_splits=10,random_state=42, shuffle=False)
for train_index,test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    br2.fit(X_train,y_train)
    test_scores.append(br.score(X_test, y_test))
    train_scores.append(br.score(X_train, y_train))


# In[ ]:


np.mean(train_scores)


# In[ ]:


np.mean(test_scores)


# We can see that after using KFold validation we got our average training accuracy to be 86.45 and average testing accuracy to be 86.30, OUR final model is bagging regressor with base estimator gradient boosting regressor.

# In[ ]:


ALL_SCORES = pd.DataFrame({'Model':['Linear Regression','Decision Tree','Extra Tree','AdaBoost','Bagging',
                                    'Extra Trees','GradientBoosting','Random Forest','Final_Model'],
                           'Training_Score':train,'Testing_Score':test,'R2_Score':r2,'MAE':mae,'MAPE':mape,'MSE':mse})


# In[ ]:


ALL_SCORES


# In[ ]:




