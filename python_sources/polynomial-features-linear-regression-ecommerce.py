#!/usr/bin/env python
# coding: utf-8

# # DATA DESCRIPTION
This dataset is having data of customers who buys clothes online. This file has
customer email, avg. session time with stylist, Time spent on the app and website, Length of Membership.
Our main objective is to predict the Yearly amount spent by the customers.
# In[ ]:


#ATTRIBUTES:
#Email: Email of the customer
#Address: Address of the customer
#Avatar: Avatar chosen by the customer
#Avg. Session Length: Average duration of the online session
#Time on App: Time spent on App
#Time on Website: Time spent on website
#Length of Membership: Time period of membership
#Yearly Amount Spent: Yearly amount spent by the customer


# # Loading Libraries

# In[ ]:


import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt   
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:


df = pd.read_csv('../input/Ecommerce.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape   ##Rows, Columns


# In[ ]:


df.dtypes ##Datatypes


# In[ ]:


df.head()


# In[ ]:


df.describe().T ##Summary for Quantitative Variables


# In[ ]:


##Above we can see the summary that contains mean, count, min, max. etc. values from the quantitative variables


# In[ ]:


df.isna().sum()


# In[ ]:


##We don't see any missing values in the data


# In[ ]:


df.describe(include=object).T ##Summary for Qualitative Variables


# In[ ]:


for i in list(df.dtypes[df.dtypes==object].index):
    print(i,'\n\n',(df[i].value_counts()/df[i].shape[0])*100,'\n\n\n\n')


# In[ ]:


##We see around 0.2% count of values in Email and Address, which shows that they are unique
##for Avatar, not all values are unique,and a few values are present as much as 1.4% in the data


# In[ ]:


##We see that address and email is unique for all the people, even though it is a qualitative variable
##Address and email won't really help us in predictions.
##Address, as a whole, could help in deciding which area has how much spending

##Avatar is a variable which has multiple counts for different people, so it looks like an important variable at the moment


# In[ ]:


for i in list(df.dtypes[df.dtypes==object].index):  ##Unique values in Categorical variables
    print(i,'\n\n',df[i].nunique(),'\n\n')


# In[ ]:


#Checking for null values
df.isna().sum()


# In[ ]:


##No missing values


# In[ ]:


#Checking outliers

#For quantitative variables

for i in list(df.dtypes[df.dtypes!=object].index):
    print(i)
    sns.boxplot(data=df,x=i,orient='v')
    plt.show()


# In[ ]:


##Our target variable, that is , Yearly amount spend, has quite a few outliers but we need to consider the outliers
##for the independant variableS:
#for the independant variables, Length of Membership has a lot of outliers, and Avg. Session Length and Time on App. have few
#outliers. We might remove outliers from some of the independant variables, at a later stage in our analysis


# In[ ]:


#Checking how the data is distributed:

##Let us check:
sns.distplot(df['Yearly Amount Spent'])
plt.show()


# In[ ]:


##Data ooks normally distributed. Let us check now:
#We will use the Jarque Bera test to check the normality of our target

#H0: Target normally distributed
#H1: Target not normally distributed
#Checking At 5% level of significance

from scipy import stats
print(stats.jarque_bera(df['Yearly Amount Spent']))


# In[ ]:


##So, the Pvalue in our case is 0.1182510779254693, which is more than 0.05 (Level of significance),and hence we will
#accept the null hypothesis, which states that Target is normally distributed


# # Checking for Correlation among variables

# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# In[ ]:


##We see that Length of Membership is highly correlated with the Target:Yearly Amount Spent(more than 0.5 value of correlation)
##Time on App's correlation with Target(Yearly Amount Spent) is 0.5(which is not very high, but still it should be kept in 
#check while building the model)


# In[ ]:


##As of now, we see that none of the independant variables are much correlated with each other. WE can check this using 
#as plot as well:
sns.heatmap(df.drop(columns='Yearly Amount Spent').corr(),annot=True)
plt.show()


# In[ ]:


##We can clearly see here that the independant variables are not much correlated with each other.
#Let's check pairplot as well
sns.pairplot(data=df)
plt.show()


# In[ ]:


##Based on this analysis alone, I choose not to discard any variables at the moment. Yes, Length of memberships is
#highly correlated with the Yearly amount spent, but for time being, I will allow it in my model. Later, if needed,
#we can remove Length of Membership to check accuracy again. But for now, we'll continue with all variables.

#Of course, we will drop/modify certain categorical variables, which aren't seen here(seen the above shows only quantitative
#variables).


# # Checking the relationship of Independant variables with the Target
# 

# In[ ]:


for i in df.drop(columns='Yearly Amount Spent').columns:
    sns.scatterplot(data=df,x=i,y='Yearly Amount Spent')
    plt.show()


# In[ ]:


## We can see that the relationships are not linear
#We can see that all the variables except Length of Membership, are evenly scattered, with Yearly Amount Spent, as the Target
##Even for Length of Membership, the relationship can't be called linear, but the correlation is very high


# In[ ]:


##Based on the above, we can later transform or combine features to extract more information out of them


# In[ ]:


##For example, we can see that graphs for email and Address look a bit similar, and that is because of the randomness of it
##We can transform Address in this case(to check for particular Areas)
#and we can actually drop Email, because it Email IDs #won't really help in predictions
##But, at the same time, we can also check the email provider and then maybe that can help us make predictions better
##Avatar can be used as it is for the time being, and later maybe we can combine it with Area where the person resides,
#or with some other variable, to extract the interaction of certain variables
#Average Session length, Time on App, and Time on Website can be used as it is, because they will act as good precitors,
#since they are not totally random, and at the same time they are not much correlated with the Target variable
##for now, we will use Length of Membership in our initial model, but later we can think of dropping it, since it is 
#highly correlated with the Target


# In[ ]:


df.head()


# In[ ]:


##Transforming Email:
df['Email'] = df['Email'].apply(lambda d: d.split('@')[1])


# In[ ]:


df['Email'].value_counts()


# In[ ]:


df['Area'] = df['Address'].apply(lambda d: d.split(',')[0])


# In[ ]:


df['Area'] = df['Area'].apply(lambda d: d.split('\n')[-1])


# In[ ]:


df['State'] = df['Address'].apply(lambda d: d.split(',')[-1][:3])


# In[ ]:


df.head()


# In[ ]:


##Dropping Address: since we can already transformed features:
df = df.drop(columns='Address')


# In[ ]:


df.head()


# In[ ]:


##Let us check quantitative summary again:
df.dtypes


# In[ ]:


df.describe(include=object)


# In[ ]:


##Checking value counts and scatterplot again:
for i in list(df.dtypes[df.dtypes==object].index):  ##Unique values in Categorical variables
    print(i,'\n\n',df[i].nunique(),'\n\n')


# In[ ]:


for i in list(df.dtypes[df.dtypes==object].index):
    print(i,'\n\n',(df[i].value_counts()/df[i].shape[0])*100,'\n\n\n\n')


# In[ ]:


##count of values in categorical columns:
for i in list(df.dtypes[df.dtypes==object].index):
    print(i,'\n\n',(df[i].value_counts()),'\n\n\n\n')


# In[ ]:


for i in df.drop(columns='Yearly Amount Spent').columns:
    sns.scatterplot(data=df,x=i,y='Yearly Amount Spent')
    plt.show()


# In[ ]:


##We see that the ditribution for email has changed a lot, but for the Area and the State it remains almost similar to
#what it was for the Address


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


##For modelling, we will have to encode(get dummies) for Email, Avatar, Area, and State:

pd.get_dummies(data=df,columns=['Email','Avatar','Area','State'],drop_first=True)


# In[ ]:


df['State'].unique()


# In[ ]:


##Above we can see that Uni, USN, Bo and USS don't really seem like States, but since we have converted our data which will
#be used for train and test, our predictions won't be biased


# In[ ]:


df['Email'].unique()


# In[ ]:


df['Area'].unique()


# In[ ]:


##We see that Area unique is also having certain Postal codes, which is not ideally desired, but of course, since we
#have converted the data for both train and test, our predictions won't be biased


# In[ ]:


##So, for now, we can actually Drop Email and Area, since they are extremely random and contain too many unique values
## we will still be using State as an independant variable, and later if we find it is not useful, we can drop it as well


# In[ ]:


df = df.drop(columns=['Area','Email'])


# In[ ]:


df2 = pd.get_dummies(data=df,columns=['Avatar','State'],drop_first=True)


# In[ ]:


df2.head()


# # Training the Model

# In[ ]:


y = df2['Yearly Amount Spent']
X = df2.drop(columns='Yearly Amount Spent')


# In[ ]:


X = sm.add_constant(X)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


##Checking if both train and test representative of the overall data


# In[ ]:


X.describe()


# In[ ]:


X_train.describe()


# In[ ]:


X_test.describe()


# In[ ]:


##We can see that the mean data from X, X_test and X_train resemble each other(means are close by to each other, for
#every independant variable,and hence we can claim that train and test are represetative of overall data)


# In[ ]:


##We can do the same for the target by doing the count
print(y.value_counts(normalize=True))
y.value_counts().plot.bar()
plt.show()
print(y_train.value_counts(normalize=True))
y_train.value_counts().plot.bar()
plt.show()
print(y_test.value_counts(normalize=True))
y_test.value_counts().plot.bar()
plt.show()


# In[ ]:


##We can see that the train and test data fairly represent the overall data, through the plots


# In[ ]:


#We can also do a ttest for the target:


# In[ ]:


from scipy.stats import ttest_1samp


# In[ ]:


#ttest_1samp(sample,pop_mean)


# In[ ]:


ttest_1samp(y_train,y.mean())


# In[ ]:


ttest_1samp(y_test,y.mean())


# In[ ]:


print('Mean is %2.1f Sd is %2.1f' % (y.mean(),np.std(y,ddof = 1)))


# In[ ]:


#PValue is greater than significance level-0.05 so accept null hypothesis, which means that train data
#and test data does represent the population


# # Base Model

# In[ ]:


import warnings 
warnings.filterwarnings('ignore')
import statsmodels.api as sm

model1 = sm.OLS(y,X).fit()
model1.summary()


# In[ ]:


##Checking the assumptions before Proceeding:


# In[ ]:


#Checking normailty of residuals


# In[ ]:


residuals = model1.resid


# In[ ]:


sns.distplot(residuals)


# In[ ]:


#Check linearity of residuals


# In[ ]:


import scipy.stats as stats
import pylab
from statsmodels.graphics.gofplots import ProbPlot
stats.probplot(residuals,dist='norm',plot=pylab)
plt.title('Probability plot')


# In[ ]:


#Checking Heteroscedacity

y_pred = model1.predict(X)

sns.regplot(x=y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.show()


# In[ ]:


#Shows homoscedacity


# In[ ]:


#Test-Goldfeld for Checking Homoscedacity

import statsmodels.stats.api as sms
test = sms.het_goldfeldquandt(y=model1.resid,x=X)
test


# In[ ]:


#Null accepted as pvalue greater than 0.05. Which means that homoscedacity exists in the data


# In[ ]:


#a. What is the overall R2? Please comment on whether it is good or not.
##overall R2 is 0.991 and Adjusted R2 is 0.984, which shows that there are certain variables which might be leading to
#overfitting.
##Also, the warnings show that a strong multi-collinearity does exist
#From the above model, we can see the separate Pvalues of the variables, which shows that the variables coming from 
#States and Avatar are not significant as their Pvalues are much greater than 0.05


# # Predictions

# In[ ]:



ypred1 = model1.predict(X_test)
ypred1


# In[ ]:


X_test


# In[ ]:


X_test['Predicted Values'] = ypred1


# In[ ]:


X_test.head()


# # What are the significant variables?

# In[ ]:


#From the above model, we see that Average Session Length, Time on App, Length of Membership are the most significant
#variables, since their Pvalues are much lesser than the level of significance i.e. 0.05.
##Almost all the States and the Avatars are insignificant variables as per the above model


# # Do you see multi-collinearity?

# In[ ]:


##Yes, multi-collinearity does exist, as we can see from the above model summary, and we can check it as well, using the
#Variance Inflation factor:

#Checking multicollearity using Heatmap and VIF:
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
pd.DataFrame({'vif': vif}, index=X.columns).sort_values(by='vif',ascending=False)


# In[ ]:


##Generally we will remove variables with VIF>4   (Except the constant: constant is not a variable, so we don't need to see the
#VIF for the constant)
##We will remove highly collinear variables for our next model


# In[ ]:





# # Evaluation Metrics: RMSE (Root Mean Square Error), MAE(Mean Absolute Error)

# In[ ]:


#Root Mean Square Error
##RMSE for test:
ypred1 = model1.predict(X_test.drop(columns='Predicted Values'))
np.sqrt(metrics.mean_squared_error(ypred1,y_test))


# In[ ]:


##RMSE for train:
ypred2 = model1.predict(X_train)
np.sqrt(metrics.mean_squared_error(ypred2,y_train))


# In[ ]:


##We see that the RMSE for test is just a bit above that of Train


# In[ ]:


##Mean absolute error:
#For test:
metrics.mean_absolute_error(y_test,ypred1)


# In[ ]:


#For test:
metrics.mean_absolute_error(y_train,ypred2)


# In[ ]:


##For Mean Absolute Error, we see that it is again a bit higher for Test when compared to Train


# In[ ]:


##MAPE: Mean Absolute Percentage Error:
##For test:
np.mean(abs(((ypred1 - y_test)/y_test)*100))


# In[ ]:


##For test:
np.mean(abs(((ypred2 - y_train)/y_train)*100))


# In[ ]:


##For Mean Absolute Percentage Error, we see that it is again a bit higher for Test when compared to Train


# # Journey Towards Improving the Model Accuracy
# 

# In[ ]:


import statsmodels.api as sm
def model(df):
    y = df['Yearly Amount Spent']
    x = df.drop(columns='Yearly Amount Spent')
    X = sm.add_constant(x)
    X_train1,X_test1,y_train1,y_test1 = train_test_split(X,y,test_size=0.3,random_state=0)
    model1 = sm.OLS(y_train1,X_train1).fit()
    ypredtrain = model1.predict(X_train1)
    rmse_train = np.sqrt(metrics.mean_squared_error(ypredtrain,y_train1))
    r2train = model1.rsquared
    r2_adjtrain = model1.rsquared_adj
    model2 = sm.OLS(y_test1,X_test1).fit()
    ypredtest = model2.predict(X_test1)
    rmse_test = np.sqrt(metrics.mean_squared_error(ypredtest,y_test1))
    r2test = model2.rsquared
    r2_adjtest = model2.rsquared_adj
    df2 = pd.DataFrame({"Train":[X_train1.shape,rmse_train,r2train,r2_adjtrain],"Test":[X_test1.shape,rmse_test,r2test,r2_adjtest]},index=['Dataset_Shape','RMSE','Rsquared','RSquared-Adjusted'])
    return df2
model(df2)


# In[ ]:


##Above, we see the full model based on all the variables.
##Now, we will use remove certain variables using VIF


# In[ ]:


# removing collinear variables using vif
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(x):
    thresh = 5.0
    output = pd.DataFrame()
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    for i in range(1,k):
        print("Iteration no.")
        print(i)
        print(vif)
        a = np.argmax(vif)
        print("Max VIF is for variable no.:")
        print(a)
        if vif[a] <= thresh :
            break
        if i == 1 :          
            output = x.drop(x.columns[a], axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        elif i > 1 :
            output = output.drop(output.columns[a],axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
    return(output)


# In[ ]:


out1 = calculate_vif(X)


# In[ ]:


## The following includes only the relevant features for creating our new model with selected variables after removing
#multi-collinearity:  The variables below don't seem important one, and hence there is a possibilty that this model won't be
#useful to us
out1.head()


# In[ ]:


##Let's create a model using out1:

import statsmodels.api as sm
def model(df):
    y = df['Yearly Amount Spent']
    x = out1
    X = sm.add_constant(x)
    X_train1,X_test1,y_train1,y_test1 = train_test_split(X,y,test_size=0.3,random_state=0)
    model1 = sm.OLS(y_train1,X_train1).fit()
    ypredtrain = model1.predict(X_train1)
    rmse_train = np.sqrt(metrics.mean_squared_error(ypredtrain,y_train1))
    r2train = model1.rsquared
    r2_adjtrain = model1.rsquared_adj
    model2 = sm.OLS(y_test1,X_test1).fit()
    ypredtest = model2.predict(X_test1)
    rmse_test = np.sqrt(metrics.mean_squared_error(ypredtest,y_test1))
    r2test = model2.rsquared
    r2_adjtest = model2.rsquared_adj
    df2 = pd.DataFrame({"Train":[X_train1.shape,rmse_train,r2train,r2_adjtrain],"Test":[X_test1.shape,rmse_test,r2test,r2_adjtest]},index=['Dataset_Shape','RMSE','Rsquared','RSquared-Adjusted'])
    return df2
model(df2)


# In[ ]:


##As guessed, we see that this model did not perform well. Let's try some other technique
#We see that our training data and testing data have performed badly in this case, and that is majorly because the VIF selection
#method could not work in this case. We can try removing the variables manually
#or else, in our first model, where we saw that variables related to States and Avatars weren't really significant.
#so, we can try dropping them or else creating polynomial features


# In[ ]:


##So, mostly in our first model, we are seeing a case of Overfitting, which can be removed. Let's try:


# In[ ]:


df2.head()


# # Using VIF to eliminate features

# In[ ]:


pd.DataFrame({'vif': vif}, index=X.columns).sort_values(by='vif',ascending=False)


# In[ ]:


##We can try removing the highest VIF variable first:


# In[ ]:


X = X.drop(columns='Avatar_SlateBlue')


# In[ ]:


X


# In[ ]:


vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
pd.DataFrame({'vif': vif}, index=X.columns).sort_values(by='vif',ascending=False)


# In[ ]:


##We can try removing State_Bo now:
X = X.drop(columns='State_ Bo')
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
pd.DataFrame({'vif': vif}, index=X.columns).sort_values(by='vif',ascending=False)


# In[ ]:


##Now, the VIF for most variables is less than 4, so let's try building a model now:


# In[ ]:


X.head()


# In[ ]:


selected = X


# In[ ]:


##Let's create a model:

import statsmodels.api as sm
def model(df):
    y = df['Yearly Amount Spent']
    x = selected
    X = sm.add_constant(x)
    X_train1,X_test1,y_train1,y_test1 = train_test_split(X,y,test_size=0.3,random_state=0)
    model1 = sm.OLS(y_train1,X_train1).fit()
    ypredtrain = model1.predict(X_train1)
    rmse_train = np.sqrt(metrics.mean_squared_error(ypredtrain,y_train1))
    r2train = model1.rsquared
    r2_adjtrain = model1.rsquared_adj
    model2 = sm.OLS(y_test1,X_test1).fit()
    ypredtest = model2.predict(X_test1)
    rmse_test = np.sqrt(metrics.mean_squared_error(ypredtest,y_test1))
    r2test = model2.rsquared
    r2_adjtest = model2.rsquared_adj
    df2 = pd.DataFrame({"Train":[X_train1.shape,rmse_train,r2train,r2_adjtrain],"Test":[X_test1.shape,rmse_test,r2test,r2_adjtest]},index=['Dataset_Shape','RMSE','Rsquared','RSquared-Adjusted'])
    return df2
model(df2)


# In[ ]:


selected.head()


# In[ ]:


import warnings 
warnings.filterwarnings('ignore')
import statsmodels.api as sm

model3 = sm.OLS(y,X).fit()
model3.summary()


# In[ ]:


##We see a good Rsquare value, but then again we see that only very few variables from Avatar and State are actually
#significant
##This model also shows Time on Website as insignificant


# # Feature Elimination: Backward Elimination

# In[ ]:


#Backward Elimination to select features
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values,index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
print(len(selected_features_BE))


# In[ ]:


##So, let's try a model with these features:

Xselect = X[['const', 'Avg. Session Length', 'Time on App', 'Length of Membership', 'Avatar_DarkGreen', 'Avatar_ForestGreen', 'Avatar_IndianRed', 'Avatar_LightSkyBlue', 'Avatar_Peru', 'Avatar_Pink', 'Avatar_PowderBlue', 'Avatar_Silver', 'Avatar_Yellow', 'State_ DC', 'State_ MI', 'State_ MT']]


# In[ ]:


import warnings 
warnings.filterwarnings('ignore')
import statsmodels.api as sm

model4 = sm.OLS(y,Xselect).fit()
model4.summary()


# In[ ]:


##On checking the Pvalues of the model built using Xselect, we see that all the independant variables have Pvalues lesser
#than 0.05, which states that all of them are significant as per this model.
##But we can still see that the multi-collinearity is present.
#Let's check multi-colinearity using VIF and Heatmap


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(Xselect.drop(columns='const').corr(),annot=True)
plt.show()


# In[ ]:


##From above itself, we can see that multi-collinearity does not exist among these variables: Still let's check
#the VIF


# In[ ]:


###Ideally, we desire a high Rsquare and low RMSE: But of course, the Rsquare should not be so high that it shows Overfitting
#So, we need to check for a model which has high Rsquare and low RMSE


# In[ ]:


##Let's create a model: using Xselect

import statsmodels.api as sm
def model(df):
    y = df['Yearly Amount Spent']
    x = Xselect
    X = sm.add_constant(x)
    X_train1,X_test1,y_train1,y_test1 = train_test_split(X,y,test_size=0.3,random_state=0)
    model1 = sm.OLS(y_train1,X_train1).fit()
    ypredtrain = model1.predict(X_train1)
    rmse_train = np.sqrt(metrics.mean_squared_error(ypredtrain,y_train1))
    r2train = model1.rsquared
    r2_adjtrain = model1.rsquared_adj
    model2 = sm.OLS(y_test1,X_test1).fit()
    ypredtest = model2.predict(X_test1)
    rmse_test = np.sqrt(metrics.mean_squared_error(ypredtest,y_test1))
    r2test = model2.rsquared
    r2_adjtest = model2.rsquared_adj
    df2 = pd.DataFrame({"Train":[X_train1.shape,rmse_train,r2train,r2_adjtrain],"Test":[X_test1.shape,rmse_test,r2test,r2_adjtest]},index=['Dataset_Shape','RMSE','Rsquared','RSquared-Adjusted'])
    return df2
model(df2)


# In[ ]:


##Let's us consider Polynomial Features now:To model using Interaction of the variables as well


# In[ ]:


df2.head()


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures

X1 = X.drop('const',axis=1)
pf = PolynomialFeatures()
Xp1 = pf.fit_transform(X1)
cols = pf.get_feature_names(X1.columns)
Xp = pd.DataFrame(Xp1,columns=cols)


# In[ ]:





# In[ ]:


##AS we can know that modelling using All the Polynomial features will not give us good results. WE don't want to burden the
#model with so much data that it can't process. And ultimately this will cause overfitting(if we build model using all the
#features created above). So, we will select certain feature(important ones)
##Let us select the important features out these created Polynomial Features


# In[ ]:


#Backward Elimination to select features out of the Polynomial Features:
ys = list(y)
cols = list(Xp.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = Xp[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(ys,X_1).fit()
    p = pd.Series(model.pvalues.values,index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
print(len(selected_features_BE))


# In[ ]:


##Again, we see that 20706 features were selected here, and we don't need so many features as we don't want to burden our data
#so, we can try to find out polynomial functions from the Xselect which we got above, by selecting the features through
#backward elimination


# In[ ]:


Xselect.head()


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures

X1 = Xselect.drop('const',axis=1)
pf = PolynomialFeatures()
Xp1 = pf.fit_transform(X1)
cols = pf.get_feature_names(X1.columns)
Xp = pd.DataFrame(Xp1,columns=cols)


# In[ ]:


#Backward Elimination to select features out of the Polynomial Features:
ys = list(y)
cols = list(Xp.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = Xp[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(ys,X_1).fit()
    p = pd.Series(model.pvalues.values,index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
print(len(selected_features_BE))


# In[ ]:


##91 seems like an okay number for features: so let's proceed with these features


# In[ ]:


Xs = Xp.loc[:,selected_features_BE]


# In[ ]:


Xs


# In[ ]:


#Model using selected features
ols = sm.OLS(ys,Xs).fit()
ols.summary()


# In[ ]:


##again, we see so many nan values in the summary(even in the Pvalues) and this is not desired, since we want to understand
#how and by how much strength are the independant variables important.
##so we can now try building a machine learning model and we can tune it to find the perfect one


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.3, random_state=1)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,y_train)


# In[ ]:


result = lr.fit(X_train,y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)


# In[ ]:


y_test_pred


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test,y_test_pred)


# In[ ]:


r2_score(y_train,y_train_pred)


# In[ ]:


from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test,y_test_pred))


# In[ ]:


np.sqrt(mean_squared_error(y_train,y_train_pred))


# In[ ]:


##We will use RFE to reduce over-fitting and the number of variables:

from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV,KFold


# In[ ]:


params = [{'n_features_to_select':list(range(1,60))}]
lr = LinearRegression()
rfe = RFE(lr)

folds = KFold(n_splits=3,random_state=1)
model_cv = GridSearchCV(rfe,param_grid=params,cv=folds)
model_cv.fit(Xp,y)


# In[ ]:


model_cv.best_params_


# In[ ]:


lr = LinearRegression()
rfe = RFE(lr,n_features_to_select=1)

rfe.fit(Xp,y)


# In[ ]:


cols = pd.DataFrame(list(zip(Xp.columns,rfe.support_,rfe.ranking_)),columns=['cols','select','rank'])
cols.sort_values(by='rank').head(8)


# In[ ]:


a = list(cols.sort_values(by='rank').head(8)['cols'])


# In[ ]:


Xp[a]


# In[ ]:


##Now building model with 8 top features(n features to select was 8)
Xp1 = Xp[a]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Xp1, y, test_size=0.3, random_state=1)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,y_train)


# In[ ]:


result = lr.fit(X_train,y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)


# In[ ]:


y_test_pred


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test,y_test_pred)


# In[ ]:


r2_score(y_train,y_train_pred)


# In[ ]:


from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test,y_test_pred))


# In[ ]:


np.sqrt(mean_squared_error(y_train,y_train_pred))


# In[ ]:





# In[ ]:


##WE see that on selecting the top 8 features, the RMSE for the test decreases and the train increases here
##So, in this case let's consider if the Test RMSE was to be of utmost importance, then we would consider this model
#since the RMSE has decreased and the Rsquare has also increased(which is good unless it is not overfitting)


# In[ ]:


##Building statistical model using Selected variables:
ols = sm.OLS(ys,Xp1).fit()
ols.summary()


# In[ ]:


import statsmodels.api as sm
def model(df):
    y = df['Yearly Amount Spent']
    x = Xp1
    X = sm.add_constant(x)
    X_train1,X_test1,y_train1,y_test1 = train_test_split(X,y,test_size=0.3,random_state=0)
    model1 = sm.OLS(y_train1,X_train1).fit()
    ypredtrain = model1.predict(X_train1)
    rmse_train = np.sqrt(metrics.mean_squared_error(ypredtrain,y_train1))
    r2train = model1.rsquared
    r2_adjtrain = model1.rsquared_adj
    model2 = sm.OLS(y_test1,X_test1).fit()
    ypredtest = model2.predict(X_test1)
    rmse_test = np.sqrt(metrics.mean_squared_error(ypredtest,y_test1))
    r2test = model2.rsquared
    r2_adjtest = model2.rsquared_adj
    df2 = pd.DataFrame({"Train":[X_train1.shape,rmse_train,r2train,r2_adjtrain],"Test":[X_test1.shape,rmse_test,r2test,r2_adjtest]},index=['Dataset_Shape','RMSE','Rsquared','RSquared-Adjusted'])
    return df2
model(df2)


# In[ ]:


##THE above model is the one which I have selected for final analysis. Please read more about it in the summary in last
#question


# In[ ]:


##We can try using Regularization technique on the above defined Xp such as LassoCV as well:
##To visualize the importance of features we can try Lasso


# In[ ]:


Xp.head()


# In[ ]:


y


# In[ ]:


from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
reg = LassoCV()
reg.fit(Xp, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(Xp,y))
coef = pd.Series(reg.coef_, index = Xp.columns)


# In[ ]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20, 20)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[ ]:


imp_coef


# In[ ]:


imp_coef.reset_index().sort_values(by=0).tail(2)


# In[ ]:


imp_coef.reset_index().sort_values(by=0,ascending=False).head(10)


# In[ ]:


##We see the variables from Lasso that show importance(either positive or negative)


# # SUMMARY
# 

# In[ ]:


##For summarizing, I have selected my final model to be the one with 8 final features(the model below which I wrote that I was
#going to explain more in the summary)
#For this model, first the significant features were selected from the original(initial) model, and then polynomial 
#features(for creatig features with interaction) were #created using the selected features. 
#Out of these polynomial features, we then selected the final 8 features #using which we build our selected model
##Out of these 8 select features, I have built both, the machine learning model and the statistical model
#For ML model:
#Test Rsquare:0.9883269083622762
#Test RMSE: 9.047655455715873

#for statistical model(Test)
#RMSE: 9.56287
#Rsquared:0.985422

#so, this statistical model(since we build statistical as initial model) has been selected as a good model because it 
#contains very less number of features as compares to other
#models that we build (and yet it gives a great Rsquare: which means that it explains the variance in the data extremely well)
#Also, as we have seen, this model shows an RMSE which is lower than a few of the other models and higher than a few of the
#other models, but it shows a good Rsquare 


# # Possible Insights for the Business Stakeholders

# In[ ]:


##Below we see variables that positively and negatively affect the total amount spend
#Please read business interpretation below:
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20, 20)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[ ]:


imp_coef.reset_index().sort_values(by=0).tail(2) ##Variables negatively affecting amount spent and the interaction of 
#these variables negatively affecting amount spent(increase in these variables will decrease amount spend)


# In[ ]:


imp_coef.reset_index().sort_values(by=0,ascending=False).head(10) ##Variables positively affecting amount spent,and
#the interaction of these variables positively affecting amount spent(increase in these variables will increase amount spent)


# # How the base model was improved.

# In[ ]:


##Most effect was done by Feature Selection and then Creating Polynomial Features

#For this model, first the significant features were selected from the original(initial) model, and then polynomial 
#features(for creatig features with interaction) were #created using the selected features. 
#Out of these polynomial features, we then selected the final 8 features #using which we build our selected model


# # End Notes

# In[ ]:


##RMSE and Rsquare have to be kept in mind at all times. My model does have a very high Rsquare which could mean that it is 
#still over fitting and then again there was another model with a lower RMSE but that model was
#not explaining the importance and interaction of the variables properly, which is why I chose this model as my final model

##Another risk could be that, is some other model/case, these variables would not have been of the same importance, and 
#the immportance of the features does not neccessaily depend on the model, but of course it does depend on how the model
#was formed in the first place(using which variables, using which techniques): all that is very important to finally
#interpret which variables would make a difference while writing the business interpretation


# In[ ]:




