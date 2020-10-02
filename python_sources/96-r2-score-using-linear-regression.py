#!/usr/bin/env python
# coding: utf-8

# # Solving this using Linear regression 

# We would be using better feature engineering for keeping things simple. This has a capability to get a score which matches other models which are more flexible models modelled using Hyper parameter tuning.

# Let's import the important libraries to operate the data.

# In[ ]:


import numpy as np
import pandas as pd
import scipy 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Let us read the Dataset using Pandas 'read_csv' command.

# In[ ]:


data=pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')

# Let us name our dataset as 'data'. And load it to see what it has init.


# In[ ]:


data.head(n=5)

# Let us load first five observations of our dataset.


# Looks like huge dataset with over 20 features. Its almost not possible to get insight from this dataset by just loading its head. Lets read the info and check for its total number of observations and examine it whether it has null values or not.

# In[ ]:


data.info()


# It has 22 columns and nearly 3000 entries and with the inequality in no. of observations from each feature it is clear that it has got some null values. Lets try to fix them up and try to solve the problem.

# It has 20 Quantitative and 2 Qualitative features.

# Let us explore and find where the null values are actually present in the Dataset. It is kind of easy to visualize it rather than having it in any other format. Luckily we have heatmaps which uses isnull command to highlight the null values.

# In[ ]:


sns.heatmap(pd.isnull(data));

# This plot highlights the null values.


# ## How do we impute NULL VALUES ?

# In genral we impute null values using the MEAN, MEDIAN Or MODE of that specific feature. Lets see how many null values does each of the features has.

# Let us write a simple algorithm which prints a dataframe which shows no. of nans from each column.

# In[ ]:


a=list(data.columns)
b=[]
for i in a:
    c=data[i].isnull().sum()
    b.append(c)
null_df=pd.DataFrame({'Feature name':a,'no. of Nan':b})
null_df


# We can see in columns like 'Life Expectancy' and 'Adult Mortality' there were few Nans values which wasn't visible in our heatmap. So lets fill them up with their average value as they are Continous features.

# In[ ]:


# This piece of code will fill the null values of the selected feature with its mean.
# This one is for filling Nans in 'Life expectancy' column.

data['Life expectancy ']=data['Life expectancy '].fillna(value=data['Life expectancy '].mean())

# Lets also do the same for Adult Mortality.

data['Adult Mortality']=data['Adult Mortality'].fillna(value=data['Adult Mortality'].mean())


# Now Lets move on to filling the other columns as well. We will move one by one by filling the Nans of each feature. Our next in target is "ALCOHOL" feature which has almost about 194 Nans.

# But wait ! We are filling about 200 Nans, Is it ok fill Nans with MEAN this time ? the answer is absolutely Yes. But I thought of doing it in a different way. As it has Nans 20 times larger than the previous one, I thought to fill it using other column which represents best this column. 

# It is always good to fill Nans using bunch of different values as we know that a feature is less likely to have a continous value about 200 or more times and Iam pretty sure that by doing this we would definitely come up with a good result.

# To do this let us first see the correlation matrix of these features using '.corr' method by pandas.

# In[ ]:


corr_data=data.corr()
corr_data


# As you can see there are some features that are quite well correlated with each other. So, we would impute these nans using the other features which are nicely correlated with eachother.

# Let us now Visualise these correlation values using a heatmap again.

# In[ ]:


sns.heatmap(corr_data)


# The DARKEST and the LIGHTEST blocks represets that there is a strong relationship between those attributes.

# From the above correlation matrix 'Alcohol' feature nicely correlates with the 'Schooling' feature. Now lets plot a Scatterplot between them and observe the trend.

# In[ ]:


sns.scatterplot(x=data['Schooling'],y=data['Alcohol']);

#The semicolon atlast in the code is to hide the address of the plot which is not that required but I personally like doing that


# - I have selected these values for imputing the Nans by observing a the trends in between selected interval.

# Ex: The mean value of Alcohol which is in between 5-10 of Schooling column is 4.0

# In[ ]:


# These values are mean values of the selected interval of other feature.

def impute_Alcohol(cols):
    al=cols[0]
    sc=cols[1]
    if pd.isnull(al):
        if sc<=2.5:
            return 4.0
        elif 2.5<sc<=5.0:
            return 1.5
        elif 5.0<sc<=7.5:
            return 2.5
        elif 7.5<sc<=10.0:
            return 3.0
        elif 10.0<sc<=15:
            return 4.0
        elif sc>15:
            return 10.0
    else:
        return al
    
data['Alcohol']=data[['Alcohol','Schooling']].apply(impute_Alcohol,axis=1)


# Now lets cross check whether values were filled or not by using the same heatmap.

# In[ ]:


sns.heatmap(pd.isnull(data))


# ##### Looks like we are still left with some Nans. But why did this exactly happened ?

# If we could observe from the heatmap the zone in which null values were failed to be filled is the same zone in which the other feature had its Null values. So for now lets fill those remaining Nans with the Mean Value.

# In[ ]:


data['Alcohol']=data['Alcohol'].fillna(value=data['Alcohol'].mean())


# In[ ]:


# Rechecking the Heatmap.
sns.heatmap(pd.isnull(data))


# Now we dont have any null values in 'alcohol' column.

# Lets look at the distribution of the alcohol column.

# In[ ]:


sns.distplot(data['Alcohol']);


# There is a huge positive skew for this distribution. Lets look at its Kurtosis value.

# In[ ]:


scipy.stats.skew(data['Alcohol'],axis=0)

# This shows that this column has positive skew.


#  Lets repeat this for imputing  Nans for all other features as well.

# #### Important observations:

# - Now our next in list is 'Hepatatis B' which highly correlates with 'Diptheria', But both of them have null values in the same zone.
# 

# - If we want to impute 'Diptheria' first with highly correlated feature 'Polio' then that again would be of no use because it has again the same problem of null values(Nans) in same Zone.

# ** But luckily we found an option of imputing 'Polio' feature with the 'Life expactancy' which nicely correlates with it. So for now lets impute 'Polio' feature firstly with 'Life expactancy' and then would impute others using this

# In[ ]:


sns.scatterplot(x=data['Life expectancy '],y=data['Polio']);

# Scattterplot between them.


# Imputing the selected values for each interval. It is the same way like we did for the 'Alcohol' feature.

# In[ ]:


def impute_polio(c):
    p=c[0]
    l=c[1]
    if pd.isnull(p):
        if l<=45:
            return 80.0
        elif 45<l<=50:
            return 67.0
        elif 50<l<=60:
            return 87.44
        elif 60<l<=70:
            return 91
        elif 70<l<=80:
            return 94.3
        elif l>80:
            return 95
    else:
        return p
    
data['Polio']=data[['Polio','Life expectancy ']].apply(impute_polio,axis=1)


# Lets have a look at the Null values again

# In[ ]:


a=list(data.columns)
b=[]
for i in a:
    c=data[i].isnull().sum()
    b.append(c)
null_df=pd.DataFrame({'Feature name':a,'no. of Nan':b})
null_df


# Now we dont have any nulls in polio section. Lets impute 'Diphtheria' using 'Polio' feature.

# In[ ]:


# Scatter plot between these features.

sns.scatterplot(x=data['Polio'],y=data['Diphtheria ']);


# In[ ]:


def impute_Diptheria(c):
    d=c[0]
    p=c[1]
    if pd.isnull(d):
        if p<=10:
            return 75.0
        elif 10<p<=40:
            return 37.0
        elif 40<p<=45:
            return 40.0
        elif 45<p<=50:
            return 50.0
        elif 50<p<=60:
            return 55.0
        elif 60<p<=80:
            return 65.0
        elif p>80:
            return 90.0
    else:
        return d
data['Diphtheria ']=data[['Diphtheria ','Polio']].apply(impute_Diptheria,axis=1)


# In[ ]:


a=list(data.columns)
b=[]
for i in a:
    c=data[i].isnull().sum()
    b.append(c)
null_df=pd.DataFrame({'Feature name':a,'no. of Nan':b})
null_df

# A look at the null values again.


# Now its time for imputing 'Hepatitis B' using 'Diptheria' feature.

# In[ ]:


sns.scatterplot(x=data['Diphtheria '],y=data['Hepatitis B']);

# Scatterplot between them.


# In[ ]:


def impute_HepatatisB(cols):
    hep=cols[0]
    dip=cols[1]
    if pd.isnull(hep):
        if dip<=15:
            return 75.0
        elif 15<dip<=30:
            return 20.0
        elif 30<dip<=45:
            return 38.0
        elif 45<dip<=60:
            return 43.0
        elif 60<dip<=80:
            return 63.0
        elif dip>80:
            return 88.4
    else:
        return hep
    
data['Hepatitis B']=data[['Hepatitis B','Diphtheria ']].apply(impute_HepatatisB,axis=1)


# In[ ]:


data[data['Diphtheria ']>80.0]['Hepatitis B'].mean()

# Mean for imputing Diptheria Nans in 80-100 interval.


# Now again cross checking the imputed values.

# In[ ]:


a=list(data.columns)
b=[]
for i in a:
    c=data[i].isnull().sum()
    b.append(c)
null_df=pd.DataFrame({'Feature name':a,'no. of Nan':b})
null_df


# Lets repeat this for all the features.

# ### Here is the list of features that we are going to impute with other features:

# - Impute 'BMI' feature with 'Life expactancy' feature.

# - Impute 'Total expenditure' with 'Alcohol' feature.

# - Impute 'GDP' feature with 'percentage expenditure	' feature.

# - Impute 'Population' feature with 'Infant death' feature.

# - Impute 'Thin 1-19' feature with 'BMI' feature.

# - Impute 'Thin 5-9' feature with 'BMI' feature.

# - Impute 'Schooling' feature and 'Income Composition of resources' feature with 'Life expactancy' feature.

# Lets begin the process of imputing. 

# In[ ]:


sns.scatterplot(x=data['Life expectancy '],y=data[' BMI ']);


# In[ ]:


def impute_BMI(c):
    b=c[0]
    l=c[1]
    if pd.isnull(b):
        if l<=50:
            return 25.0
        elif 50<l<=60:
            return 25.0
        elif 60<l<=70:
            return 32.0
        elif 70<l<=80:
            return 46.8
        elif 80<l<=100:
            return 60.0
    else:
        return b
    
data[' BMI ']=data[[' BMI ','Life expectancy ']].apply(impute_BMI,axis=1)


# In[ ]:


a=list(data.columns)
b=[]
for i in a:
    c=data[i].isnull().sum()
    b.append(c)
null_df=pd.DataFrame({'Feature name':a,'no. of Nan':b})
null_df


# In[ ]:


sns.scatterplot(y=data['Total expenditure'],x=data['Alcohol']);


# In[ ]:


def impute_Total_exp(c):
    t=c[0]
    a=c[1]
    if pd.isnull(t):
        if a<=2.5:
            return 5.08
        elif 2.5<a<=5.0:
            return 6.0
        elif 5.0<a<=10.0:
            return 6.71
        elif 10.0<a<=12.5:
            return 6.9
        elif a>12.5:
            return 6.68
    else:
        return t
    
data['Total expenditure']=data[['Total expenditure','Alcohol']].apply(impute_Total_exp,axis=1)        


# In[ ]:


sns.scatterplot(x=data['percentage expenditure'],y=data['GDP']);


# In[ ]:


def impute_GDP(c):
    g=c[0]
    p=c[1]
    if pd.isnull(g):
        if p<=1250:
            return 1100.0
        elif 1250<p<=2500:
            return 1800.0
        elif 2500<p<=3750:
            return 2900.0
        elif 3750<p<=7500:
            return 3500.0
        elif 7500<p<=8750:
            return 4500.0
        elif 8750<p<=10000:
            return 5000.0
        elif 10000<p<=11250:
            return 5700.0
        elif 11250<p<=12500:
            return 7000.0
        elif 12500<p<=15000:
            return 8000.0
        elif 15000<p<=17500:
            return 9000.0
        elif p>17500:
            return 8500.0
    else:
        return g
    
data['GDP']=data[['GDP','percentage expenditure']].apply(impute_GDP,axis=1)


# In[ ]:


sns.scatterplot(x=data['infant deaths'],y=data['Population']);


# In[ ]:


def impute_population(c):
    p=c[0]
    i=c[1]
    if pd.isnull(p):
        if i<=100:
            return 0.19*((10)**9)
        elif 100<i<=250:
            return 0.18*((10)**9)
        elif 250<i<=350:
            return 0.02*((10)**9)
        elif 350<i<=900:
            return 0.1*((10)**9)
        elif 900<i<=1100:
            return 0.18*((10)**9)
        elif 1100<i<=1250:
            return 0.05*((10)**9)
        elif 1250<i<=1500:
            return 0.19*((10)**9)
        elif 1500<i<=1750:
            return 0.05*((10)**9)
        elif i>1750:
            return 0.1*((10)**9)
    else:
        return p
    
data['Population']=data[['Population','infant deaths']].apply(impute_population,axis=1)


# In[ ]:


sns.scatterplot(x=data[' BMI '],y=data[' thinness  1-19 years']);


# In[ ]:


def impute_Thin_1(c):
    t=c[0]
    b=c[1]
    if pd.isnull(t):
        if b<=10:
            return 5.0
        elif 10<b<=20:
            return 10.0
        elif 20<b<=30:
            return 8.0
        elif 30<b<=40:
            return 6.0
        elif 40<b<=50:
            return 3.0
        elif 50<b<=70:
            return 4.0
        elif b>70:
            return 1.0
    else:
        return t
    
data[' thinness  1-19 years']=data[[' thinness  1-19 years',' BMI ']].apply(impute_Thin_1,axis=1)


# In[ ]:


sns.scatterplot(x=data[' BMI '],y=data[' thinness 5-9 years']);


# In[ ]:


def impute_Thin_1(c):
    t=c[0]
    b=c[1]
    if pd.isnull(t):
        if b<=10:
            return 5.0
        elif 10<b<=20:
            return 10.0
        elif 20<b<=30:
            return 8.0
        elif 30<b<=40:
            return 6.0
        elif 40<b<=50:
            return 3.0
        elif 50<b<=70:
            return 4.0
        elif b>70:
            return 1.0
    else:
        return t
    
data[' thinness 5-9 years']=data[[' thinness 5-9 years',' BMI ']].apply(impute_Thin_1,axis=1)


# In[ ]:


sns.scatterplot(x=data['Life expectancy '],y=data['Income composition of resources']);


# In[ ]:


def impute_Income(c):
    i=c[0]
    l=c[1]
    if pd.isnull(i):
        if l<=40:
            return 0.4
        elif 40<l<=50:
            return 0.42
        elif 50<l<=60:
            return 0.402
        elif 60<l<=70:
            return 0.54
        elif 70<l<=80:
            return 0.71
        elif l>80:
            return 0.88
    else:
        return i
        
data['Income composition of resources']=data[['Income composition of resources','Life expectancy ']].apply(impute_Income,axis=1)      


# In[ ]:


sns.scatterplot(x=data['Life expectancy '],y=data['Schooling']);


# In[ ]:


def impute_schooling(c):
    s=c[0]
    l=c[1]
    if pd.isnull(s):
        if l<= 40:
            return 8.0
        elif 40<l<=44:
            return 7.5
        elif 44<l<50:
            return 8.1
        elif 50<l<=60:
            return 8.2
        elif 60<l<=70:
            return 10.5
        elif 70<l<=80:
            return 13.4
        elif l>80:
            return 16.5
    else:
        return s
    
data['Schooling']=data[['Schooling','Life expectancy ']].apply(impute_schooling,axis=1)


# In[ ]:


data[(data['Life expectancy ']>80) & (data['Life expectancy ']<=90)]['Schooling'].mean()

# Example of how iam deciding values for filling Nans above
# You can see above in range above 80 we got avg. as 16.5 so we have imputed it that way.


# Now as we have finished filling Nans lets have a look at that null_df which shows no. of Nans.

# In[ ]:


a=list(data.columns)
b=[]
for i in a:
    c=data[i].isnull().sum()
    b.append(c)
null_df=pd.DataFrame({'Feature name':a,'no. of Nan':b})
null_df


# Now none of them above features have Nans in them, Let us split the dataset and look for fitting a model.

# As we are predicting Life Expectancy our 'Target'(y) variable will be 'Life expectancy'. And remaining attributes would be considered as X or Predictors.

# Let us check our Target Variable, also its distribution.

# In[ ]:


y=data['Life expectancy ']


# In[ ]:


sns.distplot(y);


# It almost have a normal distribution with negative skew.

# lets check our X or Predictors dataset.

# In[ ]:


X=data.drop('Life expectancy ',axis=1)


# lets see info of our X data.

# In[ ]:


X.info()


# Oops! There are 'object' type of features in our Predictor(X) dataset. Lets explore them and try to convert them into numericals.

# In[ ]:


X['Country']


# In[ ]:


# Lets see unique values of this feature.

X['Country'].unique()


# In[ ]:


# Lets see number of unique values.
X['Country'].nunique()


# In[ ]:


# Lets have look at the other object type feature.
X['Status'].unique()


# We can create dummy variables for this objects to fit well in the model. Lets build the dummy variables.

# In[ ]:


Country_dummy=pd.get_dummies(X['Country'])
# Dummy variables for Country feature.


# In[ ]:


status_dummy=pd.get_dummies(X['Status'])
# Dummy variables for status feature.


# Now lets concatenate these 'Dummies' with our X dataset.

# firstly lets drop those two object features and then concatenate it.

# In[ ]:


X.drop(['Country','Status'],inplace=True,axis=1)


# In[ ]:


X=pd.concat([X,Country_dummy,status_dummy],axis=1)


# In[ ]:


X.info()


# In[ ]:


X.head()


# Now this looks good we have about 214 columns.

# Now lets split the model

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# Lets set 30% for testing and 70% for training the model.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# Lets solve this problem by using Linear Regression and see what it gives us.

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


Linear_model= LinearRegression()


# In[ ]:


Linear_model.fit(X_train,y_train)


# We have fit our model, Now lets go for testing !

# In[ ]:


predictions1=Linear_model.predict(X_test)

# Naming it predictions1 because we are going to use some more models now.


# Looks like we have got our predictions, lets have a look at our predictions.

# In[ ]:


predictions1[0:10]

# First 10 predictions.


# Looks good but we do need some metrics to evaluate our model. In this Regression tasks nothing better than using RMSE. Lets Examine our model using RMSE.

# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


print(mean_squared_error(y_test,predictions1)**(0.5))


# Now lets have a looks at its R Square Value.

# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_test,predictions1)


# Looks like pretty good score ! Now lets see how it works if we use Ridge and Lasso Regression models.

# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


ridge_model=Ridge()


# In[ ]:


ridge_model.fit(X_train,y_train)


# In[ ]:


predictions2=ridge_model.predict(X_test)


# In[ ]:


print(mean_squared_error(y_test,predictions2)**(0.5))


# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


lasso_model=Lasso(alpha=0.00000001)

# Alpha value here was selected after choosing 8 different combinations like 0.1,0.001,0.0001...etc.


# In[ ]:


lasso_model.fit(X_train,y_train)


# In[ ]:


predictions3=lasso_model.predict(X_test)


# In[ ]:


print(mean_squared_error(y_test,predictions3)**(0.5))


# We can see Lasso almost all reached Linear regression model by using alpha=0.00000001 that means using Linear Regression is almost all ok.

# ### When to choose Linear Regression over Lasso ?

# - Even though we had good score by using Lasso we choose Linear regression over it.

# - As we choose more generalised method the flexibility in model decreases which indirectly results in the interpretability  of the model.

# - Interpretability in the model is much useful when we are solving problems, where our main goal would be knowing the relations in between the features.

#  But Lasso is also very powerful when we have alot of attributes in our data when p>n.

# Where, 'p' stands for no. of features and 'n' stands for no. of observations.

# ### Other methods to improve our model:

# - We can use interactions when the data has alot of correlation in between the features.

# - Some times it is better to have more Adjusted R2, AIC(Akaike information Criterion), BIC(Bayesian information criterion) for judging our model

# This was all about the project. 

# # THANK YOU !

# Please give this is an upvote if you really found this helpful.
