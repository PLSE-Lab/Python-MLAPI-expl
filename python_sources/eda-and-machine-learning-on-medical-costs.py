#!/usr/bin/env python
# coding: utf-8

# # Medical Costs : How your profile affects your medical charges?
# ![](http://github.com/GrejSegura/MyProjects/blob/master/MedicalInsuranceAnalysis/img/hosp.png)
# 
# - <a href='#ov'>i. Overview</a>  
# - <a href='#dta'>ii. Loading the Data</a>  
# - <a href='#eda'>1. Exploratory Data Analysis</a>  
#     - <a href='#cat'>1.2  Examining the Relationship of Charges to the Categorical Features</a>
#     - <a href='#gender'>1.3  Charges Between Gender</a>
#     - <a href='#smoker'>1.4  Charges Between Smoker and non-Smoker</a> 
#     - <a href='#regions'>1.5  Charges Among Regions</a> 
#     - <a href='#other'>1.6  In Relation to Other Variables</a> 
#     - <a href='#smokervsnon'>1.7  Smokers vs. Non-smokers</a>
# - <a href='#preprop'>2. Pre-Processing the Data</a> 
# - <a href='#linear'>3. Quantifying the effect of the features to the medical charges</a> 
# - <a href='#machine'>4. Basic Machine Learning: Comparison Between Selected Regression Models</a> 
#     - <a href='#compare'>4.1 Comparison Between Models</a> 
# - <a href='#conclusion'>Conclusion</a>

# ## <a id='ov'>i. Overview</a>
# 
# This data analysis aims to explore the factors affecting the medical costs and eventually create a predictive model.

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')


# ## <a id='dta'>ii. Loading the Data</a>
# 

# In[4]:


data = pd.read_csv("../input/insurance.csv")


# ## <a id='eda'>1. Exploratory Data Analysis</a>

# In[5]:


head = data.head(5)
print(head)


# In[5]:


describe = data.describe()
print(describe)


# The average insurance premium is at $13,270.

# In[6]:


sex = data.groupby(by = 'sex').size()
print(sex)
smoker = data.groupby(by = 'smoker').size()
print(smoker)
region = data.groupby(by = 'region').size()
print(region)


# The data is very much balanced between gender and region. On the other hand, non-smokers outnumber the smokers.

# ### <a id='cat'>1.2  Examining the Relationship of Charges to the Categorical Features</a>
# Let us first examine the distribution of charges.

# In[7]:


## check the distribution of charges
distPlot = sns.distplot(data['charges'])
plt.title("Distirbution of Charges")
plt.show(distPlot)


# The graph shows it is skewed to the right. We can tell visually that there may be outliers (the maximum charge is at $63,770). Let us examine again this time between the groups.

# #### <a id='gender'>1.3  Charges Between Gender</a>

# In[8]:


## check charges vs features
meanGender = data.groupby(by = "sex")["charges"].mean()
print(meanGender)
print(meanGender["male"] - meanGender["female"])
boxPlot1 = sns.violinplot(x = "sex", y = "charges", data = data)


# There is not much difference between gender based on the violin plot. For males, the average charge is "slightly" higher compared to female counterparts with the difference of around $1387.

# #### <a id='smoker'>1.4 Charges between Smokers and non-Smokers</a>

# In[9]:


meanSmoker = data.groupby(by = "smoker")["charges"].mean()
print(meanSmoker)
print(meanSmoker["yes"] - meanSmoker["no"])
boxPlot2 = sns.violinplot(x = "smoker", y = "charges", data = data)


# Ok, so there's around $23,615 difference between smokers and non-smokers. Smoking is very expensive indeed.

# #### <a id='regions'>1.5 Charges Among Regions</a>

# In[10]:


meanRegion = data.groupby(by = "region")["charges"].mean()
print(meanRegion)
boxPlot3 = sns.violinplot(x = "region", y = "charges", data = data)


# As with the gender, region groups also does not show much difference between them based on the plot. Even so, the individuals from the Southeast has charged more on there bills. The highest charged individual also lives in the region as shown in the chart.

# #### <a id='other'>1.6  In Relation to Other Features</a>

# The following shows the relationship of the medical charges to other numerical variables.

# In[11]:


pairPlot = sns.pairplot(data)


# Let us focus on the following relationships (first 3 charts in the bottom row):
# 1. charges vs age - it is apparent the that charges are higher to older individuals.
# 2. charges vs bmi - BMIs greater than 30 is considered obesed. The chart shows a group of individuals with BMI > 30 are charged higher.
# 3. charges vs no. children - those who has more children tends to have been charged lower than those who don't have.

# #### <a id='smokervsnon'> 1.7 Smokers vs Non- Smokers</a>

# Based on the violin plot we have noticed a big difference in charges between the smokers and non-smokers. Let us look further on this relationship complemented with other numerical variables.

# In[12]:


sns.set(style = "ticks")
smokerPairs = sns.pairplot(data, hue = "smoker")


# Focusing again on the first 3 charts in the bottom row, we can say that the higher amount of charges are dominated by blue points which are represented by smokers.

# ### <a id='preprop'>2. Pre-Processing the Data</a>
# 
# The data involves categorical variables which need to be dummified/binarized. The continuous variables are also scaled to have a more robust model in the later part.

# In[13]:


## Dummify sex, smoker and region
scaleMinMax = MinMaxScaler()
data[["age", "bmi", "children"]] = scaleMinMax.fit_transform(data[["age", "bmi", "children"]])
data = pd.get_dummies(data, prefix = ["sex", "smoker", "region"])
## retain sex = male, smoker = yes, and remove 1 region = northeast to avoid dummytrap
data = data.drop(data.columns[[4,6,11]], axis = 1)
head = data.head()
print(head)


# ### <a id='linear'> 3. Quantifying the effect of the features to the medical charges</a> 
# 
# We have already visualized the relationship of the variables to the charges. Now we will further investigate by looking at the relationships using multiple linear regression. Remember that the aim of this section is to quantify the relationship and not to create the prediction model. Let us first create a training and testing data set to proceed.
# 
# Based on the visualization, we can make a couple of hypothesis about the relationship.
# 1. There is no real difference in charges between gender or regions.
# 2. The charge for smokers are very much higher than the non-smokers.
# 3. The charge gets higher as the individual gets older.
# 4. The charge gers higher as the individual reaches over 30BMI.
# 5. Lastly, the charge is higher for those who have fewer number of children.

# In[14]:


dataX = data.drop(data.columns[[3]], axis = 1)
dataY = data.iloc[:, 3]
X_train, x_test, Y_train, y_test = train_test_split(dataX, dataY, random_state = 0)


# In[15]:


import statsmodels.api as sm
from scipy import stats

X_train2 = sm.add_constant(X_train)
linearModel = sm.OLS(Y_train, X_train2)
linear = linearModel.fit()
print(linear.summary())


# So we have generated a linear model. Let us see how our initial hypothesis fared with the actual result.
# 
# 1. There is no real difference in charges between gender or regions.<br>
#    Result: The p-value is 0.973 indicating there is no statistical difference between the gender or region group.<br>
# 2. The charge for smokers are very much higher than the non-smokers.<br>
#     Result: The p-value is 0.000 which indicates that there is a difference between the group.<br>
# 3. The charge gets higher as the individual gets older.</b>
#     Result: The p-value is 0.000 which indicates that the charge is higher as the individual gets older.<br>
# 4. The charge gers higher as the BMI gets higher.<br>
#     Result: The p-values is 0.000 which indicates that the charge is higher as the BMI gets higher.<br>
# 5. Lastly, there is significant decrease in charges as the number of children increases.<br>
#     Result: The p-value is 0.007. Interestingly, the coefficient is 2,211 which means that the charge gets higher as the individual has more number of childre. The initial hypothesis is incorrect. This is essentially the reason why we can't solely rely on visualization in generating conclusions.

# ### <a id='machine'>4. Basic Machine Learning: Comparison Between Selected Regression Models</a> 

# In this section, we will create regression models and try to compare there robustness given the data. The models considered are Linear Regression, Ridge, LASSO, and ElasticNet.

# This is basically what the following code does line-by-line:
#     1. import the library for the model
#     2. call the model
#     3. fit the model
#     4. predict the model using the test data
#     5. get the mean squared error
#     6. calculate the root mean square error
#     7. get the R-squared value

# ##### Split the data into Train/Test data set

# In[16]:


## try Linear Regression ##
from sklearn.linear_model import LinearRegression
linearModel = LinearRegression()
linear = linearModel.fit(X_train, Y_train)
linearPred = linear.predict(x_test)
mseLinear = metrics.mean_squared_error(y_test, linearPred)
rmseLinear = mseLinear**(1/2)


# In[17]:


from sklearn.linear_model import Ridge
ridgeModel = Ridge()
ridge = ridgeModel.fit(X_train, Y_train)
ridgePred = ridge.predict(x_test)
mseRidge = metrics.mean_squared_error(y_test, ridgePred)
rmseRidge = mseRidge**(1/2)


# In[18]:


from sklearn.linear_model import Lasso
lassoModel = Lasso()
lasso = lassoModel.fit(X_train, Y_train)
lassoPred = lasso.predict(x_test)
mseLasso = metrics.mean_squared_error(y_test, lassoPred)
rmseLasso = mseLasso**(1/2)


# In[19]:


from sklearn.linear_model import ElasticNet
elasticNetModel = ElasticNet(alpha = 0.01, l1_ratio = 0.9, max_iter = 20)
ElasticNet = elasticNetModel.fit(X_train, Y_train)
ElasticNetPred = ElasticNet.predict(x_test)
mseElasticNet = metrics.mean_squared_error(y_test, ElasticNetPred)
rmseElasticNet = mseElasticNet**(1/2)


# ### <a id='compare'>4.1 Comparing the Models</a> 

# In[20]:


performanceData = pd.DataFrame({"model":["linear", "lasso", "ridge", "elasticnet"], "rmse":[rmseLinear, rmseLasso, rmseRidge, rmseElasticNet]})
print(performanceData)


# Based on the table above, linear regression has a slight edge among the models considered having the least RMSE. This is not surprising as the other 3 models are known to be more robust when there are quite a number of features. We only have 8 this time.

# ### <a id='conclusion'>Conclusion</a> 
# 
# We have found out that region and gender does not bring significant difference on charges among its groups. Age, BMI, number of children and smoking are the once that drives the charges. The statistical relationship between number of children and charges is surprisingly different from our visualization.
# Meanwhile, linear regression has edged the regularized regression models in giving the best prediction. This proves that regularized regression models are not guaranteed to be superior to linear regressions.
