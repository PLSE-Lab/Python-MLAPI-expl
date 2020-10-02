#!/usr/bin/env python
# coding: utf-8

# *In this notebook, I have analyzed Airbnb data for NYC. I preprocessed data as and when required. For some columns, I created dummies so as to run regression models. I started out with OLS and improved it with Lasso regularization. I got the least RMSE with Random forest regressor. I have explained all the steps I did as well as interpretation of the model results. Please comment if you have any questions or feedback!*
# If you found this notebook useful, do leave an UPVOTE!

# ![](https://ichef.bbci.co.uk/news/1024/cpsprodpb/3E5D/production/_109556951_airbnb.png)

# In[ ]:


from sklearn import preprocessing, metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
import statsmodels.api as sm
from scipy import stats


# # Loading the data

# In[ ]:


dataset = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


dataset


# In[ ]:


### Check data types of all columns
dataset.dtypes


# Checking null values.
# Since last_review and reviews per month have same null records, lets assume that that listing never got ant review

# In[ ]:



dataset.isnull().sum()


# Let's replace reviews per month Nan by zero and null name and host_name by NoName. Also, replace last review with "Not reviewed"

# In[ ]:



dataset.fillna({'reviews_per_month':0}, inplace=True)
dataset.fillna({'name':"NoName"}, inplace=True)
dataset.fillna({'host_name':"NoName"}, inplace=True)
dataset.fillna({'last_review':"NotReviewed"}, inplace=True)


# Time to check results

# In[ ]:



dataset.isnull().sum()


# Looking at the price column

# In[ ]:



dataset["price"].describe()


# We see that the average price is 152. Price varies between 0 to 10K

# In[ ]:


### See the distribution of price
hist_price=dataset["price"].hist()
### We observe that most listings have price less than $1000


# In[ ]:


### Lets plot histogram for prices less than $2000
hist_price1=dataset["price"][dataset["price"]<1000].hist()
### This give a clearer picture!


# How many listings have price more than 1000?

# In[ ]:



dataset[dataset["price"]>1000]


# 239 listings have price per day > 1000. These are either super lavish listings or there was an error during input. Nonetheless, since this records are skewing our data a lot, we will treat them as outliers and drop them.

# In[ ]:


dataset=dataset[dataset["price"]<1000]


# In[ ]:


### We see a more Gaussian distribution here
hist_price2=dataset["price"][dataset["price"]<250].hist()


# In[ ]:


### We use 250 as threshold price 
dataset=dataset[dataset["price"]<250]


# In[ ]:


### Looking at the price column again
dataset["price"].describe()


# We see that the average price is 107. Price varies between 0 to 249

# In[ ]:


###There are 221 unique neighbourhoods in NYC as per this data set. Most listings are in Williamsburg
dataset['neighbourhood'].value_counts()


# In[ ]:


### Count how many neighbourhoods appear more than 200
dfnh =dataset.groupby("neighbourhood").filter(lambda x: x['neighbourhood'].count() > 200)


# In[ ]:


### Most data is covered. 
len(dfnh["neighbourhood"])


# In[ ]:


### Count how many neighbourhoods appear only once
dfnh =dataset.groupby("neighbourhood").filter(lambda x: x['neighbourhood'].count() == 1)
len(dfnh["neighbourhood"])


# In[ ]:


###Lets look at neighbourhood groups
dataset['neighbourhood_group'].value_counts()


# There are five major neighbourhood groups in NYC with Manhattan and Brooklyn accounting for 85% of the listings

# In[ ]:


### Lets see the average listing price by neighbourhood group
ng_price=dataset.groupby("neighbourhood_group")["price"].mean()


# In[ ]:


### Manhattan is most expensive and Bronx is the least expensive place to live
ng_price


# In[ ]:


### Lets see the distributuion of price and neighbourhood group. 
plott=sns.catplot(x="neighbourhood_group",y="price",hue="room_type", kind="swarm", data=dataset)
plott


# Here we can note that Brooklyn and Manhattan tend to have more listings with price>150.
# We also note thar most listings above price>100 are entire home type followed by private room and shared room which is the cheapest.

# In[ ]:


### Checking if there are duplicate host_ids and whats is the maximum number of listings per host_id
df = dataset.groupby(["host_id"])
max(df.size())


# In[ ]:


## Here we can see that 32K host_ids are unique appearing only once whereas some host_ids appear as much as 238 times
df.size().value_counts().head()


# In[ ]:


df.size().value_counts().tail()


# In[ ]:


### Finding the host_id with maximum listings
host_id_counts = dataset["host_id"].value_counts()
max_host = host_id_counts.idxmax()
max_host


# In[ ]:


###We see that Sonder(NYC) has the max number of listings
dataset[dataset["host_id"]==219517861]


# Listing id and Host name are not useful for our analysis so I will drop them

# In[ ]:



dataset = dataset.drop(columns = ["id","host_name"])


# In[ ]:


### Let's Analyse the listing name column
dataset["name_length"]=dataset['name'].map(str).apply(len)


# In[ ]:


###Max and Min name length
print(dataset["name_length"].max())
print(dataset["name_length"].min())
print(dataset["name_length"].idxmax())
print(dataset["name_length"].idxmin())


# In[ ]:


### Max name 
dataset.at[25832, 'name']


# In[ ]:


###Min name
dataset.at[4033, 'name']


# In[ ]:


### Let's figure if name length has an impact on how much it is noticed. We can assume higher number of reviews mean more people lived there and hence more people "noticed" the listing
#dataset["name_length"].corr(dataset["number_of_reviews"])
dataset.plot.scatter(x="name_length", y ="number_of_reviews" )


# In[ ]:


###There is hardly any relationship there. Lets try between price and name length 
dataset[dataset["name_length"]<50].plot.scatter(x="price", y ="name_length")
#dataset["name_length"].corr(dataset["price"])


# In[ ]:


dataset.name_length.hist()


# In[ ]:


### Lets look at room_type variable
dataset['room_type'].value_counts()
### Most listings are either Entire home or Private room


# In[ ]:


### Average price per room_type
rt_price = dataset.groupby("room_type")["price"].mean()


# In[ ]:


### Entire room has the highest price and shared room has lowest avg price which makes sense.
rt_price


# In[ ]:


### Analysing minimum nights

dataset["minimum_nights"].describe()


# Again, range is between 1 night to 1250 nights. Quite odd, lets investigate

# In[ ]:


### Analysing minimum nights
### We see most values are between 1 to 100
hist_mn=dataset["minimum_nights"].hist()
hist_mn


# In[ ]:


### Closer look
hist_mn1=dataset["minimum_nights"][dataset["minimum_nights"]<10].hist()
hist_mn1


# In[ ]:


dataset["minimum_nights"][dataset["minimum_nights"]>30]


# In[ ]:


### We replace all records with min nights > 30 by 30
dataset.loc[(dataset.minimum_nights >30),"minimum_nights"]=30


# In[ ]:


hist_mn2=dataset["minimum_nights"][dataset["minimum_nights"]<30].hist()
hist_mn2


# In[ ]:


### Does minimum_nights have impact on price?
dataset["minimum_nights"].corr(dataset["price"])


# In[ ]:


###Finally lets analyse availability_365 column
dataset["availability_365"].describe()


# In[ ]:


hist_av=dataset["availability_365"].hist()
hist_av


# In[ ]:


### After analysis, I have decided to drop these columns as they will not be useful in prediction
dataset.drop(["name",'last_review',"latitude",'longitude'], axis=1, inplace=True)


# In[ ]:


### Dropping host_id
dataset.drop(["host_id"], axis=1, inplace=True)


# In[ ]:


### Plotting correlation matrix 
corr = dataset.corr(method='pearson')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
dataset.columns


# In[ ]:


### Lets check out data one more time before beginning prediction. 
###Looks good!
dataset.dtypes


# ## Let us start with basic Linear Regression to create a base line model 

# Making dummies for neighbourhood group and room_type

# In[ ]:


## lets try without neighbourhood column

dataset_onehot1 = pd.get_dummies(dataset, columns=['neighbourhood_group',"room_type"], prefix = ['ng',"rt"],drop_first=True)
dataset_onehot1.drop(["neighbourhood"], axis=1, inplace=True)


# In[ ]:


### Checking dataframe shape
dataset_onehot1.shape


# In[ ]:


X1= dataset_onehot1.loc[:, dataset_onehot1.columns != 'price']


# In[ ]:


Y1 = dataset_onehot1["price"]


# ### Splitting into training and testing data

# In[ ]:



x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.20, random_state=42)


# In[ ]:


### Fitting Linear regression
reg1 = LinearRegression().fit(x_train1, y_train1)


# In[ ]:


### R squared value
reg1.score(x_train1, y_train1)


# In[ ]:


### Coefficients
reg1.coef_


# In[ ]:


### Predicting 
y_pred1 = reg1.predict(x_test1)


# In[ ]:


Coeff1 = pd.DataFrame(columns=["Variable","Coefficient"])
Coeff1["Variable"]=x_train1.columns
Coeff1["Coefficient"]=reg1.coef_
Coeff1.sort_values("Coefficient")


# In[ ]:


### Calculate RMSE
rmse1 = np.sqrt(metrics.mean_squared_error(y_test1, y_pred1))
rmse1


# In[ ]:


### Taking a closer look at the estimates
X2 = sm.add_constant(x_train1)
est = sm.OLS(y_train1, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:


## No of reviews and ng_Staten Island is not significant and does not help our model much. Drop it
x_train1.drop(["number_of_reviews","ng_Staten Island"], axis=1,inplace=True)
X2 = sm.add_constant(x_train1)
est = sm.OLS(y_train1, X2)
est2 = est.fit()
print(est2.summary())
### Does not improve our model much


# # Model Interpretation
# Now that we have the results of our model, lets try to interpret it in detail.
# 1. We first look at the Adjusted R square value since this is a Multiple linear regression. It tell us that our independent variables can explain 50.3% of variations in our dependent variable, which is price.
# 2. The constant or the y intercept has a value of 109.56. This means that putting all other x variables at 0, an Entire Apt/Home in Bronx will have a predicted price of 109.56. Remember when we created dummy variables we dropped one dummy from each column which we use as reference.
# 3. Let's now look at the coefficients. The coefficient of ng_Manhattan is 44.09. We interpret as: Everything else being constant, an Entire Apt/ Home in Manhattan will cost 44.09 more that same in Bronx. We can similarly interpret coefficient of minimum night as: With everything else being constant, with every one unit increase in minimum number of nights, the predicted price decreases by 0.8075.
# 4. The std error is nothing but sample standard deviation for each variable. the t column shows the value of t statistic which is the z score of the sample variable. Z score tells us how far a sample is from its mean. A Z score of 2 tells us that the sample is two standard deviation away from the mean.
# 5. P values suggests how significant these estimates are. Considering alpha of 0.05, we can reject the Null hypothesis for all variables except number_of_reviews, and ng_Staten Island. Alpha is the degree of error we are willing to accept. we have chosen alpha as 5% which is also most commonly used.
# 6. The confidence intervals show the upper bound and lower bound for the TRUE POPULATION coefficient with 95% confidence.
# 
# 

# ### Lets try to use the neighbourhood variable. It has more than 200 distinct values. 
# ### Hence, when we create dummies we will have large number of variables.

# In[ ]:



dataset_onehot2 = pd.get_dummies(dataset, columns=['neighbourhood_group',"neighbourhood","room_type"], prefix = ['ng',"nh","rt"],drop_first=True)


# In[ ]:


dataset_onehot2.shape


# In[ ]:


XL1= dataset_onehot2.loc[:, dataset_onehot2.columns != 'price']
YL1 = dataset_onehot2["price"]
x_trainL11, x_testL11, y_trainL11, y_testL11 = train_test_split(XL1, YL1, test_size=0.20, random_state=42)


# We will use Lasso regression because it has the ability to nullify parameters that do not improve the model.
# Also, the dataset isn't large enough and hence Lasso is a good choice as it would add a little bit bias but reduce variance greatly.
# Starting with alpha=0.1
# We can use crossvalidation and check for many values for alpha to find best one, but I will save this process for the next model
# which I think will procduce better results.

# In[ ]:



regL1 = Lasso(alpha=0.01)
regL1.fit(x_trainL11, y_trainL11) 


# In[ ]:


### R squared
### This regularised model did way better than normal linear regression
regL1.score(x_trainL11, y_trainL11)


# In[ ]:


### RMSE
### Smaller value than earlier
y_predL1= regL1.predict(x_testL11)
print(np.sqrt(metrics.mean_squared_error(y_testL11,y_predL1)))


# In[ ]:


### We can see that some parameters have zero coefficients.
regL1.coef_


# In[ ]:


CoeffLS1 = pd.DataFrame(columns=["Variable","Coefficients"])
CoeffLS1["Variable"]=x_trainL11.columns
CoeffLS1["Coefficients"]=regL1.coef_
CoeffLS1.sort_values("Coefficients", ascending = False)


# In[ ]:


### Finally, lets try Random forest regressor which I believe will give best results


# In[ ]:


### Initially, lets build a tree without any constraints.
regrRM = RandomForestRegressor(n_estimators=300)
regrRM.fit(x_trainL11, y_trainL11)


# In[ ]:


### We get R squared value at 93.6%! There is obviously a problem of overfitting:(

print(regrRM.score(x_trainL11, y_trainL11))
y_predL1= regrRM.predict(x_testL11)
print(np.sqrt(metrics.mean_squared_error(y_testL11,y_predL1)))


# In[ ]:


### Using feature importance, we can see which feature had most weight
regrRM.feature_importances_


# In[ ]:


CoeffRM1 = pd.DataFrame(columns=["Variable","FeatureImportance"])
CoeffRM1["Variable"]=x_trainL11.columns
CoeffRM1["FeatureImportance"]=regrRM.feature_importances_
CoeffRM1.sort_values("FeatureImportance", ascending = False)


# In[ ]:


regrRM.get_params()


# ### Lets see what we can do to prevent overfitting
# 1. We will set max depth to 50. This ensures that branching stops after 50th branching, otherwise each sample may have its branch and overfit.
# 2. We will use min_samples_split as 5. The default value is 2. This means that each internal node will split as long as it has a minimum of two sample. We dont want that!
# 3. We will use min_samples_leaf as 4. The default is 1. This means that a node is considered leaf node if it has just one sample. This can cause severe overfitting!

# In[ ]:



regrRM2 = RandomForestRegressor(n_estimators=200, max_depth = 50, min_samples_split = 5,min_samples_leaf =4)
regrRM2.fit(x_trainL11, y_trainL11)


# In[ ]:


### We get a smaller value for R squared
print(regrRM2.score(x_trainL11, y_trainL11))
y_predL1= regrRM2.predict(x_testL11)
print(np.sqrt(metrics.mean_squared_error(y_testL11,y_predL1)))


# In[ ]:


CoeffRM2 = pd.DataFrame(columns=["Variable","FeatureImportance"])
CoeffRM2["Variable"]=x_trainL11.columns
CoeffRM2["FeatureImportance"]=regrRM2.feature_importances_
CoeffRM2.sort_values("FeatureImportance", ascending = False)


# ## CrossValidation

# In[ ]:


### To find best values for the RF parameters, let us use cross validation
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 6)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
# Create the random grid
rm_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


print(rm_grid)


# In[ ]:


import time


# In[ ]:


# Use the random grid to search for best hyperparameters
t1 = time.time()
rf2 = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
rf2_random = RandomizedSearchCV(estimator = rf2, param_distributions = rm_grid, n_iter = 180, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf2_random.fit(x_trainL11, y_trainL11)
t2 =time.time()


# In[ ]:


### Time taken
(t2-t1)/60


# In[ ]:


### Here we can see Best parameters for the best model
rf2_random.best_params_


# In[ ]:


### Final R squared value
rf2_random.score(x_trainL11, y_trainL11)


# In[ ]:


### We finally have the least RMSE among all model!
y_predL1= rf2_random.predict(x_testL11)
print(np.sqrt(metrics.mean_squared_error(y_testL11,y_predL1)))


# In[ ]:


### Finally lets compare all models
### Including models from my previous project with pyspark
rmsedt = {"Model":["RF1_Sprk","RF2_Sprk","RF3_Sprk","LR","L1","RFR"],"RMSE":[71.55745125705758,65.7207885074504
,62.51297007998151,37.68939882420686,35.12428625156702,34.05098593042094]}
rmsedf = pd.DataFrame(rmsedt)
rsqdt = {"Model":["LR","L1","RFR"],"RSquared":[50.3,56.7,77.8]}
rsqdt = pd.DataFrame(rsqdt)


# In[ ]:


sns.catplot(x="Model", y="RMSE", linestyles=["-"],
            kind="point", data=rmsedf);


# In[ ]:


sns.catplot(x="Model", y="RSquared", linestyles=["--"], color ="green", kind="point", data=rsqdt);


# In[ ]:




