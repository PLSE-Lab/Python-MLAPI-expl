#!/usr/bin/env python
# coding: utf-8

# (Click on the below links to navigate to different sections of the notebook)
# # **Overview**  
# - <a href="#1"> 1. Importing Data</a>
# - <a href="#2"> 2. Preprocessing the Dataset</a>
#   -  <a href="#2-1" > 2.1 Check for NULLS (missing data)</a>
#   -  <a href="#2-2" > 2.2 Filling the missing data</a>
#   -  <a href="#2-3" > 2.3 Check the datatype of columns</a>
#   -  <a href="#2-4" > 2.4 Remove Outliers</a>
#   -  <a href="#2-5" > 2.5 Feature Engineering</a>
# - <a href="#3"> 3. Explanatory Data Analysis </a>
#   -  <a href="#3-1" > 3.1 Distribution of "KMs Driven", "Price", "Year" <a>
#   -  <a href="#3-2" > 3.2 Pairplot of different features with "Price"</a>
#   -  <a href="#3-3" > 3.3 Percentage of each "fuel" types for "Used" condition</a>
#   -  <a href="#3-4" > 3.4 Percentage of each "fuel" types for "New" condition</a>
#   -  <a href="#3-5" > 3.5 Count plot of "Used" and "New" having Transaction Type "Cash" for Fuel features</a>
#   -  <a href="#3-6" > 3.6 Variation of Price with Year</a>
#   -  <a href="#3-7" > 3.7 Encoding the categorical data (one hot Encoding)</a>
# - <a href="#4"> 4. Divide the data into training and testing data </a>
# - <a href="#5"> 5. Model</a>
#   -  <a href="#5-1" > A. Random forest Regressor</a>
#      -  <a href="#5-1-1" > Grid searching of hyperparameters</a>
#      -  <a href="#5-1-2" > K-Fold cross validation</a>
#      -  <a href="#5-1-3" > Correlation Matrix</a>
#      -  <a href="#5-1-4" > A. Random forest Regressor</a>
#   -  <a href="#5-2" > B. Linear Regression</a>
#      -  <a href="#5-2-1" > Train R squared score</a>
#      -  <a href="#5-2-2" > Root mean squared error</a>
#      -  <a href="#5-2-3" > K-Fold Cross Validation</a>
#      -  <a href="#5-2-4" > Scatter Plot of Predicted Price Vs Actual Price</a>  
#      -  <a href="#5-2-5" > Residual Plot</a> 
# - <a href="#6"> 6. Final prediction and Conclusion</a>
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error 

import seaborn as sns

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ## <a id="1"> 1. Importing Data </a>

# In[ ]:


df2 = pd.read_csv('../input/OLX_Car_Data_CSV.csv',encoding= 'latin1')
df2 = df2.sample(frac=1).reset_index(drop=True)# shuffle
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


display(df2.head(5))
df2.columns


# ## <a id="2" > 2. Preprocessing the Dataset</a>

# #### Checking values of categorical attributes

# In[ ]:


cat_val = ['Brand', 'Condition', 'Fuel', 'Model',
       'Registered City', 'Transaction Type']

for col in cat_val:
    print ([col]," : ",df2[col].unique())


# ### <a id="2-1" > 2.1 Check for NULLS (missing data)</a>

# Visual display of null values in each columns for quick overview

# In[ ]:


sns.heatmap(pd.DataFrame(df2.isnull().sum()),annot=True,
            cmap=sns.color_palette("cool"),linewidth=1,linecolor="white")


# In[ ]:


print("Number of Null values in train dataset\n")
print(df2.isnull().sum(axis = 0))


# There are too many missing values in the dataset. So instead of removing them <br>we will fill it median, average or most frequent
# data.<br>
# We will consider "NaN" as some data point. So let's replace it with word "unknown" <br>to treat it as a categorical 
# value in its respective columns.

# **Replace the NaN-Values with dummies <br>Replacing "NaN" with "unknown" to treat it as a categorical values in their respective columns**

# In[ ]:


#Train dataset
df2['Brand'].fillna(value='unknown', inplace=True)
df2['Condition'].fillna(value='unknown', inplace=True)
df2['Fuel'].fillna(value='unknown', inplace=True)
df2['Model'].fillna(value='unknown', inplace=True)
df2['Registered City'].fillna(value='unknown', inplace=True)
df2['Transaction Type'].fillna(value='unknown', inplace=True)


# In[ ]:


print("Number of Null values in train dataset\n")
print(df2.isnull().sum(axis = 0))


# Let's now fill the missing values in the column "KMs Driven" with average values.

# ### <a id="2-2" > 2.2 Filling the missing data</a>

# In[ ]:


#Train dataset
df2['KMs Driven'].fillna((df2['KMs Driven'].mean()),inplace = True) #average data 
# df2['Year'].fillna(df2['Year'].value_counts().index[0],inplace = True) #most frequent data


# In[ ]:


print("Number of Null values in train dataset\n")
print(df2.isnull().sum(axis = 0))


# In[ ]:


df2=df2.dropna() #drop rows with atleast a column with missing values


# In[ ]:


print("Train : ", df2.shape)


# ### <a id="2-3" > 2.3 Check the datatype of columns</a>

# In[ ]:


df2.dtypes


# In[ ]:


#Train dataset
df2.describe()


# ### <a id="2-4" > 2.4 Remove Outliers</a>

# #### Distribution of "Price" in train dataset

# In[ ]:


sns.distplot(df2["Price"])


# #### Distribution of "KMs Driven" in train dataset

# In[ ]:


sns.distplot(df2["KMs Driven"])


# #### Distribution of "Year" in train dataset

# In[ ]:


sns.distplot(df2["Year"])


# Determine outliers in dataset

# In[ ]:


cols=['Price']


for i in cols:
    quartile_1,quartile_3 = np.percentile(df2[i],[25,75])
    quartile_f,quartile_l = np.percentile(df2[i],[1,99])
    IQR = quartile_3-quartile_1
    lower_bound = quartile_1 - (1.5*IQR)
    upper_bound = quartile_3 + (1.5*IQR)
    print(i,lower_bound,upper_bound,quartile_f,quartile_l)

    df2[i].loc[df2[i] < lower_bound] = quartile_f
    df2[i].loc[df2[i] > upper_bound] = quartile_l


# In[ ]:


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

df2=remove_outlier(df2, 'Price')
# df2=remove_outlier(df2, 'KMs Driven')
# df2=remove_outlier(df2, 'Year')


# #### After removing outlier in 'Price'

# In[ ]:


sns.distplot(df2["Price"])


# ### <a id="2-5" > 2.5 Feature Engineering</a>

# In[ ]:


df2['Damaged'] = np.where(df2['KMs Driven']> 2000000, 'too_old', 'old')


# In[ ]:


df2.head()


# Price of cars whose price is more than 1500000 it considered as expensive.

# ## <a id="3">3. Explanatory Data Analysis </a>

# ### <a id="3-1" > 3.1 Distribution of "KMs Driven", "Price", "Year" in train dataset<a>

# In[ ]:


#Train Dataset
df2.hist(bins = 50 , figsize = (20,20))
plt.show()


# ### <a id="3-2" > 3.2 Pairplot of different features with "Price"</a>

# Pairplot to visualize the realtionship between the target and independent features

# In[ ]:



sns.pairplot(df2, x_vars=['Brand', 'Condition', 'Fuel', 'KMs Driven', 'Model',
       'Registered City', 'Transaction Type', 'Year'], y_vars=["Price"],aspect=1);


# It can be observed that there is some linear relationship (rougly) between dependent variable "Price"
# and independent variables "Brand", "KMs Driven", "Registered City" and "Year" .
# 
# So let's draw these plots separately for clear view.

# In[ ]:


plt.scatter(df2['Brand'], df2['Price'], color='blue')
plt.title('Price Vs Brand', fontsize=14)
plt.xlabel('Brand', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.grid(True)
plt.show()


# In[ ]:


plt.scatter(df2['KMs Driven'], df2['Price'], color='red')
plt.title('Price Vs KMs Driven', fontsize=14)
plt.xlabel('KMs Driven', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.grid(True)
plt.show()


# In[ ]:


plt.scatter(df2['Registered City'], df2['Price'], color='green')
plt.title('Price Vs Registered City', fontsize=14)
plt.xlabel('Registered City', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.grid(True)
plt.show()


# ### <a id="3-3" > 3.3 Percentage of each "fuel" types for "Used" condition</a>

# In[ ]:


df2[df2["Condition"] == "Used"]["Fuel"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["grey","orange"],startangle = 60,                                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.1,color="white")
plt.gca().add_artist(circ) 


# ### <a id="3-4" > 3.4 Percentage of each "fuel" types for "New" condition</a>

# In[ ]:


df2[df2["Condition"] == "New"]["Fuel"].value_counts().plot.pie(autopct = "%1.0f%%",colors = ["grey","orange"],startangle = 60,                                                              wedgeprops={"linewidth":2,"edgecolor":"white"},shadow =True)
circ = plt.Circle((0,0),.1,color="white")
plt.gca().add_artist(circ) 


# We can observe that new car uses CNG less than old car.

# ### <a id="3-5" > 3.5 Count plot of "Used" and "New" having Transaction Type "Cash" for Fuel features</a>

# In[ ]:


ax = sns.countplot("Condition",hue="Fuel",data=df2[df2["Transaction Type"] == "Cash"],palette=["r","b","g"])
ax.set_facecolor("white")


# ### <a id="3-6" > 3.6 Variation of Price with Year</a>

# In[ ]:


plt.figure(figsize=(18,10))
ax = sns.pointplot(df2["Year"],df2["Price"],color="w") # line is of white color
ax.set_facecolor("k") #background is black
plt.grid(True,color="grey",alpha=.3) # grid is on and its color is grey
plt.title("Average Price by year")
plt.show()


# ### <a id="3-7" > 3.7 Encoding the categorical data (one hot Encoding)</a>

# In[ ]:


# df2=df2.drop(['Transaction Type','Registered City'], axis=1)
# test=test.drop(['Transaction Type','Registered City'], axis=1)


# In[ ]:


df3=pd.get_dummies(df2,drop_first=True)
df3.head()


# ## <a id="4"> 4.  Divide the data into training and test data </a>

# In[ ]:


df_y = df3['Price'].values
df_X = df3.drop(['Price'], axis=1)


# In[ ]:


test_size = 0.30

#Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(df_X,df_y, test_size=test_size,shuffle=True, random_state = 3)


X_test.to_csv("cleaned_test_set.tsv", sep='\t', encoding='utf-8',index=False)

temp2 = pd.DataFrame(data=Y_test.flatten())
temp2.columns = temp2.iloc[0]
temp2 = temp2.reindex(temp2.index.drop(0)).reset_index(drop=True)
temp2.columns.name = None
temp2.to_csv("actual_price_test.tsv", sep='\t', encoding='utf-8',index=False)


# ## <a id="5"> 5. Model</a>

# ## <a id="5-1" > A. Random forest Regressor</a>

# In[ ]:


rf = RandomForestRegressor()

param_grid = { "criterion" : ["mse"]
              , "min_samples_leaf" : [3]
              , "min_samples_split" : [3]
              , "max_depth": [10]
              , "n_estimators": [500]}


# ### <a id="5-1-1" > Grid searching of hyperparameters</a>

# In[ ]:


gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)
gs = gs.fit(X_train, Y_train)


# In[ ]:


print(gs.best_score_)
print(gs.best_params_)
 


# In[ ]:


bp = gs.best_params_
rf_regressor = RandomForestRegressor(criterion=bp['criterion'],
                              min_samples_leaf=bp['min_samples_leaf'],
                              min_samples_split=bp['min_samples_split'],
                              max_depth=bp['max_depth'],
                              n_estimators=bp['n_estimators'])
rf_regressor.fit(X_train, Y_train)


# #### Training R squared score

# In[ ]:


print("Train R^2 Score:")
print('Score: %.2f' % rf_regressor.score(X_train, Y_train))


# ### <a id="5-1-2" > K-Fold cross validation </a>

# Now we will do cross validation . This is because we split dataset in train and test. It may happen that test and train does not have uniform distribution of samples. So to make sure our model doesn't overfit i.e to generalize it we will do cross validation.

# In[ ]:


#Predicting the Price using cross validation (KFold method)
y_pred_rf = cross_val_predict(rf_regressor, X_train, Y_train, cv=10 )

#Random Forest Regression Accuracy with cross validation
accuracy_rf = metrics.r2_score(Y_train, y_pred_rf)
print('Cross-Predicted(KFold) Random Forest Regression Accuracy: %.2f '% accuracy_rf)


# In[ ]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# #### Cross validation score

# In[ ]:


scores = cross_val_score(rf_regressor, X_train, Y_train,
                         scoring="neg_mean_squared_error", cv=5)
lin_rmse_scores = np.sqrt(-scores)

display_scores(lin_rmse_scores)


# #### Test R squared score

# In[ ]:


print("Test R^2 Score:")
print('Score: %.2f' % rf_regressor.score(X_test, Y_test))


# #### Comparison of first five predicted and actual price in train_set

# In[ ]:


y_pred=rf_regressor.predict(X_train)
y_pred[0:5]


# In[ ]:


list(Y_train[0:5])


# #### Root mean squared error

# In[ ]:


# The root mean squared error
y_pred =rf_regressor.predict(X_train)

forest_mse = mean_squared_error(Y_train, y_pred)
forest_rmse = np.sqrt(forest_mse)

print("Root Mean squared error (training): %.2f"
      % forest_rmse)


# Let's find what features are most important

# In[ ]:


ranking = np.argsort(-rf_regressor.feature_importances_)
f, ax = plt.subplots(figsize=(15, 100))
sns.barplot(x=rf_regressor.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')
ax.set_xlabel("feature importance")
plt.tight_layout()
plt.show()


# Keeping 30 most dominant features

# In[ ]:



X_train1 = X_train.iloc[:,ranking[:30]]
X_test1 = X_test.iloc[:,ranking[:30]]


# ### <a id="5-1-3" > Correlation Matrix </a>

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(X_train1.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# One thing that that the Pearson Correlation plot can tell us is that there are not too many features strongly correlated with one another. This is good from a point of view of feeding these features into our learning model because this means that there isn't much redundant or superfluous data in our training set. Here are two most correlated features are that of "Fuel_Petrol" and "Year".

# Let's run the Linear Regression to check if removing the less dominant features improved the model from earlier version.

# ## <a id="5-2" > B. Linear Regression</a>

# In[ ]:


regr = LinearRegression()

regr.fit(X_train, Y_train)


# ### <a id="5-2-1" >Train R squared score</a>

# In[ ]:


print('Train R^2 \nscore: %.2f' % regr.score(X_train, Y_train))


# ### <a id="5-2-2" >Root mean squared error</a>

# In[ ]:


# The root mean squared error
y_pred =regr.predict(X_train)

lin_mse = mean_squared_error(Y_train, y_pred)
lin_rmse = np.sqrt(lin_mse)

print("Root Mean squared error (training): %.2f"
      % lin_rmse)


# ### <a id="5-2-3" >K-Fold Cross Validation</a>

# In[ ]:


#Predicting the Price using cross validation (KFold method)
y_pred_kf = cross_val_predict(regr, X_train, Y_train, cv=10 )

#Accuracy with cross validation (KFold method)
accuracy_lf = metrics.r2_score(Y_train, y_pred_kf)
print('Cross-Predicted(KFold) Linear Regression Accuracy: %.2f' % accuracy_lf)


# #### Cross validation score

# In[ ]:


scores = cross_val_score(regr, X_train, Y_train,
                         scoring="neg_mean_squared_error", cv=5)
lin_rmse_scores= np.sqrt(-scores)

display_scores(lin_rmse_scores)


# #### Intercept and coefficients

# In[ ]:


#intercept
print('Intercept: \n', regr.intercept_)

# The coefficients
print('Coefficients: \n', regr.coef_)


# #### Comparison of first five predicted and actual price in train_set

# In[ ]:


y_pred[0:5]


# In[ ]:


list(Y_train[0:5])


# ### <a id="5-2-4" >Scatter Plot of Predicted Price Vs Actual Price</a> 

# In[ ]:


y_pred = regr.predict(X_train)
plt.figure(figsize=(12,7))
plt.grid(True)
plt.title('Scatter Plot of Predicted Price Vs Actual Price', y=1, size=20)
plt.scatter(Y_train, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predictions")


# ### <a id="5-2-5" >Residual Plot</a> 

# In[ ]:


plt.figure(figsize=(12,7)) 
plt.grid(True)
plt.title('Residual Plot for Linear Regression', y=1, size=20) 
sns.residplot(Y_train,y_pred) # regression Residual Plot for linear regression model using bootstrapping


# #### Test R squared score

# In[ ]:


print('Test R^2 \nscore: %.2f' % regr.score(X_test, Y_test))


# **Train and Cross Validation score is quite comparable. So we can say that our model in not overfitting. <br>It is generalizing better.**

# ## <a id="6"> 6. Final prediction and Conclusion</a>

# Linear Regression gives a score of  80% on final test dataset <br>
# 10-Fold Cross Validation score in case of Linear Regression =  80%<br>
# ***
# Random Forest Regressor gives a score of 85% final test dataset <br>
# 10-Fold Cross Validation score in case of Random Forest Regressor = 86%<br>

# In[ ]:


submission = rf_regressor.predict(X_test)
filename = 'submission.csv'

temp2 = pd.DataFrame(data=submission.flatten())
temp2.columns = temp2.iloc[0]
temp2 = temp2.reindex(temp2.index.drop(0)).reset_index(drop=True)
temp2.columns.name = None
temp2.to_csv("submission.tsv", sep='\t', encoding='utf-8',index=False)


# ***
