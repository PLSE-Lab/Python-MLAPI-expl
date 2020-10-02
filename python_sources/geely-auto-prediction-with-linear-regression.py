#!/usr/bin/env python
# coding: utf-8

# ### Importing And Understanding Data

# In[ ]:


# Importing all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Importing CarPrice_Assignment.csv
carDetails = pd.read_csv('../input/CarPriceAssignment.csv')


# In[ ]:


# Looking at first five rows
carDetails.head()


# In[ ]:


carDetails.info()


# In[ ]:


# Let's look at some statistical information about our dataframe.
carDetails.describe()


# ### Visualizing the data

# In[ ]:


# Let's plot a pair plot of all variables in our dataframe
sns.set(font_scale=2)
sns.pairplot(carDetails)


# In[ ]:


plt.figure(figsize = (20,10))  
sns.heatmap(carDetails.corr(),annot = True)


# ### Data Preparation

# #### Dealing with highly corelated data after data visualisation

# From above scatter plots and also the heat map as we can notice there is a high corelation between:
# 1.carlength, curbweight, wheelbase and carwidth, so we can drop 3 out of 4, so lets drop carwidth and curbweight and wheelbase
# 2.There is a high corelation of .97 between highwaympg and citympg, so lets drop highwaympg
# 

# In[ ]:


carDetails.drop(['carwidth','curbweight','wheelbase','highwaympg'], axis =1, inplace = True)
#we can also remove carID  as its just a serial number 
carDetails.drop(['car_ID'], axis =1, inplace = True)


# In[ ]:


carDetails.info()


# #### Dealing with outliers

# In[ ]:


# Plotting price 
c = [i for i in range(1,206,1)]
fig = plt.figure()
plt.scatter(c,carDetails['price'])
fig.suptitle('price vs index', fontsize=20)              # Plot heading 
plt.xlabel('index', fontsize=18)                          # X-label
plt.ylabel('price', fontsize=16)  


# In[ ]:


# # carDetails = carDetails.ix[carDetails['price'] <= 25000]
# # carDetails.describe()
# import numpy

# arr = carDetails['price']
# elements = numpy.array(arr)

# mean = numpy.mean(elements, axis=0)
# sd = numpy.std(elements, axis=0)

# final_list1 = [x for x in arr if (x < mean - 2 * sd)]
# final_list2 = [x for x in arr if (x > mean + 2 * sd)]
# print(len(final_list1))
# print(len(final_list2))

# print(final_list1)
# print(final_list2)


# carDetails = carDetails.ix[carDetails['price'] <= 30000]
# carDetails.describe()


# #### Data cleanup
# There is a variable named CarName which is comprised of two parts - the first word is the name of 'car company' and the second is the 'car model'. For example, chevrolet impala has 'chevrolet' as the car company name and 'impala' as the car model name. We need to consider only company name as the independent variable for model building. 

# In[ ]:


carDetails["CarName"] = carDetails["CarName"].str.replace('-', ' ')
carDetails.CarName.unique()

carDetails["CarName"] = carDetails.CarName.map(lambda x: x.split(" ", 1)[0])
# As we have some redundant data in carName lets fix it 
carDetails.CarName = carDetails['CarName'].str.lower()
carDetails['CarName'] = carDetails['CarName'].str.replace('vw','volkswagen')
carDetails['CarName'] = carDetails['CarName'].str.replace('vokswagen','volkswagen')
carDetails['CarName'] = carDetails['CarName'].str.replace('toyouta','toyota')
carDetails['CarName'] = carDetails['CarName'].str.replace('porcshce','porsche')
carDetails['CarName'] = carDetails['CarName'].str.replace('maxda','mazda')
carDetails['CarName'] = carDetails['CarName'].str.replace('maxda','mazda')

carDetails.CarName.unique()
# carDetails.info()


# ### Dealing with Categorical Fields 

# ##### Converting all categorical fields of two levels to binary

# In[ ]:


# Converting Yes to 1 and No to 0
carDetails['fueltype'] = carDetails['fueltype'].map({'gas': 1, 'diesel': 0})
carDetails['aspiration'] = carDetails['aspiration'].map({'std': 1, 'turbo': 0})
carDetails['doornumber'] = carDetails['doornumber'].map({'two': 1, 'four': 0})
carDetails['enginelocation'] = carDetails['enginelocation'].map({'front': 1, 'rear': 0})


# In[ ]:


carDetails.info()
# carDetails.head()


# #### Generating dummy values for categorical columns of more than 2 levels

# As we can se we have few categorial fields like carName, carbody, driveWheel, fuelsystem, cylinderNumber, engineType So lets generate dummy columns for all of these first.

# In[ ]:


df = pd.get_dummies(carDetails)
df.head()
# df.info()


# ### Rescaling the Features using Normalisation

# In[ ]:


#defining a normalisation function 
cols_to_norm = ['symboling', 'carlength', 'carheight', 
         'enginesize', 'boreratio', 'stroke', 'compressionratio','horsepower', 'peakrpm', 'citympg', 'price']
# Normalising only the numeric fields 
normalised_df = df[cols_to_norm].apply(lambda x: (x-np.mean(x))/ (max(x) - min(x)))
normalised_df.head()

df['symboling'] = normalised_df['symboling']
df['carlength'] = normalised_df['carlength']
df['carheight'] = normalised_df['carheight']
df['enginesize'] = normalised_df['enginesize']
df['boreratio'] = normalised_df['boreratio']
df['stroke'] = normalised_df['stroke']
df['price'] = normalised_df['price']
df['compressionratio'] = normalised_df['compressionratio']
df['horsepower'] = normalised_df['horsepower']
df['peakrpm']= normalised_df['peakrpm']
df['citympg'] = normalised_df['citympg']
df.head()


# ## Splitting Data into Training and Testing Sets
# 

# In[ ]:


refinedcol = df.columns
refinedcol


# In[ ]:


# Putting feature variable to X
# df.info()
# df.columns
X = df[['symboling', 'fueltype', 'aspiration', 'doornumber', 'enginelocation',
       'carlength', 'carheight', 'enginesize', 'boreratio', 'stroke',
       'compressionratio', 'horsepower', 'peakrpm', 'citympg',
       'CarName_alfa', 'CarName_audi', 'CarName_bmw', 'CarName_buick',
       'CarName_chevrolet', 'CarName_dodge', 'CarName_honda', 'CarName_isuzu',
       'CarName_jaguar', 'CarName_mazda', 'CarName_mercury',
       'CarName_mitsubishi', 'CarName_nissan', 'CarName_peugeot',
       'CarName_plymouth', 'CarName_porsche', 'CarName_renault',
       'CarName_saab', 'CarName_subaru', 'CarName_toyota',
       'CarName_volkswagen', 'CarName_volvo', 'carbody_convertible',
       'carbody_hardtop', 'carbody_hatchback', 'carbody_sedan',
       'carbody_wagon', 'drivewheel_4wd', 'drivewheel_fwd', 'drivewheel_rwd',
       'enginetype_dohc', 'enginetype_dohcv', 'enginetype_l', 'enginetype_ohc',
       'enginetype_ohcf', 'enginetype_ohcv', 'enginetype_rotor',
       'cylindernumber_eight', 'cylindernumber_five', 'cylindernumber_four',
       'cylindernumber_six', 'cylindernumber_three', 'cylindernumber_twelve',
       'cylindernumber_two', 'fuelsystem_1bbl', 'fuelsystem_2bbl',
       'fuelsystem_4bbl', 'fuelsystem_idi', 'fuelsystem_mfi',
       'fuelsystem_mpfi', 'fuelsystem_spdi', 'fuelsystem_spfi']]

# # # Putting response variable to y
y = df['price']


# In[ ]:


#random_state is the seed used by the random number generator, it can be any integer.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=100)


# #### RFE

# In[ ]:


# help(rfe)


# In[ ]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
rfe = RFE(lm, 15)             # running RFE
rfe = rfe.fit(X_train, y_train)
print(rfe.support_)           # Printing the boolean results
print(rfe.ranking_)  


# In[ ]:


X_train.columns[rfe.support_]


# In[ ]:


#variables that are to be dropped
X_train.columns
col = X_train.columns[~rfe.support_]
col


# In[ ]:


print("Before droping of columns")
X_train.columns
X_train1 = X_train.drop(col,1)
print("After Droping of columns")
X_train1.columns

df.head()


# ### Building Model By droping columns after RFE 

# In[ ]:


# Adding a constant variable 
import statsmodels.api as sm  
X_train1 = sm.add_constant(X_train1)


# In[ ]:


lm_1 = sm.OLS(y_train,X_train1).fit() # Running the linear model
print(lm_1.summary())


# #### Lets also check the VIF values 

# In[ ]:


def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.OLS(y,x).fit().rsquared  
        vif=round(1/(1-rsq),2)
        vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)


# In[ ]:


df.drop(col, axis =1, inplace = True)
# df.head()


# In[ ]:


# Calculating Vif value
# df.head()
vif_cal(input_data=df, dependent_col="price")


# ### Corelation Matrix

# In[ ]:


plt.figure(figsize = (20,10))  
sns.heatmap(df.corr(),annot = True)


# ##### Droping the values and updating the model-2 

# As we can see from heat map the cylindernumber_two and enginetype_rotor are highly corelated, the corelation is 1 and also the Vif value is pretty high for enginetype_rotor. It is infinity lets drop it.

# In[ ]:


# Dropping highly correlated variables and insignificant variables
X_train2 = X_train1.drop('enginetype_rotor', 1)


# In[ ]:


# Creating a second fitted model
lm_2 = sm.OLS(y_train,X_train2).fit()


# In[ ]:


#Let's see the summary of our second linear model
print(lm_2.summary())


# In[ ]:


df.drop('enginetype_rotor', axis =1, inplace = True)


# In[ ]:


# Calculating Vif value
vif_cal(input_data= df, dependent_col="price")


# ##### Droping the values and updating the model-3

# As we can see the cylindernumber_eight its vif is 3.43 and also we can see that it is positively corelated with enginetype_dohcv(0.44) and enginesize (0.49) lets go ahead and drop it 

# In[ ]:


# Dropping highly correlated variables and insignificant variables
X_train3 = X_train2.drop('cylindernumber_eight', 1)


# In[ ]:


# Creating a third fitted model 
lm_3 = sm.OLS(y_train,X_train3).fit()


# In[ ]:


#Let's see the summary of our third linear model
print(lm_3.summary())


# In[ ]:


df.drop('cylindernumber_eight', axis =1, inplace = True)


# In[ ]:


# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")


# ##### Droping the values and updating the model-4
# As we can see the enginetype_dohcv has very high pvalue and it is negatively corelated to CarName_porsche

# In[ ]:


# Dropping highly correlated variables and insignificant variables 
X_train4 = X_train3.drop('enginetype_dohcv', 1)


# In[ ]:


# Creating a fourth fitted model
lm_4 = sm.OLS(y_train,X_train4).fit()


# In[ ]:


#Let's see the summary of our fourth linear model
print(lm_4.summary())


# In[ ]:


df.drop('enginetype_dohcv', axis =1, inplace = True)


# In[ ]:


# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")


# ##### Droping the values and updating the model-5
# As we can see cylindernumber_four is negatively corelated to  enginesize and car width with values of -0.52 and -0.63 and also the vif is very high 30.73. Lets drop it 

# In[ ]:


# Dropping highly correlated variables and insignificant variables
X_train5 = X_train4.drop('cylindernumber_four', 1)


# In[ ]:


# Creating a fifth fitted model
lm_5 = sm.OLS(y_train,X_train5).fit()


# In[ ]:


#Let's see the summary of our fifth linear model
print(lm_5.summary())


# In[ ]:


df.drop('cylindernumber_four', axis =1, inplace = True)


# In[ ]:


# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")


# In[ ]:


plt.figure(figsize = (20,10))  
sns.heatmap(df.corr(),annot = True)


# ##### Droping the values and updating the model-6
# As we can see cylindernumber_twelve is positively corelated to enginesize  with value of 0.34 and also it has high p-value of 1.49 and also it has high negative coefficient. so lets drop it 
# 

# In[ ]:


# Dropping highly correlated variables and insignificant variables
X_train6 = X_train5.drop('cylindernumber_twelve', 1)


# In[ ]:


# Creating a sixth fitted model
lm_6 = sm.OLS(y_train,X_train6).fit()


# In[ ]:


#Let's see the summary of our sixth linear model
print(lm_6.summary())


# In[ ]:


df.drop('cylindernumber_twelve', axis =1, inplace = True)


# In[ ]:


# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")


# ##### Droping the values and updating the model-7
# As we can see stroke is positively corelated to enginesize  with value of 0.2 and also it has high p-value of 1.84 and also it has very less co relation with price. so lets drop it
# 

# In[ ]:


# Dropping highly correlated variables and insignificant variables
X_train7 = X_train6.drop('stroke', 1)


# In[ ]:


# Creating a seventh fitted model
lm_7 = sm.OLS(y_train,X_train7).fit()


# In[ ]:


#Let's see the summary of our seventh linear model
print(lm_7.summary())


# In[ ]:


df.drop('stroke', axis =1, inplace = True)


# In[ ]:


# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")


# ##### Droping the values and updating the model-8
# As we can see boreratio is positively corelated to enginesize and also the carlength with values of 0.58 and 0.61 and also it has high p-value of 1.69 and also it has vif of 2.11, so lets drop it 

# In[ ]:


# Dropping highly correlated variables and insignificant variables
X_train8 = X_train7.drop('boreratio', 1)


# In[ ]:


# Creating a eighth fitted model
lm_8 = sm.OLS(y_train,X_train8).fit()


# In[ ]:


#Let's see the summary of our eighth linear model
print(lm_8.summary())


# In[ ]:


df.drop('boreratio', axis =1, inplace = True)


# In[ ]:


# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")


# ##### Droping the values and updating the model-9
# As we can see the p-value of cylindernumber_three is high with value of 0.07 and also it has very less corelation with price. As we have other variables which have good corealtion with price. lets drop this variable 

# In[ ]:


# Dropping highly correlated variables and insignificant variables
X_train9 = X_train8.drop('cylindernumber_three', 1)


# In[ ]:


# Creating a ninth fitted model
lm_9 = sm.OLS(y_train,X_train9).fit()


# In[ ]:


#Let's see the summary of our ninth linear model
print(lm_9.summary())


# In[ ]:


df.drop('cylindernumber_three', axis =1, inplace = True)


# In[ ]:


# Calculating Vif value
vif_cal(input_data=df, dependent_col="price")


# In[ ]:


plt.figure(figsize = (20,10))  
sns.heatmap(df.corr(),annot = True)


# ### Prediction with model-9

# In[ ]:


# Adding  constant variable to test dataframe
X_test_m9 = sm.add_constant(X_test)


# In[ ]:


# Creating X_test_m12 dataframe by dropping variables from X_test_m12
X_test_m9 = X_test_m9.drop(col, axis=1)
X_test_m9 = X_test_m9.drop(['cylindernumber_three','enginetype_rotor','cylindernumber_eight',
                              'enginetype_dohcv','cylindernumber_four','cylindernumber_twelve','stroke','boreratio'], axis=1)
X_test_m9.info()


# In[ ]:


# Making predictions
y_pred_m9 = lm_9.predict(X_test_m9)
y_pred_m9


# ### Model Evaluation

# In[ ]:


# Actual vs Predicted
c = [i for i in range(1,63,1)]
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=3.5, linestyle="-")     #Plotting Actual
plt.plot(c,y_pred_m9, color="red",  linewidth=3.5, linestyle="-")  #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Car Price', fontsize=16)  


# In[ ]:


#Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred_m9)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)     


# ### Assessing the Model:
# Lets scatter plot the error and see if the error is some random noise or white noise, or it has some pattern.

# In[ ]:


# Error terms
fig = plt.figure()
c = [i for i in range(1,63,1)]
# plt.plot(c,y_test-y_pred_m9, color="blue", linewidth=2.5, linestyle="-")
plt.scatter(c,y_test-y_pred_m9)

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label


# In[ ]:


# Plotting the error terms to understand the distribution.
fig = plt.figure()
sns.distplot((y_test-y_pred_m9),bins=15)
fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 
plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label
plt.ylabel('Index', fontsize=16)             


# As we can see in the above scatter plot the error is randomly distributed and it does not follow any pattern.  I think we are good to go with this model, which has both adjusted R square and R square cloase to 0.89

# In[ ]:


import numpy as np
from sklearn import metrics
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred_m9)))


# ## Conclusion
#   
# 1. The R square and Adjusted R square value in CarModelWithOutliers is almost same which is 89.8% and 89.2% respectively which indicates that none of the parameters in CarModelWithOutliers model are reduntant.
# 
# 2. And also from Error Terms scatter plot we can see that the error (y_test-y-pred) is unequally distributed, and does not follow any pattern, as there is no curve, and shows no relation which indicates that it is just the white noise. 
# 
# 3. The RSME value is 0.06519190461262164
# 
# 4. As we can see that the model seems to be stable, The variables that can affect price are:
#  
#     1   enginesize	
#     2   carlength	
#     3	CarName_buick	
#     4	CarName_porsche	
#     5	CarName_bmw	
#     6	cylindernumber_two	
#     7	CarName_audi	
# 
# 

# In[ ]:




