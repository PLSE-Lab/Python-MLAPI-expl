#!/usr/bin/env python
# coding: utf-8

# # My first improvments to me first Kaggle tutorial method
# The other week I completed my first set of kaggle tutorials for [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning). This tutorial involved learning the basics of machine learning to making predictions of house prices bases on previous sale data. After this I experimented with trying to improve my score/predictions based on what I thought I could do to improve it.
# 
# I have shown some of the main ideas I had, those which did work, and those which did not! I hope I can share my thought process behind what I have done, and hopefully once I have learned more I will be able to come back and make some improvements!

# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.simplefilter('ignore') #supress all future warnings
#from sklearn.tree import DecisionTreeRegressor

#load training and test data
home_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



# ## Overall correlation and missing data
# The parameters in the tutorial were given to us without context. I had recently learnt about _.corr_with_, and wanted to apply it hear to choose parameters with a stronger relation to what we are tring to predict- SalePrice.
# 
# Below are two plots of the 20 highest correlation paramters to sale price. One showing the correlation coefficient of the variable with price, and the second showing the number of missing values (not-including infinites) for each variable.

# In[ ]:


corr_vals = home_data.corrwith(home_data.SalePrice) #correlating variables
corr_vals.sort_values(ascending = False,inplace = True) #sorting them from high to low

plt.rcParams["figure.figsize"]=25,5; fig, ax = plt.subplots(2)
ax[0].bar(corr_vals.index[1:21],corr_vals[1:21]);ax[0].set_ylabel('Correlation')
ax[1].bar(corr_vals.index[1:21],(home_data.shape[0]- home_data[corr_vals.index[1:21]].count()),color = 'tab:red');plt.ylabel('# of Missing Values') #total entries - present entries
plt.show()


# Looking at our high correlation variables we can we see that for the most part in our training data we have most of the top 20 not missing!
# 
# The plot below shows the number of missing datapoints for the testing data set for the same parameters. From which there appears to be missing data in similar places which will need to be filled in.

# In[ ]:


plt.rcParams["figure.figsize"]=25,2.5;fig2,ax2 = plt.subplots(); 
ax2.bar(corr_vals.index[1:21],(test_data.shape[0]- test_data[corr_vals.index[1:21]].count()),color = 'tab:red');plt.ylabel('# of Missing Values') ;plt.show()


# ### Optimizing based on parameter
# The function below will create a model based on input data and will return the Mean Average Error for that particular model. I made this function so I could get a plot of "Number of Paramaters vs. Prediction Accuracy" -where MEA is being used as my measure of prediction accuracy. As of yet, the missing data has not been replaced, so only the highet 10 correlating variables can be used so far (model error on predicitng with missing values).

# In[ ]:


def train_by_feat(y,features,home_data):
    #Trains model based on list of features, splits data into training and validation by the features.
    X = home_data[features]
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    
    model = RandomForestRegressor(random_state = 1)
    model.fit(train_X,train_y)
    feat_val_predictions = model.predict(val_X)
    feat_val_mae = mean_absolute_error(feat_val_predictions, val_y)
    
    return feat_val_mae


# In[ ]:


first_opti = pd.Series({num:train_by_feat(home_data.SalePrice,corr_vals.index[1:num+1],home_data) for num in np.arange(10)+1}) #storing MEA vs # of model params

plt.rcParams["figure.figsize"]=8,5;plt.plot(first_opti.index,first_opti.data);plt.xlabel('# of parameters');plt.ylabel('MEA');plt.title('MEA vs # of parameters')


# From this alone we can see that there are large improvements to be made in the accuracy of the model by increasing the number of parameters. Something I which I tried to improve on in a variety of ways.

# ### Possible Improvements
# Some of the ideas I had to improve the prediction inclded:
# 
# -Interpolating missing values based on top correlated variables using multi-column sorting and _fillna(method=ffill)_
# 
# -Combining month and year to date-time for year-of-sale context (i.e. housing market crash), see how this raises time values in correlation
# 
# -Converting categorical data (which there is plenty of) into dummy data then re-checking correlations. Capturing propety features which are not currently possible to model as they are non-numerical
# 

# ## Improvements
# 

# ### Combining sale Month and sale Year into single DateTime value
# In some of the forum discussions people were having on the tutorial, a common mention was how the housing market was faring post- market crash. Currently month and year are sitting pretty low in correlation. So I wondered if this is because they were separated, and wanted to know what the correlation would be if you combined them.
# 

# In[ ]:


from datetime import date
# creating date time parameters as a serial value
all_datetime_serial= pd.Series({idx:date.toordinal(pd.to_datetime(str(home_data.YrSold[idx])+'/'+str(home_data.MoSold[idx]))) for idx in home_data.index})

print("Correlation of combine Year and Month: %.3f" %(all_datetime_serial.corr(home_data.SalePrice)))
print("Correlation of Year %.3f" %(home_data.YrSold.corr(home_data.SalePrice)))
print("Correlation of Month %.3f" %(home_data.MoSold.corr(home_data.SalePrice)))


# From this we can see that the correlation of combining the two is actually quite low. Which makes sense, combining two parameters with a weak realtion to Sale Price won't increase correlation.
# 
# _We can tell that from this that there is also a very slight decrease in house sale price with the low correlation of SalePrice with year_
# 
# With this line which creates a timeseries series, we can look at how the average price moves over each year

# In[ ]:


all_datetime = pd.Series({idx:pd.to_datetime(str(home_data.YrSold[idx])+'/'+str(home_data.MoSold[idx])) for idx in home_data.index})
plt.rcParams["figure.figsize"]=16,5; plt.scatter(all_datetime,home_data.SalePrice); plt.xlabel('Date') ; plt.ylabel('Sale Price USD$');plt.title('House sale price over time');


# It is clear that there is a pattern which occurs over the months with prices peaking in second third of each year with more variance in the middle.

# In[ ]:


#calculating average house price of each month
avg_month = pd.Series({mo:home_data.SalePrice[home_data.MoSold == mo].mean() for mo in pd.Series(home_data.MoSold.unique()).sort_values()}) 
plt.rcParams["figure.figsize"]=7,5;plt.scatter(avg_month.index,avg_month.data);plt.xlabel('Month');plt.ylabel('Average Sale Price');plt.title('Avg. sale price over 12 months');


# From this it is pretty clear that there is low correlation with month due to how scattered price is with month.

# ### Converting categorial fields to dummy-numerical fields
# To increase the number of modelling parameters available, I used _pd.get_dummies()_ to allow for predictions based on the qualatative values in the data set.

# In[ ]:


category =  np.setdiff1d(home_data.columns,corr_vals.index)
category


# Above is a list of all the categorical variables that currently can't be used

# In[ ]:


numerical_home_data = home_data[corr_vals.index].copy() # creating a new dataframe but with only numerical data, to which we will append our dummy variables


# In[ ]:


numerical_home_data.shape 


# In[ ]:


num_home_data_dummy = numerical_home_data.add(pd.get_dummies(home_data[category]),fill_value =0) # creating dummy variables


# In[ ]:


num_home_data_dummy.shape


# Now the number of paramters that can be modelled from has increased from 38 to 290! 
# 
# The graph below shows the correlation chart from before, but with new entries of categorical dummies highlighted in red.

# In[ ]:


#all the top 20 correlations
dummy_corr = num_home_data_dummy.corrwith(num_home_data_dummy.SalePrice)
plt.rcParams["figure.figsize"]=28,4
fig, ax = plt.subplots()
ax = plt.bar(dummy_corr.sort_values(ascending = False).iloc[1:20].index,dummy_corr.sort_values(ascending = False).iloc[1:20]) ; plt.xlabel('Variable') ; plt.ylabel('Correlation');
for bar in ax.patches:
    bar.set_facecolor('#888888')
ax.patches[7].set_facecolor('#aa3333');ax.patches[11].set_facecolor('#aa3333');ax.patches[12].set_facecolor('#aa3333');ax.patches[16].set_facecolor('#aa3333');ax.patches[17].set_facecolor('#aa3333');ax.patches[18].set_facecolor('#aa3333')


# ## Filling missing data
# Having trying to use sale month + year, and adding categorical features, the last improvement to the data I wanted to make was to fill any missing values.

# In[ ]:


def smart_fill(to_fill,frame):
    '''
    Takes dataframe and specific variable in it. Fills the missing variables in to_fill through intepolation with correlated variables.
    '''
    top_corr = frame.corrwith(frame[to_fill]).sort_values(ascending = False).index[1:4] #top 3 correlating values to variable being
    frame.sort_values(by=list(top_corr),inplace= True) #sorting frame with highest correlating variables with missing series
    frame[to_fill].interpolate(inplace = True) #filling nans with interpolation, inferring position by the sorting
    frame[to_fill].fillna(method = 'bfill',inplace= True)
    frame[to_fill].fillna(method = 'ffill',inplace= True)
    #bfill & ffill needed as you can't interpolate for edge values, and missing edge values are left NaN by interpolate method
    return frame  


# In[ ]:


filled_cols = []
for col in num_home_data_dummy.columns:
    if num_home_data_dummy[col].isnull().any()==1: #only run for columns with missing data
        filled_cols.append(col)
        smart_fill(col,num_home_data_dummy)


# In[ ]:


filled_cols # variables which have had their missing data filled!


# Only a few variables needed filling for the training data.

# # Applying improvements to the test data
# 
# Applying both the caterogical and data cleaning changes from above to the test data.

# In[ ]:


print("Missing data in test data: {0}".format(test_data.isnull().any().count()))


# Adding categorical variables to the test data.

# In[ ]:



test_cat = test_data[corr_vals.index.drop(corr_vals.index[0])].add(pd.get_dummies(test_data[category]),fill_value = 0).copy()
#train_data = home_data[corr_vals.index].add(pd.get_dummies(home_data[category]),fill_value = 0).copy()


# Filling the missing values

# In[ ]:


filled_cols2 = []
for col in test_cat.columns:
    if test_cat[col].isnull().any()==1: #only run for columns with missing data
        filled_cols2.append(col)
        test_cat = smart_fill(col,test_cat)


# In[ ]:


print("The number of missing data filled is {0}".format(len(filled_cols2))) # a lot of missing data was filled in the categorical data!


# This time 11 variables have had missing data filled in the test data, as opposed to only 3 in the training data.

# # Creating and optimizing the model
# 
# Now that both the training and the testing data sets have been prepared, I am going to create the model using the _Random Forest Regression_ model as suggested by the tutorial. One day once I am familiar with other models I may come back and use another.
# 
# I am applying a simple optimization here- inspired by the MAE optimization in the tutorial. However, instead of tree depth, I am going to vary the number of model parameters until a minimum MAE has been found.

# In[ ]:


#creating training data
train_data = home_data[corr_vals.index].add(pd.get_dummies(home_data[category]),fill_value = 0).copy()

filled_cols2 = []
for col in train_data.columns:
    if train_data[col].isnull().any()==1: #only run for columns with missing data
        filled_cols2.append(col)
        train_data = smart_fill(col,train_data)


# Determining optimum variables...

# In[ ]:


def train_by_feat(y,features,home_data):
    '''
    Trains a model based on a specified list of features and returns the MAE value for that model.
    '''
    X = home_data[features]
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    
    model = RandomForestRegressor(random_state = 1)
    model.fit(train_X,train_y)
    feat_val_predictions = model.predict(val_X)
    feat_val_mae = mean_absolute_error(feat_val_predictions, val_y)
    
    return feat_val_mae


# _features_ is the new ordered list of parameters' correlated with SalePrice indexes.

# In[ ]:


features = train_data.corrwith(train_data.SalePrice).sort_values(ascending = False).keys()


# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Using a dictionary comprehension to loop through all the top 100 features.

# In[ ]:


opti = pd.Series({num:train_by_feat(train_data.SalePrice,features[1:num+2],train_data) for num in np.arange(100)}) #need +2 to offset the fact we are starting at index 1


# In[ ]:


plt.rcParams["figure.figsize"]=7,5;plt.plot((1+np.arange(100)),opti);plt.xlabel('Number of Features');plt.ylabel('MAE');plt.title('Optimization of Random Forrest Regressor');


# In[ ]:


print("Mininmum MAE of {0} with {1} features".format(opti[opti.idxmin()],opti.idxmin()))


# The graph above shows how MAE decreases by adding more features to the model, and the the "optimum" number of model featues is the top 85. However, unfortunately I would run into issues using this model.

# # Problems with this approach, what I learned, and my solution the the issue I created
# As mentioned, I ran into some issues with this model, I would create a model on the above 85 features, then make a submission to kaggle and my score would be extremely high (bad). I couldn't work out what the issue was for a while. Until I looked at the predictions being made and compared them to what was given in the tutorial. 
# 
# This showed just how wrong the predictions were.

# In[ ]:


#Creating model of the 85 features
y = train_data.SalePrice
X = train_data[features[1:87]]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    
model = RandomForestRegressor(random_state = 1)
model.fit(train_X,train_y)
feat_val_predictions = model.predict(val_X)
mean_absolute_error(feat_val_predictions, val_y) # shows we are assuming a low error in our predictions


# In[ ]:


X_test = test_cat[features[1:87]]
y_pred = model.predict(X_test)


# In[ ]:


output = pd.DataFrame({'Id': test_data.Id,'SalePrice': y_pred})

good_result = pd.read_csv("../input/erincb-house-submission/submission_good.csv")
good_result.rename(columns ={'SalePrice':'SalePriceCorrect'},inplace = True)
good_result['SalePriceWrong'] = y_pred
good_result.head()


# This shows just how wrong the predictions were, I eventually isolated the problem to my _smart_fill_ function. It appeared to be changing the the values in places where I didn't want them to change.
# 
# From this I learned:
# 
# -that trying to simplify my methods will make troubleshooting easier (_smart_fill_)
# 
# -MAE does not always mean that the predictions will be accurate
# 
# -and in general from writing this notebook- to keep my explanations shorter and less bloated (sorry reader!)

# # Submitted predictions:
# 
# In my latest (and last for the time being) submission to kaggle I opted to use a simpler approach for missing values, and I kept the model optimization.

# In[ ]:


#create categorical dummy sets
train_cat = pd.get_dummies(home_data)
test_cat = pd.get_dummies(test_data)

#creating list of indexes in correlating order
cat_corr = train_cat.corrwith(train_cat.SalePrice).sort_values(ascending = False)

#filling missing values
train_cat.fillna(method ='pad',inplace= True)
test_cat.fillna(method = 'pad',inplace = True)

#determining optimum number of features ()
opti = pd.Series({num:train_by_feat(train_cat.SalePrice,cat_corr.index[1:num+2],train_cat) for num in np.arange(30)}) #30 chosen arbitrarily from trial and error on kaggle score

print("Number of optimum features {0}".format(opti.idxmin()))


# In[ ]:


#creatingthe model
X = train_cat[cat_corr.index[1:28]]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    

model = RandomForestRegressor(random_state = 1)
model.fit(train_X,train_y)
feat_val_predictions = model.predict(val_X)
feat_val_mae = mean_absolute_error(feat_val_predictions, val_y)


# In[ ]:


#making predictions
predictions = model.predict(test_cat[cat_corr.index[1:28]])
outy = pd.DataFrame({'Id': test_data.Id,'SalePrice': predictions})
outy.head()


# In[ ]:


outy.to_csv('submission.csv',index = False)


# Even though this notebook was quite long (something to improve) the final bits of code were not too complex.
# 
# Even without the _smart_fill_ function, going up to >30 predictions still results in large error. This is something I may wish to look into with using random forest models.

# In[ ]:




