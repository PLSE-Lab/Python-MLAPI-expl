#!/usr/bin/env python
# coding: utf-8

# #       Regression Analysis of predicting Fish Weight

# Content:
# 
#     1.Import Relevant libraries
#     2.Import Data
#     3.Analyze the Data
#     4.Preprocessing the Data
#     5.Checking the data using StatsModels
#     6.Model Creation 
#     7.Testing the model

# # Import Relevant Libraries

# In[ ]:


import pandas as pd
pd.options.display.max_rows = 999
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

print('Import Complete')


# # Import Data

# In[ ]:


data = pd.read_csv('../input/fish-market/Fish.csv')
data.head()


# In[ ]:


print('Shape of data:',data.shape)
data.isnull().sum()


# In[ ]:


data.info(),data.describe()


# # Analysing the data

# In[ ]:


#there is zero weigth observed in weight , weight would never be zero 
#raw_data[raw_data['Weight']==0] 

raw_data = data.copy()
raw_data
#Fixing the outlier. Mean of the species weight is taken and assigned
mean_fish = data['Weight'][(data['Species'] =='Roach') & (data['Weight'] != 0)].mean()
raw_data.loc[40,'Weight'] = mean_fish
raw_data.loc[40,'Weight']


# In[ ]:


# there is no null values and next is to check the outliers

# to check Outliers
for i in raw_data.select_dtypes('float64').columns:
    plt.figure(figsize=(7,5))
    sns.distplot(raw_data[i])
    plt.show()
    print('Skewness is', raw_data[i].skew())
    #print('Kurtosis is', raw_data[i].kurtosis())                


# In[ ]:


#there is no outliers because skewness value of all columns are in acceptable range
# next we go our preprocessing of data
sns.barplot(x=raw_data['Species'],y=raw_data['Weight'])
plt.title('Species Vs Weight')


# In[ ]:


f, (ax1, ax2, ax3,ax4, ax5) = plt.subplots(5,1,sharey=True, figsize =(10,40))
ax1.scatter(raw_data['Length1'],raw_data['Weight'])
ax1.set_title('length1 and Weight')
ax2.scatter(raw_data['Length2'],raw_data['Weight'])
ax2.set_title('length2 and Weight')
ax3.scatter(raw_data['Length3'],raw_data['Weight'])
ax3.set_title('length1 and Weight')
ax4.scatter(raw_data['Height'],raw_data['Weight'])
ax4.set_title('Height and Weight')
ax5.scatter(raw_data['Width'],raw_data['Weight'])
ax5.set_title('Width and Weight')


# We take log Transformation to aceive linearity of the Model
# 
# Click the link to know more about LogTransformation
# https://www.real-statistics.com/multiple-regression/multiple-regression-log-transformations/

# In[ ]:


#take log to linearity
raw_data['LogWeight']= np.log(raw_data['Weight'])
#[np.log[i] for i in raw_data['Weight'] if i!=0 ]
#
raw_data['LogWeight']


# In[ ]:




f, (ax1, ax2, ax3,ax4, ax5) = plt.subplots(5,1,sharey=True, figsize =(10,40))
ax1.scatter(raw_data['Length1'],raw_data['LogWeight'])
ax1.set_title('length1 and Log Weight')
ax2.scatter(raw_data['Length2'],raw_data['LogWeight'])
ax2.set_title('length2 and Log Weight')
ax3.scatter(raw_data['Length3'],raw_data['LogWeight'])
ax3.set_title('length1 and Log Weight')
ax4.scatter(raw_data['Height'],raw_data['LogWeight'])
ax4.set_title('Height and Log Weight')
ax5.scatter(raw_data['Width'],raw_data['LogWeight'])
ax5.set_title('Width and Log Weight')


# In[ ]:


raw_data['LogLength1']=np.log(raw_data['Length1'])
raw_data['LogLength2']=np.log(raw_data['Length2'])
raw_data['LogLength3']=np.log(raw_data['Length3'])
raw_data['LogHeight']=np.log(raw_data['Height'])
raw_data['LogWidth']=np.log(raw_data['Width'])
raw_data.head(7)


# In[ ]:


f, (ax1, ax2, ax3,ax4, ax5) = plt.subplots(5,1,sharey=True, figsize =(10,40))
ax1.scatter(raw_data['LogLength1'],raw_data['LogWeight'])
ax1.set_title('Loglength1 and Log Weight')
ax2.scatter(raw_data['LogLength2'],raw_data['LogWeight'])
ax2.set_title('Loglength2 and Log Weight')
ax3.scatter(raw_data['LogLength3'],raw_data['LogWeight'])
ax3.set_title('Loglength1 and Log Weight')
ax4.scatter(raw_data['LogHeight'],raw_data['LogWeight'])
ax4.set_title('LogHeight and Log Weight')
ax5.scatter(raw_data['LogWidth'],raw_data['LogWeight'])
ax5.set_title('LogWidth and Log Weight')


# In[ ]:


cleaned_data = raw_data.drop(['Weight','Length1','Length2','Length3','Height','Width'],axis=1)


# #  Preprocessing the data 

# ## Dealing with Categorical Variables

# In[ ]:


# only 'species' column inthe data set is categorical ,we need to label the column 
raw_data['Species'].unique()


#     Sometimes categorical varibales also play a amjor role in predicting the results,so we have to deal the categorical variables
#  1.Drop the Categorical Columns
#  2.Imputing the categorical Variable(Label Encoder)
#  3.OneHot Encoding
#  
#  https://www.kaggle.com/alexisbcook/categorical-variables

# In[ ]:


#selct only the categorical column to impute

label_data = cleaned_data.copy()

encoder = LabelEncoder()
label_data['Species'] = encoder.fit_transform(raw_data['Species'])
#raw_data['Species'].nunique()
label_data.head(5)
#label_data['Species'].nunique()


# In[ ]:


label_data.describe()


# # Checking the data using StatsModels

# In[ ]:


#dependant and independant variable
y= label_data['LogWeight']
x= label_data.drop(['LogWeight'],axis=1)


# ### OLS Model

# In[ ]:


import statsmodels.api as sm
x_const = sm.add_constant(x)
results = sm.OLS(y,x).fit()
results.summary()


# In[ ]:


#drop the "length1" that has more p-value to get better results

y= label_data['LogWeight']
x1= label_data.drop(['LogWeight','LogLength3'],axis=1)


# In[ ]:


import statsmodels.api as sm
x_const = sm.add_constant(x1)
results2 = sm.OLS(y,x1).fit()
results2.summary()


# In[ ]:


#so we have to drop column 'LogLenth3' from the dataset
x.drop('LogLength3',axis=1)
x.head(7)


# ## Train_Test_Split

# In[ ]:


# train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=365)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# # Create the Model

# In[ ]:


lreg= LinearRegression()
lreg.fit(x_train,y_train)


# In[ ]:



y_pred = lreg.predict(x_train)
y_pred


# In[ ]:


# The simplest way to compare the targets (y_train) and the predictions (y_pred) is to plot them on a scatter plot
# The closer the points to the 45-degree line, the better the prediction
plt.scatter(y_train, y_pred , alpha=0.5)
# Let's also name the axes
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_pred)',size=18)
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
plt.xlim(0,8)
plt.ylim(0,8)
plt.show()


# In[ ]:


# Another useful check of our model is a residual plot
# We can plot the PDF of the residuals and check for anomalies
sns.distplot(y_train - y_pred)

# Include a title
plt.title("Residuals PDF", size=18)


# In[ ]:


# checking the Score

lreg.score(x_train,y_train)


# ### Wow ! Perfect... R.Score is 99% it is Perfectly fit...

# In[ ]:


lreg.coef_


# In[ ]:


lreg.intercept_


# In[ ]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_train,y_pred)


# # Testing the model

# In[ ]:


y_pred_test = lreg.predict(x_test)
y_pred


# In[ ]:


lreg.score(x_test,y_test)


# In[ ]:


lreg.intercept_


# In[ ]:


lreg.coef_


# In[ ]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,y_pred_test)


# In[ ]:


# The simplest way to compare the targets (y_test) and the predictions (y_pred) is to plot them on a scatter plot
# The closer the points to the 45-degree line, the better the prediction
plt.scatter(y_test, y_pred_test , alpha=0.5)
# Let's also name the axes
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_pred_test)',size=18)
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
plt.xlim(0,8)
plt.ylim(0,8)
plt.show()


# In[ ]:


# Another useful check of our model is a residual plot
# We can plot the PDF of the residuals and check for anomalies
sns.distplot(y_test - y_pred_test)

# Include a title
plt.title("Residuals PDF", size=18)


# # Creating the summary

# In[ ]:


# Finally, let's manually check these predictions
# To obtain the actual prices, we take the exponential of the log_price
df_pf = pd.DataFrame(np.exp(y_pred_test), columns=['Prediction'])
df_pf.head(7)


# In[ ]:


# reset the index to create the tabel
y_test = y_test.reset_index(drop=True)
# Let's overwrite the 'Target' column with the appropriate values
# Again, we need the exponential of the test log price
df_pf['Target'] = np.exp(y_test)

# Additionally, we can calculate the difference between the targets and the predictions
# Note that this is actually the residual (we already plotted the residuals)
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

# Since OLS is basically an algorithm which minimizes the total sum of squared errors (residuals),
# this comparison makes a lot of sense
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf.head(7)


# In[ ]:


# Exploring the descriptives here gives us additional insights
df_pf.describe()


# In[ ]:


# To see all rows, we use the relevant pandas syntax
pd.options.display.max_rows = 999
# Moreover, to make the dataset clear, we can display the result with only 2 digits after the dot 
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Finally, we sort by difference in % and manually check the model
df_pf.sort_values(by=['Difference%'])


# This is my first Machine Learning algorithm notebook on kaggle and  feel free to ask me any queries about this notebook
# 
# Comment your thoughts about this notebook and suggest me some methods to Improve the model
# 
# Thank You !..
