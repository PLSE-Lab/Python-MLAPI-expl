#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


beer_df = pd.read_csv("../input/beer-consumption-sao-paulo/Consumo_cerveja.csv")


# In[ ]:


beer_df


# Looks like lot of rows have all missing values and there seems to be some data discrepency with the Temperature media etc., instead of a decimal point we have a comma. This needs to be addressed. Date column can be feature engineered, to be split into days. Since the language is Portuguese, converting the column names into English would be convinient.

# In[ ]:


beer_df.info()


# In[ ]:


# Dropping rows with all NAN Values
beer_df.dropna(how = 'all', inplace = True)


# In[ ]:


beer_df.info()


# In[ ]:


# Replacing commas with period
beer_df.replace({',':'.'}, regex = True, inplace = True)


# In[ ]:


beer_df


# In[ ]:


beer_df.info()


# We could either remove the date column or convert it into days and see if beer consumption is affected by day of the week. Generally it is assumed that weekends have higher alcohol consumption than weekdays and we already have a column for that. Lets just look at the days as well before we decide to drop the date column or convert into days.

# In[ ]:


# Converting the type of Data to Date time
beer_df['Data'] = pd.to_datetime(beer_df['Data'])


# In[ ]:


beer_df.info()
days = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']


# In[ ]:


beer_df['Day'] = beer_df['Data'].apply(lambda a: days[a.weekday()])


# In[ ]:


beer_df['Day']


# In[ ]:


plt.figure(figsize=(10,5))
ax = sns.barplot(x="Day", y="Consumo de cerveja (litros)", data=beer_df)


# This shows that the days of the week doesnt matter much atleast in this data set and we can drop the 'Data' and the 'Day' columns and use the 'Final de Semana' (Weekend) to continue our analysis

# In[ ]:


beer_df.drop(['Data','Day'], axis = 1, inplace = True)


# In[ ]:


# Converting temperature and rainfall columns into float type
beer_df  = beer_df.apply(pd.to_numeric)


# In[ ]:


beer_df.info()


# Lets see how the varibles are correlated with a scatterplot

# In[ ]:


ax = sns.pairplot(beer_df)


# The plots are pretty much expected the 'Temperatura Media (C)','Temperatura Minima (C)','Temperatura Maxima (C)' are correlated to each other. There is not much correlation between 'Precipitacao (mm)' and the temperature relted variables.
# Lets build our model and see how the variables influence beer consumption

# In[ ]:


import statsmodels.api  as sm


# In[ ]:


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(beer_df, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


# Apply scaler() to all the columns except the 'yes-no' variables
num_vars = ['Temperatura Media (C)',
'Temperatura Minima (C)',
'Temperatura Maxima (C)',
'Precipitacao (mm)',
'Consumo de cerveja (litros)']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[ ]:


df_train.head()


# In[ ]:


# Dividing into X and Y sets for model building
y_train = df_train.pop('Consumo de cerveja (litros)')
X_train = df_train


# In[ ]:


# Add a constant because for stats model we need to explicitely add a constant or the line passes through origin by default
X_train_lm = sm.add_constant(X_train)


# In[ ]:


lr = sm.OLS(y_train, X_train_lm).fit()


# In[ ]:


# Print a summary of the linear regression model obtained
print(lr.summary())


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# We can see from the VIF values and the p value the Temperature Media (C) has to be dropped to get a better model

# In[ ]:


X = X_train.drop('Temperatura Media (C)', 1,)


# In[ ]:


# Building another model
X_train_lm = sm.add_constant(X)

lr_2 = sm.OLS(y_train, X_train_lm).fit()


# In[ ]:


# Print a summary of the linear regression model obtained
print(lr_2.summary())


# In[ ]:


# Calculate the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Since 'Temperatura Minima (C)' has high VIF and p value, lets drop this column and build another model

# In[ ]:


X = X.drop('Temperatura Minima (C)', 1,)


# In[ ]:


# Building another model
X_train_lm = sm.add_constant(X)

lr_3 = sm.OLS(y_train, X_train_lm).fit()


# In[ ]:


# Printing the summary of the linear regression model obtained
print(lr_3.summary())


# Lets check the performance of our model.

# In[ ]:


y_train_pred = lr_3.predict(X_train_lm)


# In[ ]:


fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)               


# In[ ]:


df_test[num_vars] = scaler.transform(df_test[num_vars])


# In[ ]:


y_test = df_test.pop('Consumo de cerveja (litros)')
X_test = df_test


# In[ ]:


X_test = sm.add_constant(X_test)


# In[ ]:


X_test = X_test.drop(['Temperatura Minima (C)', 'Temperatura Media (C)'], axis = 1)


# In[ ]:


y_pred = lr_3.predict(X_test)


# In[ ]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)     


# We see from the above test results and coeffcients obtained:
# 
# 1) The main dependence on beer consumption is on the Maximum temperature so with high temperatures the beer consumptions goes up
# 
# 2) Next is on the rainfall, if there is rain, the consumption drops down
# 
# 3) Depends on weather a particular day is a weekend or not, weekends have more consumption than weekdays.
# 
# Once again these are correlations and need not necesarrily imply causation.

# In[ ]:




