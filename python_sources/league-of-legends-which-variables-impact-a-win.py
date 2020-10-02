#!/usr/bin/env python
# coding: utf-8

# # Analysing which variables contribute the most for the Blue Team to Win.

# ## Importing Libraries & importing the csv data file

# In[ ]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#importing and defining Data Frame
df = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")


# ## Checking the Data Frame info and first look of the data/descriptive statistics

# In[ ]:


# cheking info of the df
def data_inv(df):
    print('dataframe: ',df.shape[0])
    print('dataset variables: ',df.shape[1])
    print('-'*10)
    print('dateset columns: \n')
    print(df.columns)
    print('-'*10)
    print('data-type of each column: \n')
    print(df.dtypes)
    print('-'*10)
    print('missing rows in each column: \n')
    c=df.isnull().sum()
    print(c[c>0])
data_inv(df)


# In[ ]:


# checking the df:
df.head()


# In[ ]:


# creating a copy/checkpoint before deleting unnecessary columns
df_1 = df.copy()


# ## Cleaning the Data Frame

# In[ ]:


#dropping columns
df_1 = df_1.drop(['blueGoldDiff', 'blueExperienceDiff','redGoldDiff',
       'redExperienceDiff','gameId'], axis=1)


# In[ ]:


#check number of diferent values
df_1.nunique()


# In[ ]:


#the probability of blue team winning is inversely correlated with red team, so the task is to analyse the Blue Team.
df_blue = df_1.drop(['redWardsPlaced','redWardsDestroyed',
       'redFirstBlood', 'redKills', 'redDeaths', 'redAssists',
       'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
       'redTotalGold', 'redAvgLevel', 'redTotalExperience',
       'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redCSPerMin', 'redGoldPerMin'], axis=1)


# ## Plotting some graphs to better visualize the data

# In[ ]:


x = df_blue['blueWins']
y = df_blue['blueTotalGold']
plt.bar(x, y)
plt.xticks(range(0,2))
plt.show()


# In[ ]:


# Total Gold vs Total minions killed, as we can see more minions killed doesn't equate more gold 
x1 = df_blue['blueTotalMinionsKilled']
y1 = df_blue['blueTotalGold']
plt.scatter(x1, y1)
plt.show()


# In[ ]:


# correlation between the variables,
# To avoid Multicollinearity, the independent variables must not be over 0,7 correlation or the regression output will 
        # be erroneous, for example: Blue Kills is highly correlated with Blue Assists, and one must be omitted from the model
corr = df_blue.corr()


# In[ ]:


plt.figure(figsize=(16,10))
sns.heatmap(corr, annot =  True)


# ## Standardizing the data 

# In[ ]:


# creating an input table with only the independent variables, ommiting the correlating ones,
# creating the target variable = Blue Wins
df_blue.columns


# In[ ]:


unscaled_inputs = df_blue.filter(['blueWardsPlaced', 'blueWardsDestroyed', 'blueFirstBlood',
       'blueKills', 'blueDeaths','blueEliteMonsters','blueHeralds', 'blueTowersDestroyed','blueAvgLevel','blueTotalMinionsKilled', 'blueTotalJungleMinionsKilled'], axis=1)
target = df_blue.filter(['blueWins'])


# Scale just the non categorical variables, in this case 'Blue first Blood' is categorical

# In[ ]:


# import the libraries needed to create the Custom Scaler
# note that all of them are a part of the sklearn package
# moreover, one of them is actually the StandardScaler module, 
# so you can imagine that the Custom Scaler is build on it

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# create the Custom Scaler class

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    # init or what information we need to declare a CustomScaler object
    # and what is calculated/declared as we do
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler(copy,with_mean,with_std)
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    
    # the fit method, which, again based on StandardScale
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    # the transform method which does the actual scaling

    def transform(self, X, y=None, copy=None):
        
        # record the initial order of the columns
        init_col_order = X.columns
        
        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        
        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        
        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# In[ ]:


# categorical columns to omit
columns_to_omit = ['blueFirstBlood']


# In[ ]:


# create the columns to scale, based on the columns to omit
# use list comprehension to iterate over the list
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]


# In[ ]:


blue_scaler = CustomScaler(columns_to_scale)


# In[ ]:


blue_scaler.fit(unscaled_inputs)


# In[ ]:


scaled_inputs = blue_scaler.transform(unscaled_inputs)
scaled_inputs


# ## Test, Train and Split the data

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_test_split(scaled_inputs, target)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, target, train_size=0.8, random_state=20)


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


reg = LogisticRegression()


# In[ ]:


reg.fit(x_train, y_train)


# In[ ]:


# Regression score
reg.score(x_train, y_train)


# In[ ]:


# The Intercept
intercept = reg.intercept_
intercept


# In[ ]:


# Creating a Summary Table to visualize the Variable and respective Coefficients and Odds Ratio
variables = unscaled_inputs.columns.values
variables


# In[ ]:


summary_table = pd.DataFrame(columns=['Variables'], data = variables)
summary_table['Coef'] = np.transpose(reg.coef_)
# add the intercept at index 0
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
# calculate the Odds Ratio and add to the table
summary_table['Odds Ratio'] = np.exp(summary_table.Coef)


# In[ ]:


summary_table.sort_values(by=['Odds Ratio'], ascending=False)


# ## Calculating P-values

# In[ ]:


import statsmodels.api as sm
x = sm.add_constant(x_train)
logit_model=sm.Logit(y_train,x)
result=logit_model.fit()
print(result.summary())


# # Conclusions

# 1- The Variables "Blue Wards Placed", "Blue Wards Destroyed"have no statistical significance, so the next step is to remove them from the model
# 
# 2- The Variable that has the biggest impact in the odds of winning is the number of Kills for the Blue Team, for every Kill the odds increase by 132%.
#     
# 3- The Second variable that impacts the outcome is, as expected, the number of Deaths, this time it impacts negatively:for every Death, the odds of winning decreases by 50% for the Blue team
#     
# 4- Surprizingly, killing Heralds decreases the odds of winning by about 12%
# 
# 5- Killing Elite monsters and Total minions killed are the next biggest impact in winning with 30% and 20% increase, respectively

# ## Testing the data

# In[ ]:


# testing the data is important to evalute the accuracy on a dataset that the model has never seen, to see if it's Overfitting
  # a test score 10% below the training reveals an overfitting
reg.score(x_test, y_test)


# In[ ]:


predicted_prob = reg.predict_proba(x_test)
predicted_prob[:,1]


# In[ ]:


df_blue['predicted'] = reg.predict_proba(scaled_inputs)[:,1]


# In[ ]:


df_blue

