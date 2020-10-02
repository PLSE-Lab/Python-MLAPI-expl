#!/usr/bin/env python
# coding: utf-8

# Simple model using numeric features

# In[1]:


from subprocess import check_output

#check the files available in the project directory
folder = "../input/"
print(check_output(["ls", folder]).decode("utf8")) 


# In[2]:


#Load data and summerise

import pandas as pd

#Import data
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)

#Print data description and head
print("Data description:")
print(data.describe())

print("\nData head:")
print(data.head())


# In[3]:


#Checking importance of numerical features

import matplotlib.pyplot as plt
import seaborn as sns

#Correlation map to see how features are correlated with SalePrice
corr = data.corr()
plt.subplots(figsize=(10,10))
sns.heatmap(corr, vmax=0.8, square=True)

cor_dict = corr['SalePrice'].to_dict()
del cor_dict['SalePrice']

#Print correlation dictionary
print('List of numerical features in order of correlation with Sale Price:\n')
for x in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
    print ("{0}: \t{1}".format(*x))


# For simplicity will use only most highly correlated (>0.5) numeric values. Will ignore garage area as information will be similar to garage cars. 

# In[11]:


#Format data for modeling

#Set target data as y
y = data.SalePrice

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
                        '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
X = data[features]


# Will assume all missing values are due to feature not apearing in house therefore value of feature = 0. 

# In[5]:


#Deal with missing values

def fill_NAN_0 (df):
    cols_list = list(df.columns)
    for col in cols_list:
        df[col] = df[col].fillna(0)
    return df

fill_NAN_0(X)

print("\n" + str(X.isnull().sum()))


# In[6]:


#Split into train and test sets
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y,random_state = 42)


# In[7]:


#Model data using RandomForest algorithm

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
pred_X = forest_model.predict(test_X)

mae_rand = mean_absolute_error(test_y, pred_X)

#Sanity check
print("The first 10 predictions are: " + str(pred_X[:10]))

#MAE print
print("\n The mean absolute error is {:.0f}".format(mae_rand))


# In[8]:


#Check optimum nodes
def optimum_nodes(train_X, train_y, test_X, test_y, test_range):
    #Create empty list for storing mae for a given amount of nodes
    node_list = []
    
    #Find MAE for a list of node values
    for node in test_range:
        model = RandomForestRegressor(n_estimators=node, random_state=42)
        model.fit(train_X, train_y)
        pred_y= model.predict(test_X)
        mae = mean_absolute_error(test_y, pred_y)
        node_list.append (mae)

    #Plot leaf nodes vs. MAE
    fig = plt.subplots(figsize=(5,5))
    plt.scatter(x = test_range, y = node_list)
    plt.ylabel('MAE', fontsize=13)
    plt.xlabel('Leaf Nodes', fontsize=13)
    plt.show()

test_range = range(10,400,10)

optimum_nodes(train_X, train_y, test_X, test_y, test_range)


# In[9]:


#Make predictions for test data

#Read test data
test = pd.read_csv('../input/test.csv')
test_X = test[features]

fill_NAN_0(test_X)

model = RandomForestRegressor(n_estimators=200, random_state = 42)
model.fit(train_X, train_y)
predict_y = model.predict(test_X)

# Sanity check
print("Predicted values: {}".format(predict_y))


# In[10]:


#Save predictions to csv
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predict_y})
submission.to_csv('submission.csv', index=False)
print("Finished!")

