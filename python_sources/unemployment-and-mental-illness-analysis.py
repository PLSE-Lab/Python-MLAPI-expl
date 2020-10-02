#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import math

#set up packages
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)

#import data
data = pd.read_excel('../input/Cleaned Data.xlsx', sheet_name="Sheet1", header=0)
df = pd.DataFrame(data, columns=data.columns)
target = pd.DataFrame(data, columns=["I am currently employed at least part-time"])

#Drop Variables
del df['I am unemployed']
del df['I am currently employed at least part-time']

#convert all catagorical variables to dummy variables
catagorical = ['Education','Age', 'Gender', 'Household Income', 'Region', 'Device Type']

for catagory in catagorical:
    df_dummies = pd.get_dummies(df[catagory])
    df_dummies = pd.DataFrame(df_dummies, columns=df_dummies.columns.values)
    headers = np.concatenate([df.columns.values, df_dummies.columns.values])
    df = pd.DataFrame(np.concatenate([df,df_dummies],axis=1),columns=headers)
    del df[catagory]

#set all values as numeric
for i in df.columns:
    df[i] = df[i].astype(np.float64)

#set the target variable as binary
target.astype(np.bool)

X = df
Y = target

#Impute missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)
X = pd.DataFrame(imp, columns=X.columns.values)

# Create the model (no-stepwise yet)
model = LogisticRegression(solver='lbfgs', max_iter=2000).fit(X, Y)
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
chisquare, pvalues = chi2(X, Y)
output= pd.DataFrame(np.concatenate([pd.DataFrame(df.columns.values), pd.DataFrame(np.transpose(np.round(pvalues,4))),                                      np.transpose(np.round(model.coef_,4))], axis=1), columns=['Factor','Chi-square PValue', 'Coefficient'])

output = pd.DataFrame(output.sort_values(by='Chi-square PValue', ascending=True), columns=output.columns.values)
print('Target Variable: I am currently employed at least part-time')
output


# In[ ]:


#import data
data = pd.read_excel('../input/Cleaned Data.xlsx', sheet_name="Sheet1", header=0)
df = pd.DataFrame(data, columns=data.columns)
target = pd.DataFrame(data, columns=["I am unemployed"])

#Drop Variables
del df['I am unemployed']
del df['I am currently employed at least part-time']

#convert all catagorical variables to dummy variables
catagorical = ['Education','Age', 'Gender', 'Household Income', 'Region', 'Device Type']

for catagory in catagorical:
    df_dummies = pd.get_dummies(df[catagory])
    df_dummies = pd.DataFrame(df_dummies, columns=df_dummies.columns.values)
    headers = np.concatenate([df.columns.values, df_dummies.columns.values])
    df = pd.DataFrame(np.concatenate([df,df_dummies],axis=1),columns=headers)
    del df[catagory]

#set all values as numeric
for i in df.columns:
    df[i] = df[i].astype(np.float64)

#set the target variable as binary
target.astype(np.bool)

X = df
Y = target

#Impute missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)
X = pd.DataFrame(imp, columns=X.columns.values)

# Create the model (no-stepwise yet)
model = LogisticRegression(solver='lbfgs', max_iter=2000).fit(X, Y)
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
chisquare, pvalues = chi2(X, Y)
output= pd.DataFrame(np.concatenate([pd.DataFrame(df.columns.values), pd.DataFrame(np.transpose(np.round(pvalues,4))),                                      np.transpose(np.round(model.coef_,4))], axis=1), columns=['Factor','Chi-square PValue', 'Coefficient'])

output = pd.DataFrame(output.sort_values(by='Chi-square PValue', ascending=True), columns=output.columns.values)
print('Target Variable: I am unemployed')
output


# In[ ]:


#plot unemployment rate betweeen those with mental illness and those without

df = pd.DataFrame(np.concatenate([Y,X], axis=1),columns=np.transpose(np.concatenate([Y.columns.values,X.columns.values])))

unemployed_percent = df.loc[df['I identify as having a mental illness'] == 1, 'I am unemployed'].sum(axis=0)
unemployed_percent = unemployed_percent/df.loc[df['I identify as having a mental illness'] == 1, 'I am unemployed'].count()
unemployed_percent = round(unemployed_percent,4)*100

unemployed_percent2 = df.loc[df['I identify as having a mental illness'] < 1, 'I am unemployed'].sum(axis=0)
unemployed_percent2 = unemployed_percent/df.loc[df['I identify as having a mental illness'] < 1, 'I am unemployed'].count()
unemployed_percent2 = round(unemployed_percent2,4)*100

n_1 = df.loc[df['I identify as having a mental illness'] == 1, 'I am unemployed'].count()
n_2 = df.loc[df['I identify as having a mental illness'] < 1, 'I am unemployed'].count()

SD_1 = round(math.sqrt(((unemployed_percent/100) * (1 - unemployed_percent/100))/n_1), 4) * 100
SD_2 = round(math.sqrt(((unemployed_percent2/100) * (1 - unemployed_percent2/100))/n_2), 4) * 100

CI_1 = SD_1 * 1.96
CI_2 = SD_2 * 1.96

# set width of bar
barWidth = .8

# set height of bar
bars1 = unemployed_percent
bars2 = unemployed_percent2
 
# Make the plot
plt.bar(.5,bars1, color='#191970', width=barWidth, edgecolor='white', label='Mental Illness')
plt.bar(1.5,bars2, color='#FA8072', width=barWidth, edgecolor='white', label='No Mental Illness')
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])

# Create legend & Show graphic
plt.ylabel('Percent (%)')
plt.legend()
plt.rcParams['figure.figsize'] = 8, 10
plt.rcParams['font.size'] = 16
plt.show()
print('Mental illness unemployment rate:', unemployed_percent, "+/-", CI_1, "%")
print('No mental illness unemployment rate:', unemployed_percent2, "+/-", CI_2, "%")

