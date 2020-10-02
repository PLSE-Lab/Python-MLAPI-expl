#!/usr/bin/env python
# coding: utf-8

#     Feature description - 
#     - CRIM     per capita crime rate by town
#     - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#     - INDUS    proportion of non-retail business acres per town
#     - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#     - NOX      nitric oxides concentration (parts per 10 million)
#     - RM       average number of rooms per dwelling
#     - AGE      proportion of owner-occupied units built prior to 1940
#     - DIS      weighted distances to five Boston employment centres
#     - RAD      index of accessibility to radial highways
#     - TAX      full-value property-tax rate per '$10,000'
#     - PTRATIO  pupil-teacher ratio by town
#     - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#     - LSTAT    '% lower status of the population'
#     - MEDV     Median value of owner-occupied homes in $1000's

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# In[ ]:


cols=['CRIM',
'ZN',
'INDUS',
'CHAS',
'NOX',
'RM',
'AGE',
'DIS',
'RAD',
'TAX',
'PTRATIO',
'B',
'LSTAT',
'MEDV']
df=pd.read_csv(r'/kaggle/input/boston-house-prices/housing.csv',header=None,delim_whitespace=True)
df.columns=cols
price=df['MEDV']
#df.drop('MEDV', axis=1, inplace=True)
df


# To observe correlation, I just gave a look at the pairplot for the features.

# In[ ]:


pplot=sns.pairplot(df[:-1])
pplot.fig.set_size_inches(15,15)


# We could see that there is some correlation between House Price(MEDV) and features like LSTAT,RM,CRIM. 

# In[ ]:


scaler = preprocessing.StandardScaler()
df_stand = scaler.fit_transform(df)
df_stand=pd.DataFrame(df_stand,columns=cols)
df=df_stand


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), center=0, cmap='BrBG',annot=True)
ax.set_title('Multi-Collinearity of Features')


# Using the heat map we could see that there is some collinearity between some features NOX, DIS, RAD, TAX. but for better interpretation i prefer implenting VIF.  

# Inorder to check for outliers. We are plotting BOX plot for visualization. 

# In[ ]:



print(df.columns)
plt.figure(num=None, figsize=(15,30), dpi=80, facecolor='w', edgecolor='k')
plt.figure(1)
var=1
for index,feature in enumerate(list(df.columns)):
    plt.subplot(4,4,index+1,xlabel=feature)
    sns.boxplot(y=df[feature])
    plt.grid()
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.subplots_adjust(top=0.92, bottom=0, left=0.10, right=0.95, hspace=0.5,wspace=0.5)
    var+=1


# There are outliers in few features and we remove them below.

# In[ ]:



medv_correlation={}
for i in df.columns:
    if i!='MEDV':
        corr_value=df['MEDV'].corr(df[i]) 
        medv_correlation[i]=corr_value
        print("Correlation value between MEDV and {} is {}".format(i,corr_value))
print("MAX correlated features are:", max(medv_correlation, key=medv_correlation.get),min(medv_correlation, key=medv_correlation.get))


# I have checked the correlation of the features with the dependent variable(MEDV) and found RM and LSTAT are two important features. So, I chose to remove outliers from RM feature.

# Removing Outliers from the Data for the column RM which is highly correlated with MEDV(price)

# In[ ]:


percentage=0
for k, v in df.items():
    if k=='RM':
        Q1 = np.array(v.quantile(0.25))
        Q3 = np.array(v.quantile(0.75))
        IQR = Q3 - Q1
        v_col = v[(v <= Q1 - 1.5 * IQR) | (v >= Q3 + 1.5 * IQR)]
        percentage = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
        print("Column %s outliers = %.2f%%" % (k, percentage))
print("Total number of outliers for feature RM is %d  which is %.2f%% of total data points"%(v_col.count(),percentage)) 

df=df[~(df['RM'] >= Q3 + 1.5 * IQR)|(df['RM'] <=Q1 - 1.5 * IQR)]
df.shape


# In[ ]:


df


# Standardizing the data inorder to compute correlation using Variance Inflation Factor method.

# In[ ]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
scaler = preprocessing.StandardScaler()
df_stand = scaler.fit_transform(df)
df_stand=pd.DataFrame(df_stand,columns=cols)
df_stand


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculating VIF
s=df._get_numeric_data()
vif = pd.DataFrame()
vif["variables"] = s.columns

vif["VIF"] = [variance_inflation_factor(np.array(s.values,dtype=float), i) for i in range(s.shape[1])]


# Usually, It is considered that if the VIF score is <5 they are less collinearity and near to or above 10 they have high collinearity. Best score is VIF = 1 meaning they have 0 collinearity.
# 
# Here since there are majority below 5, I have chosen only the ones that are less than the mean of all VIF's and termed them as Non Collinear and rest as collinear.

# In[ ]:


vif['Result'] = vif['VIF'].apply(lambda x: 'Non Collinear' if x <= 3.7 else 'Collinear')
vif


# In[ ]:


features=list(vif[vif['Result']=='Non Collinear']['variables'])

print(features)


# Removing the collinear features and for further analysis.
# 

# In[ ]:


for i in df.columns:
    if i!='MEDV':
        if i not in features:
            df.drop(i,axis=1,inplace=True)
df


# Analyzing the features relationship with dependent variable MEDV.

# In[ ]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
plt.figure(1)
var=1
for index,feature in enumerate(list(df.columns)):
    colors = (0,0,0)
    area = np.pi*3
    plt.subplot(4,4,index+1,xlabel=feature,ylabel='MEDV')
    sns.regplot(y='MEDV',x=feature,data=df,marker="+",ci=80)
    plt.grid()
    plt.subplots_adjust(top=0.92, bottom=0, left=0.10, right=0.95, hspace=0.5,wspace=0.5)
    var+=1


# Clearly we understand that our price(MEDV) is getting lower as LSTAT and CRIM is increasing. And we can say that the price(MEDV) is increasing as the no. of rooms(RM) is increasing.

# In[ ]:


df_plot=pd.read_csv(r'/kaggle/input/boston-house-prices/housing.csv',header=None,delim_whitespace=True)
df_plot.columns=cols


for i in df.columns:
    if i!='MEDV':
        if i not in features:
            df_plot.drop(i,axis=1,inplace=True)

from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
plt.figure(1)
var=1
for index,feature in enumerate(list(df.columns)[:-1]):
    counts,bin_edges= np.histogram(df_plot[feature],bins=10,density=True)
    pdf=counts/sum(counts)
    #a=440+var
    cdf=np.cumsum(pdf)
    plt.subplot(4,4,index+1,xlabel=feature)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.grid()
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,wspace=0.5)
    var+=1


# We could see the PDF and CDF of the features. Almost 70-80% of our dataset has CRIM less than 20 .

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


y=df_stand['MEDV'] 
df_stand.drop('MEDV',axis=1,inplace=True)
X=df_stand
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[ ]:


y_pred = regressor.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Error for our prediction is 0.55 . The Plot for our predicted and test data is below.

# In[ ]:


sns.regplot(y_test,y_pred)

