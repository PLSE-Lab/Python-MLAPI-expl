#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df_train=pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
df_test=pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
df_sub=pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

print(df_train.shape)
print(df_test.shape)
print(df_sub.shape)


# ### EDA Train Data

# In[ ]:


df_train.head()


# In[ ]:


print(f"Unique Countries: {len(df_train.Country_Region.unique())}")


# In[ ]:


train_dates=list(df_train.Date.unique())
latest_date=df_train.Date.max()
print(f"Period : {len(df_train.Date.unique())} days")
print(f"From : {df_train.Date.min()} To : {df_train.Date.max()}")


# In[ ]:


print(f"Unique Regions: {df_train.shape[0]/len(df_train.Date.unique())}")


# In[ ]:


df_train.Country_Region.value_counts()


# In[ ]:


print(f"Number of rows without Country_Region : {df_train.Country_Region.isna().sum()}")


# We will add column UniqueRegion by combining province/state name, where available to country region.

# In[ ]:


df_train["UniqueRegion"]=df_train.Country_Region
df_train.UniqueRegion[df_train.Province_State.isna()==False]=df_train.Province_State+" , "+df_train.Country_Region

region_list=df_train.UniqueRegion.unique()
print(f"Total unique regions are : {len(region_list)}")
df_train[df_train.Province_State.isna()==False]


# Dropping the unwanted labels

# In[ ]:


df_train.drop(labels=["Id","Province_State","Country_Region"], axis=1, inplace=True)
df_train


# We will add one more column, Delta (Growth Factor) which is the ratio of confirmed cases on one day to that of the previous day.

# In[ ]:


df_train["Delta"]=1.0
df_train["NewCases"]=0.0
final_train=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","NewCases","UniqueRegion", "Delta"])

for region in region_list:
    df_temp=df_train[df_train.UniqueRegion==region].reset_index()
    size_train=df_temp.shape[0]
    
    df_temp.NewCases[0]=df_temp.ConfirmedCases[1]
    for i in range(1,df_temp.shape[0]):
        df_temp.NewCases[i]=df_temp.ConfirmedCases[i]-df_temp.ConfirmedCases[i-1]
        if(df_temp.ConfirmedCases[i-1]>0):
            df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]
            
    df_temp=df_temp[["Date","ConfirmedCases","Fatalities","UniqueRegion","NewCases","Delta"]]
    final_train=pd.concat([final_train,df_temp], ignore_index=True)
final_train.shape


# In[ ]:


latest_train=final_train[final_train.Date==latest_date]
latest_train.head()


# ### Define a function to plot how confirmed cases, fatalities and Delta changes with time

# In[ ]:


def train_data_plotter(r_name, n=df_temp.shape[0]):
    """
    Inputs
    r_name : Country name 
    n      : Latest period(in days) required (default from first confirmed case) 
    
    Output
    plots confirmed cases, fatalities and delta vs date
    """
    df_temp=final_train[final_train.UniqueRegion==r_name]
    df_temp=df_temp.tail(n)
    df_temp=df_temp[df_temp.ConfirmedCases>0]
    
    sns.set(style='darkgrid')
    # Plot Confirmed Cases
    plt.figure(figsize=(10,5))
    sns.lineplot('Date', 'ConfirmedCases', data=df_temp)
    plt.xticks(rotation=90)
    plt.title("Confirmed Cases")
    plt.show()
    # Plot Fatalities
    plt.figure(figsize=(10,5))
    sns.lineplot('Date', 'Fatalities', data=df_temp)
    plt.xticks(rotation=90)
    plt.title("Fatalities")
    plt.show()
    #Plot New Cases
    plt.figure(figsize=(10,3))
    sns.lineplot('Date', 'NewCases', data=df_temp)
    plt.xticks(rotation=90)
    plt.title("NewCases")
    plt.show()
    # Plot Delta
    plt.figure(figsize=(10,3))
    sns.lineplot('Date', 'Delta', data=df_temp)
    plt.xticks(rotation=90)
    plt.title("Delta")
    plt.show()


# # Visualizing the trend in few Countries/regions

# In[ ]:


train_data_plotter("Germany")


# In[ ]:


train_data_plotter("Hubei , China")


# In[ ]:


train_data_plotter("Italy")


# In[ ]:


train_data_plotter("Korea, South")


# Observations:
# * Fatality appears to be linearly correlated with Confirmed Cases
# * When delta increases, there is exponetial increase in the number of cases

# ### Establishing the relation between Confirmed Cases and Fatalities

# In[ ]:


score_list=[]
for region in region_list:
    df_temp=final_train[final_train.UniqueRegion==region]
    X=np.array(df_temp.ConfirmedCases).reshape(-1,1)
    Y=df_temp.Fatalities
    model=LinearRegression()
    model.fit(X,Y)
    score_list.append(model.score(X,Y))
score_df=pd.DataFrame({"Region":region_list,"Score":score_list})
print(f"Average R2 score between Fatality and Confirmed Cases is :{score_df.Score.mean()}")

plt.figure(figsize=(10,6))    
plt.title("Distribution of R2 score between Confirmed Cases and Fatality")
sns.distplot(score_df.Score)
plt.show()


# ### Examining regions where r2 score is less than 0.5 

# In[ ]:


less_than_50=score_df[score_df.Score<0.5].Region.unique()
print(f"Number of countries where r2 score<0.50 : {len(less_than_50)}")
latest_train[latest_train.UniqueRegion.isin(less_than_50)]


# In[ ]:


train_data_plotter("Zhejiang , China")


# These are the regions where there are a very few number of cases (probably imported ones) and fatality rate is low.
# 
# Since the number of these regions are low, we can ignore them.

# ### We have established that Fatality varies linearly with Confirmed cases

# # Assumption delta varies linearly in short-term

# ### Analysing correlation heatmap for different periods

# In[ ]:


def heat_map_plotter(r_name):
    """
    Input: Region Name
    Output: Plots correlation heatmap for periods: full data, last 15 days, last 5 days
    """
    
    plt.title("Correlation HeatMap Full data")
    df_temp=final_train[final_train.UniqueRegion==r_name]
    df_temp=df_temp.tail(df_temp.shape[0]).reset_index()
    sns.heatmap(abs(df_temp.corr()), cmap='coolwarm', annot=True)
    plt.show()

    plt.title("Correlation HeatMap 15 days")
    df_temp=final_train[final_train.UniqueRegion==r_name]
    df_temp=df_temp.tail(15).reset_index()
    sns.heatmap(abs(df_temp.corr()), cmap='coolwarm', annot=True)
    plt.show()

    plt.title("Correlation HeatMap 5 days")
    df_temp=final_train[final_train.UniqueRegion==r_name]
    df_temp=df_temp.tail(5).reset_index()
    sns.heatmap(abs(df_temp.corr()), cmap='coolwarm', annot=True)
    plt.show()


# ### Plotting heat map for few random regions

# In[ ]:


heat_map_plotter("France")


# In[ ]:


heat_map_plotter("Iran")


# Observation:
# * Fatalities are generally highly correlated with Confirmed Cases
# * In short term New Cases is highly correlated with Delta
# * In short term Correlation between Delta and index increases

# ### Regression plot for a specific region for a specific period

# In[ ]:


def reg_plotter(r_name, n=df_temp.shape[0]):
    """
    Inputs
    r_name : Country name 
    n      : Latest period(in days) required (default from first confirmed case) 
    
    Output
    Returns R2 score
    plots regression plot between delta vs period
    """
    df_temp=final_train[final_train.UniqueRegion==r_name]
    df_temp=df_temp.tail(n).reset_index()
    date=np.arange(1,n+1)
    
    # Plot Delta
    plt.figure(figsize=(10,3))
    plt.title(f"Delta Vs Time for last {n} days")
    sns.regplot(date, df_temp.Delta)
    plt.xticks(rotation=90)
    model=LinearRegression()
    X=date.reshape(-1,1)
    Y=df_temp.Delta
    model.fit(X,Y)
    print(f"R2 Score :{round(model.score(X,Y),2)}")
    plt.show()


# In[ ]:


reg_plotter("United Kingdom",5)


# In[ ]:


reg_plotter("New York , US",5)


# In[ ]:


reg_plotter("India", 5)


# Observation:
# * There is variablily in R2 score from above examples, so assumption that delta varies linearly with time in short term needs further testing

# In[ ]:



get_ipython().run_cell_magic('time', '', 'reg_score_list=[]\nperiod=[]\nreg=[]\nfor n in range(3,10):\n    for region in region_list:\n        df_temp=final_train[final_train.UniqueRegion==region]\n        df_temp=df_temp.tail(n).reset_index()\n        date=np.arange(1,n+1)\n        model=LinearRegression()\n        X=date.reshape(-1,1)\n        Y=df_temp.Delta\n        model.fit(X,Y)\n        reg.append(region)\n        reg_score_list.append(model.score(X,Y))\n        period.append(n)\nscore_df=pd.DataFrame({"Region":reg,"Score":reg_score_list, "Period":period})\n\nfor n in range(3,10): \n    print(f"Average R2 score for {n} days period :{score_df[score_df.Period==n].Score.mean()}")\n    plt.figure(figsize=(10,6))    \n    plt.title(f"Distribution of R2 score for {n} days")\n    sns.distplot(score_df[score_df.Period==n].Score)\n    plt.show()')


# Observation:
# * Generally average R2 score is better when N=3
# * For some regions, R2 score is better when N is higer

# ### Finding Best N for regions

# In[ ]:



n_list=[]
for reg in region_list:
    temp_score_df=score_df[score_df.Region==reg]
    if temp_score_df.Score.max()==1:
        n_list.append(3)
    else:
        n_list.append(temp_score_df.Period[temp_score_df.Score==temp_score_df.Score.max()].median())
best_n_df=pd.DataFrame({"Region":region_list,"N":n_list})
sns.countplot(best_n_df.N)


# * Observation Linear Regression can be used to predict Delta for the test data however the model would be underfitting.
# * Polynomial Regression or LSTM would do a much better job in prediction

# ## Test data EDA

# In[ ]:


df_test.head()


# In[ ]:


print(f"Unique Countries: {len(df_test.Country_Region.unique())}")
test_dates=list(df_test.Date.unique())
size_test=len(df_test.Date.unique())
print(f"Period : {len(df_test.Date.unique())} days")
print(f"From : {df_test.Date.min()} To : {df_test.Date.max()}")
print(f"Unique Regions: {df_test.shape[0]/len(df_test.Date.unique())}")


# In[ ]:


df_test["UniqueRegion"]=df_test.Country_Region
df_test.UniqueRegion[df_test.Province_State.isna()==False]=df_test.Province_State+" , "+df_test.Country_Region


# In[ ]:


df_test.drop(labels=["ForecastId","Province_State","Country_Region"], axis=1, inplace=True)
df_test["ConfirmedCases"]=0
df_test["Fatalities"]=0
df_test["NewCases"]=0
df_test["Delta"]=0


# In[ ]:


final_test=df_test[["Date","ConfirmedCases","Fatalities","UniqueRegion","NewCases","Delta"]]
app_test=final_test[final_test.Date>latest_date]
app_test.shape


# ## Predicting using Linear Regression of Delta

# In[ ]:



df_pred=pd.DataFrame(columns=["ConfirmedCases","Fatalities"])
df_traintest=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","UniqueRegion","NewCases","Delta"])

for region in region_list:
    df_temp=final_train[final_train.UniqueRegion==region].reset_index()
    
    #number of days for delta trend
    n=int(best_n_df[best_n_df.Region==region].N.sum()) 
    #Delta for the period
    delta_list=np.array(df_temp.tail(n).Delta).reshape(-1,1)
    #Morality rate as on last availabe date
    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()
    
    scaler=MinMaxScaler()
    X=np.arange(1,n+1).reshape(-1,1)
    Y=scaler.fit_transform(delta_list) 
    model=LinearRegression()
    model.fit(X,Y)
    
    df_test_app=app_test[app_test.UniqueRegion==region]
    df_temp=pd.concat([df_temp,df_test_app])
    df_temp=df_temp.reset_index()
    
    for i in range (size_train, df_temp.shape[0]):
        n=n+1        
        df_temp.Delta[i]=max(1,scaler.inverse_transform(model.predict(np.array([n]).reshape(-1,1))))
        df_temp.ConfirmedCases[i]=round(df_temp.ConfirmedCases[i-1]*df_temp.Delta[i],0)
        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)
        df_temp.NewCases[i]=df_temp.ConfirmedCases[i]-df_temp.ConfirmedCases[i-1]
        
    df_traintest=pd.concat([df_traintest,df_temp],ignore_index=True)
    
    df_temp=df_temp.iloc[-size_test:,:]
    df_temp=df_temp[["ConfirmedCases","Fatalities"]]
    df_pred=pd.concat([df_pred,df_temp], ignore_index=True)


# In[ ]:


def prediction_plotter(r_name):
    pred_df=df_traintest[df_traintest.UniqueRegion==r_name]
    train_df=final_train[final_train.UniqueRegion==r_name]
    plt.figure(figsize=(10,6))
    sns.lineplot('Date','ConfirmedCases',data=pred_df, color='r', label="Predicted Cases")
    sns.lineplot('Date','ConfirmedCases',data=train_df, color='g', label="Actual Cases")
    plt.show()


# In[ ]:


prediction_plotter("Germany")


# In[ ]:


prediction_plotter("Pakistan")


# ## Prediction Where new Cases follows Linear Regression

# In[ ]:



df_pred=pd.DataFrame(columns=["ConfirmedCases","Fatalities"])
df_traintest=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","UniqueRegion","NewCases","Delta"])

for region in region_list:
    df_temp=final_train[final_train.UniqueRegion==region].reset_index()
    
    #number of days for delta trend
    n=10 
    #Delta for the period
    NewCasesList=df_temp.tail(n).NewCases 
    #Morality rate as on last availabe date
    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()
    
    X=np.arange(1,n+1).reshape(-1,1)
    Y=NewCasesList
    model=LinearRegression()
    model.fit(X,Y)
    
    df_test_app=app_test[app_test.UniqueRegion==region]
    df_temp=pd.concat([df_temp,df_test_app])
    df_temp=df_temp.reset_index()
    
    for i in range (size_train, df_temp.shape[0]):
        n=n+1        
        df_temp.NewCases[i]=round(max(0,model.predict(np.array([n]).reshape(-1,1))[0]),0)
        df_temp.ConfirmedCases[i]=df_temp.ConfirmedCases[i-1]+df_temp.NewCases[i]
        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)
        df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]
        
    df_traintest=pd.concat([df_traintest,df_temp],ignore_index=True)
    
    df_temp=df_temp.iloc[-size_test:,:]
    df_temp=df_temp[["ConfirmedCases","Fatalities"]]
    df_pred=pd.concat([df_pred,df_temp], ignore_index=True)
df_pred.shape


# In[ ]:


prediction_plotter("New York , US")


# In[ ]:


prediction_plotter("Korea, South")


# ## Prediction when confirmed cases is in polinomial regression

# In[ ]:


#"""
df_pred=pd.DataFrame(columns=["ConfirmedCases","Fatalities"])
df_traintest=pd.DataFrame(columns=["Date","ConfirmedCases","Fatalities","UniqueRegion","NewCases","Delta"])

for region in region_list:
    df_temp=final_train[final_train.UniqueRegion==region].reset_index()
    
    #number of days for delta trend
    n=7
    #Delta for the period
    ConfirmedCasesList=df_temp.tail(n).ConfirmedCases 
    #Morality rate as on last availabe date
    death_rate=df_temp.tail(1).Fatalities.sum()/df_temp.tail(1).ConfirmedCases.sum()
    polynom=PolynomialFeatures(degree=2)
    X=polynom.fit_transform(np.arange(1,n+1).reshape(-1,1))
    Y=ConfirmedCasesList
    model=LinearRegression()
    model.fit(X,Y)
    
    df_test_app=app_test[app_test.UniqueRegion==region]
    df_temp=pd.concat([df_temp,df_test_app])
    df_temp=df_temp.reset_index()
    
    for i in range (size_train, df_temp.shape[0]):
        n=n+1        
        pred=round(model.predict(polynom.fit_transform(np.array(n).reshape(-1,1)))[0],0)
        df_temp.ConfirmedCases[i]=max(df_temp.ConfirmedCases[i-1],pred)
        df_temp.NewCases[i]=df_temp.ConfirmedCases[i]+df_temp.ConfirmedCases[i-1]
        df_temp.Fatalities[i]=round(death_rate*df_temp.ConfirmedCases[i],0)
        df_temp.Delta[i]=df_temp.ConfirmedCases[i]/df_temp.ConfirmedCases[i-1]
        
    df_traintest=pd.concat([df_traintest,df_temp],ignore_index=True)
    
    df_temp=df_temp.iloc[-size_test:,:]
    df_temp=df_temp[["ConfirmedCases","Fatalities"]]
    df_pred=pd.concat([df_pred,df_temp], ignore_index=True)
df_pred.shape
#"""


# In[ ]:


prediction_plotter("India")


# In[ ]:


prediction_plotter("New York , US")


# In[ ]:


df_sub.ConfirmedCases=df_pred.ConfirmedCases
df_sub.Fatalities=df_pred.Fatalities
#df_sub.to_csv("submission.csv",index=None)

