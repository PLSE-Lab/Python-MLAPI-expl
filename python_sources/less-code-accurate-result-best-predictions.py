#!/usr/bin/env python
# coding: utf-8

# **For LSTM model you can click on the below link:**
# * https://www.kaggle.com/yashgoyal401/advanced-visualizations-and-predictions-with-lstm

# # Table of content:
#  
# * <a href='#Libraries'>Libraries</a>
# * <a href="#Data">Data</a>
# * <a href="#Model">Model</a>
# * <a href="#Out_sub">Out_sub</a>

# <a id='Libraries'></a>

# # Libraries :

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True)


# <a id='Data'></a>

# # Data :

# In[ ]:


data= pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")


# In[ ]:


data["Date"].max() ## 2020-03-20


# In[ ]:


data["Date"] = data["Date"].apply(lambda x: x.replace("-",""))
data["Date"]  = data["Date"].astype(int)


# In[ ]:


data = data.drop(['Province/State'],axis=1)
data = data.dropna()
data.isnull().sum()


# In[ ]:


data.head()


# In[ ]:


test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")   
test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))
test["Date"]  = test["Date"].astype(int)

test["Lat"]  = test["Lat"].fillna(12.5211)
test["Long"]  = test["Long"].fillna(69.9683)
test.isnull().sum()


# In[ ]:


test["Date"].min() ## 2020-03-12


# In[ ]:


x = data[['Lat', 'Long', 'Date']]
y1 = data[['ConfirmedCases']]
y2 = data[['Fatalities']]
x_test = test[['Lat', 'Long', 'Date']]


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
Tree_model = DecisionTreeClassifier(criterion='entropy')


# <a id='Model'></a>

# # Model : 

# In[ ]:


##
Tree_model.fit(x,y1)
pred1 = Tree_model.predict(x_test)
pred1 = pd.DataFrame(pred1)
pred1.columns = ["ConfirmedCases_prediction"]


# In[ ]:


pred1.head()


# In[ ]:


Tree_model.fit(x,y2)
pred2 = Tree_model.predict(x_test)
pred2 = pd.DataFrame(pred2)
pred2.columns = ["Death_prediction"]


# In[ ]:


pred2.head()


# In[ ]:


Sub = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")
Sub.columns
sub_new = Sub[["ForecastId"]]


# In[ ]:


OP = pd.concat([pred1,pred2,sub_new],axis=1)
OP.head()
OP.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']
OP = OP[['ForecastId','ConfirmedCases', 'Fatalities']]


# In[ ]:


OP["ConfirmedCases"] = OP["ConfirmedCases"].astype(int)
OP["Fatalities"] = OP["Fatalities"].astype(int)


# In[ ]:


OP.head()


# In[ ]:



#### By using DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor(random_state = 0) 


# In[ ]:


regressor.fit(x,y1)
pred_r1 = regressor.predict(x_test)
pred_r1 = pd.DataFrame(pred_r1)
pred_r1.columns = ["ConfirmedCases_prediction"]


# In[ ]:


regressor.fit(x,y2)
pred_r2 = regressor.predict(x_test)
pred_r2 = pd.DataFrame(pred_r2)
pred_r2.columns = ["Death_prediction"]


# In[ ]:


OP_dr = pd.concat([sub_new,pred_r1,pred_r2],axis=1)
OP_dr.head()
OP_dr.columns = [ 'ForecastId','ConfirmedCases', 'Fatalities']


# In[ ]:


OP_dr["ConfirmedCases"] = OP_dr["ConfirmedCases"].astype(int)
OP_dr["Fatalities"] = OP_dr["Fatalities"].astype(int)
OP_dr.head()


# In[ ]:


## Random Forest regressor
from sklearn.ensemble import RandomForestRegressor
rand_reg = RandomForestRegressor()


# In[ ]:


rand_reg.fit(x,y1)
pred_ra1 = rand_reg.predict(x_test)
pred_ra1 = pd.DataFrame(pred_ra1)
pred_ra1.columns = ["ConfirmedCases_prediction"]


# In[ ]:


rand_reg.fit(x,y2)
pred_ra2 = rand_reg.predict(x_test)
pred_ra2 = pd.DataFrame(pred_ra2)
pred_ra2.columns = ["Death_prediction"]


# In[ ]:


OP_ra = pd.concat([sub_new,pred_ra1,pred_ra2],axis=1)
OP_ra.head()
OP_ra.columns = [ 'ForecastId','ConfirmedCases', 'Fatalities']


# In[ ]:


OP_ra["ConfirmedCases"] = OP_ra["ConfirmedCases"].astype(int)
OP_ra["Fatalities"] = OP_ra["Fatalities"].astype(int)
OP_ra.head()


# <a id='Out_sub'></a>

# # Output submission :

# In[ ]:


OP_ra.to_csv("submission.csv",index=False)

