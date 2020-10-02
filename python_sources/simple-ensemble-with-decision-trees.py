#!/usr/bin/env python
# coding: utf-8

# # Acknowledgements, Changelogs & Remarks
# 
# - Built-upon https://www.kaggle.com/yashgoyal401/less-code-accurate-result-best-predictions
# - Set up simple ensemble
# - Added models

# # Import Data, Libraries & Data Preprocessing

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True)


# In[ ]:


data= pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
data["Date"] = data["Date"].apply(lambda x: x.replace("-",""))
data["Date"]  = data["Date"].astype(int)


# In[ ]:


data.head()


# In[ ]:


data = data.drop(['Province/State'],axis=1)
data = data.dropna()
data.isnull().sum()


# In[ ]:


test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")   
test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))
test["Date"]  = test["Date"].astype(int)

test["Lat"]  = test["Lat"].fillna(12.5211)
test["Long"]  = test["Long"].fillna(69.9683)
test.isnull().sum()


# In[ ]:


x = data[['Lat', 'Long', 'Date']]
y1 = data[['ConfirmedCases']]
y2 = data[['Fatalities']]
x_test = test[['Lat', 'Long', 'Date']]


# # Predictions with DecisionTreeClassifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

Tree_model = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=31)


# In[ ]:


##
Tree_model.fit(x,y1)
pred1 = Tree_model.predict(x_test)
pred1 = pd.DataFrame(pred1)
pred1.columns = ["ConfirmedCases_prediction"]


# In[ ]:


pred1.head()


# In[ ]:


##
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


# # Predicitons with DecisionTreeRegressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor  

regressor = DecisionTreeRegressor(random_state=42, max_depth=31) 


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


# # Ensemble (Baseline)

# In[ ]:


submission = OP.merge(OP_dr, on="ForecastId")
submission['ConfirmedCases'] = 0.5*submission['ConfirmedCases_x'] + 0.5*submission['ConfirmedCases_y']
submission['Fatalities'] = 0.5*submission['Fatalities_x'] + 0.5*submission['Fatalities_y']
submission = submission[['ForecastId', 'ConfirmedCases', 'Fatalities']]
submission.head()


# In[ ]:


submission.to_csv("submission.csv",index=False)

