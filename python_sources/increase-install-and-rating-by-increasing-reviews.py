#!/usr/bin/env python
# coding: utf-8

# #  **Imports and Datasets**

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from eli5.sklearn import PermutationImportance
import eli5
from pdpbox import pdp, get_dataset, info_plots
import shap


# In[ ]:


app_data = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
app_data.head()


# # Preprocessing

# In[ ]:


app_data.info()


# **Obtaining rows with unknown values and drop them**

# In[ ]:


app_data.isna().sum()


# In[ ]:


app_data.dropna(inplace = True)


# In[ ]:


app_data.info()


# Change columns with numerical data of "object" type to "int" or "float"

# In[ ]:


def toMB(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x)
        return(x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x)/1000
        return(x)
    else:
        return None

app_data["Size"] = app_data["Size"].map(toMB)

app_data.Size.fillna(method = 'ffill', inplace = True)


# In[ ]:


app_data['Installs'] = [int(i[:-1].replace(',','')) for i in app_data['Installs']]


# In[ ]:


def pricetofloat(price):
    if price == '0':
        return 0
    else:
        price = price[1:]
        price = float(price)
        return price

app_data['Price'] = app_data['Price'].map(pricetofloat).astype(float)


# In[ ]:


app_data['Reviews'] = app_data['Reviews'].astype(int)


# In[ ]:


app_data.info()


# In[ ]:


app_data.head()


# **Demonstrate the strength of the relationship between the features by the regression line**

# In[ ]:


plt.figure(figsize=(40,20))
sns.regplot(x="Reviews", y="Installs", data=app_data)
sns.set(style="white",font_scale=3)


# In[ ]:


plt.figure(figsize=(40,20))
sns.regplot(x="Rating", y="Installs", data=app_data)
sns.set(style="white",font_scale=6)


# In[ ]:


plt.figure(figsize=(40,20))
sns.regplot(x="Size", y="Installs", data=app_data)
sns.set(style="white",font_scale=5)


# In[ ]:


plt.figure(figsize=(40,20))
sns.regplot(x="Reviews", y="Rating", data=app_data)
sns.set(style="white",font_scale=5)


# In[ ]:


plt.figure(figsize=(40,20))
sns.regplot(x="Size", y="Rating", data=app_data)
sns.set(style="white",font_scale=5)


# In[ ]:


rd=app_data.groupby(['Category']).sum().sort_values(by='Installs', ascending=True).reset_index()
fig = px.bar(rd,
             x='Installs', y='Category',
             title=f'Number of installs in each category', text='Installs', color='Installs' , height=1500, orientation='h')
fig.show()


# In[ ]:


kd=app_data.groupby(['Category']).mean().sort_values(by='Rating', ascending=True).reset_index()
fig = px.bar(kd,
             x='Rating', y='Category',
             title=f'Mean of rating each category', text='Rating', color='Rating' , height=1500, orientation='h')
fig.show()


# In[ ]:


ad=app_data.groupby(['Android Ver']).sum().sort_values(by='Installs', ascending=True).reset_index()
fig = px.bar(ad,
             x='Installs', y='Android Ver',
             title=f'Number of installs for each version of Android', text='Installs', color='Installs' , height=1500, orientation='h')
fig.show()


# In[ ]:


fd=app_data.groupby(['Category']).sum().sort_values(by='Installs', ascending=True)
fig = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
fig.add_trace(go.Pie(labels=fd.index
                     , values=fd["Installs"]
                     , name="Installs"),1, 1)

fig.update_traces(hole=0.4, hoverinfo="label+percent+name")
fig.update_layout(
    
    title_text="Percentage of installation for each category",
    annotations=[dict(text='Installs', x=1, y=2, font_size=10, showarrow=False)])
fig.show()


# # Building a model and its validate with different parameters to prevent overfitting and underfitting

# In[ ]:


def get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid):
    my_model = XGBRegressor(n_estimators=max_leaf_nodes, learning_rate=0.01,n_jobs=4)
    my_model.fit(X_train, y_train)
    predictions = my_model.predict(X_valid)
    mae = mean_absolute_error(predictions, y_valid)
    return(mae)


# In[ ]:


cols_to_use = ['Installs', 'Reviews', 'Size','Category','Price']
object_cols=['Category']
X = app_data[cols_to_use]
y = app_data.Rating
my_app = pd.read_csv('../input/my-app/pre.csv')
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
label_encoder = LabelEncoder()
for col in object_cols:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_valid[col] = label_encoder.transform(X_valid[col])
    my_app[col]=label_encoder.transform(my_app[col])

for max_leaf_nodes in [5, 50, 500, 5000,10000,20000]:
    my_mae = get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid)
    print("n_estimators: %d  \t\t Mean Absolute Error:  %f" %(max_leaf_nodes, my_mae))


# In[ ]:


my_main_model = XGBRegressor(n_estimators=5000, learning_rate=0.01,n_jobs=4)
my_main_model.fit(X_train, y_train)
predictions = my_main_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))


# *We want to predict the rating of a software that in sports category and installation number is 100000 and its size is 17 MB with the reviews number 87510 and that price is 0.*

# In[ ]:


my_app


# In[ ]:


predictions = my_main_model.predict(my_app)
predictions


# **Based on this model, the rating of this software is 4.57**

# # Analyze the model and features used

# In[ ]:


perm = PermutationImportance(my_main_model, random_state=1).fit(X_valid, y_valid)
eli5.show_weights(perm, feature_names = X_valid.columns.tolist())


# In[ ]:


pdp_goals = pdp.pdp_isolate(model=my_main_model, dataset=X_valid, model_features=cols_to_use, feature='Reviews')


pdp.pdp_plot(pdp_goals, 'Reviews')
plt.show()


# In[ ]:


pdp_goals = pdp.pdp_isolate(model=my_main_model, dataset=X_valid, model_features=cols_to_use, feature='Installs')


pdp.pdp_plot(pdp_goals, 'Installs')
plt.show()


# In[ ]:


data_for_prediction = X_valid.iloc[5]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
explainer = shap.TreeExplainer(my_main_model)
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values,  data_for_prediction)


# In[ ]:


explainer = shap.TreeExplainer(my_main_model)

shap_values = explainer.shap_values(X_valid,check_additivity=False)

shap.summary_plot(shap_values, X_valid)


# In[ ]:




