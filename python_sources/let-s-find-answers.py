#!/usr/bin/env python
# coding: utf-8

# <div class="list-group" id="list-tab" role="tablist">
#   <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Let's find answers</h3>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Report-of-US-Cars" role="tab" aria-controls="messages">Report of US Cars<span class="badge badge-primary badge-pill"></span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#Description-of-Cars" role="tab" aria-controls="messages">Description of Cars<span class="badge badge-primary badge-pill"></span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#How-Price-is-distributed??" role="tab" aria-controls="settings">How Price is distributed<span class="badge badge-primary badge-pill"></span></a> 
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#Which-brand-has-expensive-cars??" role="tab" aria-controls="settings">Which brand has expensive cars<span class="badge badge-primary badge-pill"></span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Which-Color-is-popular??" role="tab" aria-controls="settings">Which Color is popular<span class="badge badge-primary badge-pill"></span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Which-brand-is-popular??" role="tab" aria-controls="settings">Which brand is popular<span class="badge badge-primary badge-pill"></span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Which-model-is-popular??" role="tab" aria-controls="settings">Which model is popular<span class="badge badge-primary badge-pill"></span></a>  
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Which-State-sell-more-cars??" role="tab" aria-controls="settings">Which State sell more cars<span class="badge badge-primary badge-pill"></span></a>  
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Which-year-has-more-selling??" role="tab" aria-controls="settings">Which year has more selling<span class="badge badge-primary badge-pill"></span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#Which-year-has-more-selling??" role="tab" aria-controls="settings">Which year has more selling<span class="badge badge-primary badge-pill"></span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#What-is-total-selling-per-year??" role="tab" aria-controls="settings">What is total selling per year<span class="badge badge-primary badge-pill"></span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#What-is-total-selling-per-year??" role="tab" aria-controls="settings">What is total selling per year<span class="badge badge-primary badge-pill"></span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#How-mileage-is-distributed" role="tab" aria-controls="settings">How mileage is distributed<span class="badge badge-primary badge-pill"></span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#How-price-is-related-to-mileage" role="tab" aria-controls="settings">How price is related to mileage<span class="badge badge-primary badge-pill"></span></a>
#       <a class="list-group-item list-group-item-action" data-toggle="list" href="#What-is-status-of-car??" role="tab" aria-controls="settings">What is status of car<span class="badge badge-primary badge-pill"></span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#How-price-is-dependent-on-status??" role="tab" aria-controls="settings">How price is dependent on status<span class="badge badge-primary badge-pill"></span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#How-mileage-is-dependent-on-status??" role="tab" aria-controls="settings">How mileage is dependent on status<span class="badge badge-primary badge-pill"></span></a>
#       

# # Library and Data

# In[ ]:


import pandas as pd
from pandas_profiling import ProfileReport 
import seaborn as sns
import matplotlib as plt
import plotly.express as px
data = pd.read_csv("../input/usa-cers-dataset/USA_cars_datasets.csv")
data = data.replace({"doors":"door"})
data.head(3)


# # Report of US Cars

# In[ ]:


report = ProfileReport(data)
report


# # Description of Cars

# In[ ]:


fig = px.treemap(data, path=["brand",'model','color'],
                  color='brand', hover_data=['model'],
                  color_continuous_scale='rainbow')
fig.show()


# # How Price is distributed??

# In[ ]:



sns.set_style("darkgrid")
sns.kdeplot(data=data['price'],label="Price" ,shade=True)


# # Which brand has expensive cars??

# In[ ]:


price = data.groupby('brand')['price'].max().reset_index()
price  = price.sort_values(by="price")
price = price.tail(10)
fig = px.pie(price,
             values="price",
             names="brand",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# # Which Color is popular??

# In[ ]:


color = data.loc[:,["color"]]
color['count'] = color.groupby([color.color])['color'].transform('count')
color = color.drop_duplicates()
color = color.sort_values(by="count",ascending = False)
color = color.head(10)
fig = px.pie(color,
             values="count",
             names="color",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# # Which brand is popular??

# In[ ]:


brand = data.loc[:,["brand"]]
brand['count'] = brand.groupby([brand.brand])['brand'].transform('count')
brand = brand.drop_duplicates()
brand = brand.sort_values(by="count",ascending = False)
brand = brand.head(10)
fig = px.pie(brand,
             values="count",
             names="brand",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# # What is Brand popularity per year??

# In[ ]:


#top brand= Ford
perb= data.loc[:,["year","brand"]]
perb['count'] = perb.groupby([perb.brand,perb.year])['brand'].transform('count')
perb= perb.drop_duplicates()
perb= perb.sort_values(by="year",ascending = False)
top_brand = ['ford', 'dodge',"nissan"] 
perb = perb.loc[perb['brand'].isin(top_brand)] 
perb = perb[perb.year>2015]
perb = perb.sort_values(by="year")

fig=px.bar(perb,x='brand', y="count", animation_frame="year", 
           animation_group="brand", color="brand", hover_name="brand")
fig.show()


# # Which model is popular??

# In[ ]:


model= data.loc[:,["model"]]
model['count'] = model.groupby([model.model])['model'].transform('count')
model= model.drop_duplicates()
model= model.sort_values(by="count",ascending = False)
model= model.head(10)
fig = px.pie(model,
             values="count",
             names="model",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# # What is model popularity per year

# In[ ]:


#top brand= Ford
perm= data.loc[:,["year","model"]]
perm['count'] = perm.groupby([perm.model,perm.year])['model'].transform('count')
perm= perm.drop_duplicates()
perm= perm.sort_values(by="year",ascending = False)
top_model = ['door', 'f-150',"caravan"] 
perm = perm.loc[perm['model'].isin(top_model)] 
perm = perm[perm.year>2016]
perm = perm.sort_values(by="year")

fig=px.bar(perm,x='model', y="count", animation_frame="year", 
           animation_group="model", color="model", hover_name="model")
fig.show()


# # Which State sell more cars??

# In[ ]:


state= data.loc[:,["state"]]
state['count'] = state.groupby([state.state])['state'].transform('count')
state= state.drop_duplicates()
state= state.sort_values(by="count",ascending = False)
state= state.head(10)
fig = px.pie(state,
             values="count",
             names="state",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# # Which state sell more cars yearly??

# In[ ]:


st= data.loc[:,["year","state"]]
st['count'] = st.groupby([st.state,st.year])['state'].transform('count')
st= st.drop_duplicates()
st= st.sort_values(by="year",ascending = False)
top_state = ['pennsylvania','florida','texas','california']
st = st.loc[st['state'].isin(top_state)] 
st = st[st.year>2012]
st = st.sort_values(by="year")

fig=px.bar(st,x='state', y="count", animation_frame="year", 
           animation_group="state", color="state", hover_name="state")
fig.show()


# # Which year has more selling??

# In[ ]:


year= data.loc[:,["year"]]
year['count'] = year.groupby([year.year])['year'].transform('count')
year= year.drop_duplicates()
year= year.sort_values(by="count",ascending = False)
year= year.head(10)
fig = px.pie(year,
             values="count",
             names="year",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# # What is total selling per year??

# In[ ]:


sp= data.loc[:,["year","price"]]
sp['count'] = sp.groupby([sp.year])['price'].transform('sum')
sp= sp.drop_duplicates()
sp= sp.sort_values(by="count",ascending = False)
fig = px.pie(sp,
             values="count",
             names="year",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# # How mileage is distributed

# In[ ]:



sns.set_style("darkgrid")
sns.kdeplot(data=data['mileage'],label="Mileage" ,shade=True)


# # How price is related to mileage

# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(data.mileage,data.price,)
plt.xlabel("Price")
plt.xlabel("Mileage")


# # What is status of car??

# In[ ]:


import seaborn as sns
sns.set(style="darkgrid")
ax = sns.countplot(x="title_status", data=data)


# # How price is dependent on status??

# In[ ]:


ax = sns.violinplot(x="title_status", y="price",
                    data=data, palette="muted")
ax = sns.catplot(x="title_status", y="price", data=data)


# # How mileage is dependent on status??

# In[ ]:


ax = sns.violinplot(x="title_status", y="mileage",
                    data=data, palette="muted")
ax = sns.catplot(x="title_status", y="mileage", data=data)

