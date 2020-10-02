#!/usr/bin/env python
# coding: utf-8

# <h1> Welcome to my weather EDA FORECAST</h1>

# In[ ]:


import pandas as pd


# In[ ]:


df=pd.read_csv("/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv")


# In[ ]:


import altair as alt
import altair_render_script


# In[ ]:


df=df[df.AvgTemperature>-99.0]


# In[ ]:


df_year=df.groupby(["Year","Country","Region"]).mean().reset_index()
list_countries=df_year["Country"].unique()
print(len(list_countries))


# In[ ]:




df_year


# In[ ]:


df_months=df.groupby(["Country","Month","Year"]).mean().reset_index()


# In[ ]:





# In[ ]:


alt.data_transformers.disable_max_rows()


# In[ ]:


# A dropdown filter
some_countries=df_year.Country.unique()
country_dropdown = alt.binding_select(options=some_countries)
country_select = alt.selection_single(fields=['Country'], bind=country_dropdown, name="Country",
                                      init={"Country":"India"})
year_dropdown = alt.binding_select(options=df_year.Year.unique())
year_select = alt.selection_single(fields=['Year'], bind=year_dropdown, name="Year",
                                      init={"Year":2000})
base=alt.Chart(df_year).mark_line(point=True).encode(
alt.X("Year:N"),
alt.Y("AvgTemperature",scale=alt.Scale(zero=False)),
    tooltip=["AvgTemperature","Country","Year"]
)
nbase=alt.Chart(df_months).mark_line(point=True).encode(
alt.X("Month:N",axis=alt.Axis(tickCount=12)),
    alt.Y("AvgTemperature",scale=alt.Scale(zero=False)),
    
    tooltip=["AvgTemperature","Country","Month"]
)
filter_countries = alt.hconcat(base.add_selection(
   country_select
).transform_filter(
   country_select
).properties(title="Dropdown Filtering"),nbase.transform_filter(country_select).add_selection(
year_select
).transform_filter(year_select))



filter_countries


# In[ ]:



#some_countries=["India","Australia","Canada","China","Japan","South Africa","Bangladesh","Brazil","North Korea"]
base=alt.hconcat()
main=alt.vconcat()
for i in range(125):
    if i%25==0 and i!=0:
        main=alt.vconcat(main,base)
        base=alt.hconcat()
        
    else:
        
        temp=alt.Chart(df_year[df_year.Country==some_countries[i]]).mark_line(  ).encode(
        alt.X("Year:N",),
        alt.Y("AvgTemperature",scale=alt.Scale(zero=False))).properties(title=some_countries[i])
        base=alt.hconcat(base,temp)
main


# In[ ]:





# In[ ]:


df_total=df_year.groupby(["Year"]).mean().reset_index()
df_total
alt.Chart(df_total).mark_line().encode(alt.X("Year"),
                                       alt.Y("AvgTemperature",scale=alt.Scale(zero=False))).properties(title="The Avg Temperature of all Regions Over the years")


# In[ ]:


df=df.sort_values(["Year","Country","City","Month","Day"])


# In[ ]:


df=df.drop(["Region"],axis=1)


# Here I add another column for called yTemperature, basically I am taking the previous day temperature and for the first entry taking it's own temperature

# In[ ]:


import numpy as np
TempList=df.AvgTemperature.values
print(TempList)
newList=np.array([64.2])
newList=np.append(newList,TempList)
print(newList)
newList=newList[:len(TempList)]
newList


# In[ ]:


df["yTemperature"]=newList


# In[ ]:


df=df.drop(["State"],axis=1)


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Xcity=le.fit_transform(df["City"])
Xcountry=le.fit_transform(df["Country"])


# In[ ]:


df["Country"]=Xcountry
df["City"]=Xcity


# In[ ]:


X_test=df[df.Year==2020]
y_test=X_test.pop("AvgTemperature")
X_train=df[df.Year!=2020]
y_train=X_train.pop("AvgTemperature")
#y=df.pop("AvgTemperature")



# In[ ]:


from xgboost import XGBRegressor
model = XGBRegressor(n_estimators = 20 , random_state = 0 , max_depth = 3)
model.fit(X_train,y_train)
print(model.score(X_train,y_train))


# In[ ]:


print(model.score(X_test,y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




