#!/usr/bin/env python
# coding: utf-8

# **This is a short analysis I have done on the cov data until June 6. You could follow the below process to create the excel for a recent data and 
# then do all the analysis just by replaceing the data**

# In[ ]:


#import matplotlib.pyplot as pt
import numpy as np
import pandas as pd
#from sklearn import linear_model
#import seaborn as sns
import plotly.express as px
from plotly import graph_objects as go


# In[ ]:


# ## This part is the code I used to create the table I used to analyse the data



# df_confirmed = pd.read_csv("time_series_covid19_confirmed_US.csv")
# df_death = pd.read_csv("US_daily_death.csv")

#  df = df.iloc[:,6:]
# df = pd.melt(df, id_vars=['Province_State', 'Country_Region',"Lat","Long_","Combined_Key"],
#         var_name='Date', value_name='Confirmed_cases')

# df_death = df.iloc[:,6:]
# df_death = pd.melt(df_death, id_vars=['Province_State', 'Country_Region',"Lat","Long_","Combined_Key"],
#         var_name='Date', value_name='Confirmed_cases')


# conf_key = df_confirmed["Combined_Key"].unique()
# for i in conf_key:
#     in_list = df_confirmed[(df_confirmed["Combined_Key"] == i)].index
#     df_confirmed.loc[in_list[0],"Daily"] = 0
#     for k in range( 1, len(in_list)):
#         df_confirmed.loc[in_list[k],"Daily"]= ( df_confirmed.loc[in_list[k], "Confirmed_cases"] - 
#                                                   df_confirmed.loc[ in_list[k-1], "Confirmed_cases"] ) 
        
        
        
# ## getting ride of negetives due to wrong data in the excel file
# for i in df_confirmed.index:
#     if df_confirmed.loc[i,"Daily"] < 0:
#         df_confirmed.loc[i,"Daily"] = 0

        
        

        
# ## create one table with daily death and confirmed cases
# conf_key = df_confirmed["Combined_Key"].unique()
# #dates = df_death["Date"].unique()
# for i in conf_key:
#     in_list = df_death[(df_death["Combined_Key"] == i)].index
#     in_list2 = df_confirmed[(df_confirmed["Combined_Key"] == i)].index
#     for k in range( 0, len(in_list)):
#         df_death.loc[in_list[k],"Daily_cases"]= df_confirmed.loc[in_list2[k],"Daily"]
        
        
        
# df_death.to_csv("US_death2.csv", index = False)


# In[ ]:


all_data = pd.read_csv("/kaggle/input/fulldata1/US_daily_death2.csv")


# In[ ]:


all_data.dtypes


# In[ ]:


all_data.columns


# In[ ]:


all_data["Date"] = pd.to_datetime(all_data["Date"])


# In[ ]:


all_data


# In[ ]:


fig = px.pie(all_data, 
             values='Daily_cases', names='Province_State',
             title='U.S Total Cases Per State')

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(height=700, uniformtext_mode='hide')
fig.show()


# In[ ]:


fig = px.pie(all_data, 
             values='Daily_death', names='Province_State',
             title='U.S Total Death Per State')

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(height=700, uniformtext_mode='hide')
fig.show()


# In[ ]:


(px.scatter( x = all_data["Date"], y = all_data["Daily_death"]
            , color=all_data["Province_State"],
            title="US Daily Deaths by State")
 .update_layout(title_font_size=29)
 .update_xaxes(showgrid=False)
 .update_traces(
     line=dict(dash="dot", width=4),
     selector=dict(type="scatter", mode="lines"))
).show()


# It is possible to unselect states and then select states that you are looking for

# In[ ]:


(px.scatter( x = all_data["Date"], y = all_data["Daily_cases"]
            , color=all_data["Province_State"],
            title="US Daily Cases by State")
 .update_layout(title_font_size=29)
 .update_xaxes(showgrid=False)
 .update_traces(
     line=dict(dash="dot", width=4),
     selector=dict(type="scatter", mode="lines"))
).show()


# In[ ]:


group_cases = all_data.groupby(["Province_State", "Date" ]).sum()["Daily_cases"]
group_cases = group_cases.reset_index()


# In[ ]:


(px.scatter( x = group_cases["Date"], y = group_cases["Daily_cases"]
            , color=group_cases["Province_State"],
            title="US Daily Cases by State")
 .update_layout(title_font_size=29)
 .update_xaxes(showgrid=False)
 .update_traces(
     
     selector=dict(type="scatter", mode="lines"))
).show()


# In[ ]:


top_five = group_cases[group_cases["Date"] == "2020-06-6"].sort_values("Daily_cases",ascending = False).head(5)
top_five= top_five.reset_index()

temp = group_cases[(group_cases["Province_State"] == top_five["Province_State"][0]) | 
          (group_cases["Province_State"] == top_five["Province_State"][1]) |
             (group_cases["Province_State"] == top_five["Province_State"][2]) | 
          (group_cases["Province_State"] == top_five["Province_State"][3]) |
         (group_cases["Province_State"] == top_five["Province_State"][4])]

(px.scatter(temp, x="Date", y="Daily_cases", color="Province_State",
            facet_col="Province_State",
           title = "Top Five States with the highest cases", hover_name ="Province_State",
           facet_col_wrap=1
            )
 .update_layout(title_font_size=20)
 .update_xaxes(showgrid=False)
 .update_traces(
     line=dict(dash="dot", width=4),
     selector=dict(type="scatter", mode="lines"))
).show()


# In[ ]:


temp_case = all_data.groupby(["Province_State"]).sum()["Daily_cases"]
states = np.sort(all_data["Province_State"].unique())

fig = px.bar(y=temp_case, x=states, text = temp_case)
fig.update_traces(texttemplate='%{text:.8s}', textposition='inside')
fig.update_layout(uniformtext_minsize=4, height=500, uniformtext_mode='hide',
                 xaxis_tickangle=-45,font=dict(
        size=8,
    )
                 )
fig.show()


# In[ ]:


temp_death = all_data.groupby(["Province_State"]).sum()["Daily_death"]
states = np.sort(all_data["Province_State"].unique())

fig = px.bar(y=temp_death, x=states, text = temp_death)
fig.update_traces(texttemplate='%{text:.8s}', textposition='inside')
fig.update_layout(uniformtext_minsize=4, height=500, uniformtext_mode='hide',
                 xaxis_tickangle=-45,font=dict(
        size=8,
    )
                 )
fig.show()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=states,
    y=temp_case,
    name = "Confirmed Cases",
    marker_color='Red'))
fig.add_trace(go.Bar(
    x=states,
    y=temp_death,
    name = "Death cases",
    marker_color='Black'))
fig.update_layout(barmode='group', xaxis_tickangle=-45,
                 font=dict(
        size=8,
    ))
fig.show()


# In[ ]:


date_death = all_data.groupby(["Date"]).sum()["Daily_death"]
#temp = temp.sort_values()
states = all_data["Date"].unique()

fig = px.bar(top_five, y=date_death, x=states, text = date_death)
fig.update_traces(texttemplate='%{text:.8s}', textposition='inside')
fig.update_layout(uniformtext_minsize=4, height=500, uniformtext_mode='hide',
                 xaxis_tickangle=-45,font=dict(
        size=8), yaxis=dict(
        title='Death per Date',
        titlefont_size=16,
        tickfont_size=9,
    )
                 )
fig.show()


# In[ ]:


temp_case_ratio = (all_data.groupby("Province_State").sum()["Daily_cases"] / all_data.groupby("Province_State").sum()["Population"]) * 100000

states = np.sort(all_data["Province_State"].unique())
fig = px.bar(y=temp_case_ratio, x=states, title='Case per 100000 population')
fig.update_traces(texttemplate='%{text:.8s}', textposition='inside')
fig.update_layout(uniformtext_minsize=4, title_font_size=30, height=500, uniformtext_mode='hide',
                 xaxis_tickangle=-45,font=dict(
        size=8,
    ), yaxis=dict(
                    title = "",
        titlefont_size=16,
        tickfont_size=9,
                 ) )
fig.show()


# In[ ]:


temp_death_ratio = (all_data.groupby("Province_State").sum()["Daily_death"] /
                    all_data.groupby("Province_State").sum()["Population"]) * 100000
states = np.sort(all_data["Province_State"].unique())

fig = px.bar(y=temp_death_ratio, x=states, title='Death per 100000 population')
fig.update_traces(texttemplate='%{text:.8s}', textposition='inside')
fig.update_layout(uniformtext_minsize=4, title_font_size=30, height=500, uniformtext_mode='hide',
                 xaxis_tickangle=-45,font=dict(
        size=8,
    ), yaxis=dict(
                    title = "",
        titlefont_size=16,
        tickfont_size=9,
                 ) )
fig.show()

