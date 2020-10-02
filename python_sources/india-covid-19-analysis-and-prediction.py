#!/usr/bin/env python
# coding: utf-8

# # Analysis and prediction of Covid-19 in India

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.express as px


# In[ ]:


df_covid = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df_covid


# In[ ]:


df_cov_ind = df_covid[df_covid["Country/Region"] == "India"]
df_cov_ind


# In[ ]:


df_cov_ind["Active"] = df_cov_ind["Confirmed"] - df_cov_ind["Deaths"] - df_cov_ind["Recovered"]
df_cov_ind


# In[ ]:


current = df_cov_ind.iloc[-1]
dead = current["Deaths"]
recov = current["Recovered"]
act = current["Active"]
patient_state = [["Active",act],["Death",dead],["Recovered",recov]]
df = pd.DataFrame(patient_state, columns=["Patient State","Count"])
fig = px.pie(df, values="Count", names="Patient State", title="State of Patients in India", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure(data=[go.Bar(name="Recovered",x = df_cov_ind["ObservationDate"],y=df_cov_ind["Recovered"]),
                      go.Bar(name="Deaths",x = df_cov_ind["ObservationDate"],y=df_cov_ind["Deaths"]),
                     go.Bar(name="Active",x = df_cov_ind["ObservationDate"],y=df_cov_ind["Active"])])
fig.update_layout(barmode='stack',title="India Covid-19 Pandemic Timeline")
fig.show()


# ## Statewise analysis

# In[ ]:


df_cov_ind1 = pd.read_csv("../input/covid19-corona-virus-india-dataset/complete.csv")
df_cov_ind1


# In[ ]:


df_cov_ind1["Date"] = pd.to_datetime(df_cov_ind1["Date"])
df_cov_ind1 = df_cov_ind1.drop(["Total Confirmed cases (Indian National)","Total Confirmed cases ( Foreign National )",                                 "Latitude","Longitude"],axis=1)


# In[ ]:


df_cov_state = df_cov_ind1[df_cov_ind1["Date"] == df_cov_ind1["Date"].max()]

df_cov_state = df_cov_state.rename(columns={"Name of State / UT":"State/UT","Total Confirmed cases":"Confirmed","Cured/Discharged/Migrated":"Cured"})
df_cov_state.index = range(0,len(df_cov_state))
df_cov_state


# ### Number of confirmed cases by State/UT

# In[ ]:


fig = px.pie(df_cov_state[df_cov_state["Confirmed"]>100], values="Confirmed", names="State/UT", title="Number of confirmed by State/UT with major infection", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


import geopandas as gpd


# In[ ]:


df_map = gpd.read_file("../input/indian-states/Indian_States.shp")
df_map["st_nm"][df_map['st_nm'] == "NCT of Delhi"] = "Delhi"
df_map = df_map.sort_values("st_nm")


# In[ ]:


df_map_state = df_cov_state[df_cov_state["State/UT"]!="Ladakh"][["Confirmed","State/UT"]]
JandK = df_cov_state["Confirmed"][df_cov_state["State/UT"] == "Jammu and Kashmir"].values[0]
df_map_state["Confirmed"][df_map_state["State/UT"] == "Jammu and Kashmir"] = JandK + df_cov_state["Confirmed"][df_cov_state["State/UT"] == "Ladakh"].values[0]


# In[ ]:


df_map["st_nm"][df_map["st_nm"] == "Jammu & Kashmir"] = "Jammu and Kashmir"
df_map["st_nm"][df_map["st_nm"] == "Andaman & Nicobar Island"] = "Andaman and Nicobar Islands"
df_map["st_nm"][df_map["st_nm"] == "Telangana"] = "Telengana"
df_map["st_nm"][df_map["st_nm"] == "Arunanchal Pradesh"] = "Arunachal Pradesh"


# In[ ]:


merged_df = df_map.set_index('st_nm').join(df_map_state.set_index('State/UT'))
merged_df = merged_df.dropna()


# In[ ]:


fig, ax = plt.subplots(1, figsize=(20, 10))
ax.axis('off')
ax.set_title('State wise distribution of Covid-19 confirmed cases', fontdict={'fontsize': '25', 'fontweight' : '3'})
merged_df.plot(column='Confirmed', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)


# ### Number of Deaths by State/UT

# In[ ]:


fig = px.pie(df_cov_state[df_cov_state["Confirmed"]>100], values="Death", names="State/UT", title="Number of deaths by State/UT with major infection", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


df_map_state["Death"] = df_cov_state["Death"]
JandK = df_cov_state["Death"][df_cov_state["State/UT"] == "Jammu and Kashmir"].values[0]
df_map_state["Death"][df_map_state["State/UT"] == "Jammu and Kashmir"] = JandK + df_cov_state["Death"][df_cov_state["State/UT"] == "Ladakh"].values[0]


# In[ ]:


merged_df = df_map.set_index('st_nm').join(df_map_state.set_index('State/UT'))
merged_df = merged_df.dropna()


# In[ ]:


fig, ax = plt.subplots(1, figsize=(20, 10))
ax.axis('off')
ax.set_title('State wise distribution of Covid-19 deaths', fontdict={'fontsize': '25', 'fontweight' : '3'})
merged_df.plot(column='Death', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)


# ### Number of recovered by State/UT

# In[ ]:


fig = px.pie(df_cov_state[df_cov_state["Confirmed"]>100], values="Cured", names="State/UT", title="Number of recovered by State/UT with major infection", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


df_map_state["Recovered"] = df_cov_state["Cured"]
JandK = df_cov_state["Cured"][df_cov_state["State/UT"] == "Jammu and Kashmir"].values[0]
df_map_state["Recovered"][df_map_state["State/UT"] == "Jammu and Kashmir"] = JandK + df_cov_state["Cured"][df_cov_state["State/UT"] == "Ladakh"].values[0]


# In[ ]:


merged_df = df_map.set_index('st_nm').join(df_map_state.set_index('State/UT'))
merged_df = merged_df.dropna()


# In[ ]:


fig, ax = plt.subplots(1, figsize=(20, 10))
ax.axis('off')
ax.set_title('State wise distribution of Covid-19 Recovered', fontdict={'fontsize': '25', 'fontweight' : '3'})
merged_df.plot(column='Recovered', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)


# ### Number of Active cases by State/UT
# 

# In[ ]:


df_map_state["Active"] = df_map_state["Confirmed"] - (df_map_state["Death"]+df_map_state["Recovered"])
df_map_state


# In[ ]:


fig = px.pie(df_map_state[df_map_state["Confirmed"]>100], values="Active", names="State/UT", title="Number of Active cases by State/UT with major infection", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


merged_df = df_map.set_index('st_nm').join(df_map_state.set_index('State/UT'))
merged_df = merged_df.dropna()


# In[ ]:


fig, ax = plt.subplots(1, figsize=(20, 10))
ax.axis('off')
ax.set_title('State wise distribution of Covid-19 Active Cases', fontdict={'fontsize': '25', 'fontweight' : '3'})
merged_df.plot(column='Active', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)


# ## Statewise testing analysis

# In[ ]:


df_state_test = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")
df_state_test


# In[ ]:


df_state_test[df_state_test["State"] == "Manipur"]


# In[ ]:


df_state_test1 = pd.DataFrame(columns=["State","TotalSamples","Negative","Positive"])

for state in df_state_test["State"].unique():
    temp_df = df_state_test[df_state_test["State"] == state]
    temp_df = temp_df.dropna()
    
    temp_df.drop(["Date"],axis=1,inplace=True)
    if(len(temp_df) > 0):
        df_state_test1 = df_state_test1.append(temp_df.iloc[-1])
df_state_test = df_state_test1
df_state_test.index = range(0,len(df_state_test))
df_state_test["PositivePercent"] = df_state_test["Positive"]/df_state_test["TotalSamples"]
df_state_test["NegativePercent"] = df_state_test["Negative"]/df_state_test["TotalSamples"]
df_state_test


# In[ ]:


fig = go.Figure(data=[go.Bar(name="Positive",x = df_state_test["State"],y=df_state_test["Positive"]),
                      go.Bar(name="Negative",x = df_state_test["State"],y=df_state_test["Negative"])])
fig.update_layout(barmode='stack',title="Statewise testing results")
fig.show()


# In[ ]:


fig = px.pie(df_state_test, values="PositivePercent", names="State", title="States by positive test result percentage", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


fig = px.pie(df_state_test, values="TotalSamples", names="State", title="States by number of tests conducted", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


df_pop = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
df_pop = df_pop.rename(columns={"State / Union Territory":"State"})
df_state_testpop = df_state_test.join(df_pop.set_index('State'),how="left",on="State", rsuffix='Population')
df_state_testpop = df_state_testpop.drop(["Sno","Rural population","Urban population","Area","Density","Gender Ratio"],axis=1)
df_state_testpop["Test/1M"] = (df_state_testpop["TotalSamples"]/df_state_testpop["Population"])*1000000
df_state_testpop


# In[ ]:


df_state_testpop["TotalSamples"][df_state_testpop["State"] == "Andhra Pradesh"] = df_state_testpop["TotalSamples"][df_state_testpop["State"] == "Andhra Pradesh"].values[0] + df_state_testpop["TotalSamples"][df_state_testpop["State"] == "Telangana"].values[0]
df_state_testpop.drop(index = df_state_testpop["TotalSamples"][df_state_testpop["State"] == "Telangana"].index[0],axis=0,inplace=True)
df_state_testpop["Test/1M"] = (df_state_testpop["TotalSamples"]/df_state_testpop["Population"])*1000000
df_state_testpop


# In[ ]:


fig = px.bar(df_state_testpop,x="State",y="Test/1M",title = "States by No. of tests per million")
fig.show()


# In[ ]:


fig = px.pie(df_state_testpop, values="Test/1M", names="State", title="States by number of tests per million", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# ### Comparison of Public Health Facilities by state

# In[ ]:


df_state_phf = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
df_state_phf


# In[ ]:


df_state_phf.drop(index=len(df_state_phf)-1,axis=0,inplace=True)
df_state_phf.drop(["Sno","NumPrimaryHealthCenters_HMIS","NumCommunityHealthCenters_HMIS","NumSubDistrictHospitals_HMIS","NumDistrictHospitals_HMIS"],axis=1,inplace=True)


# In[ ]:


fig = px.bar(df_state_phf,x = "State/UT",y="TotalPublicHealthFacilities_HMIS",title="State by Health Facilities",labels={"TotalPublicHealthFacilities_HMIS":"Total Health Facilities"})
fig.show()


# In[ ]:


df_state_phf["TotalPublicHealthFacilities_HMIS"][df_state_phf["State/UT"] == "Andhra Pradesh"] = df_state_phf["TotalPublicHealthFacilities_HMIS"][df_state_phf["State/UT"] == "Andhra Pradesh"].values[0] + df_state_phf["TotalPublicHealthFacilities_HMIS"][df_state_phf["State/UT"] == "Telangana"].values[0]
df_state_phf.drop(index = df_state_phf["TotalPublicHealthFacilities_HMIS"][df_state_phf["State/UT"] == "Telangana"].index[0],axis=0,inplace=True)


# In[ ]:


df_state_phf["State/UT"][df_state_phf["State/UT"] == "Andaman & Nicobar Islands"] = "Andaman and Nicobar Islands"
df_state_phf["State/UT"][df_state_phf["State/UT"] == "Jammu & Kashmir"] = "Jammu and Kashmir"


# In[ ]:


df_state_phf = df_state_phf.join(df_pop.set_index('State'),how="left",on="State/UT", rsuffix='Population')
df_state_phf["Facilities/1M"] = (df_state_phf["TotalPublicHealthFacilities_HMIS"]/df_state_phf["Population"])*1000000
df_state_phf


# In[ ]:


df_state_phf = df_state_phf.dropna()

fig = px.bar(df_state_phf,x = "State/UT",y="Facilities/1M",title="Distribution of Health Facilities per million")
fig.show()


# ## Spread of Covid-19 in India over time

# In[ ]:


df_cov_ind1 = df_cov_ind1.rename(columns={"Name of State / UT":"State/UT"})
df_cov_ind1 = df_cov_ind1.rename(columns={"Name of State / UT":"State/UT","Total Confirmed cases":"Confirmed","Cured/Discharged/Migrated":"Cured"})


# In[ ]:


from PIL import Image
import datetime
from matplotlib.animation import FuncAnimation


# In[ ]:


len(df_cov_ind1["Date"].unique())


# In[ ]:



states = df_cov_ind1["State/UT"].unique()

#fig,ax = plt.subplots(figsize=(15,15))
for date in df_cov_ind1["Date"].unique()[::4]:
    
    df_temp = df_cov_ind1[df_cov_ind1["Date"] == date]
    
    for s in states:
        if s not in df_temp["State/UT"].values:
            df_temp = df_temp.append({"Date":date,"State/UT":s,"Cured":0,"Confirmed":0,"Death":0},ignore_index=True)
    df_temp1 = df_temp[df_temp["State/UT"]!="Ladakh"][["Date","Confirmed","Cured","Death","State/UT"]]
    JandK = df_temp["Confirmed"][df_temp["State/UT"] == "Jammu and Kashmir"].values[0]
    df_temp1["Confirmed"][df_temp1["State/UT"] == "Jammu and Kashmir"] = JandK + df_temp["Confirmed"][df_temp["State/UT"] == "Ladakh"].values[0]
    JandK = df_temp["Cured"][df_temp["State/UT"] == "Jammu and Kashmir"].values[0]
    df_temp1["Cured"][df_temp1["State/UT"] == "Jammu and Kashmir"] = JandK + df_temp["Cured"][df_temp["State/UT"] == "Ladakh"].values[0]
    JandK = df_temp["Death"][df_temp["State/UT"] == "Jammu and Kashmir"].values[0]
    df_temp1["Death"][df_temp1["State/UT"] == "Jammu and Kashmir"] = JandK + df_temp["Death"][df_temp["State/UT"] == "Ladakh"].values[0]
    
    df_temp = df_temp1
    #df_temp["Active"] = df_temp["Confirmed"] - df_temp["Death"] - df_temp["Cured"]
    
    merged_df = df_map.set_index('st_nm').join(df_temp.set_index('State/UT'))
    merged_df = merged_df.dropna()
    
    fig,ax = plt.subplots(figsize=(15,15))
    fig.patch.set_facecolor("blue")
    ax.set_facecolor("blue")
    fig1 = merged_df.plot(column='Confirmed', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    fig1.axis('off')
    fig1.set_title('State wise distribution of Covid-19 Confirmed Cases', fontdict={'fontsize': '25', 'fontweight' : '3'})
    
    date1 = datetime.datetime.utcfromtimestamp(date.tolist()/1e9)
    
    date1 = date1.strftime('%Y.%m.%d')
    fig1.annotate(date1,xy=(0.1, .225), xycoords='figure fraction',horizontalalignment='left', verticalalignment='top',fontsize=35)
    
    chart = fig1.get_figure()
    chart.patch.set_facecolor("blue")
    #plt.close(chart)
    chart.savefig(date1+"img.jpg",facecolor='blue')
    plt.close(fig)
    fig1.clear()
    fig.clear()
    ax.clear()
    #images.append(Image.open("img.jpg"))
    
#images[0].save("map.gif",save_all=True,append_images=images[1:],optimize=False,duration=100,loop=0)
    


# In[ ]:


images = []
for date in df_cov_ind1["Date"].unique()[::4]:
    date1 = datetime.datetime.utcfromtimestamp(date.tolist()/1e9)
    date1 = date1.strftime('%Y.%m.%d')
    images.append(Image.open(date1+"img.jpg"))
    
images[0].save("map.gif",save_all=True,append_images=images[1:],optimize=False,duration=1000,loop=0)


# ![SegmentLocal](map.gif "map")
# 

# ## Districtwise Covid-19 zones in India

# In[ ]:


df_zones = pd.read_csv("../input/covid19-corona-virus-india-dataset/zones.csv")
df_zones


# In[ ]:


df_dist = gpd.read_file("../input/districts1/gadm36_IND_2.shp")
df_dist = df_dist[["NAME_2","geometry"]]
df_dist.index = range(len(df_dist))
df_dist = df_dist.rename(columns={"NAME_2":"district"})
df_dist


# In[ ]:


df_dist["district"][df_dist["district"] == "Ahmadabad"] = "Ahmedabad"
df_dist["district"][df_dist["district"] == "Ahmadnagar"] = "Ahmednagar"

df_zones["district"][df_zones["district"] == "Warangal Urban"] = "Warangal"
df_zones = df_zones[df_zones["district"] != "Warangal Rural"]


# In[ ]:


df_zones.drop(["lastupdated","source","state","statecode","districtcode"],axis=1,inplace=True)


# In[ ]:


df_distzones = df_dist.set_index("district").join(df_zones.set_index("district"))
df_distzones


# In[ ]:


from matplotlib.colors import LinearSegmentedColormap
cmap = [(0,1,0),(1,0.65,0),(1,0,0)]
cmap = LinearSegmentedColormap.from_list("zones",cmap)


# In[ ]:


df_distzones = df_distzones.fillna("Orange")
fig = df_distzones.plot(column="zone", cmap=cmap, linewidth=0.8,figsize=(15,15), edgecolor='0.8', legend=True)
fig.axis("off")
fig.set_title("Coronavirus Pandemic zones in India(Maps of some zones are not available as per the zone, hence such places are filled Orange)")


# ## Distribution of cases by age

# In[ ]:


df_age = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")

fig = px.bar(df_age,x = "AgeGroup",y="TotalCases",title="Distribution of Covid-19 cases by age group")
fig.show()


# In[ ]:


fig = px.pie(df_age, values="TotalCases", names="AgeGroup", title="Distribution of Covid-19 by age in India", template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# ## Genderwise distribution

# In[ ]:


df_pat = pd.read_csv("../input/covid19-corona-virus-india-dataset/patients_data.csv")
df_pat = df_pat[["gender","current_status"]]
df_pat = df_pat.dropna()
#df_pat.index = range(len(df_pat))
df_pat


# In[ ]:


m_pat = (df_pat["gender"] == "M").sum()
f_pat = (df_pat["gender"] == "F").sum()

fig = go.Figure(data=[
    go.Bar(name='Male', x=["Count"], y=[m_pat]),
    go.Bar(name='Female', x=["Count"], y=[f_pat])
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# In[ ]:


m = df_pat["gender"] == "M"
f = df_pat["gender"] == "F"
m_hosp = (df_pat["current_status"][m] == "Hospitalized").sum()
f_hosp = (df_pat["current_status"][f] == "Hospitalized").sum()
m_dec = (df_pat["current_status"][m] == "Deceased").sum()
f_dec = (df_pat["current_status"][f] == "Deceased").sum()
m_rec = (df_pat["current_status"][m] == "Recovered").sum()
f_rec = (df_pat["current_status"][f] == "Recovered").sum()

fig = go.Figure(data=[
    go.Bar(name='Male', x=["Hospitalized","Deceased","Recovered"], y=[m_hosp,m_dec,m_rec]),
    go.Bar(name='Female', x=["Hospitalized","Deceased","Recovered"], y=[f_hosp,f_dec,f_rec])
])
# Change the bar mode
fig.update_layout(barmode='group',title="Distribution of patient state by gender(Patient with not known gender not included)")
fig.show()


# ## Prediction of the Pandemic

# In[ ]:


import statsmodels.api as sm
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.neural_network import MLPRegressor
from datetime import timedelta


# ### Using Regressor models
# 1) MLP Regressor

# ### i. Total cases

# In[ ]:


x = np.arange(len(df_cov_ind)).reshape(-1,1)
y = df_cov_ind["Confirmed"].values


# In[ ]:


mlp_model1 = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)
mlp_model1.fit(x, y)


# In[ ]:


pred = mlp_model1.predict(x)

fig = go.Figure(data = [go.Scatter(x = df_cov_ind["ObservationDate"], y = pred, mode="lines+markers", name = "Predicted"),
               go.Scatter(x = df_cov_ind["ObservationDate"], y = y, mode="lines+markers", name = "Actual")])
fig.update_layout(title = "Predicted values vs Actual values")
fig.show()


# In[ ]:


df_cov_ind["ObservationDate"] = pd.to_datetime(df_cov_ind["ObservationDate"])


# In[ ]:


fut_pred = mlp_model1.predict(np.arange(len(df_cov_ind),len(df_cov_ind)+90).reshape(-1,1))
fut_time = [df_cov_ind["ObservationDate"].iloc[-1] + timedelta(days=i) for i in range(1,len(fut_pred))]

fig = go.Figure(data=go.Scatter(x=fut_time, y=fut_pred, mode="lines+markers"))
fig.update_layout(title="Forecast 3 months")
fig.show()


# ### ii. Deaths

# In[ ]:


x = np.arange(len(df_cov_ind)).reshape(-1,1)
y = df_cov_ind["Deaths"].values


# In[ ]:


mlp_model2 = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)
mlp_model2.fit(x, y)


# In[ ]:


pred = mlp_model2.predict(x)

fig = go.Figure(data = [go.Scatter(x = df_cov_ind["ObservationDate"], y = pred, mode="lines+markers", name = "Predicted"),
               go.Scatter(x = df_cov_ind["ObservationDate"], y = y, mode="lines+markers", name = "Actual")])
fig.update_layout(title = "Predicted values vs Actual values")
fig.show()


# In[ ]:


fut_pred = mlp_model2.predict(np.arange(len(df_cov_ind),len(df_cov_ind)+90).reshape(-1,1))
fut_time = [df_cov_ind["ObservationDate"].iloc[-1] + timedelta(days=i) for i in range(1,len(fut_pred))]

fig = go.Figure(data=go.Scatter(x=fut_time, y=fut_pred, mode="lines+markers"))
fig.update_layout(title="Forecast 3 months")
fig.show()


# ### iii. Recovered

# In[ ]:


x = np.arange(len(df_cov_ind)).reshape(-1,1)
y = df_cov_ind["Recovered"].values


# In[ ]:


mlp_model3 = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)
mlp_model3.fit(x, y)


# In[ ]:


pred = mlp_model3.predict(x)

fig = go.Figure(data = [go.Scatter(x = df_cov_ind["ObservationDate"], y = pred, mode="lines+markers", name = "Predicted"),
               go.Scatter(x = df_cov_ind["ObservationDate"], y = y, mode="lines+markers", name = "Actual")])
fig.update_layout(title = "Predicted values vs Actual values")
fig.show()


# In[ ]:


fut_pred = mlp_model3.predict(np.arange(len(df_cov_ind),len(df_cov_ind)+90).reshape(-1,1))
fut_time = [df_cov_ind["ObservationDate"].iloc[-1] + timedelta(days=i) for i in range(1,len(fut_pred))]

fig = go.Figure(data=go.Scatter(x=fut_time, y=fut_pred, mode="lines+markers"))
fig.update_layout(title="Forecast 3 months")
fig.show()


# ### Using Prophet

# ### i. Confirmed

# In[ ]:


pr_data= pd.DataFrame()
pr_data["ds"] = df_cov_ind["ObservationDate"]
pr_data["y"] = df_cov_ind["Confirmed"]
pr_data.index = range(len(pr_data))
pr_data.head()


# In[ ]:


pr_data['y'] = np.log(pr_data['y'] + 1)
m1=Prophet()
m1.fit(pr_data)


# In[ ]:


# compare actual vs predicted
pred_date = pd.DataFrame(df_cov_ind["ObservationDate"])
pred_date.columns = ['ds']
pred = m1.predict(pred_date)
pred['yhat'] = np.exp(pred['yhat']) - 1


fig = go.Figure(data = [go.Scatter(x = df_cov_ind["ObservationDate"], y = pred.yhat, mode="lines+markers", name = "Predicted"),
               go.Scatter(x = df_cov_ind["ObservationDate"], y = df_cov_ind["Confirmed"], mode="lines+markers", name = "Actual")])
fig.update_layout(title = "Predicted values vs Actual values")
fig.show()


# In[ ]:


future=pd.DataFrame([df_cov_ind["ObservationDate"].iloc[-1] + timedelta(i+1) for i in range(120)])
future.columns = ['ds']
forecast=m1.predict(future)
forecast['yhat'] = np.exp(forecast['yhat']) - 1

fig = go.Figure(data=go.Scatter(x=future["ds"], y=forecast.yhat, mode="lines+markers"))
fig.update_layout(title="Forecast 4 months")
fig.show()


# ### ii. Deaths

# In[ ]:


pr_data= pd.DataFrame()
pr_data["ds"] = df_cov_ind["ObservationDate"]
pr_data["y"] = df_cov_ind["Deaths"]
pr_data.index = range(len(pr_data))
pr_data.head()


# In[ ]:


pr_data['y'] = np.log(pr_data['y'] + 1)
m2=Prophet()
m2.fit(pr_data)


# In[ ]:


pred_date = pd.DataFrame(df_cov_ind["ObservationDate"])
pred_date.columns = ['ds']
pred = m2.predict(pred_date)
pred['yhat'] = np.exp(pred['yhat']) - 1


fig = go.Figure(data = [go.Scatter(x = df_cov_ind["ObservationDate"], y = pred.yhat, mode="lines+markers", name = "Predicted"),
               go.Scatter(x = df_cov_ind["ObservationDate"], y = df_cov_ind["Deaths"], mode="lines+markers", name = "Actual")])
fig.update_layout(title = "Predicted values vs Actual values")
fig.show()


# In[ ]:


future=pd.DataFrame([df_cov_ind["ObservationDate"].iloc[-1] + timedelta(i+1) for i in range(120)])
future.columns = ['ds']
forecast=m2.predict(future)
forecast['yhat'] = np.exp(forecast['yhat']) - 1

fig = go.Figure(data=go.Scatter(x=future["ds"], y=forecast.yhat, mode="lines+markers"))
fig.update_layout(title="Forecast 4 months")
fig.show()


# ### iii. Recovered

# In[ ]:


pr_data= pd.DataFrame()
pr_data["ds"] = df_cov_ind["ObservationDate"]
pr_data["y"] = df_cov_ind["Recovered"]
pr_data.index = range(len(pr_data))
pr_data.head()


# In[ ]:


pr_data['y'] = np.log(pr_data['y'] + 1)
m3=Prophet()
m3.fit(pr_data)


# In[ ]:


pred_date = pd.DataFrame(df_cov_ind["ObservationDate"])
pred_date.columns = ['ds']
pred = m3.predict(pred_date)
pred['yhat'] = np.exp(pred['yhat']) - 1


fig = go.Figure(data = [go.Scatter(x = df_cov_ind["ObservationDate"], y = pred.yhat, mode="lines+markers", name = "Predicted"),
               go.Scatter(x = df_cov_ind["ObservationDate"], y = df_cov_ind["Recovered"], mode="lines+markers", name = "Actual")])
fig.update_layout(title = "Predicted values vs Actual values")
fig.show()


# In[ ]:


future=pd.DataFrame([df_cov_ind["ObservationDate"].iloc[-1] + timedelta(i+1) for i in range(120)])
future.columns = ['ds']
forecast=m3.predict(future)
forecast['yhat'] = np.exp(forecast['yhat']) - 1

fig = go.Figure(data=go.Scatter(x=future["ds"], y=forecast.yhat, mode="lines+markers"))
fig.update_layout(title="Forecast 4 months")
fig.show()


# ### Using ARIMA

# ### i. Confirmed

# In[ ]:


arima_data = pd.DataFrame()
arima_data["confirmed_data"] = df_cov_ind["ObservationDate"]
arima_data["count"] = df_cov_ind["Confirmed"]
arima_data.index = range(len(arima_data))
arima_data.head()


# In[ ]:


get_ipython().system('pip install pmdarima')


# In[ ]:


from pmdarima import auto_arima

stepwise_fit = auto_arima(arima_data['count'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',    
                          suppress_warnings = True,  
                          stepwise = True)           
stepwise_fit.summary()


# In[ ]:


model1= SARIMAX(arima_data['count'],order=(1,2,2),seasonal_order=(1,1,1,12)) #Change the model as per the result of above as the dataset is updated
fit_model1 = model1.fit(full_output=True, disp=True)
fit_model1.summary()


# In[ ]:


pred = fit_model1.predict(0,len(arima_data)-1)

fig = go.Figure(data = [go.Scatter(x = df_cov_ind["ObservationDate"], y = pred, mode="lines+markers", name = "Predicted"),
               go.Scatter(x = df_cov_ind["ObservationDate"], y = df_cov_ind["Confirmed"], mode="lines+markers", name = "Actual")])
fig.update_layout(title = "Predicted values vs Actual values")
fig.show()


# In[ ]:


forecast = fit_model1.forecast(steps=90)
fig = go.Figure(data=go.Scatter(x=fut_time, y=forecast, mode="lines+markers"))
fig.update_layout(title="Forecast 3 months")
fig.show()


# ### ii. Deaths

# In[ ]:


arima_data = pd.DataFrame()
arima_data["confirmed_data"] = df_cov_ind["ObservationDate"]
arima_data["count"] = df_cov_ind["Deaths"]
arima_data.index = range(len(arima_data))
arima_data.head()


# In[ ]:


stepwise_fit = auto_arima(arima_data['count'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',    
                          suppress_warnings = True,  
                          stepwise = True)           
stepwise_fit.summary()


# In[ ]:


model2= SARIMAX(arima_data['count'],order=(0,2,1),seasonal_order=(0,1,1,12)) #Change the model as per the result of above as the dataset is updated
fit_model2 = model2.fit(full_output=True, disp=True)
fit_model2.summary()


# In[ ]:


pred = fit_model2.predict(0,len(arima_data)-1)

fig = go.Figure(data = [go.Scatter(x = df_cov_ind["ObservationDate"], y = pred, mode="lines+markers", name = "Predicted"),
               go.Scatter(x = df_cov_ind["ObservationDate"], y = df_cov_ind["Deaths"], mode="lines+markers", name = "Actual")])
fig.update_layout(title = "Predicted values vs Actual values")
fig.show()


# In[ ]:


forecast = fit_model2.forecast(steps=90)
fig = go.Figure(data=go.Scatter(x=fut_time, y=forecast, mode="lines+markers"))
fig.update_layout(title="Forecast 3 months")
fig.show()


# ### iii. Recovered

# In[ ]:


arima_data = pd.DataFrame()
arima_data["confirmed_data"] = df_cov_ind["ObservationDate"]
arima_data["count"] = df_cov_ind["Recovered"]
arima_data.index = range(len(arima_data))
arima_data.head()


# In[ ]:


stepwise_fit = auto_arima(arima_data['count'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',    
                          suppress_warnings = True,  
                          stepwise = True)           
stepwise_fit.summary()


# In[ ]:


model3= SARIMAX(arima_data['count'],order=(0,2,0),seasonal_order=(0,1,0,12)) #Change the model as per the result of above as the dataset is updated
fit_model3 = model3.fit(full_output=True, disp=True)
fit_model3.summary()


# In[ ]:


pred = fit_model3.predict(0,len(arima_data)-1)

fig = go.Figure(data = [go.Scatter(x = df_cov_ind["ObservationDate"], y = pred, mode="lines+markers", name = "Predicted"),
               go.Scatter(x = df_cov_ind["ObservationDate"], y = df_cov_ind["Recovered"], mode="lines+markers", name = "Actual")])
fig.update_layout(title = "Predicted values vs Actual values")
fig.show()


# In[ ]:


forecast = fit_model3.forecast(steps=90)
fig = go.Figure(data=go.Scatter(x=fut_time, y=forecast, mode="lines+markers"))
fig.update_layout(title="Forecast 3 months")
fig.show()


# 
