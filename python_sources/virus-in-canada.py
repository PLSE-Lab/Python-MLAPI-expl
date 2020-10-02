#!/usr/bin/env python
# coding: utf-8

# ![](https://media.giphy.com/media/dVuyBgq2z5gVBkFtDc/giphy.gif)

# **Coronaviruses are a large family of viruses which may cause illness in animals or humans. In humans, several coronaviruses are known to cause respiratory infections ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS) and Severe Acute Respiratory Syndrome (SARS). The most recently discovered coronavirus causes coronavirus disease COVID-19.COVID-19 is the infectious disease caused by the most recently discovered coronavirus. This new virus and disease were unknown before the outbreak began in Wuhan, China, in December 2019.**
# * [Source](https://www.who.int/news-room/q-a-detail/q-a-coronaviruses)

# # Coronavirus in the world

# In[ ]:


import pandas as pd 
cases = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
import plotly.offline as py
import plotly.express as px


py.init_notebook_mode(connected=True)

grp = cases.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
grp = grp.reset_index()
grp['Date'] = pd.to_datetime(grp['ObservationDate'])
grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')
grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']
grp['Country'] =  grp['Country/Region']

fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                     color="Confirmed", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="natural earth",
                     animation_frame="Date",width=1000, height=700,
                     color_continuous_scale='Reds',
                     range_color=[1000,100000],

                     title='World Map of Coronavirus')

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)


# # **Related Work**
#  *  For Analysis and Prediction on Coronavirus(South-Korea), Click [here](https://www.kaggle.com/vanshjatana/analysis-on-coronavirus)
# * For Analysis and Prediction on Coronavirus(Italy), Click [here](https://www.kaggle.com/vanshjatana/analysis-and-prediction-on-coronavirus-italy?scriptVersionId=29892166)
# *  For Analysis and Prediction on Coronavirus(Iran), Click [here](https://www.kaggle.com/vanshjatana/analysis-and-prediction-on-coronavirus-iran)
# *  For Machine Learning on Cornovirus, Click [here](https://www.kaggle.com/vanshjatana/machine-learning-on-coronavirus)
# *  For report on Coronavirus, Click [here](https://www.researchgate.net/publication/339738108_Analysis_On_Coronavirus)

# # **Canada**

# **Importing Libraries**

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
from datetime import date, timedelta
from sklearn.cluster import KMeans
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import plotly.offline as py
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# Symptoms of Coronavirus
# 
# 

# In[ ]:


symptoms={'symptom':['Fever',
        'Dry cough',
        'Fatigue',
        'Sputum production',
        'Shortness of breath',
        'Muscle pain',
        'Sore throat',
        'Headache',
        'Chills',
        'Nausea or vomiting',
        'Nasal congestion',
        'Diarrhoea',
        'Haemoptysis',
        'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}

symptoms=pd.DataFrame(data=symptoms,index=range(14))
symptoms


# **Symptom of  Coronavirus**

# In[ ]:


fig = px.bar(symptoms[['symptom', 'percentage']].sort_values('percentage', ascending=False), 
             y="percentage", x="symptom", color='symptom', 
             log_y=True, template='ggplot2', title='Symptom of  Coronavirus')
fig.show()


# In[ ]:


fig = px.pie(symptoms,
             values="percentage",
             names="symptom",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:



fig = px.treemap(symptoms, path=['symptom'], values='percentage',
                  color='percentage', hover_data=['symptom'],
                  color_continuous_scale='Rainbow')
fig.show()


# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in symptoms.symptom)
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()


# **Reading data **

# In[ ]:


screening  = pd.read_csv("../input/covid19-in-canada/Testing_Canada .csv")
confirm = pd.read_csv("../input/covid19-in-canada/Public_COVID-19_Canada .csv")
cnfrm_age = pd.read_csv("../input/covid19-confirmed-cases-by-country-and-age/COVID-19_Age.csv")
recovery = pd.read_csv("../input/covid19-in-canada/Recovered_Canada .csv")
death = pd.read_csv("../input/covid19-in-canada/Mortality_Canada .csv")
data_cnfrm = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
data_rec = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
data_dth = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
data = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
comp = pd.read_excel('/kaggle/input/covid19327/COVID-19-3.27-top30-500.xlsx')


# **Processing**

# In[ ]:


screening.head()


# In[ ]:


screening.isna().sum()


# In[ ]:


screening  = screening.drop("province_source",axis=1)


# In[ ]:


screening = screening.dropna()


# In[ ]:


screening = screening.rename(columns={"date_testing":"Date"})


# In[ ]:


cnfrm_age = cnfrm_age[cnfrm_age["Country"]=="Canada"]
cnfrm_age['age'] = ((cnfrm_age['Age_start'] + cnfrm_age['Age_end']+1)/2)


# # **Screening in Canada**

# In[ ]:


screening.head()


# **State wise Screening**

# In[ ]:


Alberta = screening[screening["province"] == "Alberta"]
BC = screening[screening["province"] == "BC"]
New_Brunswick = screening[screening["province"] == "New Brunswick"]
NL = screening[screening["province"] == "NL"]
Nova_Scotia = screening[screening["province"] == "Nova Scotia"]
Ontario = screening[screening["province"] == "Ontario"]
PEI = screening[screening["province"] == "PEI"]
Quebec = screening[screening["province"] == "Quebec"]
Saskatchewan = screening[screening["province"] == "Saskatchewan"]
NWT = screening[screening["province"] == "NWT"]
Nunavut = screening[screening["province"] == "Nunavut"]
Yukon = screening[screening["province"] == "Yukon"]


# In[ ]:


s = screening['province']
s = s.dropna()
s=s.unique()


# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in s)
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()


# **Graphical Representaion of Screening in Canadian Province**

# In[ ]:


#Alberta
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="Date", y="cumulative_testing", data=Alberta,
             color="blue")
plt.plot(Alberta.Date,Alberta.cumulative_testing,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Alberta',fontsize=70)


plt.figure(figsize=(100,30))
plt.bar(Alberta.Date, Alberta.cumulative_testing,label="Test")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Alberta',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()


#BC
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="Date", y="cumulative_testing", data=BC,
             color="blue")
plt.plot(BC.Date,BC.cumulative_testing,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in BC',fontsize=70)


plt.figure(figsize=(100,30))
plt.bar(BC.Date, BC.cumulative_testing,label="Test")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in BC',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()



#New Brunswick
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="Date", y="cumulative_testing", data=New_Brunswick,
             color="blue")
plt.plot(New_Brunswick.Date,New_Brunswick.cumulative_testing,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in New Brunswick',fontsize=70)


plt.figure(figsize=(100,30))
plt.bar(New_Brunswick.Date, New_Brunswick.cumulative_testing,label="Test")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in New Brunswick',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()



#NL
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="Date", y="cumulative_testing", data=NL,
             color="blue")
plt.plot(NL.Date,NL.cumulative_testing,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in NL',fontsize=70)


plt.figure(figsize=(100,30))
plt.bar(NL.Date, NL.cumulative_testing,label="Test")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Nl',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()



#Nova Scotia
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="Date", y="cumulative_testing", data=Nova_Scotia,
             color="blue")
plt.plot(Nova_Scotia.Date,Nova_Scotia.cumulative_testing,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Nova Scotia',fontsize=70)


plt.figure(figsize=(100,30))
plt.bar(Nova_Scotia.Date, Nova_Scotia.cumulative_testing,label="Test")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Nova Scotia',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()




#Ontario
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="Date", y="cumulative_testing", data=Ontario,
             color="blue")
plt.plot(Ontario.Date,Ontario.cumulative_testing,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Ontario',fontsize=70)


plt.figure(figsize=(100,30))
plt.bar(Ontario.Date, Ontario.cumulative_testing,label="Test")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Ontario',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()



#PEI
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="Date", y="cumulative_testing", data=PEI,
             color="blue")
plt.plot(PEI.Date,PEI.cumulative_testing,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in PEI',fontsize=70)



plt.figure(figsize=(100,30))
plt.bar(PEI.Date, PEI.cumulative_testing,label="Test")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in PEI',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()



#Saskatchewan
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="Date", y="cumulative_testing", data=Saskatchewan,
             color="blue")
plt.plot(Saskatchewan.Date,Saskatchewan.cumulative_testing,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Saskatchewan',fontsize=70)


plt.figure(figsize=(100,30))
plt.bar(Saskatchewan.Date, Saskatchewan.cumulative_testing,label="Test")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Saskatchewan',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()



#NWT
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="Date", y="cumulative_testing", data=NWT,
             color="blue")
plt.plot(NWT.Date,NWT.cumulative_testing,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in NWT',fontsize=70)


plt.figure(figsize=(100,30))
plt.bar(NWT.Date, NWT.cumulative_testing,label="Test")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in NWT',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()


#Nunavut
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="Date", y="cumulative_testing", data=Nunavut,
             color="blue")
plt.plot(Nunavut.Date,Nunavut.cumulative_testing,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Nunavut',fontsize=70)

plt.figure(figsize=(100,30))
plt.bar(Nunavut.Date, Nunavut.cumulative_testing,label="Test")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Nunavut',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()


#Yukon
f, ax = plt.subplots(figsize=(100, 30))
ax=sns.scatterplot(x="Date", y="cumulative_testing", data=Yukon,
             color="blue")
plt.plot(Yukon.Date,Yukon.cumulative_testing,zorder=1)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Yukon',fontsize=70)


plt.figure(figsize=(100,30))
plt.bar(Yukon.Date, Yukon.cumulative_testing,label="Test")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.title('Screening in Yukon',fontsize=70)
plt.legend(frameon=True, fontsize=12)
plt.show()


# # **Latest data of testing in Canada**

# In[ ]:


daily = screening.sort_values(['Date','province'])
latest = screening[screening.Date == daily.Date.max()]
latest = latest.sort_values("cumulative_testing",ascending = False)
latest.head()


# **Graphical Representation of Testing in Canada**

# In[ ]:


fig = px.bar(latest[['province', 'cumulative_testing']].sort_values('cumulative_testing', ascending=False), 
             y="cumulative_testing", x="province", color='province', 
             log_y=True, template='ggplot2', title='Province vs Test Performed')
fig.show()


# In[ ]:


fig = px.pie(latest,
             values="cumulative_testing",
             names="province",
             title="Screening in Canada",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:



fig = px.treemap(latest, path=['province'], values='cumulative_testing',
                  color='cumulative_testing', hover_data=['province'],
                  color_continuous_scale='Rainbow')
fig.show()


# # **Confirm Cases in Canada**

# In[ ]:


data_cnfrm['latest'] = data_cnfrm[data_cnfrm.columns[len(data_cnfrm.columns)-1]]
data_cnfrm = data_cnfrm[data_cnfrm["Country/Region"] ==  "Canada"]
data_cnfrm = data_cnfrm.loc[:,['Lat','Long','Province/State','latest']]
data_cnfrm.head()


# In[ ]:


c =  data_cnfrm.loc[:,["Province/State","latest"]]
c= c.rename(columns={"latest":"Confirm"})
fig = px.treemap(c, path=['Province/State'], values='Confirm',
                  color='Confirm', hover_data=['Province/State'],
                  color_continuous_scale='burgyl')
fig.show()


# In[ ]:


fig = px.pie(c,
             values="Confirm",
             names="Province/State",
             title="Confirmation in Canada",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


fig = px.bar(c[['Province/State', 'Confirm']].sort_values('Confirm', ascending=False), 
             y="Confirm", x="Province/State", color='Province/State', 
             log_y=True, template='ggplot2', title='State vs Confirmation')
fig.show()


# **Clusterning**

# In[ ]:


K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = data_cnfrm[['Lat']]
X_axis = data_cnfrm[['Long']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Score vs Cluster')
plt.show()


# In[ ]:


clus = data_cnfrm


# In[ ]:


kmeans = KMeans(n_clusters = 4, init ='k-means++')
kmeans.fit(clus[clus.columns[1:2]])
clus['cluster_label'] = kmeans.fit_predict(clus[clus.columns[1:2]])
centers = kmeans.cluster_centers_
labels = kmeans.predict(clus[clus.columns[1:2]])


# **Plotting Clusters**

# In[ ]:


clus.plot.scatter(x = 'Lat', y = 'Long', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 0], c='black', s=100, alpha=0.5)


# # **Confirmation on world map **

# In[ ]:


data_cnfrm.head()


# In[ ]:


import folium
map = folium.Map(location=[45.4,-75.666667 ], zoom_start=3,tiles='Stamen Toner')

for lat, lon,city, latest in zip(data_cnfrm['Lat'], data_cnfrm['Long'],data_cnfrm['Province/State'],data_cnfrm['latest']):
   folium.CircleMarker([lat, lon],
                       color='red',
                
                popup =('City: ' + str(city) + '<br>'
                       'Confirm : ' + str(latest) + '<br>'),

                       fill_color='red',
                       fill_opacity=0.7 ).add_to(map)
map


# **Number of caes in Canada**

# In[ ]:


confirmbydate = cnfrm_age.loc[:,["Date","Positive"]]
confirmbydate = confirmbydate.groupby("Date")[["Positive"]].sum().reset_index()
confirmbydate.plot()


# In[ ]:


confirm.head()


# In[ ]:


print("Total Cases in Canada : " + str(confirm.case_id.max()))
print("Maximum Cases in Province : " + str(confirm.provincial_case_id.max()))


# **Confirmation by gender**

# In[ ]:


gender =confirm['sex']
gender = gender.dropna()
gender = gender[gender!='Not Reported']


# In[ ]:


plt.figure(figsize=(15, 5))
plt.title('Confirmation by gender')
gender.value_counts().plot.bar();


# In[ ]:


agc = confirm[confirm.sex!="Not Reported"]
agc = agc['sex'].dropna()
fig = px.pie(agc,
             names="sex",
             title="Gender of Confirm person",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# # **Confirmation by age**

# In[ ]:


age = confirm['age']
age= age.dropna()
age = age[age!='Not Reported']


# In[ ]:


plt.figure(figsize=(15, 5))
plt.title('Confirmation by age')
age.value_counts().plot.bar();


# In[ ]:


agc = confirm[confirm.age!="Not Reported"]
agc = agc['age'].dropna()
fig = px.pie(agc,
             names="age",
             title="Age of Confirm person",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


ax = sns.kdeplot(cnfrm_age.age,cnfrm_age.Positive, n_levels=30, cmap="Purples_d")


# # **Number of  patients in province**

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Number of  patients in province')
confirm.province.value_counts().plot.bar();


# In[ ]:


fig = px.pie( values=confirm.groupby(['province']).size().values,names=confirm.groupby(['province']).size().index)
fig.update_layout(
    font=dict(
        size=15,
        color="#242323"
    )
    )   
    
py.iplot(fig)


# **Age vs Province**

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(confirm,  size = 20).map(plt.scatter, 'age', 'province').add_legend()
plt.title('Age vs Province',fontsize=40)
plt.xticks(fontsize=18)
plt.yticks(fontsize=28)


plt.show()


# **Patiennts in Cities**

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Patiennts in Cities')
confirm.health_region.value_counts().plot.bar();


# In[ ]:


fig = px.pie( values=confirm.groupby(['health_region']).size().values,names=confirm.groupby(['health_region']).size().index)
fig.update_layout(
    font=dict(
        size=15,
        color="#242323"
    )
    )   
    
py.iplot(fig)


# In[ ]:


c = confirm['health_region']
c = c.dropna()
c = c.unique()


# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in c)
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()


# **Age vs City**

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(confirm,  size = 20).map(plt.scatter, 'age', 'health_region').add_legend()
plt.title('Age vs City',fontsize=40)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(confirm,  size = 20).map(plt.scatter, 'province', 'health_region').add_legend()
plt.title('Province vs City',fontsize=30)
plt.xticks(fontsize=12)
plt.yticks(fontsize=20)
plt.show()


# **Travel History of Patiennts**

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Travel History of Patiennts')
confirm.travel_history_country.value_counts().plot.bar();


# In[ ]:


cnf = confirm[confirm.travel_history_country!="Not Reported"]
cnf = cnf['travel_history_country'].dropna()
fig = px.pie(cnf,
             names="travel_history_country",
             title="Travel history of Patients",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:


t = confirm['travel_history_country']
t = t.dropna()
t = t.unique()


# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in t)
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()


# **Provicne vs Travel Country**

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(confirm,  size = 20).map(plt.scatter, 'province', 'travel_history_country').add_legend()
plt.title('Provicne vs Travel Country',fontsize=30)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)
plt.show()


# **'Reason of Infection**

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Reason of Infection')
confirm.locally_acquired.value_counts().plot.bar();


# **Recovery in Canada**

# **Preprocessing**

# In[ ]:


data_rec['latest'] = data_rec[data_rec.columns[len(data_rec.columns)-1]]
data_rec = data_rec[data_rec["Country/Region"] ==  "Canada"]
data_rec = data_rec.loc[:,['Lat','Long','Province/State','latest']]
data_rec.head()


# **Clustering**

# In[ ]:


"""
K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = data_rec[['Lat']]
X_axis = data_rec[['Long']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Score vs Cluster')
plt.show()
"""


# In[ ]:


clus = data_rec


# In[ ]:


""""
kmeans = KMeans(n_clusters = 5, init ='k-means++')
kmeans.fit(clus[clus.columns[1:2]])
clus['cluster_label'] = kmeans.fit_predict(clus[clus.columns[1:2]])
centers = kmeans.cluster_centers_
labels = kmeans.predict(clus[clus.columns[1:2]])
"""


# **Graphical representaion of recovery**

# In[ ]:


#clus.plot.scatter(x = 'Lat', y = 'Long', c=labels, s=50, cmap='viridis')
#plt.scatter(centers[:, 0], centers[:, 0], c='black', s=100, alpha=0.5)


# In[ ]:


data_rec1 = data_rec[data_rec.latest>0]


# **Recovery on World Map**

# In[ ]:


import folium
map = folium.Map(location=[45.4,-75.666667 ], zoom_start=3,tiles='Stamen Toner')

for lat, lon,city, latest in zip(data_rec1['Lat'], data_rec1['Long'],data_rec1['Province/State'],data_rec1['latest']):
   folium.CircleMarker([lat, lon],
                       color='red',
                
                popup =('City: ' + str(city) + '<br>'
                       'Recovery : ' + str(latest) + '<br>'),

                       fill_color='red',
                       fill_opacity=0.7 ).add_to(map)
map


# ****Death in Canada****

# **Preprocessing**

# In[ ]:


data_dth['latest'] = data_dth[data_dth.columns[len(data_dth.columns)-1]]
data_dth = data_dth[data_dth["Country/Region"] ==  "Canada"]
data_dth = data_dth.loc[:,['Lat','Long','Province/State','latest']]
data_dth.head()


# In[ ]:


k =  data_dth.loc[:,["Province/State","latest"]]
k= k.rename(columns={"latest":"Death"})


# In[ ]:


fig = px.bar(k[['Province/State', 'Death']].sort_values('Death', ascending=False), 
             y="Death", x="Province/State", color='Province/State', 
             log_y=True, template='ggplot2', title='State vs Death')
fig.show()


# In[ ]:


fig = px.pie(k,
             values="Death",
             names="Province/State",
             title="Death in Canada",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
fig.show()


# In[ ]:



fig = px.treemap(k, path=['Province/State'], values='Death',
                  color='Death', hover_data=['Province/State'],
                  color_continuous_scale='Rainbow')
fig.show()


# **Clustering for Death**

# In[ ]:


K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = data_dth[['Lat']]
X_axis = data_dth[['Long']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Score vs Cluster')
plt.show()


# In[ ]:


clus = data_dth


# In[ ]:


kmeans = KMeans(n_clusters = 4, init ='k-means++')
kmeans.fit(clus[clus.columns[1:2]])
clus['cluster_label'] = kmeans.fit_predict(clus[clus.columns[1:2]])
centers = kmeans.cluster_centers_
labels = kmeans.predict(clus[clus.columns[1:2]])


# **Graphical Representation of Death**

# In[ ]:


clus.plot.scatter(x = 'Lat', y = 'Long', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 0], c='black', s=100, alpha=0.5)


# In[ ]:


data_dth1= data_dth[data_dth.latest>0]


# **Death on world map **

# In[ ]:


import folium
map = folium.Map(location=[45.4,-75.666667 ], zoom_start=3,tiles='Stamen Toner')

for lat, lon,city, latest in zip(data_dth['Lat'], data_dth1['Long'],data_dth1['Province/State'],data_dth1['latest']):
   folium.CircleMarker([lat, lon],
                       color='red',
                
                popup =('City: ' + str(city) + '<br>'
                       'Death : ' + str(latest) + '<br>'),

                       fill_color='red',
                       fill_opacity=0.7 ).add_to(map)
map


# In[ ]:


death.head()


# **Death by gender**

# In[ ]:


plt.figure(figsize=(15, 5))
plt.title('Death by gender')
death.sex.value_counts().plot.bar();


# In[ ]:


fig = px.pie( values=death.groupby(['sex']).size().values,names=death.groupby(['sex']).size().index)
fig.update_layout(
    font=dict(
        size=15,
        color="#242323"
    )
    )   
    
py.iplot(fig)


# **Death by province**

# In[ ]:


plt.figure(figsize=(15, 5))
plt.title('Death by province')
death.province.value_counts().plot.bar();


# In[ ]:


fig = px.pie( values=death.groupby(['province']).size().values,names=death.groupby(['province']).size().index)
fig.update_layout(
    font=dict(
        size=15,
        color="#242323"
    )
    )   
    
py.iplot(fig)


# **Death by City**

# In[ ]:


plt.figure(figsize=(15, 5))
plt.title('Death by City')
death.health_region.value_counts().plot.bar();


# In[ ]:


fig = px.pie( values=death.groupby(['health_region']).size().values,names=death.groupby(['health_region']).size().index)
fig.update_layout(
    font=dict(
        size=15,
        color="#242323"
    )
    )   
    
py.iplot(fig)


# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(death,  size = 20).map(plt.scatter, 'province', 'health_region').add_legend()
plt.title('Province vs City',fontsize=30)
plt.xticks(fontsize=12)
plt.yticks(fontsize=20)
plt.show()


# **Total death in Canada**

# In[ ]:


print("Total death in Canada till " + str(death.date_death_report.max()) + " is "+ str(death.death_id.max()))


# **Processing data**

# In[ ]:


data_cnfrm =  data_cnfrm.rename(columns={"latest":"Confirm"})
data_rec =  data_rec.rename(columns={"latest":"Recovery"})
data_dth =  data_dth.rename(columns={"latest":"Death"})


# In[ ]:


crd = data_cnfrm.loc[:,["Lat","Long","Province/State","Confirm"]]
crd['Recovery'] = data_rec['Recovery']
crd['Death'] = data_dth['Death']


# In[ ]:


crd.head()


# **World Map including Confirm, Recovery and Death**

# In[ ]:


import folium
map = folium.Map(location=[45.4,-75.666667 ], zoom_start=3,tiles='Stamen Toner')

for lat, lon,city,Confirm,Recovery,Death in zip(crd['Lat'], crd['Long'],crd['Province/State'],crd['Confirm'],crd['Recovery'],crd['Death']):
   folium.CircleMarker([lat, lon],
                       color='red',
                
                popup =('City: ' + str(city) + '<br>'
                        'Confirm : '+ str(Confirm) + '<br>'
                        'Recover : ' + str(Recovery) + '<br>'
                       'Death : ' + str(Death) + '<br>'),

                       fill_color='red',
                       fill_opacity=0.7 ).add_to(map)
map


# **Growth Factor and Ratio**

# This snippet of code is taken by [covid-19-predictions-growth-factor-and-calculus](https://www.kaggle.com/dferhadi/covid-19-predictions-growth-factor-and-calculus) , do check this kernel by [Daner Ferhadi](https://www.kaggle.com/dferhadi)

# In[ ]:


global_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")


# In[ ]:


# This functions smooths data, thanks to Dan Pearson. We will use it to smooth the data for growth factor.
def smoother(inputdata,w,imax):
    data = 1.0*inputdata
    data = data.replace(np.nan,1)
    data = data.replace(np.inf,1)
    #print(data)
    smoothed = 1.0*data
    normalization = 1
    for i in range(-imax,imax+1):
        if i==0:
            continue
        smoothed += (w**abs(i))*data.shift(i,axis=0)
        normalization += w**abs(i)
    smoothed /= normalization
    return smoothed

def growth_factor(confirmed):
    confirmed_iminus1 = confirmed.shift(1, axis=0)
    confirmed_iminus2 = confirmed.shift(2, axis=0)
    return (confirmed-confirmed_iminus1)/(confirmed_iminus1-confirmed_iminus2)

def growth_ratio(confirmed):
    confirmed_iminus1 = confirmed.shift(1, axis=0)
    return (confirmed/confirmed_iminus1)

# This is a function which plots (for in input country) the active, confirmed, and recovered cases, deaths, and the growth factor.
def plot_country_active_confirmed_recovered(country):
    
    # Plots Active, Confirmed, and Recovered Cases. Also plots deaths.
    country_data = global_data[global_data['Country/Region']==country]
    table = country_data.drop(['SNo','Province/State', 'Last Update'], axis=1)
    table['ActiveCases'] = table['Confirmed'] - table['Recovered'] - table['Deaths']
    table2 = pd.pivot_table(table, values=['ActiveCases','Confirmed', 'Recovered','Deaths'], index=['ObservationDate'], aggfunc=np.sum)
    table3 = table2.drop(['Deaths'], axis=1)
   
    # Growth Factor
    w = 0.5
    table2['GrowthFactor'] = growth_factor(table2['Confirmed'])
    table2['GrowthFactor'] = smoother(table2['GrowthFactor'],w,5)

    # 2nd Derivative
    table2['2nd_Derivative'] = np.gradient(np.gradient(table2['Confirmed'])) #2nd derivative
    table2['2nd_Derivative'] = smoother(table2['2nd_Derivative'],w,7)


    #Plot confirmed[i]/confirmed[i-1], this is called the growth ratio
    table2['GrowthRatio'] = growth_ratio(table2['Confirmed'])
    table2['GrowthRatio'] = smoother(table2['GrowthRatio'],w,5)
    
    #Plot the growth rate, we will define this as k in the logistic function presented at the beginning of this notebook.
    table2['GrowthRate']=np.gradient(np.log(table2['Confirmed']))
    table2['GrowthRate'] = smoother(table2['GrowthRate'],0.5,3)
    
    # horizontal line at growth rate 1.0 for reference
    x_coordinates = [1, 100]
    y_coordinates = [1, 1]
    f, ax = plt.subplots(figsize=(15,5))
    table2['Deaths'].plot(title='Deaths')
    plt.show()
    f, ax = plt.subplots(figsize=(15,5))
    table2['GrowthFactor'].plot(title='Growth Factor')
    plt.plot(x_coordinates, y_coordinates) 
    plt.show()
    f, ax = plt.subplots(figsize=(15,5))
    table2['2nd_Derivative'].plot(title='2nd_Derivative')
    plt.show()
    f, ax = plt.subplots(figsize=(15,5))
    table2['GrowthRatio'].plot(title='Growth Ratio')
    plt.plot(x_coordinates, y_coordinates)
    plt.show()
    f, ax = plt.subplots(figsize=(15,5))
    table2['GrowthRate'].plot(title='Growth Rate')
    plt.show()

    return 


# In[ ]:


plot_country_active_confirmed_recovered('Canada')


# **Prediciton of Confirmation**

# In[ ]:


data = data[data["Country/Region"] == "Canada"]
data = data.loc[:,["Date","Confirmed"]]
data = data.groupby("Date")[["Confirmed"]].max().reset_index()


# In[ ]:


data.head()


# In[ ]:


data.columns = ['ds','y']


# In[ ]:


data1 = data.cumsum()
data1 = data1.loc[:,["y"]]
data1 = data1[20:]
x = np.arange(len(data1)).reshape(-1, 1)
y = data1.values.reshape(-1, 1)


# **Regression Model**

# In[ ]:


from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)
_=model.fit(x, y.ravel())


# In[ ]:


test = np.arange(len(data1)+7).reshape(-1, 1)
pred = model.predict(test)
prediction = pred.round().astype(int)


# In[ ]:


prediction = pd.DataFrame(prediction)


# In[ ]:


prediction.plot()


# ****Prophet Algorithm****

# In[ ]:


m=Prophet()
m.fit(data)
future=m.make_future_dataframe(periods=30)
forecast_cm=m.predict(future)
forecast_cm


# In[ ]:


cnfrm = forecast_cm.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm.head()
cnfrm=cnfrm.tail(30)
cnfrm.columns = ['Date','Confirm']
cnfrm.head()


# In[ ]:


fig_cm = plot_plotly(m, forecast_cm)
py.iplot(fig_cm) 

fig_cm = m.plot(forecast_cm,xlabel='Date',ylabel='Confirmed Count')


# **LSTM**

# In[ ]:


dataset = data
dataset.columns = ['confirmed_date','Confirmed']
dataset = dataset.drop("confirmed_date",axis=1)


# **Splitting Data**

# In[ ]:


data = np.array(dataset).reshape(-1, 1)
train_data = dataset[:len(dataset)-5]
test_data = dataset[len(dataset)-5:]


# In[ ]:


scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)
n_input =5
n_features =1
                             
generator = TimeseriesGenerator(scaled_train_data,scaled_train_data, length=n_input, batch_size=1)

lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (n_input, n_features)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units = 1))
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
lstm_model.fit(generator, epochs = 50)


# In[ ]:


lstm_predictions_scaled = []

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)


# In[ ]:


prediction = pd.DataFrame(scaler.inverse_transform(lstm_predictions_scaled))
prediction.head()


# **Comparision**

# In[ ]:


comp.head()


# In[ ]:


comp_table = pd.DataFrame(comp.describe().T)


# In[ ]:


comp_table


# In[ ]:


comp.columns


# In[ ]:


comp = comp.loc[:,["Canada","US","Italy","Switzerland"]]


# In[ ]:


comp.plot()


# **Prevention**
# To avoid the critical situation people are suggested to do following things
# 
# * Avoid contact with people who are sick.
# * Avoid touching your eyes, nose, and mouth.
# * Stay home when you are sick.
# * Cover your cough or sneeze with a tissue, then throw the tissue in the trash.
# * Clean and disinfect frequently touched objects and surfaces using a regular household
# * Wash your hands often with soap and water, especially after going to the bathroom; before eating; and after blowing your nose, coughing, or sneezing. If soap and water are not readily available, use an alcohol-based hand sanitizer.

# 
