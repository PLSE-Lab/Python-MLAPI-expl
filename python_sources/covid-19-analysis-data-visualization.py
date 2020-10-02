#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Loading the Required Packages 
# ### 1. One thing that is coming in my mind is importing Matplotlib (Pandas and Numpy are always there), but I think tnhat Interactive Graphs cannot be Visualized with Matplotlib. So,we will do it with Plotly (I don't know how to use it but will try as much as I can).
# ### 2. Now, in order to make this Journey a memorable one, let us also create a story which will make us enjoy this Journey
# ### 3. Let us consider a Scenario, where a person has heard a story on News Channel X where he heard first about the new pandemic 'COVID-19'
# ### 4. Now, let us assume that he was too curious, so he did not believe the news and went on Internet to find the Data and then he would (by Self) analyze the Report and then draws some conclusion about the same news.

# ## Let us start this Journey

# In[ ]:


# Importing Libraries and the Data
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
df=pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')


# In[ ]:


df.head()


# 1. The first thing that I (as the person) observes the NaN values in the County and Province_State columns. 
# 2. As I already told, the person is too curious!! So, he is going to analyze each and every column

# In[ ]:


# Obtaining the Columns of the Data
df.columns


# In[ ]:


# Analyzing the Data Type of the Columns
df.dtypes


# ## If I would be the person knowing some coding, the first thing that would come to my mind would be converting the Column 'Date' into Date Time
# ### 1. So, let us proceed further and the name of the Variable also does not suit, so let us name the variable as 'covid' (that would be much Interesting !!!)
# ### 2. Next Step, as the person would be to check the Null Values in the Different Column (The person is quite intellignet regarding the Knowledge in the Analytics!!!)

# In[ ]:


df['Date']=pd.to_datetime(df['Date'])
covid=df


# In[ ]:


print("Size/Shape of the dataset: ",covid.shape)
print("Checking for null values:\n",covid.isnull().sum())
print("Checking Data-type of each column:\n",covid.dtypes)


# ### So, from the above code we can see that there are a lot of Empty Values in the Province_State and Country_Region (and that is actually a Fact as these columns would be labelled mostly when the Counry is USA )
# ### Hence, we would not Drop this column and would deal with it Seperately
# 1. We would now, set the Index of the Dataframe as 'Id'
# 2. Remember that df and covid variables are equal

# In[ ]:


df.set_index('Id')


# ## My thinking goes to plot the top 10 countries (in terms of Confirmed Cases) excluding America (because the confirmed cases are too High such that the Analysis of America could be done seperately!!)
# ### Now, we are going to plot the confirm cases of the top 10 countries except for America (you know the reason!!)

# In[ ]:


data_df=df.sort_values(by=['Population'])     # Sorting the DataFrame by Popuation
xx=data_df.groupby(['Country_Region'])        # Grouping the Sorted Dataframe by ''Country_Region
country_name=xx['Target'].value_counts()      # Counting the number of COnfirmed Cases and Fatalities for Each Country
country_name                                  # Displaying the Outcome of the process


# In[ ]:


l={}                                        # Creating an empty Dicitonary for Storing the values of Countries and Number of Fatalities
for i in df['Country_Region'].unique():
    l[i]=country_name[i]['Fatalities'] 
m={}                                        # Creating an empty Dicitonary for Storing the values of Countries and Number of Confirmed Cases
for i in df['Country_Region'].unique():
    m[i]=country_name[i]['ConfirmedCases']


# In[ ]:


ll=pd.DataFrame(l.values(),l.keys())   # Transforming the Dictionary into Data Frame
ll=ll.reset_index()                    # Resetting the Index
ll
mm=pd.DataFrame(m.values(),m.keys())   # Transforming the Dictionary into Data Frame
mm=mm.reset_index()                     # Resetting the Index
mm


# In[ ]:


ll.columns=['Country','Fatalities']            # Labelling the Columns as Country and Fatalities
mm.columns=['Country','Confirmed Cases']      # Labelling the Columns as Country and Fatalities
mm.nlargest(10,'Confirmed Cases')             # Displaying the Top 10 Country according the Confirmed Cases


# In[ ]:


g=ll.nlargest(11,'Fatalities')                 # Selecting the Top 11 Countries on the basis of Number of Fatalities
h=g.drop([173])                                # Dropping the America Row (due to tehreason I had mentioned above
h


# In[ ]:


kk=h.set_index('Country')                       # Setting the Index as Country which will be useful in the Plotting 
oo=mm.nlargest(11,'Confirmed Cases')            # Selecting the Top 11 Countries on the basis of Number of Confirmed Cases
oo=oo.drop([173])                               # Considering only the countries except for America


# ## So, the labour work is done and now, ket us go for our Favourite Part i.e. Plotting the Top 10 countries and their Confirmed Cases (We will go with plotly as it is quite Interactive)

# In[ ]:


import plotly.express as px
data = px.data.gapminder()

data_canada = oo
fig = px.bar(data_canada, x='Country', y='Confirmed Cases',
             hover_data=['Country', 'Confirmed Cases'], color='Confirmed Cases', height=400)
fig.show()


# # Conclusions:
# 1. The Graph is pretty cool and the Colors are also quite Vibrant and now we can observe the amount of Confirmed Cases in the Top - 10 COuntries
# 2. China has the Highest amount of Confirmed Cases after America followed by Canada and France

# In[ ]:


gg=ll.nlargest(15,'Fatalities')    # Selecting the Top 15 countries on the basis of Number of Fatalities
gg.drop([173],inplace=True) # Taking only the countries except for US


# ## So, the labour work is done and now, ket us go for our Favourite Part i.e. Plotting the Top 15 countries and their Fatalities.

# In[ ]:


import plotly.express as px
data = px.data.gapminder()

data_canada = gg
fig = px.bar(data_canada, x='Country', y='Fatalities',
             hover_data=['Country', 'Fatalities'], color='Fatalities', width=1000, height=600)
fig.show()


# # From the above figure, some of the conclusions that can be drawn are
# 
# 1. United States have the highest amount of Fatalities and is comparatively very large as compared to the rest of the countries
# 2. Analyzing the rest of the Countries, China has the highest amount of Fatalities (i.e 4760) and then followed by Canada and France (the neighbouring country of the United States of America)
# 3. Now, let us analyze the County Province of United States

# In[ ]:


usa=df[df['Country_Region']=='US']  # Selecting the Values whose Country Region belongs to US
usa11=usa.dropna()                  # Dropping Null Values


# In[ ]:


usa11                               # Displaying the Obtained DataFrame


# ### Now the thing that is coming in my mind, is to analyze the Confirmed Cases in US in order of the Month 
# So, let us not wait and excite the curiosity of the Plot !!!!

# In[ ]:


usa1=usa.groupby('Date')                                 # Grouping by the Dat
january=usa[usa['Date']<'2020-02-01']                    # Creating the DataFrame object having the Data in the month of January
february=usa[usa['Date']<'2020-03-01']                   # Creating the DataFrame object having the Data in the month of February
march=usa[usa['Date']<'2020-04-01']                      # Creating the DataFrame object having the Data in the month of March
april=usa[usa['Date']<'2020-05-01']                      # Creating the DataFrame object having the Data in the month of April
may=usa[usa['Date']<'2020-06-01']                        # Creating the DataFrame object having the Data in the month of May
june=usa[usa['Date']<'2020-07-01']                       # Creating the DataFrame object having the Data in the month of June


# ### In the Next cell, we are going to create the Dictionary containg the Month and the Number of Confirmed Cases for the US country

# In[ ]:


cases_of_january={'Fatalities':january['Target'].value_counts()['Fatalities'],'Confirmed Cases':january['Target'].value_counts()['ConfirmedCases']}
cases_of_february={'Fatalities':february['Target'].value_counts()['Fatalities'],'Confirmed Cases':february['Target'].value_counts()['ConfirmedCases']}
cases_of_march={'Fatalities':march['Target'].value_counts()['Fatalities'],'Confirmed Cases':march['Target'].value_counts()['ConfirmedCases']}
cases_of_april={'Fatalities':april['Target'].value_counts()['Fatalities'],'Confirmed Cases':april['Target'].value_counts()['ConfirmedCases']}
cases_of_may={'Fatalities':may['Target'].value_counts()['Fatalities'],'Confirmed Cases':may['Target'].value_counts()['ConfirmedCases']}
cases_of_june={'Fatalities':june['Target'].value_counts()['Fatalities'],'Confirmed Cases':june['Target'].value_counts()['ConfirmedCases']}


# In[ ]:


x=['January','February','March','April','May','June']
y=[cases_of_january['Confirmed Cases'],cases_of_february['Confirmed Cases'],cases_of_march['Confirmed Cases'],cases_of_april['Confirmed Cases'],cases_of_may['Confirmed Cases'],cases_of_june['Confirmed Cases']]


# ## Let us create a Bar Chart of the Month and the number of Confirm Cases

# In[ ]:


# Let us plot for Confirmed cases versus the month for USA
cc={'Month':x,'Confirm Cases':y}
dff=pd.DataFrame(cc,columns=['Month','Confirm Cases'])


# In[ ]:


fig = px.bar(dff, x='Month', y='Confirm Cases')
fig.show()


# ## Conclusion: 
# ### 1. The Number of Confirmed Cases has been increased linearly and has been Maximum in the month of June (according to the Data Available)

# In[ ]:


country=df['Country_Region'].unique()   # Getting the name of Countries in the world


# In[ ]:


total_cases=df.groupby('Country_Region')['Target'].value_counts()   # Grouping the original DataFrame on the basis of Country
total_cases


# In[ ]:


# Making a Dictionary Confirmed Cases in the World 
ff=[]
jjj={}
for i,j in zip(total_cases,total_cases.index):
    if j not in ff:
        ff.append(j)
        jjj[j[0]]=i
jjj


# In[ ]:


# Converting Dictionary into Dataframe
df1 = pd.DataFrame(list(zip(list(jjj.keys()), list(jjj.values()))), 
               columns =['Country', 'Confirm Cases']) 
ghg=df1.nlargest(15,'Confirm Cases')                   # Taking the top 15 countries in terms of the Confirm Cases
dfff=df1.nlargest(11,'Confirm Cases').drop([173])      # Again Dropping the America Will be useful for Comparison of the World versus America


# ## In the Next Cell, we will be creating a DataFrame on the basis of the Confirmed Cases and the Month

# In[ ]:


usa=df
january=usa[usa['Date']<'2020-02-01']
february=usa[usa['Date']<'2020-03-01']
march=usa[usa['Date']<'2020-04-01']
april=usa[usa['Date']<'2020-05-01']
may=usa[usa['Date']<'2020-06-01']
june=usa[usa['Date']<'2020-07-01']
january=january[df['Target']=='ConfirmedCases']['Target'].shape[0]
february=february[df['Target']=='ConfirmedCases']['Target'].shape[0]
march=march[df['Target']=='ConfirmedCases']['Target'].shape[0]
april=april[df['Target']=='ConfirmedCases']['Target'].shape[0]
may=may[df['Target']=='ConfirmedCases']['Target'].shape[0]
june=june[df['Target']=='ConfirmedCases']['Target'].shape[0]


# In[ ]:


x=['January','February','March','April','May','June']
y=[january,february,march,april,may,june]
# Let us plot for Confirmed cases versus the month for USA
cc={'Month':x,'Confirm Cases':y}                               # Creating a Dicitonary of Month and Confirm Cases
dff=pd.DataFrame(cc,columns=['Month','Confirm Cases'])         # Converting the Dictionary into DataFrame 
value=dff['Confirm Cases'].to_list()                           # Converting the Series Object into list
number=[i for i in range(1,7)]                                 # Naming the Month on the Basis of Number


# ## Time for the Line Plot of the Confirm Cases in the world (except America) and The Month
# 
# ## I think this man has some personal reason for deselecting America!!

# In[ ]:


import plotly.graph_objects as go
import numpy as np
fig = go.Figure(data=go.Scatter(x=number, y=value))
fig.update_layout(title='Analysis of Confirm Cases across the whole world by the month',
                   xaxis_title='Month',
                   yaxis_title='Confirm Cases')


# In[ ]:


dfff                                        # Displaying the DataFrame of Confirm Cases and the respective Country


# ## Time to see the Contribtuion of the Different Countries in the Confirm Cases (except for America) and the best to be done is through Pie-Chart

# In[ ]:


fig = px.pie(dfff, values='Confirm Cases', names='Country',title='Percentage Proportion of Confirmed Cases in top 10 countries (except US)')
fig.update_traces(textposition='inside', textinfo='percent+label')
labels=dfff['Country'].to_list()
values=dfff['Confirm Cases'].to_list()
fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.2, 0, 0, 0])])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20)
)
fig.show()


# In[ ]:


usa11                                           # Coming back to America


# In[ ]:


usa=df[df['Country_Region']=='US']
usa11=usa.dropna()


# In[ ]:


a=lambda x:x[0]


# In[ ]:


li=list(usa11['Province_State'].unique())


# In[ ]:


li


# ## Now there does not seem much error. Let us move on!!!

# ## But the Problem in the above result is the repetition of the Province Name, so let us fix this error

# In[ ]:


usa11


# In[ ]:


b=lambda x:x[0]


# In[ ]:


usa11[usa11['Province_State']=='Alabama']['Target'].value_counts()


# In[ ]:


gun={}
for i in li:
    cool=usa11[usa11['Province_State']==i]['Target'].value_counts()['ConfirmedCases']
    gun[i]=cool
gun                              # Not much Description needed and you can analyze it yourslef


# In[ ]:


lon=sorted(list(gun.items()),key=lambda x:x[1],reverse=True)
lon=lon[:10]                             # Considering only the top 10 states of US


# In[ ]:


aa=lambda x:x[0]
ab=lambda x:x[1]


# In[ ]:


city=list(map(aa,lon))                           # Labelling the City
cases=list(map(ab,lon))                          # Labelling the Values i.e the Confirmed Cases


# In[ ]:


labels


# ## Now, the Final comparison between the Contribution of the Countries in the World and the Contribution of Different States in the US

# In[ ]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=city, values=cases, name="US"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=values, name="World"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    title_text="Comparison of Top 10 Province of US vs top 10 countries of World (in case of Confirmed Cases)",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='US', x=0.18, y=0.5, font_size=20, showarrow=False),
                 dict(text='World', x=0.82, y=0.5, font_size=20, showarrow=False)])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(
    margin=dict(l=20, r=20, t=40, b=20))
fig.show()


# # Final Conclusion
# ## 1. In the world, about 97% Contribution is donated by America (not in the Diagram, but you can verify!!).
# ## 2. In America, the highest contribtuion comes from Texas(19.8%) followed by Georgia (12.4%), Virginia (10.4%).
# ## 3. In the world (excluding America) the maximum contribution comes from China (38.2%),origin of the cause and then followed by Canada (14.6%) and France (12.4%)
# ## 4. So, now from the above discussion my country of residence does not seem to be much of the trouble (in terms of Confirmed Cases) which is opposite to the hyoe that is generated in the media of my country, but I will take all the precautuonay measures to be safe!! 

# ## Hey, Guys this is my first Notebook on Visualization (and I am a begineer as well) and I am open to feedbacks and hence, please feel free to provide feedbacks.
# ## I have put most of the efforts which I have learnt so far, and expecting to grow much more in the future.
# 
# ## Thanks

# In[ ]:





# In[ ]:




