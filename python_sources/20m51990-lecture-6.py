#!/usr/bin/env python
# coding: utf-8

# # Exercise 6
# 
# At this time, I look at Corona cases for South Korea.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='South Korea'
df=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(np.unique(df['Country/Region'].values))
df = df[df['Country/Region']==selected_country]
df=df.groupby('ObservationDate').sum()
print(df)


# # Daily increase

# In[ ]:


df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
df['daily_deaths'].plot()


# In[ ]:


print(df)


# # Interactive chart

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovery_object=go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily recoveries')

layout_object= go.Layout(title='South Korea daily cases 20M51990',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('South Korea_daily_cases_20M51990.html')


# # Informative table

# In[ ]:



df1 = df#[['daily_confirmed']]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_20M51990.html','w')
f.write(styled_object.render())


# # Analysis
# 
# From the figures, we could say that the corona cases in South Korea are decreasing with time. There was a peak value in 03/03/2020 that 851 people were confirmed. On the other hands, we can find that the recovery cases exceeded the confirmed cases from 03/21/2020 to 05/09/2020 and the largest recovery cases number was in 03/22 that 1369 people recovered. The number of deaths was kept to less than 10 a day except 03/02/2020 that 11 people died in this day. After 04/17/2020, the death number was kept to less than 2. So far (up to 06/10/2020),11947 people have been confirmed and 10654 people have recovered. The number of deaths is 276. So, I think the government of South Korea do have well to control the spread infection.
# 
# The number of confirmed cases has decreased into single digit from late April to early May. However, it increased obviously in late May and there was a peak value in 05/27 that 79 people confirmed. The number of confirmed was increasing slightly from May. According to my search, these confirmed cases are mainly injected in the capital region. Most of the cases were caused by the aggregation infection in large indoor entertainment venues.(1) Government said in the 06/12 press conference that they will broaden more restrictions on large indoor entertainment venues and will make a particular list of daily participants by using QR code in order to prevent the aggregation infection effectively and accurately.(2)
# 
# In my opinion, the restrictions on the import of foreign country in South Korea was timely. The restriction on the import, depart and even transfer of China started in 02/04 and I think the government has avoided a larger measure of infection from the beginning. (3)
# 
# And the government also appealed the public to wear the mask timely as well. There are not only signs to warn people to wear masks in public facilities but also there are policies in some areas (Daegu) that people will impose a fine without mask. (4) Meanwhile, public can get some free masks from the governmental organizations. The government also made great efforts to make the infected path clearly as well. (5)
# 
# By the way, the global ranking (total confirmed cases number) of South Korea is 57, I count it by myself without using programming. (6)
# 
# References
# 
# (1) What South Korea's Nightclub Coronavirus Outbreak Can Teach Other Countries as They Reopen,https://time.com/5834991/south-korea-coronavirus-nightclubs/
# 
# (2) South Korea to use QR codes for entering 'high-risk areas' to contain COVID-19,https://www.zdnet.com/article/south-korea-to-use-qr-codes-to-contain-covid-19/
# 
# (3)Koreanair Entry restrictions by countries,https://www.koreanair.com/global/en/2020_02_TSA_detail.html?fbclid=IwAR0JnLH-Ldp2UOzSLXX0cq65rtfLnVPYeRdpPXWMNhtpouIJiD7V5olpBK8
# 
# (4) Though outbreak has peaked, masks becoming mandatory in Korea,http://www.koreaherald.com/view.php?ud=20200517000260
# 
# (5)Report materials from South Korea Government,http://ncov.mohw.go.kr/cn/tcmBoardList.do?brdId=22&brdGubun=225&dataGubun=&ncvContSeq=&contSeq=&board_id=
# 
# (6) Data from BBC News 06/10/2020,https://www.bbc.com/zhongwen/simp/world-52932320
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# # Global ranking

# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df1 = df.groupby(['ObservationDate','Country/Region']).sum()
df2 = df[df['ObservationDate']=='06/10/2020'].sort_values(by=['Confirmed'],ascending=False).reset_index()
print(df2[df2['Country/Region']=='South Korea'])


# # Additional practice 
# I also tried to make the global ranking refer to Rynazal's way (https://www.kaggle.com/ryzary/covid-lecture6).

# In[ ]:


import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data.head()


# In[ ]:


data.index=data['ObservationDate']
data = data.drop(['SNo','ObservationDate'],axis=1)
data.head()


# In[ ]:


data_SK = data[data['Country/Region']=='South Korea']
data_SK.tail()


# In[ ]:


latest = data[data.index=='06/08/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 
print('Ranking of South Korea: ', latest[latest['Country/Region']=='South Korea'].index.values[0]+1)


# The ranking of South Korea which I have counted was 57 and the result of programming was 56.
# I think there are 2 reasons of the error: my count mistakes or the different data(for I counted the ranking by using the data from BBC News).
