#!/usr/bin/env python
# coding: utf-8

# ## Introduction:
# Hello Kagglers!!
# I am here to contribute towards COVID- 19. I am little bit late in contribution to COVID19, But i will make sure this notebook will be worth to some extent. I am going to analyze COVID19 for Rajasthan State only in my first notebook, Later i will cover more!! Thank you for being here, Happy Reading !!

# ## Covid-19:
# ![banner-2.png](attachment:banner-2.png)
# 
# Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.
# Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.
# 

# ## HOW IT SPREADS?
#    The virus that causes COVID-19 is mainly transmitted through droplets generated when an infected person coughs, sneezes, or exhales. These    droplets are too heavy to hang in the air, and quickly fall on floors or surfaces. You can be infected by breathing in the virus if you       are within close proximity of someone who has COVID-19, or by touching a contaminated surface and then your eyes, nose or mouth.
#    
#    
# ## To prevent infection and to slow transmission of COVID-19, do the following:
# 
# 1. Wash your hands regularly with soap and water, or clean them with alcohol-based hand rub.
# 2. Maintain at least 1 metre distance between you and people coughing or sneezing.
# 3. Avoid touching your face.
# 4. Cover your mouth and nose when coughing or sneezing.
# 5. Stay home if you feel unwell.
# 6. Refrain from smoking and other activities that weaken the lungs.
# 7. Practice physical distancing by avoiding unnecessary travel and staying away from large groups of people.(by WHO)

# # Symptoms
# ## Most common symptoms:
# 
# 1. fever.
# 2. dry cough.
# 3. tiredness.
# ## Less common symptoms:
# 
# 1. aches and pains.
# 2. sore throat.
# 3. diarrhoea.
# 4. conjunctivitis.
# 5. headache.
# 6. loss of taste or smell.
# 7. a rash on skin, or discolouration of fingers or toes.
# ## Serious symptoms:
# 
# 1. difficulty breathing or shortness of breath.
# 2. chest pain or pressure.
# 3. loss of speech or movement.
# ##### Seek immediate medical attention if you have serious symptoms.  Always call before visiting your doctor or health facility. 
# 
# 

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objects as go
cf.go_offline()


# data= pd.read_csv("../input/corona_report.csv")
# data.drop('Unnamed: 0', axis= 1, inplace= True)
# data.head()

# ## EDA(Exploatry Data Analysis):
#   let's find statistical evidence from data

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=data['District'], y= data['Cumulative_Positive'], name='Positive_Cases',
                         line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=data['District'], y=data['Recoverd'], name='Recovered',
                         line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=data['District'], y=data['Discharged'], name='Discharged',
                         line=dict(color='green', width=2)))

fig.update_layout(
    title='Corona Virus Trend in Rajasthan',
     yaxis=dict(
        title='Number of Cases')
    )

fig.show()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=data['District'], y= data['Cumulative_Positive'], name='Positive_Cases',
                         line=dict(color='red', width=2)))


# ## Top 10 Districts with Total Confirmed Cases

# In[ ]:


data.sort_values('Cumulative_Positive',ascending=False)[:10].iplot(kind='bar',
                                                                               x='District',
                                                                               color = ['blue','green','red'],
                                                                               title='Top 10 Districts with Total Confirmed Cases',
                                                                               xTitle='Districts',
                                                                               yTitle = 'Cases Count')


# ## Top 10 Districts with Total Recoverd Cases

# In[ ]:


data.sort_values('Recoverd',ascending=False)[:10].iplot(kind='bar',
                                                                               x='District',
                                                                               color = ['blue','green','red'],
                                                                               title='Top 10 Districts with Total Recovered Cases',
                                                                               xTitle='Districts',
                                                                               yTitle = 'Cases Count')


# ## Correlation:

# In[ ]:


corr = data[['Cumulative_Positive','Recoverd','Discharged']].corr()
mask = np.triu(np.ones_like(corr,dtype = bool))

plt.figure(dpi=100)
plt.title('Correlation Analysis')
sns.heatmap(corr,mask=mask,annot=True,lw=1,linecolor='white',cmap='Reds')
plt.xticks(rotation=0)
plt.yticks(rotation = 0)
plt.show()


# ### Well, I will end this notebook here only!! I have not done any Forecasting in this notebook, My main aim was to do some EDA and Reporting. I will forecast in my second notebook .

# In[ ]:




