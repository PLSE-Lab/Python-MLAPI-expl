#!/usr/bin/env python
# coding: utf-8

# <h3><i><font color = "#1e88e5">In this kernel, I'm going to explore the situation of water in Chennai based on the quantity of water in the reservoirs and rainfall that it received in the last 15 years.</font></i></h3>

# In[ ]:


# Loading the required libraries 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading the required datasets and using the 'Date' as index 

df_chennai_rs = pd.read_csv('../input/chennai_reservoir_levels.csv', parse_dates=['Date'], index_col='Date')
df_chennai_rainfall = pd.read_csv('../input/chennai_reservoir_rainfall.csv', parse_dates=['Date'], index_col='Date')


# In[ ]:


# Exploring the reservoirs dataset 

df_chennai_rs.head()


# In[ ]:


# Statistical facts about the reservoirs dataset 

df_chennai_rs.describe()


# In[ ]:


# More info about the reservoirs dataset like varible type, number of observations, etc

df_chennai_rs.info()


# In[ ]:


# Plotting for POONDI with date for reservoir quantity from 2004 to 2018

sns.set(rc={'figure.figsize':(15,7)})
poondi = df_chennai_rs[['POONDI']]
poondi.plot()
plt.show()


# In[ ]:


# Plotting for CHOLAVARAM with date for reservoir quantity from 2004 to 2018

sns.set(rc={'figure.figsize':(15,7)})
cholavaram = df_chennai_rs['CHOLAVARAM']
cholavaram.plot()
plt.show()


# <h3>It is clearly visible that the quantity of water in the reservoir keeps on changing depending on the monsoon. Its a seasonal data.

# In[ ]:


# Plotting for REDHILLS with date for reservoir quantity from 2004 to 2018

sns.set(rc={'figure.figsize':(15,7)})
redhills = df_chennai_rs[['REDHILLS']]
redhills.plot()
plt.show()


# In[ ]:


# Plotting for CHEMBARAMBAKKAM with date for reservoir quantity from 2004 to 2018

sns.set(rc={'figure.figsize':(15,7)})
chembarambakkam = df_chennai_rs[['CHEMBARAMBAKKAM']]
chembarambakkam.plot()
plt.show()


# In[ ]:


# Plotting all the reservoirs' storage (in million cubic feet) for years 2004 to 2018 in a single plot

sns.set(rc={'figure.figsize':(15,7)}) 
All = df_chennai_rs[['POONDI', 'CHOLAVARAM', 'CHEMBARAMBAKKAM', 'REDHILLS']]
All.plot()
plt.xticks(rotation=30)
plt.show()


# <h3>The years from 2012 to 2015 show a drastic descrease in the quantity of all reservoirs which is a result of severe droughts faced by Tamil Nadu during these last 3 years. <br></h3>
# <b>Source: <a href='https://www.thehindu.com/news/national/tamil-nadu/3-deficit-years-and-an-impending-drought/article6230312.ece'>Droughts from 2012 to 2015</a>

# <i><h3>Now, let us analyse the rainfall data.

# In[ ]:


# Statistical facts about the rainfall data 

df_chennai_rainfall.describe()


# In[ ]:


# More info about the rainfall data 

df_chennai_rainfall.info()


# In[ ]:


# Plotting rainfall data for all the places on a single plot to better understand the patterns

sns.set(rc={'figure.figsize':(15,10)}) 
All = df_chennai_rainfall[['POONDI', 'CHOLAVARAM', 'CHEMBARAMBAKKAM', 'REDHILLS']]
All.plot()
plt.xticks(rotation=30)
plt.show()


# <h4>The unit of measurement for rainfall is mm.</h4>

# In[ ]:


# Combining rainfall and reservoir data for each location 

# POONDI

df_POONDI = pd.DataFrame({
        'Rainfall': df_chennai_rainfall.POONDI * 5,
        'Reservoir': df_chennai_rs.POONDI
    })

# CHEMBARAMBAKKAM 

df_CHEMBARAMBAKKAM = pd.DataFrame({
        'Rainfall': df_chennai_rainfall.CHEMBARAMBAKKAM * 5,
        'Reservoir': df_chennai_rs.CHEMBARAMBAKKAM
    })

# CHOLAVARAM 

df_CHOLAVARAM = pd.DataFrame({
        'Rainfall': df_chennai_rainfall.CHOLAVARAM * 5,
        'Reservoir': df_chennai_rs.CHOLAVARAM
    })

# REDHILLS 

df_REDHILLS = pd.DataFrame({
        'Rainfall': df_chennai_rainfall.REDHILLS * 5,
        'Reservoir': df_chennai_rs.REDHILLS
    })


# In[ ]:


# Plotting 3 years rainfall and reservoirs quantity data from 2015 to 2018 to better understand the patterns.
# CHEMBARAMBAKKAM

sns.set(rc={'figure.figsize':(15,10)}) 
df_three_years_CHEMBARAMBAKKAM = df_CHEMBARAMBAKKAM['01-01-2015':'01-01-2018']
df_three_years_CHEMBARAMBAKKAM.plot()
plt.show()


# In[ ]:


# Plotting 3 years rainfall and reservoirs quantity data from 2015 to 2018 to better understand the patterns.
# POONDI

sns.set(rc={'figure.figsize':(15,10)}) 
df_three_years_POONDI = df_POONDI['01-01-2015':'01-01-2018']
df_three_years_POONDI.plot()
plt.show()


# In[ ]:


# Plotting 3 years rainfall and reservoirs quantity data from 2015 to 2018 to better understand the patterns.
# REDHILLS

sns.set(rc={'figure.figsize':(15,10)}) 
df_three_years_REDHILLS = df_REDHILLS['01-01-2015':'01-01-2018']
df_three_years_REDHILLS.plot()
plt.show()


# In[ ]:


# Plotting 3 years rainfall and reservoirs quantity data from 2015 to 2018 to better understand the patterns.
# CHOLAVARAM

sns.set(rc={'figure.figsize':(15,10)}) 
df_three_years_CHOLAVARAM = df_CHOLAVARAM['01-01-2015':'01-01-2018']
df_three_years_CHOLAVARAM.plot()
plt.show()


# <h3><font color = '#00897b'>So it can be clearly observed that Chennai receives its maximum rainfall during the winters around <i><strong><font color = '#004d40'>October - January</font></strong></i>, and as expected, the amount of water in the reservoirs also tend to increase during this time.</font></h3>
# <h3><i><font color = '#2e7d32'>Now, lets explore monthly data around October - January to learn more about the trends.</font></i></h3>

# In[ ]:


# Plotting for time period between October 2015 to January 2016 for CHOLAVARAM

sns.set(rc={'figure.figsize':(15,10)}) 
df_three_months_CHOLAVARAM = df_CHOLAVARAM['10-01-2015':'01-01-2016']
df_three_months_CHOLAVARAM.plot()
plt.show()


# In[ ]:


# Plotting for time period between October 2015 to January 2016 for POONDI

sns.set(rc={'figure.figsize':(15,10)}) 
df_three_months_POONDI = df_POONDI['10-01-2015':'01-01-2016']
df_three_months_POONDI.plot()
plt.show()


# In[ ]:


# Plotting for time period between October 2015 to January 2016 for REDHILLS

sns.set(rc={'figure.figsize':(15,10)}) 
df_three_months_REDHILLS = df_REDHILLS['10-01-2015':'01-01-2016']
df_three_months_REDHILLS.plot()
plt.show()


# In[ ]:


# Plotting for time period between October 2015 to January 2016 for CHEMBARABAKKAM 

sns.set(rc={'figure.figsize':(15,10)}) 
df_three_months_CHEMBARAMBAKKAM = df_CHEMBARAMBAKKAM['10-01-2015':'01-01-2016']
df_three_months_CHEMBARAMBAKKAM.plot()
plt.show()


# <h3><font color = '#01579b'>It can be clearly observed that there's a good amount of rainfall during these 3 months and hence our assumption is correct.</font></h3>

# <h3><font color = '#263238'><b>Plotting Reservoir v/s Rainfall for the same location to understand any occuring pattern.</b></font></h3>

# <b><i><font color = '#7e57c2'>For the purpose of comparison on a plot, I have multiplied the values of rainfall by 5 because it has really small unit to be able to compare to the quantity of water in the reservoirs.</font></i></b>

# In[ ]:


# POONDI

df_POONDI.plot()
plt.show()


# In[ ]:


# CHEMBARAMBAKKAM 

df_CHEMBARAMBAKKAM.plot()
plt.show()


# In[ ]:


# CHOLAVARAM 

df_CHOLAVARAM.plot()
plt.show()


# In[ ]:


# REDHILLS 

df_REDHILLS.plot()
plt.show()


# <ul><font color = '#37474f'><h3>Observations from the above analysis: </h3>
# <li><h3>Monsoon is at its peak during the months of October - January in Chennai, which is quite late when compared to the rest of the country.</h3></li>
# <li><h3>It is observed that as the monsoon approaches, the level of water in the reserviors increase. This has been the trend (also, expectation) from the last 15 years.</h3></li>
# <li><h3>There was a severe drought for 3 years (2012 - 2015) which shows the upredictable nature of monsoon in Chennai, which is quite concerning given that millions of people reside there. </h3></li>
# <li><h3>But it can also be observed that as the monsson gets over, the quantity of water in the reservoirs deplete sharply.</h3></li></font></ul>

# <i><h3><font color = '#ef5350'>Hence, need of the hour is to find optimal ways to conserve water by methods like rainwater harvesting so that it can be used throughout the year and not only during or after few months from monsoon. 
