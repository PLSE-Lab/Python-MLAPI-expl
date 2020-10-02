#!/usr/bin/env python
# coding: utf-8

# # Predicting the trend of the Coronavirus using Facebook Prophet
# Initially designed to predict the creation of events on Facebook, [Prophet](https://facebook.github.io/prophet/) is a very useful tool to analyse timeseries data. Naturally, any spread of disease is such a timeseries. Prophet fits an additive model to the timeseries based on seasonality, whether they are daily, weekly etc. It is available both for Python and R. The underlying language is Stan a Bayesian statistical modelling language that is extremely fast. <br>
# 
# So basically the model is based on a smoothed addition of the following 4 components:
# * The trend of the data
# * The weekly variations
# * The annual seasonality
# * And optionally holiday effects
# 
# Keep in mind what it was designed for, to predict human behaviour. And the research by Facebook showed that most events were created on Mondays. <br>
# Now a virus is unlikely to be affected by holidays, or weekdays although people were worried about the Chinese new year celebrations that could have worsened the outbreak. However, it is unlikely that the limited data that is currently available (about two months) is sufficient to pick up any such trend. The same is true for annual seasonality as we need a minimum of one year to be able to pick this up. <br> 
# 
# Ok so let us do Prophet do its magic and see what we can learn. 
# 

# In[ ]:


import pandas as pd

df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
df.Date = pd.to_datetime(df.Date, format='%m/%d/%y')
df.head()


# So we start by summing up the daily numbers as that is what is interesting to predict a global trend.

# In[ ]:


total = df.groupby(['Date']).sum().loc[:,['Confirmed','Deaths','Recovered']].reset_index()

total.head()


# We switch off the daily component as we only have cases per day and not per hour so all cases are reported at midnight each day. That makes a daily variation useless. <br>
# 
# *Before you can start you need to rename the columns of the dataframe to ds for the timestamps and y for the target.*

# In[ ]:


import fbprophet
from fbprophet.plot import add_changepoints_to_plot

df_prophet= total.rename(columns={'Date': 'ds', 'Confirmed': 'y'})

# Make a future dataframe for X days
m_global = fbprophet.Prophet(changepoint_prior_scale=0.05,changepoint_range=0.95,
                      daily_seasonality=False, 
                      weekly_seasonality=True,
                     mcmc_samples=300)
# Add seasonlity
m_global.add_seasonality(name='monthly', period=30.5, fourier_order=5)

m_global.fit(df_prophet)

# Make predictions
future_global = m_global.make_future_dataframe(periods=30, freq='D')

forecast_global = m_global.predict(future_global)

m_global.plot_components(forecast_global);


# 
# ### Interpreting the trend plot
# What we can see (in March 2020 but this will change over time) is that the model predicts a linear increase in cases into infinity. This is obviously not possible as that would require an infinite "supply" of infectable humans. However, at the moment that is the case as the virus makes its way across the globe. In reality, what we would expect is an exponential growth curve that will eventually plateau and drop again. You can see an example of the US flu season (2018-19) recorded by the [CDC](https://www.cdc.gov/mmwr/volumes/68/wr/mm6824a3.htm).
# 
# ![](https://www.cdc.gov/mmwr/volumes/68/wr/figures/mm6824a3-F1.gif)
# 
# As we are only two months into the Coronavirus data this trend is not yet apparent but will eventually set in. This means any model we create has to be updated very frequently as we will eventually get astronomical predictions.
# 
# ### Interpreting the weekly plot
# What is apparent is that the error margin is very large suggesting that there is no variation over the week. This would be expected as the spread of a virus is not affected by weekdays. <br>
# But what about Wednesday? It appears there is a drop of cases on Wednesdays. That does not really make sense as mentioned before a virus does not care what day of the week it is. <br>
# What we see here is a change in case definition on Wednesday 19th of February. On that day China changed its reporting standards, which was later reversed. This caused a drop of about two-thirds in new cases on this particular Wednesday, explaining the weekly model.
# 
# ### Interpreting the monthly model
# Again this has to be taken with a pinch of salt because we only have two months worth of data. However, the model suggests that more new cases occur around the middle of the month. This could potentially be an actual behaviour of the outbreak. <br>
# What epidemiologists call an index case or patient zero is the first human that became infected. This most likely was a once-off event. The source of the virus is not known but the go-to culprits based on investigations into SARS are always bats on a meat market. That is an educated guess and might well prove wrong. Whether or not the origin are bats the virus must have jumped species, from an animal to humans. This is a rarerish event and this caused the infection of patient-zero. Patient-zero will then have continued their normal daily routine for approximately two weeks before they showed symptoms. That means that an increase in roughly two-week waves is imaginable. <br>
# 
# Contributing to this will be the guideline changes described in the interpretation of the weekly plot, as these occurred in the middle of the month.
# 

# So far so good that are the components of our quick model. But what are the predictions for the future? 

# In[ ]:


fig =m_global.plot(forecast_global)


# This is what our prediction looks like. The direction of overall case numbers is probably true. If you look closely in March you can see a shoulder in the data. This looks suspiciously similar to the shoulder seen in February. I have explained in detail how it came to this ledge in this [post](https://medium.com/@treich112/tracking-the-coronavirus-without-the-panic-53b580dee5c1). In summary, the definition of what constitutes a coronavirus case changed a couple of times making it first less than more sensitive. So this ominous shoulder is not "real" as the cases will have continued to grow unhindered by legislation change. <br>
# 
# Our model, however, does not know these details and it appears to have learned that the coronavirus spreads with these characteristic shoulders. This highlights the issue with real-life data, that can be affected by simple guideline changes. An option could be to interpolate the affected area to maybe get a model mirroring reality more closely. <br>
# 
# If you come back to this post in a few weeks it will have become apparent how good the model was and it will most likely have changed then as well.

# The prophet model identifies points at which the trend changes and those can be plotted.

# In[ ]:


fig = m_global.plot(forecast_global)
a = add_changepoints_to_plot(fig.gca(), m_global, forecast_global)


# The changepoints are shown in red by default Prophet used the first 80% of the dataset to establish the changepoints. I have deliberately changed this to 95% as we have so little data so far. In this graph, it is nicely visible how the uncertainty increases into the future shown as light blue shade. 

# I am aware that predicting the global caseload has only limited value as it is at best a nice to know figure. Ideally, a single country-trend and in a perfect world a prediction for individual provinces would be desirable. To see if that is possible we will look at China simply because this is where the outbreak started and we have the dataset with the longest history. <br>
# 
# **Give it a go and try a different country!**

# In[ ]:


# restrict to one country
df_china = df[df['Country/Region']=='China']
total_china = df_china.groupby(['Date']).sum().loc[:,['Confirmed','Deaths','Recovered']].reset_index()
total_china.head()


# In[ ]:


(100*total_china.Confirmed.sum()) / 1339724852


# In[ ]:


china_prophet= total_china.rename(columns={'Date': 'ds', 'Confirmed': 'y'})

# Make a future dataframe for X days
m_china = fbprophet.Prophet(changepoint_prior_scale=0.05,changepoint_range=0.95,
                      daily_seasonality=False, 
                      weekly_seasonality=True,
                     mcmc_samples=300)
# Add seasonlity
m_china.add_seasonality(name='monthly', period=30.5, fourier_order=5)

m_china.fit(china_prophet)

# Make predictions
future_china = m_china.make_future_dataframe(periods=30, freq='D')

forecast_china = m_china.predict(future_china)

m_china.plot_components(forecast_china);


# Although the graph looks rather similar and there are hardly any changes on the weekly and monthly graphs the change might be subtle but interesting. <br>
# The trend curve is now an actual curve rather than a linear trend as the global scale suggests. This suggests that the cases in China might be about to plateau and case numbers might soon even reduce. Have a look at the absolute case numbers for China below vs the global numbers. <br>
# 
# The similarity of the weekly and monthly curves are expected if we assume the analysis especially my interpretation that the drop on Wednesdays and the increase in the middle of the month could be caused by the change in the case definition. Furthermore, the model obviously has learned from the past and at the beginning of the data, the cases in China dominated the dataset. <br>

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,6))

fig = sns.set_palette('viridis')
fig = sns.lineplot(x='Date', y='Confirmed',data = total, label='Global')
fig = sns.lineplot(x='Date', y='Confirmed',data = total_china, label='China')

sns.despine()
plt.legend()
plt.tight_layout()
plt.xticks(rotation=40)


# What you can see on the graph above is that new cases in China are indeed decreasing and it looks like the plateau might have been reached. What is striking though are the very few cases not in China. When I am writing this (08/03/2020) 93.6 % are Chinese cases. This means the global model is trained predominantly on case number from China. See the current percentage below.

# In[ ]:


print('Percentage of global cases in China: %s' %((total_china.Confirmed.sum()*100)/total.Confirmed.sum()))


# So let's see what the prediction looks like.

# In[ ]:


fig = m_china.plot(forecast_china)


# So the suspicion that the model might be inclined to assume that the spread in China could slow down is confirmed from the predictions above. This is interesting as the slow down is predicted to occur after another "shoulder" in the curve. Prophet also allows looking at the predictions. <br>
# 
# Below you will find the predictions for the next six days. Time will tell if the models we have built are of any use. I will update this conclusion over the next few days so come back and see if we got it right.

# In[ ]:


forecast_china[len(total_china):].loc[:,['ds', 'yhat_lower' ,'yhat_upper', 'yhat']].iloc[:7]


# In[ ]:


forecast_global[len(total):].loc[:,['ds', 'yhat_lower' ,'yhat_upper', 'yhat']].iloc[:7]


# In summary, the analysis shown here illustrates that contrary to western media at the moment only 7% of cases occurred outside China. This means any model trained on these data can mostly learn from the Chinese cases at this stage. Therefore it would be expected that the model predictions for China might be better than global estimates. Besides, the data emphasised what impact a change in case numbers can have. And what troubles this can cause for us data scientists.
