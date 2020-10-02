#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Forecasting Challenge (Week 2) Data Prep
# 
# This notebook prepared the data in Kaggle's [COVID-19 Global Forecasting Competition (Week 2)](https://www.kaggle.com/c/covid19-global-forecasting-week-2) that was used to launch the competition. The source data comes from [JHU CSSE's COVID-19 data repository on GitHub](https://github.com/CSSEGISandData/COVID-19).
# 
# I re-ran this notebook on updated data to add descriptive comments, so it won't output precisely the same as the original launch data. I saved the original launch data [to this dataset](https://www.kaggle.com/benhamner/covid19-forecasting-week-two-launch-data).
# 
# The data for the submission period for the forecasting challenges is also updated every day, alongside leaderboard rescores. I use [this notebook](https://www.kaggle.com/benhamner/covid-19-forecasting-ongoing-data-updates/) to run the ongoing data updates.

# In[ ]:


from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

confirmed = pd.read_csv("../input/jhucovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
deaths   = pd.read_csv("../input/jhucovid19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")


# In[ ]:


launch_date = date(2020, 3, 26)
latest_train_date = date(2020, 3, 25)

public_leaderboard_start_date = launch_date - timedelta(7)
close_date = launch_date + timedelta(7)
final_evaluation_start_date = launch_date + timedelta(8)


# Move to ISO 8601 dates

# In[ ]:


confirmed.columns = list(confirmed.columns[:4]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in confirmed.columns[4:]]
deaths.columns    = list(deaths.columns[:4])    + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in deaths.columns[4:]]


# In[ ]:


# Filter out problematic data points (The West Bank and Gaza had a negative value, cruise ships were associated with Canada, etc.)
removed_states = "Recovered|Grand Princess|Diamond Princess"
removed_countries = "US|The West Bank and Gaza"

confirmed.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)
deaths.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"}, inplace=True)
confirmed = confirmed[~confirmed["Province_State"].replace(np.nan, "nan").str.match(removed_states)]
deaths    = deaths[~deaths["Province_State"].replace(np.nan, "nan").str.match(removed_states)]
confirmed = confirmed[~confirmed["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]
deaths    = deaths[~deaths["Country_Region"].replace(np.nan, "nan").str.match(removed_countries)]

confirmed.drop(columns=["Lat", "Long"], inplace=True)
deaths.drop(columns=["Lat", "Long"], inplace=True)


# In[ ]:


confirmed


# In[ ]:


deaths


# Starting to pull in US state data, since this was saved separately

# In[ ]:


us_keys = pd.read_csv("../input/jhucovid19/csse_covid_19_data/csse_covid_19_daily_reports/03-25-2020.csv")
us_keys = us_keys[us_keys["Country_Region"]=="US"]
us_keys = us_keys.groupby(["Province_State", "Country_Region"])[["Confirmed", "Deaths"]].sum().reset_index()

us_keys = us_keys[~us_keys.Province_State.str.match("Diamond Princess|Grand Princess|Recovered|Northern Mariana Islands|American Samoa")].reset_index(drop=True)
us_keys


# In[ ]:


confirmed = confirmed.append(us_keys[["Province_State", "Country_Region"]], sort=False).reset_index(drop=True)
deaths = deaths.append(us_keys[["Province_State", "Country_Region"]], sort=False).reset_index(drop=True)


# In[ ]:


for col in confirmed.columns[2:]:
    confirmed[col].fillna(0, inplace=True)
    deaths[col].fillna(0, inplace=True)


# In[ ]:


confirmed


# Adding in daily US state data

# In[ ]:


us_start_date = date(2020, 3, 10)
day_date = us_start_date

while day_date <= latest_train_date:
    day = pd.read_csv("../input/jhucovid19/csse_covid_19_data/csse_covid_19_daily_reports/%s.csv" % day_date.strftime("%m-%d-%Y"))
    
    if "Country/Region" in day.columns:
        day.rename(columns={"Country/Region": "Country_Region", "Province/State": "Province_State"}, inplace=True)
    
    us = day[day["Country_Region"]=="US"]
    us = us.groupby(["Province_State", "Country_Region"])[["Confirmed", "Deaths"]].sum().reset_index()
    
    unused_data = []
    untouched_states = set(confirmed[confirmed["Country_Region"]=="US"]["Province_State"])
    
    for (i, row) in us.iterrows():
        if confirmed[(confirmed["Country_Region"]=="US") & (confirmed["Province_State"]==row["Province_State"])].shape[0]==1:
            confirmed.loc[(confirmed["Country_Region"]=="US") & (confirmed["Province_State"]==row["Province_State"]), day_date.strftime("%Y-%m-%d")] = row["Confirmed"]
            deaths.loc[(deaths["Country_Region"]=="US") & (deaths["Province_State"]==row["Province_State"]), day_date.strftime("%Y-%m-%d")] = row["Deaths"]
            untouched_states.remove(row["Province_State"])
        else:
            unused_data.append(row["Province_State"])
            
    print(day_date, "Untouched", untouched_states)
    print(day_date, "Unused", unused_data)

    day_date = day_date + timedelta(1)


# In[ ]:


confirmed


# In[ ]:


deaths


# Filtering out any data on or after the launch date for the competition

# In[ ]:


dates_on_after_launch = [col for col in confirmed.columns[4:] if col>=launch_date.strftime("%Y-%m-%d")]
print("Removing %d columns: %s" % (len(dates_on_after_launch), str(dates_on_after_launch)))

cols_to_keep = [col for col in confirmed.columns if col not in dates_on_after_launch]

confirmed = confirmed[cols_to_keep]
deaths = deaths[cols_to_keep]


# Adding the rows to be forecast

# In[ ]:


for i in range(36):
    this_date = (launch_date + timedelta(i)).strftime("%Y-%m-%d")
    confirmed.insert(len(confirmed.columns), this_date, np.NaN)
    deaths.insert(len(deaths.columns), this_date, np.NaN)


# Melting the data to a version that will be friendlier to Kaggle's evaluation system.

# In[ ]:


confirmed_melted = confirmed.melt(confirmed.columns[:2], confirmed.columns[2:], "Date", "ConfirmedCases")
#confirmed_melted.insert(5, "Type", "Confirmed")
deaths_melted = deaths.melt(deaths.columns[:2], deaths.columns[2:], "Date", "Fatalities")
#deaths_melted.insert(5, "Type", "Deaths")

confirmed_melted.sort_values(by=["Country_Region", "Province_State", "Date"], inplace=True)
deaths_melted.sort_values(by=["Country_Region", "Province_State", "Date"], inplace=True)

assert confirmed_melted.shape==deaths_melted.shape
assert list(confirmed_melted["Province_State"])==list(deaths_melted["Province_State"])
assert list(confirmed_melted["Country_Region"])==list(deaths_melted["Country_Region"])
assert list(confirmed_melted["Date"])==list(deaths_melted["Date"])

cases = confirmed_melted.merge(deaths_melted, on=["Province_State", "Country_Region", "Date"], how="inner")
cases = cases[["Country_Region", "Province_State", "Date", "ConfirmedCases", "Fatalities"]]

cases.sort_values(by=["Country_Region", "Province_State", "Date"], inplace=True)
cases.insert(0, "Id", range(1, cases.shape[0]+1))
cases


# In[ ]:


forecast = cases[cases["Date"]>=public_leaderboard_start_date.strftime("%Y-%m-%d")]
forecast.drop(columns="Id", inplace=True)
forecast.insert(0, "ForecastId", range(1, forecast.shape[0]+1))
forecast.insert(6, "Usage", "Ignored")
forecast.loc[forecast["Date"]<launch_date.strftime("%Y-%m-%d"),"Usage"]="Public"
forecast.loc[forecast["Date"]>=final_evaluation_start_date.strftime("%Y-%m-%d"),"Usage"]="Private"
forecast


# ## Global competition data

# In[ ]:


train = cases[cases["Date"]<launch_date.strftime("%Y-%m-%d")]
train.to_csv("train.csv", index=False)
train


# In[ ]:


test = forecast[forecast.columns[:-3]]
test.to_csv("test.csv", index=False)
test


# In[ ]:


solution = forecast[["ForecastId", "ConfirmedCases", "Fatalities", "Usage"]]
solution["ConfirmedCases"].fillna(1, inplace=True)
solution["Fatalities"].fillna(1, inplace=True)
solution.to_csv("solution.csv", index=False)
solution


# In[ ]:


submission = forecast[["ForecastId", "ConfirmedCases", "Fatalities"]]
submission["ConfirmedCases"] = 1
submission["Fatalities"] = 1
submission.to_csv("submission.csv", index=False)

submission


# In[ ]:




