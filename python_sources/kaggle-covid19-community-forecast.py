#!/usr/bin/env python
# coding: utf-8

# # Kaggle COVID19 Forecasting - Cumulative Cases and Fatalities
# 
# This selects the top 20 Kaggle Community forecasts to-date from the [COVID19 Week 4 forecasting challenge](https://www.kaggle.com/c/covid19-global-forecasting-week-4), aggregates them, and visualizes the forecasts by location. The forecasts were submitted on or before Wednesday, April 15.

# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[ ]:


import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as subplots
import warnings

def build_graphs(submission_path, solution_path, test_path, ihme_path, mode="all"):
    submission_files = os.listdir(submission_path)
    submissions_list = []

    for f in submission_files:
        submission = pd.read_csv(os.path.join(submission_path, f))
        submission.insert(0, "SubmissionId", int(f[:-4]))
        if "ForecastID" in submission.columns:
            submission.rename(columns={"ForecastID": "ForecastId"}, inplace=True)
        submissions_list.append(submission)

    submissions = pd.concat(submissions_list, ignore_index=True, sort=False)

    # Read in solution/test files
    solution = pd.read_csv(solution_path)
    test = pd.read_csv(test_path)
    if "Province/State" in test.columns:
        test = test.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"})
    test.loc[:,"LocationName"] = [row["Country_Region"] if type(row["Province_State"]) is not str else ("%s - %s" % (row["Country_Region"], row["Province_State"])) for (i, row) in test.iterrows()]

    # Merge submissions to solution and test files
    forecasts = submissions.merge(test, on="ForecastId", how="left")
    forecasts = forecasts.merge(solution, on="ForecastId", how="left", suffixes=("", "Actual"))

    # Add daily information
    forecasts.sort_values(by=["SubmissionId", "LocationName", "ForecastId"], inplace=True)
    forecasts.loc[forecasts["Usage"]=="Ignored", "ConfirmedCasesActual"] = np.nan
    forecasts.loc[forecasts["Usage"]=="Ignored", "FatalitiesActual"] = np.nan
    forecasts.loc[:, "DailyConfirmedCases"] = np.nan
    forecasts.loc[forecasts.index!=forecasts.index[0], "DailyConfirmedCases"] = forecasts.loc[forecasts.index!=forecasts.index[0], "ConfirmedCases"].to_numpy()-forecasts.loc[forecasts.index!=forecasts.index[-1], "ConfirmedCases"].to_numpy()
    forecasts.loc[:, "DailyConfirmedCasesActual"] = np.nan
    forecasts.loc[forecasts.index!=forecasts.index[0], "DailyConfirmedCasesActual"] = forecasts.loc[forecasts.index!=forecasts.index[0], "ConfirmedCasesActual"].to_numpy()-forecasts.loc[forecasts.index!=forecasts.index[-1], "ConfirmedCasesActual"].to_numpy()
    forecasts.loc[:, "DailyFatalities"] = np.nan
    forecasts.loc[forecasts.index!=forecasts.index[0], "DailyFatalities"] = forecasts.loc[forecasts.index!=forecasts.index[0], "Fatalities"].to_numpy()-forecasts.loc[forecasts.index!=forecasts.index[-1], "Fatalities"].to_numpy()
    forecasts.loc[:, "DailyFatalitiesActual"] = np.nan
    forecasts.loc[forecasts.index!=forecasts.index[0], "DailyFatalitiesActual"] = forecasts.loc[forecasts.index!=forecasts.index[0], "FatalitiesActual"].to_numpy()-forecasts.loc[forecasts.index!=forecasts.index[-1], "FatalitiesActual"].to_numpy()

    # This conversion to daily causes some forecasts to go negative. Set these to 0
    forecasts.loc[forecasts["DailyConfirmedCases"]<0, "DailyConfirmedCases"] = 0.0
    forecasts.loc[forecasts["DailyFatalities"]<0, "DailyFatalities"] = 0.0

    # Add ranks and scores to the submissions
    # Some submission input values to logarithm were invalid
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    forecasts.loc[:,"FatalitiesSLE"] = (np.log(1+forecasts["Fatalities"])-np.log(1+forecasts["FatalitiesActual"]))**2
    forecasts.loc[:,"ConfirmedCasesSLE"] = (np.log(1+forecasts["ConfirmedCases"])-np.log(1+forecasts["ConfirmedCasesActual"]))**2
    warnings.filterwarnings("default", category=RuntimeWarning) 
    scores = forecasts[forecasts["Usage"]=="Private"][["SubmissionId", "FatalitiesSLE", "ConfirmedCasesSLE"]].groupby("SubmissionId").mean().reset_index()
    scores.loc[:, "FatalatiesRMSLE"] = np.sqrt(scores["FatalitiesSLE"])
    scores.loc[:, "ConfirmedCasesRMSLE"] = np.sqrt(scores["ConfirmedCasesSLE"])
    scores.loc[:, "OverallRMSLE"] = (scores["FatalatiesRMSLE"]+scores["ConfirmedCasesRMSLE"])/2.0
    scores = scores.sort_values(by="OverallRMSLE")
    forecasts = forecasts.merge(scores[["SubmissionId", "FatalatiesRMSLE", "ConfirmedCasesRMSLE", "OverallRMSLE"]], on="SubmissionId")
    ranks = forecasts[["SubmissionId", "OverallRMSLE"]].drop_duplicates().sort_values(by="OverallRMSLE")
    ranks["Rank"] = list(range(1, ranks.shape[0]+1))
    forecasts = forecasts.merge(ranks[["SubmissionId", "Rank"]], on="SubmissionId", how="inner")


    kaggle_community_forecast = (forecasts[forecasts["Rank"]<=20][["ForecastId", "LocationName", "Country_Region", "Province_State", "Date", "Usage", "ConfirmedCases", "Fatalities", "DailyConfirmedCases", "DailyFatalities"]]
                                 .groupby(["ForecastId", "LocationName", "Country_Region", "Province_State", "Date", "Usage"])
                                 .median()
                                 .reset_index()
                                 .sort_values(by="ForecastId"))

    # Read and process IHME forecasts
    ihme = pd.read_csv(ihme_path)
    min_date = min(forecasts[forecasts["Usage"]!="Public"]["Date"])
    max_date = max(forecasts[forecasts["Usage"]!="Public"]["Date"])
    ihme = ihme[(ihme["date"]>=min_date) & (ihme["date"]<=max_date)] # Filter IHME to date ranges of Kaggle predictions

    if mode=="develop":
        locations = [("US", "California", "US - California"), ("US", "New York", "US - New York")]
    else:
        locations = sorted(set([(row["Country_Region"], row["Province_State"] if type(row["Province_State"]) is str else "", row["LocationName"]) for (i, row) in test.iterrows()]))
    location_names = [l[2] for l in locations]
    default_location = "US - California"

    fig = subplots.make_subplots(rows=4, cols=1,
                                 subplot_titles=["Cumulative Cases", "Cumulative Fatalities", "Daily Cases", "Daily Fatalities"],
                                 vertical_spacing=0.05)

    region_plot_names = []
    buttons=[]

    for (country_name, state_name, location_name) in locations:
        if state_name!="":
            region = forecasts[(forecasts["Country_Region"]==country_name) & (forecasts["Province_State"]==state_name) & (forecasts["Usage"]!="Public") & (forecasts["Rank"]<=20)]
        else:
            region = forecasts[(forecasts["Country_Region"]==country_name) & (forecasts["Usage"]!="Public") & (forecasts["Rank"]<=20)]

        kaggle_region = kaggle_community_forecast[(kaggle_community_forecast["LocationName"]==location_name) & (kaggle_community_forecast["Usage"]!="Public")]
        region_actual = region[(region["SubmissionId"]==region["SubmissionId"][region.index[1]]) & (region["Usage"]=="Private")]

        #fig.add_trace(go.Box(x=region["Date"], y=region["ConfirmedCases"], visible=(location_name==default_location), name="Kaggle Forecast", marker=dict(opacity=0), marker_color="dodgerblue"), row=1, col=1)
        fig.add_trace(go.Scatter(x=kaggle_region["Date"], y=kaggle_region["ConfirmedCases"], line={"color":"dodgerblue"}, visible=(location_name==default_location), name="Kaggle Forecast"), row=1, col=1)
        fig.add_trace(go.Scatter(x=region_actual["Date"], y=region_actual["ConfirmedCasesActual"], line={"width":4, "color":"indianred"}, visible=(location_name==default_location), name="Reported"), row=1, col=1)
        #fig.add_trace(go.Box(x=region["Date"], y=region["Fatalities"], visible=(location_name==default_location), name="Kaggle Forecast", marker=dict(opacity=0), marker_color="dodgerblue", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=kaggle_region["Date"], y=kaggle_region["Fatalities"], line={"color":"dodgerblue"}, visible=(location_name==default_location), name="Kaggle Forecast", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=region_actual["Date"], y=region_actual["FatalitiesActual"], line={"width":4, "color":"indianred"}, visible=(location_name==default_location), name="Reported", showlegend=False), row=2, col=1)
        #fig.add_trace(go.Box(x=region["Date"], y=region["DailyConfirmedCases"], visible=(location_name==default_location), name="Kaggle Forecast", marker=dict(opacity=0), marker_color="dodgerblue", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=kaggle_region["Date"], y=kaggle_region["DailyConfirmedCases"], line={"color":"dodgerblue"}, visible=(location_name==default_location), name="Kaggle Forecast", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=region_actual["Date"], y=region_actual["DailyConfirmedCasesActual"], line={"width":4, "color":"indianred"}, visible=(location_name==default_location), name="Reported", showlegend=False), row=3, col=1)
        #fig.add_trace(go.Box(x=region["Date"], y=region["DailyFatalities"], visible=(location_name==default_location), name="Kaggle Forecast", marker=dict(opacity=0), marker_color="dodgerblue", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=kaggle_region["Date"], y=kaggle_region["DailyFatalities"], line={"color":"dodgerblue"}, visible=(location_name==default_location), name="Kaggle Forecast", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=region_actual["Date"], y=region_actual["DailyFatalitiesActual"], line={"width":4, "color":"indianred"}, visible=(location_name==default_location), name="Reported", showlegend=False), row=4, col=1)

        if state_name in set(ihme["location_name"]):
            ihme_region = ihme[ihme["location_name"]==state_name]
            fig.add_trace(go.Scatter(x=ihme_region["date"], y=ihme_region["totdea_mean"], visible=(location_name==default_location), name="IHME Forecast", line={"color": "mediumseagreen"}), row=2, col=1)
            fig.add_trace(go.Scatter(x=ihme_region["date"], y=ihme_region["deaths_mean"], visible=(location_name==default_location), name="IHME Forecast", line={"color": "mediumseagreen"}, showlegend=False), row=4, col=1)
            region_plot_names.extend([location_name]*10)        
        elif country_name in set(ihme["location_name"]):
            ihme_region = ihme[ihme["location_name"]==country_name]
            fig.add_trace(go.Scatter(x=ihme_region["date"], y=ihme_region["totdea_mean"], visible=(location_name==default_location), name="IHME Forecast", line={"color": "mediumseagreen"}), row=2, col=1)
            fig.add_trace(go.Scatter(x=ihme_region["date"], y=ihme_region["deaths_mean"], visible=(location_name==default_location), name="IHME Forecast", line={"color": "mediumseagreen"}, showlegend=False), row=4, col=1)
            region_plot_names.extend([location_name]*10)        
        else:
            region_plot_names.extend([location_name]*8)

    for (country_name, state_name, location_name) in locations:
        buttons.append(dict(method='update',
                            label=location_name,
                            args = [{'visible': [location_name==r for r in region_plot_names],
                                     "title": location_name}]))


    # add dropdown menus to the figure
    fig.update_layout(updatemenus=[{"buttons": buttons, "direction": "down", "active": location_names.index(default_location), "showactive": True, "x": 0.58, "y": 1.1}],
                      height=2000)
    fig.show()


# In[ ]:


build_graphs("../input/covid19-global-forecasting-submissions/week_4", "../input/covid-19-forecasting-ongoing-data-updates/wk4_solution.csv", "../input/covid19-forecasting-week-four-launch-data/test.csv", "../input/ihme-covid19-forecasts/ihme_2020_04_16.csv")


# In[ ]:


submission_path = "../input/covid19-global-forecasting-submissions/week_4"
solution_path   = "../input/covid-19-forecasting-ongoing-data-updates/wk4_solution.csv"
test_path       = "../input/covid19-forecasting-week-four-launch-data/test.csv"
ihme_path       = "../input/ihme-covid19-forecasts/ihme_2020_04_16.csv"
mode            = "develop"

submission_files = os.listdir(submission_path)
submissions_list = []

for f in submission_files:
    submission = pd.read_csv(os.path.join(submission_path, f))
    submission.insert(0, "SubmissionId", int(f[:-4]))
    if "ForecastID" in submission.columns:
        submission.rename(columns={"ForecastID": "ForecastId"}, inplace=True)
    submissions_list.append(submission)

submissions = pd.concat(submissions_list, ignore_index=True, sort=False)

# Read in solution/test files
solution = pd.read_csv(solution_path)
test = pd.read_csv(test_path)
if "Province/State" in test.columns:
    test = test.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"})
test.loc[:,"LocationName"] = [row["Country_Region"] if type(row["Province_State"]) is not str else ("%s - %s" % (row["Country_Region"], row["Province_State"])) for (i, row) in test.iterrows()]

# Merge submissions to solution and test files
forecasts = submissions.merge(test, on="ForecastId", how="left")
forecasts = forecasts.merge(solution, on="ForecastId", how="left", suffixes=("", "Actual"))

# Add daily information
forecasts.sort_values(by=["SubmissionId", "LocationName", "ForecastId"], inplace=True)
forecasts.loc[forecasts["Usage"]=="Ignored", "ConfirmedCasesActual"] = np.nan
forecasts.loc[forecasts["Usage"]=="Ignored", "FatalitiesActual"] = np.nan
forecasts.loc[:, "DailyConfirmedCases"] = np.nan
forecasts.loc[forecasts.index!=forecasts.index[0], "DailyConfirmedCases"] = forecasts.loc[forecasts.index!=forecasts.index[0], "ConfirmedCases"].to_numpy()-forecasts.loc[forecasts.index!=forecasts.index[-1], "ConfirmedCases"].to_numpy()
forecasts.loc[:, "DailyConfirmedCasesActual"] = np.nan
forecasts.loc[forecasts.index!=forecasts.index[0], "DailyConfirmedCasesActual"] = forecasts.loc[forecasts.index!=forecasts.index[0], "ConfirmedCasesActual"].to_numpy()-forecasts.loc[forecasts.index!=forecasts.index[-1], "ConfirmedCasesActual"].to_numpy()
forecasts.loc[:, "DailyFatalities"] = np.nan
forecasts.loc[forecasts.index!=forecasts.index[0], "DailyFatalities"] = forecasts.loc[forecasts.index!=forecasts.index[0], "Fatalities"].to_numpy()-forecasts.loc[forecasts.index!=forecasts.index[-1], "Fatalities"].to_numpy()
forecasts.loc[:, "DailyFatalitiesActual"] = np.nan
forecasts.loc[forecasts.index!=forecasts.index[0], "DailyFatalitiesActual"] = forecasts.loc[forecasts.index!=forecasts.index[0], "FatalitiesActual"].to_numpy()-forecasts.loc[forecasts.index!=forecasts.index[-1], "FatalitiesActual"].to_numpy()

# This conversion to daily causes some forecasts to go negative. Set these to 0
forecasts.loc[forecasts["DailyConfirmedCases"]<0, "DailyConfirmedCases"] = 0.0
forecasts.loc[forecasts["DailyFatalities"]<0, "DailyFatalities"] = 0.0

# Add ranks and scores to the submissions
# Some submission input values to logarithm were invalid
warnings.filterwarnings("ignore", category=RuntimeWarning) 
forecasts.loc[:,"FatalitiesSLE"] = (np.log(1+forecasts["Fatalities"])-np.log(1+forecasts["FatalitiesActual"]))**2
forecasts.loc[:,"ConfirmedCasesSLE"] = (np.log(1+forecasts["ConfirmedCases"])-np.log(1+forecasts["ConfirmedCasesActual"]))**2
warnings.filterwarnings("default", category=RuntimeWarning) 
scores = forecasts[forecasts["Usage"]=="Private"][["SubmissionId", "FatalitiesSLE", "ConfirmedCasesSLE"]].groupby("SubmissionId").mean().reset_index()
scores.loc[:, "FatalatiesRMSLE"] = np.sqrt(scores["FatalitiesSLE"])
scores.loc[:, "ConfirmedCasesRMSLE"] = np.sqrt(scores["ConfirmedCasesSLE"])
scores.loc[:, "OverallRMSLE"] = (scores["FatalatiesRMSLE"]+scores["ConfirmedCasesRMSLE"])/2.0
scores = scores.sort_values(by="OverallRMSLE")
forecasts = forecasts.merge(scores[["SubmissionId", "FatalatiesRMSLE", "ConfirmedCasesRMSLE", "OverallRMSLE"]], on="SubmissionId")
ranks = forecasts[["SubmissionId", "OverallRMSLE"]].drop_duplicates().sort_values(by="OverallRMSLE")
ranks["Rank"] = list(range(1, ranks.shape[0]+1))
forecasts = forecasts.merge(ranks[["SubmissionId", "Rank"]], on="SubmissionId", how="inner")


kaggle_community_forecast = (forecasts[forecasts["Rank"]<=20][["ForecastId", "LocationName", "Country_Region", "Province_State", "Date", "Usage", "ConfirmedCases", "Fatalities", "DailyConfirmedCases", "DailyFatalities"]]
                             .groupby(["ForecastId", "LocationName", "Country_Region", "Province_State", "Date", "Usage"])
                             .median()
                             .reset_index()
                             .sort_values(by="ForecastId"))

# Read and process IHME forecasts
ihme = pd.read_csv(ihme_path)
min_date = min(forecasts[forecasts["Usage"]!="Public"]["Date"])
max_date = max(forecasts[forecasts["Usage"]!="Public"]["Date"])
ihme = ihme[(ihme["date"]>=min_date) & (ihme["date"]<=max_date)] # Filter IHME to date ranges of Kaggle predictions

if mode=="develop":
    locations = [("US", "California", "US - California"), ("US", "New York", "US - New York")]
else:
    locations = sorted(set([(row["Country_Region"], row["Province_State"] if type(row["Province_State"]) is str else "", row["LocationName"]) for (i, row) in test.iterrows()]))
location_names = [l[2] for l in locations]
default_location = "US - California"

fig = subplots.make_subplots(rows=4, cols=1,
                             subplot_titles=["Cumulative Cases", "Cumulative Fatalities", "Daily Cases", "Daily Fatalities"],
                             vertical_spacing=0.05)

region_plot_names = []
buttons=[]

for (country_name, state_name, location_name) in locations:
    if state_name!="":
        region = forecasts[(forecasts["Country_Region"]==country_name) & (forecasts["Province_State"]==state_name) & (forecasts["Usage"]!="Public") & (forecasts["Rank"]<=20)]
    else:
        region = forecasts[(forecasts["Country_Region"]==country_name) & (forecasts["Usage"]!="Public") & (forecasts["Rank"]<=20)]
    
    kaggle_region = kaggle_community_forecast[(kaggle_community_forecast["LocationName"]==location_name) & (kaggle_community_forecast["Usage"]!="Public")]
    region_actual = region[(region["SubmissionId"]==region["SubmissionId"][region.index[1]]) & (region["Usage"]=="Private")]

    #fig.add_trace(go.Box(x=region["Date"], y=region["ConfirmedCases"], visible=(location_name==default_location), name="Kaggle Forecast", marker=dict(opacity=0), marker_color="dodgerblue"), row=1, col=1)
    fig.add_trace(go.Scatter(x=kaggle_region["Date"], y=kaggle_region["ConfirmedCases"], line={"color":"dodgerblue"}, visible=(location_name==default_location), name="Kaggle Forecast"), row=1, col=1)
    fig.add_trace(go.Scatter(x=region_actual["Date"], y=region_actual["ConfirmedCasesActual"], line={"width":4, "color":"indianred"}, visible=(location_name==default_location), name="Reported"), row=1, col=1)
    #fig.add_trace(go.Box(x=region["Date"], y=region["Fatalities"], visible=(location_name==default_location), name="Kaggle Forecast", marker=dict(opacity=0), marker_color="dodgerblue", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=kaggle_region["Date"], y=kaggle_region["Fatalities"], line={"color":"dodgerblue"}, visible=(location_name==default_location), name="Kaggle Forecast", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=region_actual["Date"], y=region_actual["FatalitiesActual"], line={"width":4, "color":"indianred"}, visible=(location_name==default_location), name="Reported", showlegend=False), row=2, col=1)
    #fig.add_trace(go.Box(x=region["Date"], y=region["DailyConfirmedCases"], visible=(location_name==default_location), name="Kaggle Forecast", marker=dict(opacity=0), marker_color="dodgerblue", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=kaggle_region["Date"], y=kaggle_region["DailyConfirmedCases"], line={"color":"dodgerblue"}, visible=(location_name==default_location), name="Kaggle Forecast", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=region_actual["Date"], y=region_actual["DailyConfirmedCasesActual"], line={"width":4, "color":"indianred"}, visible=(location_name==default_location), name="Reported", showlegend=False), row=3, col=1)
    #fig.add_trace(go.Box(x=region["Date"], y=region["DailyFatalities"], visible=(location_name==default_location), name="Kaggle Forecast", marker=dict(opacity=0), marker_color="dodgerblue", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=kaggle_region["Date"], y=kaggle_region["DailyFatalities"], line={"color":"dodgerblue"}, visible=(location_name==default_location), name="Kaggle Forecast", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=region_actual["Date"], y=region_actual["DailyFatalitiesActual"], line={"width":4, "color":"indianred"}, visible=(location_name==default_location), name="Reported", showlegend=False), row=4, col=1)
    
    if state_name in set(ihme["location_name"]):
        ihme_region = ihme[ihme["location_name"]==state_name]
        fig.add_trace(go.Scatter(x=ihme_region["date"], y=ihme_region["totdea_mean"], visible=(location_name==default_location), name="IHME Forecast", line={"color": "mediumseagreen"}), row=2, col=1)
        fig.add_trace(go.Scatter(x=ihme_region["date"], y=ihme_region["deaths_mean"], visible=(location_name==default_location), name="IHME Forecast", line={"color": "mediumseagreen"}, showlegend=False), row=4, col=1)
        region_plot_names.extend([location_name]*10)        
    elif country_name in set(ihme["location_name"]):
        ihme_region = ihme[ihme["location_name"]==country_name]
        fig.add_trace(go.Scatter(x=ihme_region["date"], y=ihme_region["totdea_mean"], visible=(location_name==default_location), name="IHME Forecast", line={"color": "mediumseagreen"}), row=2, col=1)
        fig.add_trace(go.Scatter(x=ihme_region["date"], y=ihme_region["deaths_mean"], visible=(location_name==default_location), name="IHME Forecast", line={"color": "mediumseagreen"}, showlegend=False), row=4, col=1)
        region_plot_names.extend([location_name]*10)        
    else:
        region_plot_names.extend([location_name]*8)

for (country_name, state_name, location_name) in locations:
    buttons.append(dict(method='update',
                        label=location_name,
                        args = [{'visible': [location_name==r for r in region_plot_names],
                                 "title": location_name}]))
   

# add dropdown menus to the figure
fig.update_layout(updatemenus=[{"buttons": buttons, "direction": "down", "active": location_names.index(default_location), "showactive": True, "x": 0.58, "y": 1.1}],
                  height=2000)
fig.show()


# In[ ]:


us_state_abbreviations = pd.read_csv("../input/us-state-abbreviations/us_state_abbreviations.csv")

latest_us = test.merge(solution, on="ForecastId")
latest_us = latest_us[latest_us["Usage"]!="Ignored"]
latest_us.loc[:, "DailyConfirmedCases"] = np.nan
latest_us.loc[latest_us.index!=latest_us.index[0], "DailyConfirmedCases"] = latest_us.loc[latest_us.index!=latest_us.index[0], "ConfirmedCases"].to_numpy()-latest_us.loc[latest_us.index!=latest_us.index[-1], "ConfirmedCases"].to_numpy()
latest_us.loc[:, "DailyFatalities"] = np.nan
latest_us.loc[latest_us.index!=latest_us.index[0], "DailyFatalities"] = latest_us.loc[latest_us.index!=latest_us.index[0], "Fatalities"].to_numpy()-latest_us.loc[latest_us.index!=latest_us.index[-1], "Fatalities"].to_numpy()
latest_date = max(latest_us["Date"])
latest_us = latest_us[latest_us["Date"]==latest_date]
latest_us = latest_us[latest_us["Country_Region"]=="US"]
latest_us = latest_us.merge(us_state_abbreviations, left_on="Province_State", right_on="State", how="inner")

stats = ["DailyConfirmedCases", "ConfirmedCases", "DailyFatalities", "Fatalities"]
stat_labels = ["Daily Confirmed Cases", "Cumulative Confirmed Cases", "Daily Fatalities", "Cumulative Fatalities"]
default_stat="DailyFatalities"
buttons = []

fig = go.Figure()

for (stat, label) in zip(stats, stat_labels):
    fig.add_trace(go.Choropleth(
        locations=latest_us["Code"],
        z = latest_us[stat],
        locationmode = 'USA-states', 
        hovertemplate = "%{text}",
        text = ["<br />".join(["<b>{State}</b><br />",
                               "Daily Cases: {DailyConfirmedCases:,.0f}",
                               "Cumulative Cases: {ConfirmedCases:,.0f}<br />",
                               "Daily Fatalities: {DailyFatalities:,.0f}",
                               "Cumulative Fatalities: {Fatalities:,.0f}",]).format(**row) for (i, row) in latest_us.iterrows()],
        colorbar_title = stat,
        name="",
        visible=(stat==default_stat)))
    buttons.append(dict(method='update',
                    label= label + " (data from %s)" % latest_date,
                    args = [{'visible': [stat==s for s in stats]}]))

fig.update_layout(
    updatemenus=[{"buttons": buttons, "direction": "down", "active": stats.index(default_stat), "showactive": True, "x": 0.58, "y": 1.15}],
    geo_scope='usa', # limite map scope to USA
)

fig.show()


# In[ ]:




