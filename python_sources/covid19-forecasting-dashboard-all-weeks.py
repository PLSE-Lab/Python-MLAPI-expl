#!/usr/bin/env python
# coding: utf-8

# # Kaggle COVID19 Forecasting - Cumulative Cases and Fatalities
# 
# This selects the top 25 best team's selected submissions to-date in each of the four weekly COVID19 forecasting competitions, and visualizes the predictions by US state.

# In[ ]:


import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings

def build_graphs(week_num, submission_path, solution_path, test_path):
    submission_files = os.listdir(submission_path)
    submissions_list = []

    for f in submission_files:
        submission = pd.read_csv(os.path.join(submission_path, f))
        submission.insert(0, "SubmissionId", int(f[:-4]))
        submissions_list.append(submission)

    submissions = pd.concat(submissions_list, ignore_index=True, sort=False)

    # Read in solution/test files
    wk1_solution = pd.read_csv(solution_path)
    wk1_test = pd.read_csv(test_path)
    if "Province/State" in wk1_test.columns:
        wk1_test = wk1_test.rename(columns={"Province/State": "Province_State", "Country/Region": "Country_Region"})

    # Merge submissions to solution and test files
    wk1 = submissions.merge(wk1_test, on="ForecastId", how="left")
    wk1 = wk1.merge(wk1_solution, on="ForecastId", how="left", suffixes=("", "Actual"))

    # Add ranks and scores to the submissions
    # Some submission input values to logarithm were invalid
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    wk1.loc[:,"FatalitiesSLE"] = (np.log(1+wk1["Fatalities"])-np.log(1+wk1["FatalitiesActual"]))**2
    wk1.loc[:,"ConfirmedCasesSLE"] = (np.log(1+wk1["ConfirmedCases"])-np.log(1+wk1["ConfirmedCasesActual"]))**2
    warnings.filterwarnings("default", category=RuntimeWarning) 
    wk1_scores = wk1[wk1["Usage"]=="Private"][["SubmissionId", "FatalitiesSLE", "ConfirmedCasesSLE"]].groupby("SubmissionId").mean().reset_index()
    wk1_scores.loc[:, "FatalatiesRMSLE"] = np.sqrt(wk1_scores["FatalitiesSLE"])
    wk1_scores.loc[:, "ConfirmedCasesRMSLE"] = np.sqrt(wk1_scores["ConfirmedCasesSLE"])
    wk1_scores.loc[:, "OverallRMSLE"] = (wk1_scores["FatalatiesRMSLE"]+wk1_scores["ConfirmedCasesRMSLE"])/2.0
    wk1_scores = wk1_scores.sort_values(by="OverallRMSLE")
    wk1 = wk1.merge(wk1_scores[["SubmissionId", "FatalatiesRMSLE", "ConfirmedCasesRMSLE", "OverallRMSLE"]], on="SubmissionId")
    wk1_ranks = wk1[["SubmissionId", "OverallRMSLE"]].drop_duplicates().sort_values(by="OverallRMSLE")
    wk1_ranks["Rank"] = list(range(1, wk1_ranks.shape[0]+1))
    wk1 = wk1.merge(wk1_ranks[["SubmissionId", "Rank"]], on="SubmissionId", how="inner")
    
    states = sorted(set(wk1[wk1["Country_Region"]=="US"]["Province_State"]))

    # FATALITIES
    fig_fatalities=go.Figure()

    region_plot_names = []
    updatemenu=[]
    buttons=[]

    default_state = "California"

    for region_name in states:
        region = wk1[(wk1["Province_State"]==region_name) & (wk1["Usage"]!="Public") & (wk1["Rank"]<=25)]
        region_actual = region[(region["SubmissionId"]==region["SubmissionId"][region.index[1]]) & (region["Usage"]=="Private")]

        fig_fatalities.add_trace(go.Box(x=region["Date"], y=region["Fatalities"], visible=(region_name==default_state)))
        fig_fatalities.add_trace(go.Scatter(x=region_actual["Date"], y=region_actual["FatalitiesActual"], line={}, visible=(region_name==default_state)))

        region_plot_names.extend([region_name]*2)

    for region_name in states:
        buttons.append(dict(method='update',
                            label=region_name + (" Fatalities, Week %d" % week_num),
                            args = [{'visible': [region_name==r for r in region_plot_names],
                                     "title": region_name + " Fatalities"}]))

    # add dropdown menus to the figure
    fig_fatalities.update_layout(showlegend=False, updatemenus=[{"buttons": buttons, "direction": "down", "active": states.index(default_state), "showactive": True, "x": 0.5, "y": 1.15}])
    
    # CASES
    fig_cases=go.Figure()

    region_plot_names = []
    updatemenu=[]
    buttons=[]

    default_state = "California"

    for region_name in states:
        region = wk1[(wk1["Province_State"]==region_name) & (wk1["Usage"]!="Public") & (wk1["Rank"]<=25)]
        region_actual = region[(region["SubmissionId"]==region["SubmissionId"][region.index[1]]) & (region["Usage"]=="Private")]

        fig_cases.add_trace(go.Box(x=region["Date"], y=region["ConfirmedCases"], visible=(region_name==default_state)))
        fig_cases.add_trace(go.Scatter(x=region_actual["Date"], y=region_actual["ConfirmedCasesActual"], line={}, visible=(region_name==default_state)))

        region_plot_names.extend([region_name]*2)

    for region_name in states:
        buttons.append(dict(method='update',
                            label=region_name + (" Cases, Week %d" % week_num),
                            args = [{'visible': [region_name==r for r in region_plot_names],
                                     "title": region_name + " Cases"}]))

    # add dropdown menus to the figure
    fig_cases.update_layout(showlegend=False, updatemenus=[{"buttons": buttons, "direction": "down", "active": states.index(default_state), "showactive": True, "x": 0.5, "y": 1.15}])
    
    return (fig_fatalities, fig_cases)


# # Week 1 Predictions
# 
# These forecasts came from the [COVID19 Week 1 forecasting challenge](https://www.kaggle.com/c/covid19-global-forecasting-week-1) and were submitted by Wednesday March 25.

# In[ ]:


(fig_wk1_fatalities, fig_wk1_cases) = build_graphs(1, "../input/covid19-global-forecasting-submissions/week_1", "../input/covid-19-forecasting-ongoing-data-updates/wk1_solution.csv", "../input/covid19-forecasting-week-one-launch-data/test.csv")
fig_wk1_fatalities.show()


# In[ ]:


fig_wk1_cases.show()


# # Week 2 Predictions
# 
# These forecasts came from the [COVID19 Week 2 forecasting challenge](https://www.kaggle.com/c/covid19-global-forecasting-week-2) and were submitted by Wednesday April 1.

# In[ ]:


(fig_wk2_fatalities, fig_wk2_cases) = build_graphs(2, "../input/covid19-global-forecasting-submissions/week_2", "../input/covid-19-forecasting-ongoing-data-updates/wk2_solution.csv", "../input/covid19-forecasting-week-two-launch-data/test.csv")
fig_wk2_fatalities.show()


# In[ ]:


fig_wk2_cases.show()


# # Week 3 Predictions
# 
# These forecasts came from the [COVID19 Week 3 forecasting challenge](https://www.kaggle.com/c/covid19-global-forecasting-week-3) and were submitted by Wednesday April 8.

# In[ ]:


(fig_wk3_fatalities, fig_wk3_cases) = build_graphs(3, "../input/covid19-global-forecasting-submissions/week_3", "../input/covid-19-forecasting-ongoing-data-updates/wk3_solution.csv", "../input/covid19-forecasting-week-three-launch-data/test.csv")
fig_wk3_fatalities.show()


# In[ ]:


fig_wk3_cases.show()


# # Week 4 Predictions
# 
# These forecasts came from the [COVID19 Week 4 forecasting challenge](https://www.kaggle.com/c/covid19-global-forecasting-week-4) and were submitted by Wednesday April 15.

# In[ ]:


(fig_wk4_fatalities, fig_wk4_cases) = build_graphs(4, "../input/covid19-global-forecasting-submissions/week_4", "../input/covid-19-forecasting-ongoing-data-updates/wk4_solution.csv", "../input/covid19-forecasting-week-four-launch-data/test.csv")
fig_wk4_fatalities.show()


# In[ ]:


fig_wk4_cases.show()

