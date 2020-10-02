#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_columns', 99)
pd.set_option('display.max_rows', 99)
from tqdm import tqdm
import plotly.express as px
import plotly.subplots as subplots
import plotly.graph_objects as go
import os

from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# # Read train test files

# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
train = train.fillna('')
train['Location'] = train.Country_Region + '-' + train.Province_State + '-' + train.County
train = train[['Location', 'Date', 'Target', 'Weight', 'TargetValue']]
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
test = test.fillna('')
test['Location'] = test.Country_Region + '-' + test.Province_State + '-' + test.County
test = test[['Location', 'Date', 'Target', 'ForecastId']]


# In[ ]:


train.shape, test.shape
train.head(2)
test.head(2)


# In[ ]:


train.Date.max()


# # Read Submission Files

# In[ ]:


PRIVATE_START = '2020-05-13'

DATA_DIR = '/kaggle/input/covid19belugaw5/'

top_submission_files = [f for f in os.listdir(DATA_DIR) if f.startswith('submission_')]
top_submission_files


# In[ ]:


top_submissions = pd.concat([
    pd.read_csv(DATA_DIR + f).set_index('ForecastId_Quantile') for f in top_submission_files],
    axis=1)
top_teams = [f.split('_')[1][:-4] for f in top_submission_files]
top_submissions.columns = [f'pred_{f}' for f in top_teams]
top_submissions['pred_blend'] = top_submissions.median(axis=1)
top_submissions['ForecastId'] = top_submissions.index.map(lambda s: s.split('_')[0]).astype(int)
top_submissions['q'] = top_submissions.index.map(lambda s: s.split('_')[1]).astype('float64')
top_submissions = top_submissions.merge(test, on='ForecastId')
top_submissions.Date.max()
top_submissions = top_submissions.merge(train, on=['Date', 'Location', 'Target'])


# In[ ]:


top_submissions = top_submissions[top_submissions.Date >= PRIVATE_START]
top_submissions.shape
top_submissions.head(3)


# # Calculate Weighted Pinball Loss
# 

# In[ ]:


def loss(preds, actual, weights, qs):
    l = 1 * (actual >= preds) * qs * (actual - preds) - 1 * (actual < preds) * (1 - qs) * (actual - preds)
    return l * weights


# In[ ]:


predictions = top_teams + ['blend']
for p in predictions:
    top_submissions[f'loss_{p}'] = loss(top_submissions[f'pred_{p}'],
                                        top_submissions.TargetValue,
                                        top_submissions.Weight,
                                        top_submissions.q)
top_submissions.head(3)


# # Check correlations

# In[ ]:


predictions = [c for c in top_submissions.columns if c.startswith('pred_')]
weighted_predictions = top_submissions[predictions].copy()
for c in weighted_predictions.columns:
    weighted_predictions[c] *= top_submissions.Weight.values
corr = weighted_predictions.corr()


# In[ ]:


fig = px.imshow(corr,
                labels=dict(x="Teams", y="Teams", color="Correlation"),
                x=[c.split('_')[1] for c in corr.columns],
                y=[c.split('_')[1] for c in corr.index],
                color_continuous_scale='Viridis'
               )
_ = fig.update_xaxes(side="top", title_text='Correlation among top teams')
fig.show()


# In[ ]:


wp = pd.concat([weighted_predictions, top_submissions[['q', 'Location', 'Date', 'Target']]], axis=1)
low = wp[wp.q == 0.05].drop(columns=['q'])
high = wp[wp.q == 0.95].drop(columns=['q'])

spread = pd.merge(low, high, on=['Location', 'Date', 'Target'], suffixes=['_l', '_h'])
for p in predictions:
    spread[f'spread_{p}'] = spread[[f'{p}_l', f'{p}_h']].max(axis=1) - spread[[f'{p}_l', f'{p}_h']].min(axis=1)
mean_weighted_spread = spread[[c for c in spread.columns if c.startswith('spread')]].mean().reset_index()
mean_weighted_spread.columns = ['p', 'mean_weighted_spread']
mean_weighted_spread.sort_values(by='mean_weighted_spread')


# # Private Leaderboard

# In[ ]:


losses = [c for c in top_submissions.columns if c.startswith('loss_')]
top_submissions[losses].mean().reset_index().sort_values(by=0)


# In[ ]:


daily_losses = top_submissions.groupby('Date')[losses].mean()
daily_losses.tail()


# In[ ]:


plot_data = daily_losses.reset_index().melt(id_vars='Date')
lb = top_submissions[losses].mean().reset_index()
lb.columns = ['variable', 'LB']
plot_data = plot_data.merge(lb, on='variable')
plot_data.variable = plot_data.variable.str.replace('loss_', '')
plot_data['name'] = plot_data.variable + '(' + plot_data.LB.round(4).astype(str) + ')'
fig = px.line(plot_data, x='Date', y='value', color='name')
_ = fig.update_layout(title_text='Daily Prediction Performance')
fig.show()


# # Unexpected Regions

# In[ ]:


loss_cols = [c for c in top_submissions.columns if c.startswith('loss_')]
locations = top_submissions.groupby('Location')[loss_cols].mean().reset_index()
locations['relative_loss'] = locations.loss_blend / locations.loss_blend.sum()
locations = locations.sort_values(by='loss_blend', ascending=False)

locations.head(10)
locations.head(10).relative_loss.sum()


# In[ ]:


top = locations.head(10)
fig = px.imshow(top[losses],
                labels=dict(x="Teams", y="Locations", color="Pinball Loss"),
                x=[c.split('_')[1] for c in losses],
                y=top.Location,
                color_continuous_scale='Reds'
               )
_ = fig.update_xaxes(side="top", title_text='Model Performance by Location')
fig.show()


# In[ ]:


top_submissions.loc[top_submissions.Location != 'Brazil--', losses].mean()


# In[ ]:


top_submissions.loc[top_submissions.Location.str.startswith('US'), losses].mean()


# ## Top regions with select button

# In[ ]:


def draw_predictions(location, target):
    selected_loc = top_submissions[
        (top_submissions.Location == location) & (top_submissions.Target == target)]
    team_colors = {t: c for t, c in zip(top_teams, px.colors.qualitative.Plotly)}
    fig = go.Figure()

    low = data=selected_loc[selected_loc.q == 0.05]
    median = data=selected_loc[selected_loc.q == 0.5]
    high = data=selected_loc[selected_loc.q == 0.95]

    for team in top_teams:
        _ = fig.add_trace(go.Scatter(
            x=low.Date, y=low[f'pred_{team}'], mode='lines', name=team,
            line=dict(color=team_colors[team], dash='dash')
        ))
        _ = fig.add_trace(go.Scatter(
            x=median.Date, y=median[f'pred_{team}'], mode='lines', name=team,
            line=dict(color=team_colors[team], width=3)
        ))
        _ = fig.add_trace(go.Scatter(
            x=high.Date, y=high[f'pred_{team}'], mode='lines', name=team,
            line=dict(color=team_colors[team], dash='dash')
        ))
    _ = fig.add_trace(go.Scatter(
        x=median.Date, y=median['TargetValue'], mode='lines',
        name='Actual', line=dict(color='black', width=3)))
    _ = fig.update_layout(
        title=f'{location} - {target}', yaxis_title=target, height=700
    )
    return fig


# In[ ]:


location_names = list(locations.Location.values[:50])
default_location = location_names[0]
targets = ['ConfirmedCases', 'Fatalities']
fig = go.Figure()
fig = subplots.make_subplots(
    rows=2, cols=1,
    subplot_titles=["Daily Cases", "Daily Fatalities"], vertical_spacing=0.05)
region_plot_names = []
team_colors = {t: c for t, c in zip(top_teams, px.colors.qualitative.Plotly)}
for location in location_names:
    for row, target in enumerate(targets):
        selected_loc = top_submissions[
            (top_submissions.Location == location) & (top_submissions.Target == target)]

        low = data=selected_loc[selected_loc.q == 0.05]
        median = data=selected_loc[selected_loc.q == 0.5]
        high = data=selected_loc[selected_loc.q == 0.95]

        for team in top_teams:
            _ = fig.add_trace(go.Scatter(
                x=low.Date, y=low[f'pred_{team}'], mode='lines', name=team,
                line=dict(color=team_colors[team], dash='dash'),
                visible=(location==default_location),
                showlegend=(row==0)
            ), row=row + 1, col=1)
            _ = fig.add_trace(go.Scatter(
                x=median.Date, y=median[f'pred_{team}'], mode='lines', name=team,
                line=dict(color=team_colors[team], width=3),
                visible=(location==default_location),
                showlegend=(row==0)
            ), row=row + 1, col=1)
            _ = fig.add_trace(go.Scatter(
                x=high.Date, y=high[f'pred_{team}'], mode='lines', name=team,
                line=dict(color=team_colors[team], dash='dash'),
                visible=(location==default_location),
                showlegend=False
            ), row=row + 1, col=1)
        _ = fig.add_trace(go.Scatter(
            x=median.Date, y=median['TargetValue'], mode='lines',
            name='Actual', line=dict(color='black', width=3),
            visible=(location==default_location),
            showlegend=(row==0)
        ), row=row + 1, col=1)
        number_of_lines = len(top_teams) * 3 + 1
        region_plot_names.extend([location] * number_of_lines)

buttons = []
for location in location_names:
    buttons.append(dict(
        method='update',
        label=location,
        args = [{
            'visible': [location==r for r in region_plot_names],
            'title': location
        }]
    ))

_ = fig.update_layout(
    updatemenus=[{'buttons': buttons,
                  'direction': 'down',
                  'active': location_names.index(default_location),
                  'showactive': True, 'x': 0.58, 'y': 1.1
                }],
    yaxis_title=targets[0],
    yaxis2_title=targets[1],
    height=1000
)
fig.show()


# 
# ## A couple of more charts...

# In[ ]:


for location in locations.Location.values[:20]:
    for target in ['ConfirmedCases', 'Fatalities']:
        fig = draw_predictions(location, target)
        fig.show()


# In[ ]:


for location in ['Hungary--']:
    for target in ['ConfirmedCases', 'Fatalities']:
        fig = draw_predictions(location, target)
        fig.show()


# In[ ]:




