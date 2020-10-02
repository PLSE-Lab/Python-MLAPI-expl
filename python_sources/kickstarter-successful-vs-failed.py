#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The aim of this notebook is to find out, if there are any features, which influence the Kickstarter project to succeed or fail. That is why we are going to focus only data which has state 'successful' or 'failed'.<br>
# The notebook is divided into 2 parts. In Part 1 we are going to visualize some patters and perform a data analysis. In Part 2, we will create a simple model to predict, if a project is going to be successful or not based on chosen features.

# # Content

# [Part 1 - Data Analysis](#part1)<br>
#     1.1. [Process data](#process-data)<br>
#     1.2. [Main categories' impact](#main-categories)<br>
#     1.3. [Amount pledged](#amount-pledged)<br>
#     1.4. [Campaign length](#campaign-length)<br>
#     1.5. [Countries' impact](#countries)<br>
#     1.6. [Overview](#part1-overview)<br>
# [Part 2 - Modelling](#part2)<br>
#     2.1. [Prepare data](#prepare-data)<br>
#     2.2. [Logistic Regression model](#lr-model)<br>
#     2.3. [What features contribute to successfulness?](#features-positive)<br>
#     2.4. [What feature has the largest impact on failing?](#features-negative)<br>
#     2.5. [Overview](#part2-overview)<br>
# [Conclusion](#conclusion)

# <a id='part1'></a>
# # Part 1

# In[1]:


# Libraries
import numpy as np
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# <a id='process-data'></a>
# ### 1.1. Process data

# In[2]:


projects = pd.read_csv('../input/ks-projects-201801.csv')
# Slicing just successful and failed projects
projects = projects[(projects['state'] == 'failed') | (projects['state'] == 'successful')]


# In[3]:


projects.head()


# Having look at the data, we will begin with 'main category' and its influence on successfulness of the projects.

# In[4]:


main_colors = dict({'failed': 'rgb(200,50,50)', 'successful': 'rgb(50,200,50)'})


# <a id='main-categories'></a>
# ### 1.2. Main categories' impact

# In[5]:


data = []
annotations = []

rate_success_cat = projects[projects['state'] == 'successful'].groupby(['main_category']).count()['ID']                / projects.groupby(['main_category']).count()['ID'] * 100
rate_failed_cat = projects[projects['state'] == 'failed'].groupby(['main_category']).count()['ID']                / projects.groupby(['main_category']).count()['ID'] * 100
    
rate_success_cat = rate_success_cat.sort_values(ascending=False)
rate_failed_cat = rate_failed_cat.sort_values(ascending=True)

bar_success = go.Bar(
        x=rate_success_cat.index,
        y=rate_success_cat,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

bar_failed = go.Bar(
        x=rate_failed_cat.index,
        y=rate_failed_cat,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='stack',
    title='% of successful and failed projects by main category',
    autosize=False,
    width=800,
    height=400,
    annotations=annotations
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='main_cat')


# It looks that some categories are more likely to be successful than others. It might be a matter of people's interest or other factors. For example, the goal amount in some categories on average may be lower than in others and projects of this category are more likely to succeed. That is why the 'goal amount' by category is the next thing we should look at.

# In[6]:


data = []

goal_success = projects[projects['state'] == 'successful'].groupby(['main_category'])                    .median()['usd_goal_real'].reindex(rate_success_cat.index)
goal_failed = projects[projects['state'] == 'failed'].groupby(['main_category'])                    .median()['usd_goal_real'].reindex(rate_success_cat.index)

bar_success = go.Bar(
        x=goal_success.index,
        y=goal_success,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

bar_failed = go.Bar(
        x=goal_failed.index,
        y=goal_failed,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='group',
    title='Median goal of successful and failed projects by main category (in USD)',
    autosize=False,
    width=800,
    height=400
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='median_goal_main_cat')


# We are not looking at 'mean' since single outliers may have an impact we do not want to capture.<br>
# First thing we notice is that the projects with higher goal amount are more likely to fail.<br>
# Another thing is that the most successful categories have relatively low goal amount. However, this dependency is not that strong and it is difficult to rely on this pattern too much.

# We may also assume that some categories are less successful since they have too many 'outliers' with too high goal amounts. That is why we will look at the difference of median goal amounts of failed and successful projects.<br>
# Since $1000 may be a significant sum for one category(eg. Craft) and less significant for another(eg. Technology), we will calculate relative differences.

# In[7]:


goal_dif = (goal_failed - goal_success)/goal_failed

bar_goal = go.Bar(
        x=goal_dif.index,
        y=goal_dif,
        name='failed',
        marker=dict(
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_goal]
layout = go.Layout(
    barmode='group',
    title='Relative difference of median goal of failed and successful projects (in USD)',
    autosize=False,
    width=800,
    height=400
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='dif_goal_cat')


# From this chart, I would reject our assumption, since there is no an obvious tendency in relative differences.

# <a id='ammount-pledged'></a>
# ### 1.3. Amount pledged

# It might also be interesting to have a look at how much failing projects are pledged before failure.

# In[8]:


pleged_failed = projects[projects['state'] == 'failed']['usd_pledged_real']                        /projects[projects['state'] == 'failed']['usd_goal_real']*100
data = [go.Histogram(x=pleged_failed, marker=dict(color=main_colors['failed']))]

layout = go.Layout(
    title='% pledged of the goal amount for failed projects',
    autosize=False,
    width=800,
    height=400
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='pleged_failed')


# It seems that the majority of failing projects do not get even several percent of the goal amount before deadline.

# <a id='campaign-length'></a>
# ### 1.4. Campaign length

# Maybe failed projects just do not have enough time to collect their goal amounts.

# In[10]:


# Calculating the length of campaign
projects['length_days'] = (pd.to_datetime(projects['deadline']) - pd.to_datetime(projects['launched'])).dt.days + 1


# In[11]:


data = [go.Histogram(x=projects[projects['state'] == 'failed']['length_days'], 
                     marker=dict(color=main_colors['failed']),
                     name='failed'),
        go.Histogram(x=projects[projects['state'] == 'successful']['length_days'], 
                     marker=dict(color=main_colors['successful']),
                     name='successful')]

layout = go.Layout(
    barmode='stack',
    title='Campaign length distribtuion',
    autosize=False,
    width=800,
    height=400
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='length_distribution')


# In[12]:


print('Mean days for failed projects: {0}'
      .format(round(projects[projects['state'] == 'failed']['length_days'].mean(), 2)))
print('Mean days for successful projects: {0}'
      .format(round(projects[projects['state'] == 'successful']['length_days'].mean(), 2)))


# Based on the plot, I would not say that the length of campaign is one of the reasons to fail.<br>
# Even, if we compare the mean campaign lengths, we can notice that the failed projects' campaings are on average a bit longer. 

# <a id='countries'></a>
# ### 1.5. Countries' impact

# Let us have a look at how projects from different countries perform.

# In[13]:


# Replacing unknown value to nan
projects['country'] = projects['country'].replace('N,0"', np.nan)

data = []
total_expected_values = []
annotations = []
shapes = []

rate_success_country = projects[projects['state'] == 'successful'].groupby(['country']).count()['ID']                / projects.groupby(['country']).count()['ID'] * 100
rate_failed_country = projects[projects['state'] == 'failed'].groupby(['country']).count()['ID']                / projects.groupby(['country']).count()['ID'] * 100
    
rate_success_country = rate_success_country.sort_values(ascending=False)
rate_failed_country = rate_failed_country.sort_values(ascending=True)

bar_success = go.Bar(
        x=rate_success_country.index,
        y=rate_success_country,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

bar_failed = go.Bar(
        x=rate_failed_country.index,
        y=rate_failed_country,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

for country in rate_success_country.index:
    weights = projects[projects['country'] == country]['main_category'].value_counts().sort_index()
    expected_values = weights * (rate_success_cat.sort_index()/100)
    total_expected_value = round(expected_values.sum() / weights.sum() * 100, 2)
    total_expected_values.append(total_expected_value)
    
for i, cat in enumerate(rate_success_country.index):
    shape = dict({
            'type': 'line',
            'x0': i-0.5,
            'y0': total_expected_values[i],
            'x1': i+1-0.5,
            'y1': total_expected_values[i],
            'line': {
                'color': 'rgb(255, 255, 255)',
                'width': 2,
            }})
    annot = dict(
            x=i,
            y=total_expected_values[i]+5,
            xref='x',
            yref='y',
            text='{0}'.format(int(total_expected_values[i])),
            font=dict({'color': 'rgb(255,255,255)'}),
            showarrow=False
        )
    annotations.append(annot)
    shapes.append(shape)

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='stack',
    title='% of successful and failed projects by country',
    autosize=False,
    width=800,
    height=400,
    annotations=annotations,
    shapes=shapes
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='main_cat')


# From the first sight, it is possible to conclude that projects from some countries are more likely to succeed, however, as we have seen, different categories have different probabilities to succeed. Similarly, some countries may have less projects from categories that are more likely to succeed. That is why we have calculated expected amount of successful projects, taking into consideration the weights of each category, based on the probabilities from the chart '% of successful and failed projects by main category'. This number is denoted in the chart above in white color. It helps us to see, that category is probably not the reason for projects from some countries to succeed more frequently.

# Let us check, if a higher goal amount may be the reason. <br>Note! In the chart below, there are means of median goals amounts in the main categories. However, we do not include categories, where there are less than 10 projects. That is why, for example, Luxemburg and Japan have no mean (zero).

# In[14]:


country_medians = []
for country in rate_success_country.index:
    medians = projects[(projects['country'] == country)]                .groupby(['main_category']).median()['usd_goal_real'].sort_index()
    values_count = projects[(projects['country'] == country)]['main_category']                .value_counts().sort_index()
    median = medians[values_count > 10].mean()
    country_medians.append(median)

bar_median = go.Bar(
        x=rate_success_country.index,
        y=country_medians,
        name='failed',
        marker=dict(
            line=dict(
                width=1,
            )
        ),
    )

data = [bar_median] 
layout = go.Layout(
    barmode='group',
    title='Average median goal amount (in USD)',
    autosize=False,
    width=800,
    height=400,
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='median')


# It is difficult to conclude with absolute certainty, that the goal amount is the reason. But it seems, that countries with lower median mean on average are more likely to have more successful projects.

# <a id='part1-overview'></a>
# ### 1.6. Overview

# * some categories are more likely to be successful
# * projects with higher goal amount are more likely to fail
# * 83% of failing projects are pledged less than 20% of the goal amount
# * country of a project may be one of the reasons of its successfulness

# <a id='part2'></a>
# # Part 2

# In his part, we will try to make a simple model to predict, if a project is going to be successful based on its main category, country and goal.

# <a id='prepare-data'></a>
# ### 2.1. Prepare data

# In[15]:


# Selecting features
projects_data = projects[['state', 'main_category', 'country', 'usd_goal_real']]

# Modifing value of dependent variable from categorical to numerical
projects_data.loc[projects_data['state'] == 'failed', 'state'] = 0
projects_data.loc[projects_data['state'] == 'successful', 'state'] = 1

# Scaling goal amount since it behaves differently in each category
for cat in projects_data['main_category'].unique():
    scaler = StandardScaler()
    new_values = scaler.fit_transform(projects_data[projects_data['main_category'] == cat][['usd_goal_real']])
    projects_data.loc[projects['main_category'] == cat, 'usd_goal_real'] = new_values.transpose()[0]

# Modifing independent variables to dummies
projects_data = pd.get_dummies(projects_data)


# <a id='lr-model'></a>
# ### 2.2. Logistic Regression model

# We will use a binary logistic regression for prediction.

# In[16]:


# Spliting data
train_X, test_X, train_y, test_y = train_test_split(projects_data.drop('state', axis=1), projects_data['state'], 
                                                    test_size=0.1, random_state=7)

# Creating model
LR = LogisticRegression()

# Fitting model
LR.fit(train_X, train_y)

# Scoring
print("Model's accuracy is {0}%".format(round(LR.score(test_X, test_y)*100, 2)))


# Our model's accuracy is around 63%, which is better than just a random guess, however, still not very accurate. You may try to include other features such as 'category' or 'currency' and try to use another ML algorithm such as SVC or Decision Tree, but it is not the purpose of this notebook.

# <a id='features-positive'></a>
# ### 2.3. What features contribute to successfulness?

# Let us have a look at features, which had a positive coefficient, so contributed to a project's successfulness.

# In[17]:


from_largest = np.argsort(LR.coef_)[0][::-1]
positive_coef_inds = []
for index in from_largest:
    if LR.coef_[0][index] > 0:
        positive_coef_inds.append(index)
    else:
        break
print(train_X.iloc[:, positive_coef_inds].columns)


# As we see, being in category 'Theater', 'Comics', 'Dance' and 'Music' or being from Hong Kong had a positive influence of being a successful project.

# <a id='features-negative'></a>
# ### 2.4. What feature has the largest impact on failing? 

# Now, let us have a look at a feature, which had the lowest coefficient, so contributed the most for a project to fail.

# In[18]:


print(train_X.iloc[:, np.argmin(LR.coef_[0])].name)


# As we see, the higher the goal, the more likely the project is going to fail.

# <a id='part2-overview'></a>
# ### 2.5. Overview

# Running a simple model we found out that some categories contribute more for a project's success than others. Similarly, the goal amount is a parameter, which contributes the most for a project's failure.

# <a id='conclusion'></a>
# # Conclusion

# In this notebook we tried to understand, which projects are more likely to be successful and which are more likely to fail. We have done a data analysis and created a model. Our conclusions in data analysis correspond to the model we have created. Particularly, the impact of the main category and goal amount on successfulness of a project.
