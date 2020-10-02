#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')
df.head()


# In[ ]:


df.shape


# # 1. Basic Data Cleaning

# In[ ]:


#splitting date into its components. It might help us later in the analysis. 
df['month'] = pd.to_datetime(df['date']).dt.month
df['year'] = pd.to_datetime(df['date']).dt.year
df.head()


# ## 1.1. Investigating Missing Values
# 
# Here are the columns that contain missing values: age (4.3%), race (9.6%), flee (4.6%), gender (0.03%).
# 
# These aren't a lot of missing values. Let's see if we can tackle them individually. 
# 
# Armed: In my opinion, I think we can assign the value 'unarmed' to all these values. My reasoning behind this is simple: In cases where the victim was armed, the police has an overwhelming desire to get this fact out there. Since the suspect being armed diminishes the blame for the shooting officers. Hence, if this information is unknown to us, we can assume that police weren't able to find any evidence of the suspect being armed.
# 
# Race: This is a tricky one. Since we want to investigate racism and prejudice, I don't think we should impute the missing values here. So we will drop all values that don't contain information about the victim's race.
# 
# Age: Like race, I will drop the missing values here. Very hard to impute them.
# 
# Flee: I will apply the 'armed' logic here as well. I believe if the victim was fleeing, the police would've logged that into the report and it would've made its way into this dataset.
# 
# Gender: The missing values for this only constitute 0.03% of the values. We can look at these observations individual and impute the gender by the victim's name. If it isn't possible, we will drop the value.

# In[ ]:


#lets identify the columns that have a lot of missing values
print(df.isna().sum()*100/5416)


# In[ ]:


#race
df_clean = df.dropna(subset = ['race', 'age'])

df_clean['armed'].fillna('unarmed', inplace=True)
df_clean['flee'].fillna('Not fleeing', inplace=True)

print(df_clean[df_clean['gender'].isna() == True]) #missing value's name is 'Scout Schultz'. He was an intersex, non-binary student.
#We don't have the right to decide his gender for him. So we will drop his name from the data.

df_clean = df_clean.dropna(subset = ['gender'])
df_clean.shape


# # 2. EDA
# 
# ## 2.1. Racial Distribution of Police Shootings
# 
# 
# There are six different racial classifications in the data: 'W' = White, 'A' = Asian, 'B' = Black, 'H' = Hispanic', 'N' = Native American and 'O' = Others.
# 
# First, we plot all shootings by race. The histogram represents the total count for each race.
# 
# Then, using the 2015 census data regarding actual racial distribution in the US, we determine how many shootings will take place for each race if members of that race were shot according to ther proportion in the population. This number is represented by the scatter line.
# 
# For the first graph, we can see that Black people are much more likely to be shot by the police while White and Asian people are less likely to be shot by the police. The difference is astounding.
# 
# 
# Then, we specifically look for the distribution of police shootings in cases where the victim was 'unarmed'. We discover a similar pattern here but the differences are even more pronounced. Black people are much more likely to be shot by the police in cases where they are unarmed.

# In[ ]:


print(df['race'].unique())


# In[ ]:


shootings_race = df_clean[df_clean['manner_of_death'] == 'shot']
shootings_race.shape #4564 shootings


# In[ ]:


fig = px.histogram(shootings_race, x = 'race',
                  title = "Distribution of Police Shootings by Race")

#let's add a line to represent the number of shootings for each race if they were proportional to their population in the US
#white: 0.618, b: 0.132, h: 0.178, a: 0.053, n: 0.01, o: 0.009 These are rough estimates based on 2015 values.
races = ['A','W','H','B','N','O']
race_prop = np.array([0.053, 0.618, 0.178, 0.132, 0.01, 0.009])
fig.add_trace(go.Scatter(x = races, y = shootings_race.shape[0] * race_prop))

fig.show()


# Let's look at unarmed shootings.

# In[ ]:


unarmed = df_clean[df_clean['armed'] == 'unarmed']
fig2 = px.histogram(unarmed, x = 'race',
                   title = 'Distribution of Police Shootings by Race where the Suspect was Unarmed') #this is messing the order on the x axis compared to the last graph, idk why
races2 = ['H','W','B','N','O','A'] 
race_prop2 = np.array([0.178, 0.618, 0.132, 0.01, 0.009, 0.053])
fig2.add_trace(go.Scatter(x = races2, y = unarmed.shape[0] * race_prop2))
fig2.show()


# ## 2.2. Distribution by Age
# 
# Let's look at the distribution of shootings by age.
# 
# First, we plot the distribution for the entire dataset. From this we observe that the most victims are between 25 and 40 years old. The distribution has long right tail but the left tail is much thicker comparatively.
# 
# Secondly, we subset the distribution for only black victims and we plot the age distribution for this. From this we observe that the distribution is relatively similar. However, we look at the lower end of the distribution (for victims under 20), we can see that young black people are slightly more likely to be shot by the police.

# In[ ]:


import plotly.figure_factory as ff
np.random.seed(1)
age_data = df_clean['age']
age_labels = ['Age']
fig = ff.create_distplot([age_data], age_labels)
fig.update_layout(title_text='Distribution of Police Shooting Victims by Age')

fig.show()


# In[ ]:


df_black = df_clean[df_clean['race'] == 'B']
black_age_data = df_black['age']
fig = ff.create_distplot([black_age_data], age_labels)
fig.update_layout(title_text='Distribution of Black Police Shooting Victims by Age')

fig.show()


# ## 2.3. Distribution by States
# 
# We plot the % of total shootings in the Top 20 most trigger-happy states on the left side of the graph. On the right side, we have the actual proportion of the total US population representated by the state. This gives us some interesting insights.
# 
# 1. States like California, New Mexico and Arizona are very trigger happy. Their proportion of shootings is much higher than their actual proportion of the population. It is interesting to note that all three of these states lie along the border with Mexico. Maybe there is a relationship there. We know Arizona's police/sherrif force is extremely conservative and trigger happy. In California, LAPD has a similar reputation. 
# 
# 2. New York surprisingly fares better. Although it has had some really high-profile shooting cases (i.e. Eric Garner), it generally fares better proportionally. It constitutes 6% of the US population by only over 2% of the shootings. This can be explained by the fact that NYPD places a greater importance on body-cams. Moreover, relatively speaking, it is a much liberal city. Same can be observed for Pennsylvania and Illinois.
# 
# 3. Oklahoma is very trigger happy. For a relatively small population proportion, it does have a lot of police shootings.

# In[ ]:


from plotly.subplots import make_subplots

trigger_states = df_clean['state'].value_counts()[:20]
trigger_states = pd.DataFrame(trigger_states)
trigger_states = trigger_states.reset_index()
trigger_states['percent'] = trigger_states['state']*100/df_clean.shape[0]
s = [11.91, 8.74, 6.47, 2.19, 1.74, 3.2, 1.19, 3.52, 3.16, 2.29, 1.85, 2.06, 1.40, 3.86, 3.82, 1.48, 0.63, 2.03, 5.86, 2.57]
trigger_states['state_prop'] = s

fig = make_subplots(rows = 1, cols = 2, specs = [[{},{}]], shared_xaxes = True,
                  shared_yaxes = False, vertical_spacing = 0.001)

fig.append_trace(go.Bar(
    x = trigger_states['percent'],
    y = trigger_states['index'],
    marker = dict(
        color = 'red',
        line = dict(
            color = 'red',
            width = 1),
    ),
    name = 'Distribution of Shootings by State',
    orientation = 'h',
), 1, 1)

fig.append_trace(go.Bar(
    x = trigger_states['state_prop'],
    y = trigger_states['index'],
    marker = dict(
        color = 'goldenrod',
        line = dict(
            color = 'goldenrod',
            width = 1),
    ),
    name = '% of total US populaiton in the State',
    orientation = 'h',
), 1, 2)


fig.update_layout(
    title = 'Twenty US States with the most Police Shootings',
    yaxis = dict(
        showgrid = False,
        showline = False,
        showticklabels = True,
        domain = [0, 0.85],
    ),
    yaxis2 = dict(
        showgrid = False,
        showline = True,
        showticklabels = True,
        linecolor = 'rgba(102, 102, 102, 0.8)',
        domain = [0, 0.85],
    ),
    xaxis = dict(
        zeroline = False,
        showline = False,
        showticklabels = True,
        showgrid = True,
        domain = [0, 0.42],
    ),
    xaxis2 = dict(
        zeroline = False,
        showline = False,
        showticklabels = True,
        showgrid = True,
        domain = [0.5, 0.92],
    ),
    legend = dict(x = 0.029, y = 1.038, font_size = 10),
    margin = dict(l = 100, r = 20, t = 70, b = 70),
    paper_bgcolor = 'rgb(248, 248, 255)',
    plot_bgcolor = 'rgb(248, 248, 255)',
)

fig.show()


# ## 2.4. Shootings Over Time
# 
# Let's look at the trend over time.
# 
# Generally, there is a slight drop in the total number of police shootings over time. However, the racial breakdown is much more interesting. If you look at police shootings of black people, it dropped between 2015 and 2016 but from 2015 to 2019, it steadily increased. We can't say anything yet about 2020 since the year isn't done yet. However, hopefully, given the popular protest movement, I hope we see a decrease. 

# In[ ]:


years = df_clean[['year','race']]
years['shootings'] = 1
years = years.groupby(['year','race']).sum()
years = years.reset_index()
fig = px.bar(years , y = 'shootings', x = 'year',color = 'race', barmode = 'group', title = 'Shootings from 2015-2020',
            color_discrete_sequence = px.colors.qualitative.D3)
fig.show()


# # 3. Conclusion
# 
# From the basic analysis, we can see that there is definitely a racist bent in law enforcement in the United States. Proportionally speaking, black people are much more likely to be shot, no matter how you slice it. 
# 
# However, there are some caveats here.
# 
# 1. We require much more data to effectively make a conclusion.
# 2. Black people are more likely to be suspected by the police. That obviously contributes to the higher shooting rate. However, we cannot quantify these biases effectively unless we have data regarding police searchers.
# 
# 
# NOTE: I will update the notebook. This was just a very basic rudimentary analysis.
