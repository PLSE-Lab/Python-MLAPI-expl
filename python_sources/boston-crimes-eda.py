#!/usr/bin/env python
# coding: utf-8

# <h3>Import Libraries & Data</h3>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')
import random


# Let's take a look at the first five items in our dataset.

# In[ ]:


df = pd.read_csv("../input/boston-ds/crime.csv", index_col = None, encoding='windows-1252', parse_dates = ['OCCURRED_ON_DATE', 'YEAR', 'DAY_OF_WEEK'], engine='python')
df.head()


# Let's find out how many entries there are in our dataset and what type the variables are.

# In[ ]:


print(f'The dataset contains %s rows and %s columns' % (df.shape[0],df.shape[1]), '\n')
print('The columns and the its values types:\n')
df.info()


# <h3>Task1 - Top Offense Code</h3>
# <br>Here I am investigating the most "popular" report registered. This will help us to better understand the structure and dataset in general. Additionally, this will be useful for our further discoveries.

# Let's load the offense code dataset which will help us with code decoding

# In[ ]:


codes = pd.read_csv("../input/boston-ds/offense_codes.csv", index_col = None, encoding='windows-1252', engine='python')
codes.CODE.value_counts() #This line is for checking of whether codes are unique or not
codes.drop_duplicates(subset=['CODE'], keep='first', inplace=True) #Since there are duplicates, let's drop them


# Let's take a look on this dataset briefly

# In[ ]:


codes.head()


# In[ ]:


print(f'The dataset contains %s rows and %s columns' % (codes.shape[0],codes.shape[1]), '\n')
print('The columns and the its values types:\n')
codes.info()


# Then, lets' sumup the codes from the main dataset and sort them descending

# In[ ]:


top = df.OFFENSE_CODE.value_counts().to_frame().reset_index(level=0)


# In[ ]:


top.columns.values[0] = 'CODE'
top.columns.values[1] = 'TOTAL_AMOUNT'
top.head(5)


# And now we are merging <i>top</i> dataset with <i>codes</i> on <i>'CODE'</i> column as a key

# In[ ]:


code_top = top.merge(codes, on='CODE', how = 'left')

Brief observation of that everything went smoothly:
# In[ ]:


code_top.head(10)


# <b>Visualization of Task1</b>. <br>
# Here, for the further exploratory purpuse, I added OFFENSE_CODE_GROUP column to our decoded top list.

# In[ ]:


gr = df.loc[:,['OFFENSE_CODE','OFFENSE_CODE_GROUP']]
gr.info()
code_tg = pd.merge(code_top, gr, left_on='CODE', right_on='OFFENSE_CODE', how='inner')
code_tg.drop_duplicates(subset=['CODE'], keep='first', inplace=True)
code_tg.reset_index(drop=True,inplace=True)
code_tg.head(5)


# The most interesting part of Task1 - visualization! Look how the plot is built in code.<br>
# Here only first 20 codes are pictured.

# In[ ]:


code_tg.head(20).plot(kind = 'barh', x = 'NAME', y = 'TOTAL_AMOUNT', figsize=(12, 12))

plt.gca().invert_yaxis() 

plt.xlabel('Number of Reports')
plt.ylabel('Offense Type')
df.sort_values(['OCCURRED_ON_DATE'], ascending=True, inplace=True)
plt.title('Boston Offense Rating: ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[0]) + ' : ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[-1]))

# This loop automatically add the value of each position to the each bar:
for index, value in enumerate(code_tg.head(20)['TOTAL_AMOUNT']):
    label = format(int(value), ',')
    plt.annotate(label, xy=(value - 100, index + 0.10), ha='right', color='white')


# In addition, I'd also like to know the top reports rating, grouped by OFFENSE_CODE_GROUP.<br>
# For this reason we need to make some additional manipulations:

# In[ ]:


top_gr = code_tg.groupby(['OFFENSE_CODE_GROUP'], as_index=False).sum(axis=1)
top_gr = top_gr[['OFFENSE_CODE_GROUP','TOTAL_AMOUNT']].sort_values(['TOTAL_AMOUNT'], ascending=False).reset_index(drop=True)


# In[ ]:


top_gr.head(10)


# Now let's plot it!

# In[ ]:


top_gr.head(10).plot(kind = 'barh', x = 'OFFENSE_CODE_GROUP', y = 'TOTAL_AMOUNT', figsize=(12, 5))

plt.gca().invert_yaxis() 

plt.xlabel('Number of Reports')
plt.ylabel('Offense Group')
plt.title('Boston Offense Groups Rating: ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[0]) + ' : ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[-1]))


# This loop automatically add the value of each position to the each bar:
for index, value in enumerate(top_gr.head(10)['TOTAL_AMOUNT']):
    label = format(int(value), ',')
    plt.annotate(label, xy=(value - 300, index + 0.13), ha='right', color='white')


# Here we see a slightly different picture rather than comparing offense names individually. The most-reported here are incidents related to Motor Vehicle Incidents.

# <h3>Task2 - Most Shooting Areas</h3>
# <br>This part is about the understanding of which boroughs are most dangerous in the meaning of shooting level. The library I used here is <b>folium</b> - it works with maps and coordinates, can combine the close locations into groups and many other things

# Firstly, let's extract the rows we are interested in. We need those with a 'Y' shooting mark and the filled district value.

# In[ ]:


shtng = df[(df.SHOOTING == 'Y') & (df.DISTRICT.notnull())]


# After that, we can plot the incidents using folium library:

# In[ ]:


import folium
import folium.plugins as plugins

latitude = list(shtng.Lat)[1] # This is to initiate the latitude start point for the map
longitude = list(shtng.Long)[1] # This is to initiate the longitude start point for the map

latitudes = list(shtng.Lat) #create the list of all reported latitudes
longitudes = list(shtng.Long) #create the list of all reported longitudes

shooting_map = folium.Map(location = [latitude, longitude], zoom_start = 12) # instantiate a folium.map object

shooting = plugins.MarkerCluster().add_to(shooting_map) # instantiate a mark cluster object for the incidents in the dataframe

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(shtng.Lat, shtng.Long, shtng.DISTRICT):
    if (not np.isnan(lat)) & (not np.isnan(lng)): # also, we check a non-nullness of the coordinates
        folium.Marker(
            location=[lat, lng],
#             icon=None,
            popup=label,
            icon=folium.Icon(icon='exclamation-sign')
        ).add_to(shooting)

# display the map
shooting_map


# After a short time, we can see the map with all reported incidents with the shooting. The most frequently mentioned areas are colored in dark orange. <br>
# According to the plot, the most  "dangerous" boroughs are situated in the south part of the city, mainly along the railroad.

# Now I want to range the districts to highlight those where I wouldn't recommend settling.

# In[ ]:


# re-assemble the dataset for the more convenient plotting process
top_sh = shtng.DISTRICT.value_counts().to_frame().reset_index(level=0)
top_sh.columns.values[0] = 'DISTRICT'
top_sh.columns.values[1] = 'NUMBER'
top_sh.plot(kind = 'barh', x = 'DISTRICT', y = 'NUMBER', figsize=(12, 7))

# invert y-axis
plt.gca().invert_yaxis()

# Name axis and title
plt.xlabel('Number of Shooting Reports')
plt.ylabel('Districts')

plt.title('Boston Top Shooting Districts: ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[0]) + ' : ' + str(df.OCCURRED_ON_DATE.dt.date.iloc[-1]))

# Lop for values plotting
for index, value in enumerate(top_sh['NUMBER']):
    label = format(int(value), ',')
    plt.annotate(label, xy=(value - 1, index + 0.11),
                 ha='right', 
                 color='white'
                )
    
# Loop for arrows plotting. Notice that the arrowhead will always point on the bottom-right bar's corner.
# Also, here I separately defined a starting arrows' point to maximize the procedural plotting 
xy_label = (250,5)
for index, value in enumerate(top_sh['NUMBER']):
    plt.annotate('',
             xy=(value, index + 0.3),
             xytext=xy_label,
             xycoords='data',
             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2',
                             connectionstyle='arc3', 
                             color='xkcd:blue', 
                             lw=2
                            )
            )
    if index == 2: # We want to plot only top 3 the most shooting districts, so we need to interrupt the loop here.
        break

# This dictionary I built using googling method.
dict0 = {'C11' : 'DORCHESTER', 'B3' : 'MATTAPAN', 'B2' : 'ROXBURY'} 

# Plot the district name decoding it using our dictionary.
for index, value in enumerate(top_sh['NUMBER']):
    v = top_sh.loc[top_sh['NUMBER']==value]['DISTRICT'].astype('str')
    plt.annotate('[ ' + dict0[v[index]] + ' ]',
             xy=(value - 15, index + 0.13),
             rotation=0,
             va='bottom',
             ha='right',
             color = 'white'
            )
    if index == 2:
        break
        
# Plot the annotation text. Here I used xy_label defined earlier for automation.
plt.annotate('The Most Shooting Districts', # text to display
             xy=(xy_label[0],xy_label[1] + 0.5),
             rotation=0,
             va='bottom',
             ha='center',
            )
        
plt.show()


# The figure above clearly shows that Roxbury, Mattapan and Dorchester are the most restless districts.

# To finalize with folium library, I'd like to show how else we can depict the incidents using a heat map.<br>
# In this case, all shooting incidents are colored depending on the intensity at any particular area. For the time-saving purpose, I limited the number of points to be pictured.

# In[ ]:


shooting_hmap = folium.Map(location=[df.Lat[100],df.Long[100]], 
                       tiles = "Stamen Toner",
                      zoom_start = 12)

from folium.plugins import HeatMap   

hm = df.loc[:,['Lat','Long','SHOOTING']]
hm.dropna(axis=0, inplace=True)
hlimit = hm.shape[0]
hm = hm.sample(hlimit)
hdata = []
for ln, lt in zip(hm.Lat, hm.Long):
    hdata.append((ln,lt))
HeatMap(hdata, 
        gradient = {0.01: 'blue', 0.15: 'lime', 0.25: 'red'},
        blur = 15,
        radius=5).add_to(shooting_hmap)

shooting_hmap


# <h3>Task3 - Most Dangerous Time in Boston</h3>
# <br>To continue the previous chapter, let's consider the question of determining the time of the day when It is better to stay at home. In particular, I want to know, when the chance to catch a bullet is peaked.

# Let's prepare the data first: we're interested only in the rows where 'Y' shooting mark is and the 'HOUR' values are filled.

# In[ ]:


shtngh = df[(df.SHOOTING == 'Y') & (df.HOUR.notnull())]
shtngh.sort_values(['OCCURRED_ON_DATE'], ascending=True, inplace=True)
shtngh.head()


# The dataset we need seems to be relatively small, so we will plot all the observations.<br>
# This time I use histogram plot, which shows how frequently each particular hour, when the incident with the shooting was reported, appears in the dataset.<br>
# I am going to use different kinds of arrows to point the main features.

# In[ ]:


# plot the histogram
plt.figure(figsize=(9, 5))
plt.hist(shtngh.HOUR, bins=range(24))
plt.title('Shooting Time Distribution in Boston: ' + str(shtngh.OCCURRED_ON_DATE.dt.date.iloc[0]) + ' : ' + str(shtngh.OCCURRED_ON_DATE.dt.date.iloc[-1]))
plt.xticks(range(24))

# Decrease Arrow
plt.annotate('',
             xy=(10, 25), # Arrow head
             xytext=(1, 120), # Starting point
             xycoords='data', # Use the coordinate system of the object being annotated 
             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2', connectionstyle='angle3, angleA=110,angleB=0', color='xkcd:blue', lw=2)
            ) # Arrow props

# After Midday Arrow
plt.annotate('',
             xy=(16, 85),
             xytext=(15, 100),
             xycoords='data',
             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2', connectionstyle='arc3', color='xkcd:blue', lw=2)
            )

# Latenight Madness Arrow
plt.annotate('',
             xy=(21.5, 190),
             xytext=(20, 65),
             xycoords='data',
             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2', connectionstyle='arc3', color='xkcd:blue', lw=2)
            )

# Annotate Text
plt.annotate('Gradual decrease ',
             xy=(2.5, 38),
             rotation=-40,
             va='bottom',
             ha='left',
            )

# Annotate Text
plt.annotate('After midday peak', # text to display
             xy=(15, 100),
             rotation=0,
             va='bottom',
             ha='right',
            )

# Annotate Text
plt.annotate('Latenight madness',
             xy=(19.5, 90),
             rotation=79,
             va='bottom',
             ha='left',    
            )

plt.show()


# What we can conclude from the histogram is:
# 1. Nobody shoots at 7 am (too sleepy, probably)
# 2. There is a midday shooting peak (too hot?)
# 3. And the late-night madness at midnight (everybody has fun)
# 4. Then the fun gradually goes down and fades in the early morning.

# Well, usually the shooting follows by the verbal disputes. I want to check, is the correlation between these two events or not.<br>
# First of all, I want to check it graphically, using the old good histogram.

# Let's bultd a dataset we want to examine:

# In[ ]:


vd = df[(df.OFFENSE_CODE_GROUP == 'Verbal Disputes') & (df.HOUR.notnull())]
vd.sort_values(['OCCURRED_ON_DATE'], ascending=True, inplace=True)
vd.shape


# It contains over 13k observations so that our data look statistically significant.

# This is how I plot it:

# In[ ]:


plt.figure(figsize=(9, 5))
plt.hist(vd.HOUR, bins=range(24))
plt.title('Verbal Disputes Rate in Boston:'  + str(vd.OCCURRED_ON_DATE.dt.date.iloc[0]) + ' : ' + str(vd.OCCURRED_ON_DATE.dt.date.iloc[-1]))
plt.xticks(range(24))
plt.annotate('',
             xy=(21.5, 1400),
             xytext=(5, 200),
             xycoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc, angleA=90, angleB=-95, armA=40, armB=60, rad=45.0', color='xkcd:blue', lw=2)
            )
plt.annotate('Verbal disputes gradually increases', # text to display
             xy=(3, 1200),
             rotation=0,
             va='bottom',
             ha='left',
            )
plt.annotate('all day long and reaches its peak', # text to display
             xy=(3, 1100),
             rotation=0,
             va='bottom',
             ha='left'
            )
plt.annotate('at midnight', # text to display
             xy=(3, 1000),
             rotation=0,
             va='bottom',
             ha='left',
            )
plt.show()


# My theory of that incidents with shooting are preceded by verbal disputes is confirmed graphically.<br>But I need more evidence.<br>
# Suddenly, I've decided to check whether we could build a prediction model that could predict the probability of the shooting using only OFFENSE_CODE_GROUP features.<br>Then we will see which OFFENSE_CODE_GROUP codes are the most related to incidents with shooting.

# Data preparation step:

# In[ ]:


pred = df.loc[:,['SHOOTING', 'OFFENSE_CODE_GROUP']]
pred.replace(np.nan, 0, inplace=True)
pred.replace('Y', 1, inplace=True)
groups_dummy = pd.get_dummies(pred['OFFENSE_CODE_GROUP'])
pred = pd.concat([pred,groups_dummy], axis=1)
pred.drop(['OFFENSE_CODE_GROUP'], inplace=True, axis=1)


# Model Development

# I chose the Logistic Regression model since we need to predict the probability of 0 or 1 and it usually suits perfectly for this purpose.

# In[ ]:


y = pred[['SHOOTING']]
X = pred.iloc[:,1:]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)
LR


# So, the model is trained. It's time to evaluate it.

# In[ ]:


yhat = LR.predict(x_test)
yhat


# In[ ]:


# Evaluation using Jaccard Index
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# Outstanding result! Seems like our model fits test data perfectly.

# Now let's see the probalility of the classes:

# In[ ]:


yhat_prob = LR.predict_proba(x_test)
yhat_prob


# Here, <b>predict_proba</b> returns evaluations for all classes, ordered by the label of the classes. <br>
# So, the first column is the probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X):

# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# This is how confusion matrix looks like. All wee need to do here is to measure Log Loss for this model. The lower value, the better model.

# In[ ]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# Great result, let's find the importance for the each feature.

# Hare are some data manipulations:

# In[ ]:


feature_importance=pd.concat([pd.DataFrame(X.columns), pd.DataFrame(LR.coef_.T)], axis = 1)
feature_importance.columns = ['features', 'importance']
feature_importance.sort_values(['importance'], ascending=False, inplace=True)
feature_importance = feature_importance.reset_index(drop=True)
feature_importance.head(8)


# What I see here is that there is no place for Verbal Disputes! However, even these outcomes do not exclude the relationship between shooting and verbal disputes reported. They just could follow one after another since they are both in different observations.

# <h3>Task4 - The tendency in the incedents with shooting</h3>
# <br>Ok, now let's switch to a slightly different question - I want to know how is everything going with the number of shooting incidents - is it increasing or decreasing over the recent years?

# For this purpose, we need to transform our data, parse the date column and leave only year-month value because I don't want to plot everyday-dots.

# In[ ]:


shtng_gr = shtng.loc[:,['OCCURRED_ON_DATE']]
shtng_gr['Amount'] = 1
shtng_gr['Date'] = pd.DatetimeIndex(shtng_gr.OCCURRED_ON_DATE).normalize()
shtng_gr.drop(['OCCURRED_ON_DATE'], axis=1, inplace=True)
shtng_gr['YM'] = pd.to_datetime(shtng_gr["Date"], format='%Y00%m').apply(lambda x: x.strftime('%Y-%m'))
shtng_gr['YM'] = pd.to_datetime(shtng_gr["YM"])
shtng_gr.drop(['Date'], axis=1, inplace=True)
shtng_gr = shtng_gr.groupby(['YM'], as_index=False).sum()
shtng_gr.reset_index(drop=False, inplace=True)
shtng_gr.head()


# Now, I want to plot just a regression line to see the trend briefly

# In[ ]:


from numpy.polynomial.polynomial import polyfit
px = np.asarray(shtng_gr.index)
b, m = polyfit(shtng_gr.index, shtng_gr.Amount, 1)
shtng_gr.plot(kind='scatter',x='index', y='Amount', rot='90', figsize=(10, 6), alpha = 1, c='xkcd:salmon')
plt.plot(px, b + m * px, '-', c='xkcd:blue')
plt.xticks(shtng_gr.index, shtng_gr.YM.dt.date, rotation=90)
plt.ylabel('Amount of Shooting Reports / Month')
plt.xlabel('Months')
plt.title('Total Shooting Reports in Boston: ' + str(shtng_gr.YM.dt.date.iloc[0]) + ' : ' + str(shtng_gr.YM.dt.date.iloc[-1]))

plt.annotate('Regression Line : Insignificant growth',                      # s: str. Will leave it blank for no text
             xy=(20, 27),             # place head of the arrow at point (year 2012 , pop 70)
             xytext=(12, 46),         # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',         # will use the coordinate system of the object being annotated 
             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2', connectionstyle='arc3', color='xkcd:blue', lw=2)
            )

plt.show()


# In[ ]:





# Great! Even though the number of shooting reports experiences slow growth, it still seems like a lateral trend.

# In comparison to the trend above, I would like to show realy disturbing tendency - the tendency in Medical Assistance reports.

# In[ ]:


ma = df[(df.OFFENSE_CODE_GROUP == 'Medical Assistance') & (df.OCCURRED_ON_DATE.notnull())]


# In[ ]:


ma_gr = ma.loc[:,['OCCURRED_ON_DATE']]
ma_gr['Amount'] = 1
ma_gr['Date'] = pd.DatetimeIndex(ma_gr.OCCURRED_ON_DATE).normalize()
ma_gr.drop(['OCCURRED_ON_DATE'], axis=1, inplace=True)
ma_gr['YM'] = pd.to_datetime(ma_gr["Date"], format='%Y00%m').apply(lambda x: x.strftime('%Y-%m'))
ma_gr['YM'] = pd.to_datetime(ma_gr["YM"])
ma_gr.drop(['Date'], axis=1, inplace=True)
ma_gr = ma_gr.groupby(['YM'], as_index=False).sum()
ma_gr.reset_index(drop=False, inplace=True)
ma_gr.head()


# To diverse the project, I use a seaborn library here to plot a regression line. From my perspective, it makes it more spectacular and informative.

# In[ ]:


import seaborn as sns
plt.figure(figsize=(15, 10))
ax = sns.regplot(x='index', y='Amount', data=ma_gr, color='green', marker='+', scatter_kws={'s': 50,'color':'xkcd:salmon', 'alpha' : 1})

ax.set(xlabel='Months', ylabel='Amount of Medical Assistance Reports / Month')
ax.set_title('Total Medical Assistance Reports in Boston: ' + str(ma_gr.YM.dt.date.iloc[0]) + ' : ' + str(ma_gr.YM.dt.date.iloc[-1]))
ax.set_ylim(200)
plt.xticks(ma_gr.index, ma_gr.YM.dt.date, rotation=90)

plt.annotate('Steady Growth',
             xy=(25, 600),
             xytext=(12, 350),
             xycoords='data',
             arrowprops=dict(arrowstyle='fancy ,head_length=0.4,head_width=0.4,tail_width=0.2', connectionstyle='arc3', color='xkcd:salmon', lw=2)
            )


plt.show()


# What I see here is a clear uptrend. There is more than a 50% increase in 'the Medical Assistance reports' in just 3 years.
# <br>However, to investigate the reasons for this we need other datasets.
