#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# **DATA AND INFO DATA**

# In[2]:


calendar = pd.read_csv("../input/calendar.csv")
calendar.head()


# In[3]:


calendar.info()


# In[4]:


listing = pd.read_csv("../input/listings.csv")
listing.head(1)


# In[5]:


reviews = pd.read_csv("../input/reviews.csv")
reviews.head()


# * **DESCRIPTIVE ANALYTIS & DIAGNOSTIC ANALYTICS**
# 

# 1. ***Selection***

# > Loc numberic columns for Selection

# In[6]:


df_listing = listing[listing.applymap(np.isreal)]
df_listing.dropna(how = "all", axis = 1, inplace = True)
df_listing.head()


# 2. ****Preprocessing**** 

# Fill NAN and relace them by mean

# In[7]:


df_listing.fillna(listing.mean(), inplace = True)
df_listing.drop(["latitude", "longitude"], axis = 1, inplace= True)
df_listing.head()


# In[8]:


calendar.head()


# 3. **Transformation**

# In[9]:


calendar["price"] = calendar["price"].apply(lambda x: str(x).replace("$", ""))
calendar["price"] = pd.to_numeric(calendar["price"] , errors="coerce")
df1  = calendar.groupby("date")[["price"]].sum()
df1["mean"]  = calendar.groupby("date")[["price"]].mean()
df1.columns = ["Total", "Average"]
df1.head()


# In[10]:


df2 = calendar.set_index("date")
df2.index = pd.to_datetime(df2.index)
df2 =  df2[["price"]].resample("M").mean()
df2.head()


# 4. **Visualization**

# In[11]:


import plotly as py
from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.offline as offline


# In[12]:


trace1 = go.Scatter(
    x = df1.index,
    y = df1["Total"]
)
data = [trace1]
layout = go.Layout(
    title = "Price by each time",
    xaxis  = dict(title = "Time"),
    yaxis = dict(title = "Total ($)")
)
trace2 = go.Scatter(
    x = df1.index,
    y = df1["Average"]
)

data2 = [trace2]
layout2 = go.Layout(
    title = "Price by each time",
    xaxis  = dict(title = "Time"),
    yaxis = dict(title = "Mean ($)")
)
fig = go.Figure(data = data, layout = layout)
fig2 = go.Figure(data = data2, layout = layout2)
offline.iplot(fig)


# In[13]:


offline.iplot(fig2)


# In[14]:


trace3 = go.Scatter(
    x = df2.index[:-1],
    y = df2.price[:-1]
)
layout3 = go.Layout(
    title = "Average price by month",
    xaxis = dict(title = "time"),
    yaxis = dict(title = "Price")
)
data3 = [trace3]
fig3 = go.Figure(data= data3, layout= layout3)
offline.iplot(fig3)


# 5.** Time series stationarity and using statistic test**

# In[15]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[16]:


def draw_interactive_graph(mode):
    df1.index = pd.to_datetime(df1.index)
    decomposition = seasonal_decompose(df1[[mode]])
    trace4_1 = go.Scatter(
        x = decomposition.observed.index, 
        y = decomposition.observed[mode],
        name = "Observed"
    )
    trace4_2 = go.Scatter(
        x = decomposition.trend.index,
        y = decomposition.trend[mode],
        name = "Trend"
    )
    trace4_3 = go.Scatter(
        x = decomposition.seasonal.index,
        y = decomposition.seasonal[mode],
        name = "Seasonal"
    )
    trace4_4 = go.Scatter(
        x = decomposition.resid.index,
        y = decomposition.resid[mode],
        name = "Resid"
    )

    fig = py.tools.make_subplots(rows=4, cols=1, subplot_titles=('Observed', 'Trend',
                                                              'Seasonal', 'Residiual'))
    # append trace into fig
    fig.append_trace(trace4_1, 1, 1)
    fig.append_trace(trace4_2, 2, 1)
    fig.append_trace(trace4_3, 3, 1)
    fig.append_trace(trace4_4, 4, 1)

    fig['layout'].update( title='Descompose with TimeSeri')
    offline.iplot(fig)


# In[17]:


draw_interactive_graph("Average")


# In[18]:


draw_interactive_graph("Total")


# In[ ]:





# 
# ****

# **PREDICTIVE ANALYTICS**

# **PRESCRIPTIVE ANALYTIS**

# In[ ]:





# **QUESTION**

# 1. Can you describe the vibe of each Seattle neighborhood using listing descriptions?

# In[19]:


def loc_city(x):
    if "," not in str(x):
        return x
    if "live" in str(x) or "Next door to" in str(x) or "live" in str(x) or "having" in str(x):
        return "USA"
    return str(x).split(",")[0]
a = listing["host_location"].apply(lambda x: loc_city(x))


# In[20]:


df_listing["City"]  = a
df_listing.head(1)


# In[21]:


df_seattle = df_listing[df_listing["City"] == "Seattle"]
df_seattle.head()


# In[ ]:





# 2. What are the busiest times of the year to visit Seattle? By how much do prices spike?

# In[22]:


calendar_clean = calendar.dropna()
calendar_clean.set_index("date", inplace = True)
calendar_clean.head()


# In[23]:


calendar_clean.index = pd.to_datetime(calendar_clean.index)
number_hire_room = calendar_clean.resample("M")[["price"]].count()
total_price_each_month  = calendar_clean.resample("M")[["price"]].sum()


# In[24]:


trace5 = go.Scatter(
    x = number_hire_room.index[:-1],
    y = number_hire_room.price[:-1]
)
data5 = [trace5]
layout5 = go.Layout(
    title = "Number of Hire Room by Month in Seattle",
    xaxis = dict(title = "Month"),
    yaxis = dict(title = "Number hirde")
)
fig5  = go.Figure(data = data5, layout = layout5)


# In[25]:


trace6 = go.Scatter(
    x = number_hire_room.index[:-1],
    y = number_hire_room.price[:-1]/number_hire_room.price[0]
)
data6 = [trace6]
layout6 = go.Layout(
    title = "the ratio of the number of rooms compare with the first month",
    xaxis = dict(title = "Month"),
    yaxis = dict(title = "Ratio")
)
fig6 = go.Figure(data = data6, layout = layout6)


# In[26]:


offline.iplot(fig5)


# In[27]:


offline.iplot(fig6)


# As we can see,
# 
# In figure 5, there are graph show number of visting by each month.
# 
# There are high in March, May, October,November and December.
# 
# And lower in April, June, July, August, Septemper.
# 
# In figure 6, there are graph show ration of hiring room by each month compare with January.
# 
# 

# **Using statistic test to test number visitor**

# In[28]:


from scipy import stats


# In[67]:


a = calendar_clean.index.month
# calendar_clean["Month"] = a
calendar_clean = calendar_clean.assign(Month = a)
calendar_clean.head()


# In[30]:


result = []
for i in range(1,13):
    result.append(np.array([calendar_clean[calendar_clean["Month"] == i].price]))


# In[59]:


data_score = []
for i in range(11):
    score = stats.ttest_rel(result[i][0][:64911],result[-1][0][:64911])
    data_score.append((score[0], score[1]))


# In[66]:


score_board = pd.DataFrame(data = data_score, columns = ["Test Statistic", "P_value"])
score_board["Month"] = range(1, 12)
score_board.set_index("Month", inplace = True)
score_board


# Using  Paired sample t-test compares means from the same group at different times.
# In this case, i compare January to November with December.
# * H0 : Mean of December > Mean of other month.
# * H1 : Mean of December < Mean of other month
# 
# We can see that, every p_value is smaller 0.05 too.
# So H0 is true and we can conclude : Number of visitor in December is really high compare with other month.

# In[50]:


offline.iplot(fig3)


# 3. Is there a general upward trend of both new Airbnb listings and total Airbnb visitors to Seattle?

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




