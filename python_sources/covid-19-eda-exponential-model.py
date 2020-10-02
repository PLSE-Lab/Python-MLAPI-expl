#!/usr/bin/env python
# coding: utf-8

# ### CONTENTS
# 
# 1. [Importing those libraries](#Imp) 
# 2. [Preprocessing](#prp)
# 3. [Univariate Analysis](#uv)
# 4. [Multivariate Analysis](#mu)
# 5. [Exponential Graph](#ex)

# In[ ]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import plotly.offline as py
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import statsmodels.api as sm
from plotly.offline import init_notebook_mode, iplot


# In[ ]:


### Reading the data

case_data = pd.read_csv('../input/coronavirusdataset/case.csv')
patient_data = pd.read_csv('../input/coronavirusdataset/patient.csv')
route_data = pd.read_csv('../input/coronavirusdataset/route.csv')
trend_data = pd.read_csv('../input/coronavirusdataset/trend.csv')
time_data = pd.read_csv('../input/coronavirusdataset/time.csv')


# ### There are several datasets, so lets go one by one

# <a id="prp"></a>
# 
# ### Preprocessing 

# In[ ]:


print(f"Dataset is of {patient_data.shape[0]} rows and {patient_data.shape[1]} columns")


# In[ ]:


patient_data.head(10)


# In[ ]:


patient_data['birth_year'] = patient_data.birth_year.fillna(0.0).astype(int)

patient_data['birth_year'] = patient_data['birth_year'].map(lambda val: val if val > 0 else np.nan)


# In[ ]:


patient_data.confirmed_date = pd.to_datetime(patient_data.confirmed_date)
daily_count = patient_data.groupby(patient_data.confirmed_date).patient_id.count()
accumulated_count = daily_count.cumsum()


# In[ ]:


patient_data['age'] = 2020 - patient_data['birth_year'] 

import math

def group_age(age):
    if age >= 0: # not NaN
        if age % 10 != 0:
            lower = int(math.floor(age / 10.0)) * 10
            upper = int(math.ceil(age / 10.0)) * 10 - 1
            return f"{lower}-{upper}"
        else:
            lower = int(age)
            upper = int(age + 9) 
            return f"{lower}-{upper}"
    return "Unknown"


# In[ ]:


patient_data["age_range"] = patient_data["age"].apply(group_age)


# <a id="uv"></a>
# 
# ## Univariate Analysis
# 
# 
# 
# 
# ### Male vs Female patient Count

# In[ ]:


import plotly.graph_objects as go

source_counts = patient_data.sex.value_counts().reset_index()

labels = source_counts['index'].values
values = source_counts.sex.values

layout = dict(title= "Male Vs Female", width = 590, height = 500)

fig = go.Figure(data=[go.Pie(labels=labels, values=values)], layout=layout)

fig.show()


# ### Reason for Infection?

# In[ ]:


def _generate_bar_plot_hor(df, col, title, color, w=None, h=None, lm=0, limit=100):
    cnt_srs = df[col].value_counts()[:limit]

    trace = go.Bar(y=cnt_srs.index[::-1], x=cnt_srs.values[::-1], orientation = 'h', marker=dict(color=color))

    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)

    data = [trace]
    
    fig = go.Figure(data=data, layout=layout)
    
    iplot(fig)


# In[ ]:


patient_data


# In[ ]:


cols = ['infection_reason']
_generate_bar_plot_hor(patient_data, cols[0], "Distribution of Infection Reason", '#f2b5bc', 1000, 600, 200)


# ### Distribution of age-range

# In[ ]:


cols = ['age_range']
_generate_bar_plot_hor(patient_data, cols[0], "Distribution of Age Range", '#CD5C5C', 1000, 600, 400)


# <a id="mu"></a>
# 
# ## Multivariate

# ### Infection reason by Gender

# In[ ]:


sns.set(rc={'figure.figsize':(18, 10)})

sns.boxplot(x="infection_reason", y="contact_number", hue="sex",  palette="PRGn", data=patient_data)

plt.title("#of people contacted by Infection Reason & gender")

plt.xticks(rotation='vertical')
plt.show()


# ### How many patients are there under different age ranges and different infections ?

# In[ ]:


cols = ['infection_reason', 'age_range']

sns.set(rc={'figure.figsize':(18, 10)})

colmap = sns.light_palette("#ff4284", as_cmap=True)

pd.crosstab(patient_data[cols[0]], patient_data[cols[1]]).style.background_gradient(cmap = colmap)


# ### How many patients are there under different states and different infections ?

# In[ ]:


cols = ['infection_reason', 'state']

colmap = sns.light_palette("#7FB3D5", as_cmap=True)

pd.crosstab(patient_data[cols[0]], patient_data[cols[1]]).style.background_gradient(cmap = colmap)


# ### What is the Mean Age of Patients having different States and different cause of infection

# In[ ]:


def _create_bubble_plots(col1, col2, aggcol, func, title, cs):

    tempdf = patient_data.groupby([col1, col2]).agg({aggcol : func}).reset_index()
    tempdf[aggcol] = tempdf[aggcol].apply(lambda x : int(x))
    tempdf = tempdf.sort_values(aggcol, ascending=False)

    sizes = list(reversed([i for i in range(10,31)]))
    intervals = int(len(tempdf) / len(sizes))
    size_array = [9]*len(tempdf)
    
    st = 0
    for i, size in enumerate(sizes):
        for j in range(st, st+intervals):
            size_array[j] = size 
        st = st+intervals
    tempdf['size_n'] = size_array

    cols = list(tempdf['size_n'])

    trace1 = go.Scatter(x=tempdf[col1], y=tempdf[col2], mode='markers', text=tempdf[aggcol],
        marker=dict( size=tempdf.size_n, color=cols, colorscale=cs ))
    data = [trace1]
    layout = go.Layout(title=title)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

_create_bubble_plots('infection_reason', 'state', 'age', 'mean', '', 'Electric')


# ### Correlation among the variables

# In[ ]:


patient_data_corr = patient_data[['age', 'infection_order', 'contact_number']]


# In[ ]:


corr = patient_data_corr.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(style="white")

f, ax = plt.subplots(figsize=(15, 12))

cmap = sns.diverging_palette(30, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.1, cbar_kws={"shrink": .5});


# ### Infection reason and Infection order

# In[ ]:


import plotly.graph_objects as go

ct = pd.crosstab(patient_data['infection_reason'] , patient_data['infection_order'])
ctr = ct.index.tolist()


fig = go.Figure(data=[
    go.Bar(name='1.0', x=ctr, y=ct[1.0].tolist()),
    go.Bar(name='2.0', x=ctr, y=ct[2.0].tolist()),
    go.Bar(name='3.0', x=ctr, y=ct[3.0].tolist()),
    go.Bar(name='4.0', x=ctr, y=ct[4.0].tolist()),
    go.Bar(name='5.0', x=ctr, y=ct[5.0].tolist()),

])

fig.update_layout(barmode='group')
fig.show()


# ### Forecasting using FB-Prophet

# In[ ]:


data = daily_count.resample('D').first().fillna(0).cumsum()


# In[ ]:


prophet= pd.DataFrame(data)

prophet

pr_data = prophet.reset_index()

pr_data.columns = ['ds','y']


# In[ ]:


m=Prophet()

m.fit(pr_data)

future = m.make_future_dataframe(periods=30)

forecast=m.predict(future)


# In[ ]:


figure = m.plot(forecast, xlabel='Date', ylabel='Confirmed Count')


# In[ ]:


df = time_data.set_index('date').sort_index()['confirmed'].cumsum().loc[lambda x: x > 0].to_frame('cases_cs')
df['t'] = df.reset_index().reset_index().index + 1


# In[ ]:


df.iloc[-10:]


# <a id="ex"></a>
# 
# ### Exponential Function

# In[ ]:


def exp_func(a, b, t):
    return a*np.exp(b*t)


# In[ ]:


x = df.t.tolist()
y = df.cases_cs.tolist()


# In[ ]:


import scipy as sp

pop, params = sp.optimize.curve_fit(lambda t, a, b: exp_func(a, b, t),  x,  y, p0=[0.018, -0.243])

a_hat, b_hat = pop

print(a_hat, b_hat)


# In[ ]:


df['cases_hat'] = df.t.apply(lambda t: exp_func(a_hat, b_hat, t))


# In[ ]:


ts = [
    {
        't': t,
        'cases_hat': exp_func(a_hat, b_hat, t)
    }
    for t in range(26, 91)
]
df_hat = pd.DataFrame(ts)


# In[ ]:


df_proj = pd.concat([
    df.drop(['cases_hat'], axis=1),
    df_hat
], sort=False).set_index('t')

df_proj.rename(
    columns={
        'cases_cs': 'Confirmed',
        'cases_hat': 'Projected'
    }, inplace=True
)


# In[ ]:


today_index = 53


# In[ ]:


delta_forecast = 150.0
today_forecast = df.set_index('t').loc[today_index-1, 'cases_cs'] + delta_forecast


# In[ ]:


today_actual = df_proj.loc[today_index].Confirmed.dropna().iloc[0]


# In[ ]:


today_abs_error = np.abs(today_actual - today_forecast)
today_abs_percentage_error = round(today_abs_error / today_actual * 100,1)

error_string = ("12/03/2020: "
                f"\nConfirmed={today_actual} "
                f"\nForecast={today_forecast} "
                f"\nError={today_abs_error} ({today_abs_percentage_error}%)")

print(error_string)


# ### Fitting the model on Known data

# In[ ]:


_, _, r_value, _, _ = sp.stats.linregress(df_proj.groupby('t').min().dropna()['Confirmed'],
                                          df_proj.groupby('t').min().dropna()['Projected'])
r_2 = round(r_value, 4)
r_2


# In[ ]:


df_proj.loc[:today_index + 10].plot(alpha=0.5, lw=5)

plt.text(x=today_index, y=100, s=f"4 days ago")
plt.axvline(x=today_index, color="darkgreen", linestyle="dotted")

plt.xlabel("Days from Outbreak")
plt.ylabel("# Covid-19 Cases")
plt.title(r"$y = 0.033 e ^{- 0.227 t}$; $R^2=0.9965$")


# In[ ]:




