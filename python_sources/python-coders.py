#!/usr/bin/env python
# coding: utf-8

# # Python Coders: Lone Warriors or at the Center of Attention?
# 
# What is the most important programming language in data science - when it comes to frequency of use and recommended go-to langauge?
# 
# Python, or R? This question has the potential to kick off lively, controverse debates, as can be seen on [medium](https://medium.com/@data_driven/python-vs-r-for-data-science-and-the-winner-is-3ebb1a968197) or [quora](https://www.quora.com/Which-is-the-right-place-to-start-for-AI-ML-Data-science-career-Python-or-R-or-something-else).
# 
# Or can another language jump in?
# 
# Kagglers, at least, can give quite a clear answer...

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot


# In[ ]:


def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.12.12.min.js?noext',
            },
          });
        </script>
        '''))


# **Python is used most often**
# 
# When asked for the one specific, most often used programming language, the majority of the 15222 respondents name Python, namely 53.7%.
# 
# *8637 participants didn't answer.*

# In[ ]:


df = pd.read_csv('../input/multipleChoiceResponses.csv', delimiter=',')
lang1 = df.loc[1:, "Q17"].replace('NaN', np.NaN) # no answer
lang1 = lang1.dropna(axis=0, how='any')


# In[ ]:


configure_plotly_browser_state()
init_notebook_mode(connected=True) # connected=False if this fails

data_trace = dict(
              type='bar',
              value_format = ".2f",
              x = lang1.value_counts().index,
              y = lang1.value_counts().values / lang1.shape[0],
              marker = dict(color=['blue'] + ['darkgrey']*(len(lang1.value_counts().index) - 1)),
              opacity=0.8
)

data = [data_trace]
layout = dict(
    title = "Programming Language Used Most Often",
    yaxis = dict(title='(Relative) Number of Answers'),
    height = 600,
    width = 900,
    font = dict(size = 10),
)

fig = dict(data=[data_trace], layout=layout)
iplot(fig, validate=False)


# **Python recommended even by users of other languages**
# 
# Even of those participants that do not use Python most often, a high number would recommend it to data science aspirants as first language to learn.

# In[ ]:


lang = df.loc[1:, ["Q17", "Q18"]].replace('NaN', np.NaN) # no answer for both
lang = lang.dropna(axis=0, how='all')
# if only one answer -> replace with "None"
lang = lang.replace(np.NaN, 'None')

#https://datascience.stackexchange.com/questions/29840/how-to-count-grouped-occurrences
from itertools import product
suggestion_counts = lang.groupby(["Q17", "Q18"]).size().to_frame("Counts").reset_index()
q17_types = list(np.unique(suggestion_counts["Q17"]))
q18_types = list(np.unique(suggestion_counts["Q18"]))

suggestion_counts["Q17"] = suggestion_counts["Q17"].astype('category')
suggestion_counts["Q18"] = suggestion_counts["Q18"].astype('category')

suggestion_counts["Q17"] = suggestion_counts["Q17"].cat.codes
suggestion_counts["Q18"] = suggestion_counts["Q18"].cat.codes


# In[ ]:


configure_plotly_browser_state()
init_notebook_mode(connected=True) # connected=False if this fails

data_trace = dict(
    type='sankey',
    orientation="h",
    valueformat =".0f",
    node = dict(
        pad = 10,
        thickness = 30,
        line = dict(color="black", width=0.5),
        label = q17_types + q18_types,
        color = ["blue"] * (len(q17_types) + len(q18_types))
    ),
    link = dict(
        source = suggestion_counts["Q17"],
        target = suggestion_counts["Q18"] + len(q17_types),
        value = suggestion_counts["Counts"],
    )
)

layout = dict(
    title = "Programming Language Used vs. Sugessted",
    height = 750,
    width = 900,
    font = dict(size = 10),
)

fig = dict(data=[data_trace], layout=layout)
iplot(fig, validate=False)


# *If one of the two questions was not answered by a respondent, it was counted as "None". Participants who didn't answer both questions where excluded (n=5052).  Thus, a total of 18807 responses were analyzed.*

# **Majority recommends Python as first language to learn**
# 
# Python is recommended by 75.48% of the 18788 respondents as first langauge to learn when doing data science.
# 
# *5071 participants didn't answer.*

# In[ ]:


lang2 = df.loc[1:, "Q18"].replace('NaN', np.NaN) # no answer
lang2 = lang2.dropna(axis=0, how='any')


# In[ ]:


configure_plotly_browser_state()
init_notebook_mode(connected=True) # connected=False if this fails

data_trace = dict(
              type='bar',
              value_format = ".2f",
              x = lang2.value_counts().index,
              y = lang2.value_counts().values / lang2.shape[0],
              marker = dict(color=['blue'] + ['darkgrey']*(len(lang2.value_counts().index) - 1)),
              opacity=0.8
)

layout = dict(
    title = "Programming Language Used Most Often",
    yaxis = dict(title='(Relative) Number of Answers'),
    height = 600,
    width = 900,
    font = dict(size = 10),
)

fig = dict(data=[data_trace], layout=layout)
iplot(fig, validate=False)


# **Python increasingly popular**
# 
# Python's popularity has increased, as becomes obvious when looking at how long respondents are doing data science. While "only" 39.18% of respondents with a data science experience of 30-40 years claim to most often use Python, this value increases up to 59.89% for respondents with 1-2 years of experience. For newcomers (< 1 year or never written code to analyze data), having coding experience from different fields, we can observe a slightly more diverse range of programming languages that are currently used most often - but still with a dominance of Python.
# 
# *5326 participants didn't answer. 41 participants who have never written code and do not want to learn were also excluded.*

# In[ ]:


# sort data items
experience = ['< 1 year', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20-30 years', '30-40 years', '40+ years', 'I have never written code but I want to learn']
languages = df.loc[1:, "Q17"].value_counts().sort_index().index


# In[ ]:


configure_plotly_browser_state()
init_notebook_mode(connected=True) # connected=False if this fails

#https://community.plot.ly/t/custom-button-to-modify-data-source-in-plotly-python-offline/5915
def changeData(filter_name):
  filtered = df[df["Q24"] == filter_name].loc[:, "Q17"]
  filtered_counts = filtered.value_counts()
  for l in languages:
    if l not in filtered_counts.index:
      filtered_counts.at[l] = 0
  filtered_counts = filtered_counts.sort_index()
  return [ {'x': [filtered_counts.index], 'y':[filtered_counts.values / np.sum(filtered_counts.values)]}, {'title':'Programming Language Used Most Often with Data Science Experience of '+filter_name + " (n=" + str(filtered.shape[0]) + ")"} ]



filtered = df[df["Q24"] == experience[0]].loc[:, "Q17"]
filtered_counts = filtered.value_counts()
for l in languages: 
  if l not in filtered_counts.index:
    filtered_counts.at[l] = 0
filtered_counts = filtered_counts.sort_index()
  
data_trace = dict(
    type='bar',
    value_format = ".2f",
    x = filtered_counts.index,
    y = filtered_counts.values / np.sum(filtered_counts.values),
    marker = dict(color=['darkgrey']*10 + ['blue'] + ['darkgrey']*6),
    opacity=0.8
)

layout = dict(
    title = "Programming Language Used Most Often with Data Science Experience of < 1 Year (n=" + str(filtered.shape[0]) + ")",
    yaxis = dict(title='(Relative) Number of Answers',
                range=[0.0, 0.7]),
    height = 600,
    width = 950,
    font = dict(size = 10),
)

updatemenus=list([
    dict(
    buttons=list([
        dict(label = exp,
            method = 'update',
            args=changeData(exp)) for exp in experience]),
    direction='left',
    showactive = True,
    type='buttons',
    x=-0.1,
    xanchor='left',
    y=1.1,
    yanchor='top'
)
])

layout['updatemenus'] = updatemenus

fig = dict(data=[data_trace], layout=layout)
iplot(fig, validate=False)


# 
