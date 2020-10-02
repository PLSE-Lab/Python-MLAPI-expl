#!/usr/bin/env python
# coding: utf-8

# This notebook is part of the lesson from [Dashboarding with Notebooks series](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-1?utm_medium=email&utm_source=intercom&utm_campaign=dashboarding-event) that will be updated as the new classes are available.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# In[ ]:


notices = pd.read_csv("../input/procurement-notices.csv", parse_dates=["Publication Date", "Deadline Date"])
notices.sample(5)


# Maybe we want to monitor the number of notice types at each time period, so we could make a dashboard to keep track of it. The unique types are:

# In[ ]:


np.unique(notices["Notice Type"])


# Now we can plot the data using [plotly](https://plot.ly/feed/#/).

# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

grouped = notices.groupby(["Publication Date", "Notice Type"])["Project ID"].count().unstack()

data = list()

# for each Notice Type, plot the count of Project IDs
for column in grouped.columns:
    data.append(go.Scatter(
        x = grouped.index,
        y = grouped[column],
        name = column
        )
    )
iplot(data)
# grouped["Contract Award"]


# Now let's use some contry data. How many IDs there are for each country?

# In[ ]:


project_by_country = notices.groupby("Country Name", as_index = False)["Project ID"].count()
project_by_country.sample(5)


# *Note:  The Python code was adapted from [Theofanis' notebook](https://www.kaggle.com/faniseng/dashboard-world-bank-procurements-py).*

# In[ ]:


import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

iplot([go.Choropleth(
    locationmode='country names',
    locations=project_by_country["Country Name"].values,
    text=project_by_country["Country Name"],
    z=project_by_country["Project ID"]
)])


# In[ ]:




