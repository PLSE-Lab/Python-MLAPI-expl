#!/usr/bin/env python
# coding: utf-8

# # First contact with dataviz packages for python
# ## Matplotlib
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


plt.plot(3,2,'x')
plt.show()


# In[ ]:


plt.plot(5,4, '.')
plt.plot(3,2, '^')
plt.show()


# In[ ]:


plt.plot([1,2,3],[1,2,3])


# ## Pandas.plot()

# In[ ]:


import pandas as pd
import numpy  as np
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2018', periods=1000))
ts.head()


# In[ ]:


ts = ts.cumsum()
ts.head()


# In[ ]:


ts.plot()


# In[ ]:


df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
df = df.cumsum()
plt.figure(); df.plot();


# 

# ## Seaborn

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "total_bill", color="steelblue", bins=bins)


# ## Plotly

# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)

trace0 = go.Scatter(
    x=[1,   2,  3, 4],
    y=[10, 15, 13, 17]
)
trace1 = go.Scatter(
    x=[1, 2, 3, 4],
    y=[16, 5, 11, 9]
)
data = [trace0, trace1]

py.iplot(data)


# In[ ]:


df = pd.DataFrame(np.random.randn(1000, 4), index=pd.date_range('1/1/2018', periods=1000), columns=list('ABCD'))
df = df.cumsum()  

trace1 = go.Scatter(
    y=df['A'],
    x=df.index
)
data = [trace1]

py.iplot(data)


# In[ ]:


df = pd.DataFrame(np.random.randn(1000, 4), index=pd.date_range('1/1/2018', periods=1000), columns=list('ABCD'))
df = df.cumsum()  

trace1 = go.Scatter(
    y=df['A'],
    x=df.index
)
trace2 = go.Scatter(
    y=df['B'],
    x=df.index
)
trace3 = go.Scatter(
    y=df['C'],
    x=df.index
)
trace4 = go.Scatter(
    y=df['D'],
    x=df.index
)



data = [trace1,trace2,trace3,trace4]

py.iplot(data)


# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go

import pandas as pd
import numpy as np

N = 40
x = np.linspace(0, 1, N)
y = np.random.randn(N)
df = pd.DataFrame({'x': x, 'y': y})
df.head()

data = [
    go.Bar(
        x=df['x'], # assign x as the dataframe column 'x'
        y=df['y']
    )
]
py.iplot(data)


# In[ ]:





# In[ ]:




