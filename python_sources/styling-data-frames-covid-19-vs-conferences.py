#!/usr/bin/env python
# coding: utf-8

# # Styling Data Frames: COVID-19 vs Conferences
# 
# Credit to @parulpandey for the respective dataset.

# ## Imports and reading in data

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.express as px
import sqlite3 as sq


# In[ ]:


df = pd.read_csv("../input/2020-conferences-cancelled-due-to-coronavirus/2020 Conferences Cancelled Due to Coronavirus - Sheet1.csv")


# In[ ]:


df.head()


# # Part 1: Simple colormaps

# In[ ]:


x = df['Country'].value_counts()
x = pd.DataFrame(x)
x.style.background_gradient(cmap='Reds')


# In[ ]:


x = df['Venue'].value_counts()
x = pd.DataFrame(x)
x.style.background_gradient(cmap='Reds')


# In[ ]:


x = df['Status'].value_counts()
x = pd.DataFrame(x)
x.style.background_gradient(cmap='Greens')


# # Part 2: More detailed colormaps

# In[ ]:


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


# In[ ]:


df.style.apply(highlight_max)


# In[ ]:


def highlight_min(s):
    '''
    highlight the minimum in a series blanched almond (no, I do not know what that is).
    '''
    is_max = s == s.min()
    return ['background-color: blanchedalmond' if v else '' for v in is_max]


# In[ ]:


df.style.apply(highlight_min)


# # Part 3: More intricate displays

# *Credit: pandas.pydata.org*

# In[ ]:


import pandas as pd
from IPython.display import HTML

# Test series
test1 = pd.Series([-100,-60,-30,-20], name='All Negative')
test2 = pd.Series([10,20,50,100], name='All Positive')
test3 = pd.Series([-10,-5,0,90], name='Both Pos and Neg')

head = """
<table>
    <thead>
        <th>Align</th>
        <th>All Negative</th>
        <th>All Positive</th>
        <th>Both Neg and Pos</th>
    </thead>
    </tbody>

"""

aligns = ['left','zero','mid']
for align in aligns:
    row = "<tr><th>{}</th>".format(align)
    for serie in [test1,test2,test3]:
        s = serie.copy()
        s.name=''
        row += "<td>{}</td>".format(s.to_frame().style.bar(align=align,
                                                           color=['#d65f5f', '#5fba7d'],
                                                           width=100).render()) #testn['width']
    row += '</tr>'
    head += row

head+= """
</tbody>
</table>"""


# In[ ]:


HTML(head)


# In[ ]:


# Interactivity
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "4pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]


# In[ ]:


np.random.seed(25)
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
bigdf = pd.DataFrame(np.random.randn(20, 25)).cumsum()

bigdf.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '1pt'})    .set_caption("Hover to magnify")    .set_precision(2)    .set_table_styles(magnify())

