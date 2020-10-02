#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import sqlite3
conn = sqlite3.connect("/kaggle/input/uw-madison-courses/database.sqlite3")
cur = conn.cursor()
cur.execute("select name, course_offering_uuid from subjects, subject_memberships where subjects.code=subject_memberships.subject_code  limit 5;")
results = cur.fetchall()
print(results)
# cur.execute("select name, subject_memberships.course_offering_uuid, a_count, ab_count, b_count, bc_count, c_count, d_count, f_count   " \
#             "from subjects, subject_memberships, grade_distributions " \
#             "where subjects.code=subject_memberships.subject_code " \
#             "and subject_memberships.course_offering_uuid=grade_distributions.course_offering_uuid " )
#   #         "limit 5;")
# results = cur.fetchall()
# print(results)


# In[ ]:


query = "select name, subject_memberships.course_offering_uuid, a_count, ab_count, b_count, bc_count, c_count, d_count, f_count   "             "from subjects, subject_memberships, grade_distributions "             "where subjects.code=subject_memberships.subject_code "             "and subject_memberships.course_offering_uuid=grade_distributions.course_offering_uuid " 
df = pd.read_sql_query(query, conn)
print(df.head())
print(df.name.nunique())
print(df.name.value_counts()[:5])


# In[ ]:


df.name.unique()


# In[ ]:


names = ['Chemistry', 'Electrical and Computer Engineering', 'Counseling Psychology', 'History']
df['a_count'] = df['a_count'].astype(int)
df['ab_count'] = df['ab_count'].astype(int)
df['b_count'] = df['b_count'].astype(int)
df['bc_count'] = df['bc_count'].astype(int)
df['c_count'] = df['c_count'].astype(int)
df['d_count'] = df['d_count'].astype(int)
df['f_count'] = df['f_count'].astype(int)
df_filtered = df.loc[df['name'].isin(names)].copy()


# In[ ]:


print(df_filtered.info())
df_filtered.columns


# In[ ]:


df_filtered.loc[:,'total'] = df_filtered['a_count'] + df_filtered['ab_count'] + df_filtered['b_count'] + df_filtered['bc_count'] + df_filtered['c_count'] + df_filtered['d_count'] + df_filtered['f_count']
df_filtered.loc[:,'total'].describe()


# In[ ]:


# https://registrar.wisc.edu/grades-and-gpa/
# grades = {}


# In[ ]:


import seaborn as sns
sns.set(style="ticks")
_=sns.pairplot(df_filtered.drop(['course_offering_uuid'], axis=1), hue="name")


# In[ ]:


df_filtered_copy = df_filtered[df_filtered['total']>0].copy()
df_filtered_copy.loc[:, [ 'a_count', 'ab_count', 'b_count',
       'bc_count', 'c_count', 'd_count', 'f_count']] = df_filtered_copy.loc[:, [ 'a_count', 'ab_count', 'b_count',
       'bc_count', 'c_count', 'd_count', 'f_count']].div(df_filtered_copy['total'], axis=0)
_=sns.pairplot(df_filtered_copy.drop(['course_offering_uuid','total'], axis=1), hue="name")


# In[ ]:


df_filtered_copy_melt = pd.melt(df_filtered_copy.drop(['course_offering_uuid','total'], axis=1), "name", var_name="grade type count")


# In[ ]:


df_filtered_copy_melt.describe()


# In[ ]:


import matplotlib.pyplot as plt
plt.subplots(figsize=(16, 7))
ax = sns.boxplot(x="grade type count", y="value",
            hue="name", 
            data=df_filtered_copy_melt)


# In[ ]:


from numpy import median
ax = sns.catplot(x="grade type count", y="value",row='name', data=df_filtered_copy_melt, estimator=median,aspect=2,height=4, kind='bar')


# In[ ]:


# pal = sns.cubehelix_palette(10, rot=-.25, light=.7) ,palette=pal aspect=2, height=2,
g = sns.FacetGrid(df_filtered_copy_melt, row="grade type count", col='name',  hue="name", margin_titles=True)

# Draw the densities in a few steps
g.map(sns.kdeplot, "value", clip_on=False, shade=True, alpha=0.2, lw=1.5, bw=.2).add_legend()
# def label(x, color, label):
#     ax = plt.gca()
#     ax.text(0, .2, label, color=color,
#             ha="left", va="center", transform=ax.transAxes)
# g.map(label, "name")
# g.set_titles("")


# In[ ]:


import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


init_notebook_mode(connected=True)


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_filtered['a_count'],
    y=df_filtered['name'],
    name='a_count',
    marker=dict(
        color='rgba(50, 165, 196, 0.95)',
        line_color='rgba(156, 165, 196, 0.5)',
    )
))


fig.update_traces(mode='markers', marker=dict(line_width=1, symbol='circle', size=5))

fig.update_layout(
    title="a_count",
    xaxis=dict(
        showgrid=False,
        showline=True,
        linecolor='rgb(102, 102, 102)',
        tickfont_color='rgb(102, 102, 102)',
        showticklabels=True,
        dtick=10,
        ticks='outside',
        tickcolor='rgb(102, 102, 102)',
    ),
    margin=dict(l=140, r=40, b=50, t=80),
    legend=dict(
        font_size=10,
        yanchor='middle',
        xanchor='right',
    ),
    width=800,
    height=400,
    paper_bgcolor='white',
    plot_bgcolor='white',
    hovermode='closest',
)
fig.show()

