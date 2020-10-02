#!/usr/bin/env python
# coding: utf-8

# Simple kernel for of Two sigma predict stock movements results. Later I'll add some animation based on later LB results.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


leaderboard_prev = pd.read_csv('../input/lb20190401/publicleaderboarddata20190401/two-sigma-financial-news-publicleaderboard.csv')
leaderboard = pd.read_csv('../input/lb20190418/publicleaderboarddata20190418/two-sigma-financial-news-publicleaderboard.csv')


# In[3]:


leaderboard.head()


# > We have 959 kernels in competition and 693 teams.

# In[4]:


leaderboard.shape[0], leaderboard.TeamId.nunique()


# This is a distribution of all kernels and team max values

# In[5]:


plt.figure(figsize=(15,6))
leaderboard.Score.plot.density(color='green')
leaderboard.groupby('TeamId').Score.max().plot.density(color='red');


# There are 427 teams with one kernel and 266 teams with 2 kernels

# In[6]:


leaderboard.groupby('TeamName').size().value_counts()


# In[7]:


leaderboard.groupby('TeamName').size().plot.hist();


# Results of two previous scorings

# In[8]:


lb_sorted = leaderboard.groupby('TeamName').Score.max().sort_values(ascending=False).reset_index()
lb_sorted_prev = leaderboard_prev.groupby('TeamName').Score.max()
lb_sorted['Score_prev'] = lb_sorted.TeamName.map(lb_sorted_prev) 
lb_sorted['Score_prev_rank'] = lb_sorted['Score_prev'].rank()
lb_sorted['Score_rank'] = lb_sorted['Score'].rank()
lb_sorted['Score_prev_rank'] = lb_sorted['Score_prev_rank'].max() - lb_sorted['Score_prev_rank'] + 1
lb_sorted['Score_rank'] = lb_sorted['Score_rank'].max() - lb_sorted['Score_rank'] + 1


# In[ ]:


from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper
from bokeh.layouts import row,column,gridplot
from bokeh.models.widgets import Tabs,Panel
output_notebook()

source = ColumnDataSource(lb_sorted)
plot = figure(
    x_axis_label = "Score",
    y_axis_label = "Prev Score",
    tools="crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,",
    plot_width=1000,
    plot_height=1000,
)
plot.circle(x="Score",y="Score_prev",source = source, radius=.01, alpha=0.5)
hover = HoverTool(tooltips = [
    ('Team', '@TeamName'), 
    ('Score/Prev', '@Score / @Score_prev'), 
    ('LB/Prev', '@Score_rank / @Score_prev_rank')])
plot.add_tools(hover)

plot.text(x=lb_sorted['Score'],y=lb_sorted['Score_prev'], text=lb_sorted['TeamName'],
       text_baseline="middle", text_align="left", text_font_size='8pt', text_font='Arial', alpha=0.5)
    
show(plot)


# And animation :)

# In[ ]:


df = lb_sorted.dropna().copy()
df.columns


# In[ ]:


from IPython.display import HTML
import matplotlib.animation as animation

df = lb_sorted.dropna().copy()
df.columns = ['z', 'y', 'x', 'x_rank', 'y_rank']

fig, ax = plt.subplots(figsize=(15,15))
ax.grid()
i=0
sc, = ax.plot((df.x+i*(df.y-df.x)/50).values, (df.x_rank+i*(df.y_rank-df.x_rank)/50).values, marker="o", ls="") # set linestyle to none
ax.set_ylim(ax.get_ylim()[::-1])
texts = []
for ind in df.index.values:
    texts += [plt.text(df.loc[ind, 'x']+0.01, df.loc[ind, 'x_rank'], df.loc[ind, 'z'])]

FRAMES = 32
FRAMES_2 = FRAMES // 2

def plot(i):
    delta = ((df.y_rank - df.x_rank)/700 * (FRAMES_2 - np.abs(i - FRAMES_2))/FRAMES_2).values
    
    X = [(df.x+i*(df.y-df.x)/FRAMES).values+delta*2, (df.x_rank+i*(df.y_rank-df.x_rank)/FRAMES).values+delta*300]
    sc.set_data(X[0], X[1])
    for ind, t in enumerate(texts):
        t.set_x(X[0][ind])
        t.set_y(X[1][ind])
#     ax.relim()
    ax.set_xlim(np.min(X[0]), np.max(X[0]))
    ax.set_ylim(np.max(X[1]), np.min(X[1]))
    ax.autoscale_view(True,True,True)

ani = animation.FuncAnimation(fig, plot,
            frames=FRAMES+1, interval=100, repeat=False) 
HTML(ani.to_jshtml())


# Kernels with the same results:

# In[ ]:


scores_count = leaderboard.groupby('Score').size().sort_values(ascending=False)
scores_count[scores_count>1]


# In[ ]:


for s in scores_count[scores_count>1].index.values:
    print(f'Score: {s}, Kernels count: {scores_count[s]}')
    print(f'Teamnames:', leaderboard[leaderboard.Score==s].TeamName.values)
    print()


# All results (Later we will have more information)

# In[ ]:


plt.figure(figsize=(9,96))
sns.barplot(x='Score', y='TeamName', data=lb_sorted);

