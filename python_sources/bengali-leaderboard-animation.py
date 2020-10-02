#!/usr/bin/env python
# coding: utf-8

# # Top 25 Over Time Animation

# In[ ]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pylab as plt\n\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport matplotlib.ticker as ticker\nimport matplotlib.animation as animation\nfrom IPython.display import HTML\n\nimport matplotlib.colors as mcolors\n\nimport seaborn as sns')


# In[ ]:


# subs = pd.read_csv('../input/meta-kaggle/Submissions.csv', low_memory=False)
# teams = pd.read_csv('../input/meta-kaggle/Teams.csv', low_memory=False)
# comps = pd.read_csv('../input/meta-kaggle/Competitions.csv', low_memory=False)
mypal = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Grab the color pal
cm = plt.get_cmap('tab20')

NUM_COLORS = 20
mypal = [mcolors.to_hex(cm(1.*i/NUM_COLORS)) for i in range(NUM_COLORS)]
# mypal = [mcolors.CSS4_COLORS[val] for val in mcolors.CSS4_COLORS]


# In[ ]:


df = pd.read_csv('../input/bengalileaderboard/0316_bengaliai-cv19-publicleaderboard.csv')
df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
df = df.set_index(['TeamName','SubmissionDate'])['Score'].unstack(-1).T
df.columns = [name for name in df.columns]

FIFTEENTH_SCORE = df.max().sort_values(ascending=True)[15]
FIFTYTH_SCORE = df.max().sort_values(ascending=True)[50]
TOP_SCORE = df.max().sort_values(ascending=True)[0]

ALL_TEAMS = df.columns.values
df_ffill = df[ALL_TEAMS].ffill()

my_df = df.T

min_sub_dict = {}
for c in df.columns:
    min_sub_dict[c] =  df[c].dropna().index.min()
    

my_df['colors'] = [np.random.choice(mypal) for c in range(len(my_df))]
color_map = my_df['colors'].to_dict()


# In[ ]:


def draw_barchart(mydate):
    mydate = pd.to_datetime(mydate)
    dff = df_ffill.loc[df_ffill.index <= mydate]         .iloc[-1]         .sort_values(ascending=True)         .dropna()         .tail(25)

    last_sub_date = {}
    df2 = df.loc[df.index <= mydate]
    for c in df2.columns:
        last_sub_date[c] = df2[c].dropna().index.max()

    ax.clear()
    ax.barh(dff.index, dff.values, color=[color_map[x] for x in dff.index])
    ax.set_xlim(dff.min()-0.002, dff.max()+0.001)
    dx = dff.values.max() / 10000
    for i, (value, name) in enumerate(zip(dff.values, dff.index)):
        ax.text(value-dx,
                i,
                name,
                size=14, weight=600, ha='right', va='bottom')
        ax.text(value-dx,
                i-.25,
                f'first sub: {min_sub_dict[name]:%d-%b-%Y} / last sub {last_sub_date[name]:%d-%b-%Y}',
                size=10,
                color='#444444',
                ha='right',
                va='baseline')
        ax.text(value+dx, i,     f'{value:,.4f}',  size=14, ha='left',  va='center')
    # ... polished styles
    ax.text(1, 0.4, mydate.strftime('%d-%b-%Y'), transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.text(0, 1.06, 'Score', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.4f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.12, 'Bengali Kaggle Competition Top 25',
            transform=ax.transAxes, size=24, weight=600, ha='left')
    plt.box(False)

# fig, ax = plt.subplots(figsize=(15, 13))
# draw_barchart('2020-03-05')


# In[ ]:


dates = [pd.to_datetime(x) for x in pd.Series(df.index.date).unique() if x > pd.to_datetime('12-31-2019')]
dates = dates + [dates[-1] + pd.Timedelta('1 day')]
fig, ax = plt.subplots(figsize=(15, 20))
animator = animation.FuncAnimation(fig,
                                   draw_barchart,
                                   frames=dates,
                                   interval=300)
ani = HTML(animator.to_jshtml())
# _ = animator.to_html5_video()
# or use animator.to_html5_video() or animator.save()


# In[ ]:


ani

