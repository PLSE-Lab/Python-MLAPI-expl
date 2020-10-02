#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import sklearn as skl
import plotly as plt
import csv

plt.offline.init_notebook_mode(connected=True)


# First, we load the data, doing some preprocessing, preserve the columns we are interested, and rename them in the process: 

# In[ ]:


data_path = "/kaggle/input/pokemon-database/Pokemon Database.csv"

df_raw = pd.read_csv(data_path, encoding='cp1252')
df_raw = df_raw.set_index('Pokemon Id')
df_raw.loc[df_raw['Original Pokemon ID'].notna(),'Legendary Type'] =     list(df_raw.loc()[df_raw[df_raw['Original Pokemon ID'].notna()]['Original Pokemon ID']]['Legendary Type'])


# In[ ]:


column_name_dict = {
    'Pokedex Number': 'nid', 
    'Pokemon Name': 'name', 
    'Alternate Form Name': 'form', 
    'Legendary Type': 'legendary', 
    'Pokemon Height': 'height', 
    'Pokemon Weight': 'weight', 
    'Primary Type': 'type_1', 
    'Secondary Type': 'type_2',
    'Health Stat': 'hp', 
    'Attack Stat': 'atk', 
    'Defense Stat': 'def', 
    'Special Attack Stat': 'satk', 
    'Special Defense Stat': 'sdef', 
    'Speed Stat': 'spd', 
    'Base Stat Total': 'bst', 
    'EV Yield Total': 'ev_total', 
}

df = df_raw[column_name_dict.keys()]
df.columns = column_name_dict.values()
df = df.fillna(value={'form': '', 'legendary': ''})
df.reset_index(level=0, inplace=True)
df.head()


# Next, we extract the numeric columns we are going to use and normalize them: 

# In[ ]:


TYPE_LIST = sorted(list(set(df.type_1)))
COL_STATS = ['hp', 'atk', 'def', 'satk', 'sdef', 'spd']

df_stats = df[COL_STATS+['ev_total']]
display(df_stats.describe().loc()[['mean', 'std']].style.set_caption('Before normalization'))

def normalize(df, population=None):
    if population is None:
        population = df
    df_desc = population.describe().loc()[['mean', 'std']]
    return (df-df_desc.loc['mean'])/df_desc.loc['std']

display(normalize(df_stats).describe().loc()[['mean', 'std']].style.set_caption('After normalization'))


# After that we can perform PCA (Principal Component Analysis) on them: 

# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(random_state=227)
pca.fit(normalize(df_stats))
pcs = pca.components_

df_var_r = pd.DataFrame(pca.explained_variance_ratio_[:,np.newaxis], columns=['var_r'])
df_var_r.index = [f"pc{i}" for i in range(len(pcs))]

df_pc = pd.DataFrame(pca.components_, columns=COL_STATS+['ev_total'])
df_pc.index = [f"pc{i}" for i in range(len(pcs))]

display(
    pd.concat([df_pc, df_var_r],axis=1).style\
        .background_gradient(cmap='bwr_r', axis=None)\
        .format("{:.3}")
)


df_pc_revNorm = df_pc*df_stats.describe().loc['std']
df_pc_revNorm.loc['pc0'] /= df_pc_revNorm.loc['pc0'].ev_total 
display(df_pc_revNorm.loc()[['pc0']].style.format("{:.4}"))


# Here we can see some interesting results. I include the **total EV (effort value) yield** becuase it usually coorelates to what "stage" that Pokemon is, for example for 3-stage evolution lines they are 1 to 3 for each stages, and for single stage legendary Pokemon it's usually 3. Also, this also works for the "intermediate" single stage Pokemon like Skarmory, Druddigon, and Torkoal, all of which have 2 total EV yield. 
# 
# `pc0` clearly shows how the stats of a Pokemon grows with stages. It also explains nealy 50% of the variation in the input data. By reverse the normalization and doing some calcuations with `pc0`, we can see that when total EV yield go up by 1, the total stat increases by about 176.9 in general.
# 
# In components besides `pc0`, `pc1` and `pc2` take up the main portion of the variation, both of which have nearly 0 on total EV yield, and explains 10% of the data variation. `pc1` shows the tradeoff between mainly **defence** and **speed**, which appears in game as the difference between *tanks* and *sweepers*. That is, Pokemon with more positive `pc1` components tent to be defensive, more balky, and take less damage, but at the same time slower and usually have less attack. On the other hand, Pokemon with more negative `pc1` components are faster, can hit before the opponents more easily, with the cost of lower defence and HP. 
# 
# `pc2` shows the difference between *physical-oriented* Pokemon and *special-oriented* Pokemon. In the game, there are 3 types of moves: *physical*, *special*, and *status*. *Physical moves* usually calculates its damage by physical stats (**attack** and **defense**), and *special move* by physical stats (**special attack** and **special defense**). *Status moves* inflict status on various objects, for example the user itself, the target, or even the game field. Some example of the effects of status moves includes increasing the attack of the user, decreasing the defense of the target, or alter the weather of the game field. Since effects of different types of moves depends on different stats, it's normal that player will catagorize Pokemon as more physical-oriented or more special-oriented. One example is the infamous "SkarBliss" combination, which includes Skarmory and Blissey:  

# In[ ]:


df[df.name.isin(['Skarmory', 'Blissey'])]


# Since the reason given above, attackers are often either physical attacker of special attacker based on thier stats, which are then usually given the cooresponing type of moves. In the "SkarBliss" combination, Skarmory has high defense (physical wall) and Blissey has high special defense (special wall) and high HP. At the same time, Skarmory also has move that can set up entry hazard (damages opposing Pokemon when switch in) and force the oppenct to switch respectively (a role which is often called phaser). 
# 
# `pc5` and `pc6` are somewhat interesting. Although only take up 6% of the explained variation, both of them mostly increase the base stats while decreasing total EV yield. Further more, the increased stats are compliment in `pc5` and `pc6`, meaning Pokemon with positive `pc5` or `pc6` components will have higher stats compared to other pokemon with the same total EV yield. What does this mean? We see after we analyze deeper...

# To explore more about the data and take other factors into account, for example types of the Pokemon, and whether the Pokemon is legenday, I chart the data using a interactive scatter plot. This also uses the dropdown widgets from `ipywidgets` for selecting the x axis, y axis, and the color. You proably need to edit the notebook to make the widgets working. The size of the dot represents the total EV yield of each Pokemon. 

# In[ ]:


import plotly.express as px
import ipywidgets as widgets

TYPE_COLOR_MAP = {
    'Bug': 'lightgreen', 
    'Dark': 'black', 
    'Dragon': 'blue', 
    'Electric': 'yellow', 
    'Fairy': 'fuchsia', 
    'Fighting': 'orange', 
    'Fire': 'red', 
    'Flying': 'skyblue', 
    'Ghost': 'midnightblue', 
    'Grass': 'green', 
    'Ground': 'brown', 
    'Ice': 'aqua', 
    'Normal': 'gray', 
    'Poison': 'purple', 
    'Psychic': 'violet', 
    'Rock': 'teal', 
    'Steel': 'silver', 
    'Water': 'navy', 
}

df_plot = pd.concat([df[['nid', 'name', 'form', 'type_1', 'type_2', 'ev_total', 'legendary']], 
                     pd.DataFrame(pca.transform(normalize(df_stats)), 
                                  columns=[f"pc{i}" for i in range(len(pcs))])],
                    axis=1).copy()
pd.options.mode.chained_assignment = None
df_plot.type_2[df_plot.type_2.isna()] = df_plot.type_1[df_plot.type_2.isna()] 
df_plot.legendary[df_plot.legendary == ''] = 'None'
df_plot.form[df_plot.form == ''] = 'None'

def show_pcs_fig(df):
    def show_pcs_fig_df(x_axis, y_axis, color):
        fig = px.scatter(df, x=x_axis, y=y_axis, 
                         color=color, size='ev_total', 
                         hover_data=['name','form','legendary'],
                         size_max=6, 
                         color_discrete_map=TYPE_COLOR_MAP,
                         category_orders={'type_1': TYPE_LIST,
                                          'type_2': TYPE_LIST})
        return fig
    return show_pcs_fig_df


pcs_str = [f'pc{i}' for i in range(len(pcs))]
x_dropdown = widgets.Dropdown(options=pcs_str, value=pcs_str[1])
y_dropdown = widgets.Dropdown(options=pcs_str, value=pcs_str[2])
class_dropdown = widgets.Dropdown(options=['type_1', 'type_2', 'legendary'], value='type_1')

_ = widgets.interact(show_pcs_fig(df_plot), x_axis=x_dropdown, y_axis=y_dropdown, color=class_dropdown)


# Besides that, we can also calculate the statistics of the components of Pokemon of different types. Here to be counted as a specific type, the Pokemon need any of its type being that type. For example, Skarmory is included in both Steel type and Flying type. 

# In[ ]:


display(pd.DataFrame({t: pd.DataFrame(pca.transform(normalize(df_stats[(df.type_1==t) | (df.type_2==t)],df_stats)), 
                                      columns=[f"pc{i}" for i in range(len(pcs))])\
                         .describe().loc()['mean']
                         for t in TYPE_LIST
                     }).style\
                       .background_gradient(cmap='bwr_r', axis=1)\
                       .format("{:.3}")
                       .set_caption('mean')
       )
display(pd.DataFrame({t: pd.DataFrame(pca.transform(normalize(df_stats[(df.type_1==t) | (df.type_2==t)],df_stats)), 
                                      columns=[f"pc{i}" for i in range(len(pcs))])\
                         .describe().loc()['std']
                         for t in TYPE_LIST
                     }).style\
                       .background_gradient(cmap='OrRd', axis=1)\
                       .format("{:.3}")
                       .set_caption('std')
       )


# In the statisics, one thing that immediately stand out is how high the mean of `pc0` componenet of Dragon type Pokemon is. This is expected however, considering that how many Dragon type Pokemon are legendary, all of which has 3 total EV yield. 
# 
# There are also some other interesting things related to my intepretation of the components and the type of the Pokemons. To see that in detail, first we need to define functions for filtering Pokemon by types: 

# In[ ]:


df_prop_ = df[['nid', 'name', 'form', 'legendary', 'height', 'weight', 'type_1', 'type_2', 'ev_total']]
df_pca_ = pd.DataFrame(pca.transform(normalize(df_stats)), 
                      columns=[f"pc{i}" for i in range(len(pcs))])

df_pca = pd.concat([df_prop_, df_pca_], axis=1)


# In[ ]:


def type_filter(df, t):
    df_type_ = df_pca[(df.type_1==t) | (df.type_2==t)].copy()
    df_type_.insert(len(df_type_.columns)-7, 'type', t)
    return df_type_

def types_filter(df, ts):
    return pd.concat([type_filter(df,t) for t in ts], axis=0)

types_filter(df_pca, ['Flying', 'Steel'])


# For example, we can see using `pc1` in the following chart that Flying types and Electric types are generaly faster, while Rock types and Steel types are more defensive: 

# In[ ]:


show_pcs_fig(types_filter(df_pca, ['Flying', 'Electric', 'Steel', 'Rock']))('pc1', 'pc2', 'type')


# In this chart, we can see 2 outliners for `pc1`: Ninjask for being negative and Shuckle for being positive. It's not hard to see why by looking at their stats: 

# In[ ]:


df[df.name.isin(['Ninjask', 'Shuckle'])]


# Likewise, we can see using `pc2` in the following chart that Fighting types and Ground types are generaly more physical-oriented, while Fairy types and Physic types are more special-oriented:

# In[ ]:


show_pcs_fig(types_filter(df_pca, ['Fighting', 'Ground', 'Fairy', 'Psychic']))('pc1', 'pc2', 'type')


# Now let's try to answer what do `pc5` and `pc6` represent. Ploting using `pc5` and `pc6` give us these: 

# In[ ]:


show_pcs_fig(df_plot)('pc5', 'pc6', 'form')


# Some Pokemon with large positive `pc5` or `pc6` are in Mega form, Primal form, or Eternamax form. These form in game are temporary and need some requirements, for example needing to have special item on Pokemon, but at the same time give the Pokemon adventage by increasing stats, changing ability and such. Thus, we can see that `pc5` and `pc6` can be use to represent how high the stats of the Pokemon is compared to Pokemon of the same stage. Further more, we can plot legenday Pokemon using `pc5` and `pc6`: 

# In[ ]:


fig = show_pcs_fig(df_plot[df_plot.legendary == 'Legendary'])('pc5', 'pc6', 'form')
fig.add_shape(
            type="line",x0=-2,y0=2,x1=2,y1=-2,
            line={'color': 'MediumPurple','width': 1, 'dash': "dot"}
)


# In this chart be can see 2 groups, one above the diagonal (`pc5+pc6>0`) and one below it (`pc5+pc6<0`). Most of the Pokemon above the line are either so-called "box legenday" i.e. legenday Pokemon that appeared as the box art of the game, or some form of the other legenday, for example Mega Latias and Mega Latios.  

# In[ ]:




