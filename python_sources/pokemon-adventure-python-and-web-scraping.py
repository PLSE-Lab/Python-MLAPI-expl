#!/usr/bin/env python
# coding: utf-8

# # My Pokemon adventure (with Python!)

# ![Charizard and pikatch](https://media.giphy.com/media/HZpCCbcWc0a3u/giphy.gif)

# ## Introduction
# 
# This is my first Kaggle kernel, so I decided to start working on something that occupied a lot of my child-teenage years, Pokemon!  
# I wanted to analyse how various pokemon types and attributes compare in Gens 1 and 2 (as these are the ones I actually got to play) with some visuals and in the process I ended up doing some webscraping to enhance a bit the dataset.  
# #### Let's get started!
# 

# In[ ]:


# Data manipulation libraries
import numpy as np 
import pandas as pd 

# Data visualisation 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Web scraping 
from bs4 import BeautifulSoup, Comment
from requests import get

## Reading the data
pokemon = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')


# ### Table of contents
# 
# 1.0 [Data preparation and cleaning](#1)  
# &nbsp;&nbsp;&nbsp;1.1 [Dataset structure](#2)  
# &nbsp;&nbsp;&nbsp;1.2 [Dataset cleaning and filtering](#3)
# 
# 2.0 [Pokemon attributes and how they relate to each other](#4)  
# &nbsp;&nbsp;&nbsp;2.1 [Correlation between attributes](#5)  
# 
# 3.0 [Analysis of attributes per pokemon and type](#6)  
# &nbsp;&nbsp;&nbsp;3.1 [Web scraping to improve our dataset ](#7)  
# &nbsp;&nbsp;&nbsp;3.2 [Attributes per pokemon and type for only last evolution pokemon](#8)  
# &nbsp;&nbsp;&nbsp;3.3 [Addressing the elephant in the room and dealing with dual types](#9)  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3.1 [A brief overview of dual types in the dataset](#10)  
# &nbsp;&nbsp;&nbsp;3.4 [Attributes per pokemon and type for last evolution pokemon taking into account their dual type](#11)
# 
# 4.0 [Attributes of the average pokemon of each type](#12)  
# &nbsp;&nbsp;&nbsp;4.1 [Last evolution pokemon taking into account their dual type](#13)
# 
# 5.0 [Next steps](#14)
# 

# ## Data preparation and cleaning <a id="1"></a>
# After loading the dataframe we should start with some basic exploration of the dataset.

# In[ ]:


print(pokemon.head())
print(pokemon.info())
# 800 pokemon are present, the types of the columns seem to match what we would expect in terms of numeric and non-numeric


# ### Dataset structure <a id="2"></a>
# 
# Our dataset is structured as follows:
# 1. Name of the pokemon
# 2. Type 1 --> The main type of the pokemon
# 3. Type 2 --> The second type of a pokemon (some but not all have dual types)
# 4. Total --> The sum of all the pokemon attributes that are specified next
# 5. HP-Speed --> Individual pokemon attributes
# 6. Generation --> To which generation does each pokemon belong
# 7. Legendary --> Whether the pokemon is legendary or not
# 
# As I am getting older, I would probably like to narrow it down to the first and second pokemon generations which I actually got to play (Yellow and Crystal!)

# ### Dataset cleaning and filtering <a id="3"></a>

# In[ ]:


# We see that the first column is the same as the index so we can safely drop it
pokemon.drop(['#'], axis = 'columns', inplace = True)

my_pokemon = pokemon[pokemon.Generation.isin([1, 2])].copy()
my_pokemon.Generation.unique()
my_pokemon.shape


# Ok this seems kinda odd, the second generation had 251 Pokemon in total if I remember correctly so there have to be overlaps here. Let's see what might the problem be and resolve it.

# In[ ]:


print('There are ',my_pokemon.duplicated('Name').sum(), 'duplicated names in the dataset, so it must be something else\n')
print(my_pokemon.loc[(my_pokemon.Generation == 1) & (my_pokemon.Legendary == True), 'Name'],  '''\nAha! We didn't have MEGA stuff back in my day. Let's exclude those\n''')


# In[ ]:


mega_filter = (my_pokemon.Name.str.contains('Mega ')) #Put a space there to exclude meganium which is a legit second generation pokemon
my_pokemon = my_pokemon[~mega_filter]
print(my_pokemon.groupby('Generation').size(),  '\nOk now everything matches my expectations!')


# Alright! We have the 251 Pokemon now as we were expecting! Let's try to explore the attributes a bit to see just broadly how they relate to each other.

# ## Pokemon attributes and how they relate to each other <a id="4"></a>
# 
# Since we have 251 pokemon and 6 attributes for each of them I thought I would do a breakdown how they relate to each other.  
# Do pokemon with higher defence have also higher attack? Are fast pokemon also good defensively?  
# To that end I decided to use a pairgrid from Seaborn. Its quite a customisable grid where you can map different types of plots in the upper triangle, lower triangle and diagonal axes.  
# The figure produced has three plots:
# * Upper triangle --> Scatter plot for each pokemon with orange indicating legendary pokemon (Seaborn scatterplot)
# * Diagonal --> Kernel density plot for each attribute (Seabon kdplot)
# * Lower triangle --> Scatter plot with a linear regression model fit (Seaborn regplot)

# In[ ]:


def my_pairgrid(input_df):
    mpl.rcParams["axes.labelsize"] = 20 #Increases the axis titles and the legend size
    g = sns.PairGrid(input_df, vars=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'])
    g = g.map_diag(sns.kdeplot, shade=True)
    g = g.map_lower(sns.regplot, scatter_kws={'alpha':0.3})
    g.hue_vals = input_df["Legendary"]
    g.hue_names = input_df["Legendary"].unique()
    g.palette = sns.color_palette("Set2", len(g.hue_names))
    g = g.map_upper(sns.scatterplot).add_legend(title='Legendary', fontsize= 14)
    return(g)
_ = my_pairgrid(my_pokemon)


# So let's break down some findings from this overview chart:
# 1. Legendary pokemon are on the higher end of the spectrum for the majority of attribtues 
# 2. I can see some signs of positive relationship between Attack and Defence, Sp.Def and HP, Speed and Attack as well as Speed and Sp.Atk 
# 3. Speed seems to be the attribute that is the least concentrated amongst all 6

# ### Correlation between attributes <a id="5"></a>
# 
# 
# While the first chart was useful, perhaps a better way to visualise the correlation between attributes is a heatmap with stronger blue indicating higher correlation.

# In[ ]:


my_corr = my_pokemon.loc[:, 'HP':'Speed'].corr() 
mask = np.zeros_like(my_corr) 
mask[np.triu_indices_from(mask)] = True ## This part is to blank out the upper diagonal
with sns.axes_style("white"):
     ax = sns.heatmap(my_corr, mask=mask, square=True, cmap=sns.color_palette("Blues"), vmax=np.max(np.sort(my_corr.values)[:,-2]), linewidths=0.3)


# Turns out I missed Sp. Def and Defence, Sp. Def and Sp. Atk as well as Speed and Sp. Attack! 

# ## Analysis of attributes per pokemon and type <a id="6"></a>
# 
# Let's do some analysis of how attributes varry with type.  
# My first idea was to use a Seaborn Swarmplot since we don't have so much data that it will be overcrowded.

# In[ ]:


sns.reset_defaults()
plt.figure(figsize=(16,12))
g = sns.swarmplot(x='HP', y='Type 1', hue = 'Generation', data=my_pokemon)
g.yaxis.label.set_visible(False)
g.set_title('HP broken down by Type and Generation')
_ = g.annotate('These are Chansey and Blissey', xy=(253, 3.5), xytext = (250, 2), fontsize = 12, arrowprops=dict(facecolor='black') )
_ = g.annotate('Wooooobbufet', xy=(190, 9.5), xytext = (191, 8.5), fontsize = 12, arrowprops=dict(facecolor='green'))
g.title.set_fontsize(20)
g.xaxis.label.set_fontsize(18)
g.tick_params(axis='x', labelsize=16)
g.tick_params(axis='y', labelsize=16)


# But honestly the chart seems very distracting, I cannot distinguish a lot from the plot besides the outliers plus I noticed that there is a fairy type!  This was not the case in Gens 1 and 2, fairy pokemon were normal pokemon!

# In[ ]:


my_pokemon.loc[my_pokemon['Type 1'] == "Fairy", 'Type 1'] = 'Normal'
my_pokemon.loc[my_pokemon['Type 2'] == "Fairy", 'Type 2'] = np.nan


# So after fixing this, let's try to get a more comprehensive (to my eye) view of things by combining it with a boxplot!

# In[ ]:


def attr_per_type(input_df, y_attr, hue_attr):
    mpl.rcParams["font.size"] = 18
    f, axes = plt.subplots(2, 3, figsize=(20, 20), sharey=True, sharex=True)
    #f.tight_layout()
    poke_attributes = ['HP', 'Defense', 'Sp. Def', 'Speed', 'Attack', 'Sp. Atk']
    counter = 0
    #axes[0,1].tick_params(labelleft=False)
    for row in range(2):
        for col in range(3):
            sns.boxplot(y = y_attr, x = poke_attributes[counter], data=input_df, ax=axes[row, col], showfliers=False)
            sns.swarmplot(y = y_attr, x = poke_attributes[counter], data=input_df, ax=axes[row, col], hue = hue_attr, palette=['black', 'orange'])
            axes[row, col].set_title(axes[row, col].get_xlabel())
            axes[row, col].xaxis.label.set_visible(False)
            axes[row, col].get_legend().remove()
            counter += 1
            if (col == 0):
                axes[row, col].yaxis.label.set_visible(False)
                continue
            axes[row, col].yaxis.set_visible(False)
    return(f, axes)


# In[ ]:


(f, axes) = attr_per_type(my_pokemon, 'Type 1', 'Legendary')
_ = axes[0, 1].annotate('Shuckle the defense \nmonster', xy=(230, 3.2), xytext = (140, 5), arrowprops=dict(facecolor='blue'))
_ = axes[0, 2].annotate('Ho-oh so much \nspecial defence', xy=(154, 1.1), xytext=(150, 2.8), arrowprops=dict(facecolor='red'))
_ = axes[1, 2].annotate('Mewtwo destroyer \nof worlds', xy=(155, 8.8), xytext= (100, 8), arrowprops=dict(facecolor='purple'))


# From the above chart we can draw some conclusions:
# * Defensive attributes (HP, Def and Sp. Def) have many more outliers compared to offensive attributes (Speed, Atk and Sp. Atk) such as Cloyster, Chansey, Blissey, Forretress etc.
# * Steel, rock and ground pokemon tend to have higher defense
Alright this looks much nicer! The problem with the current dataset is that it includes lower evolution pokemon.  For example Charmander, Charmeleon and Charizard are all included. I would prefer it if only last evolution pokemon where included because the selection is mostly done based on those.  
Let's try to deal with this problem by webscraping the pokemon evolution tree from the pokemondb website. I am going to break down each step so its more understandable.
# ### Web scraping to improve our dataset <a id="7"></a>
# 
# *Step 1: Finding the URL*
# 
# The website I chose (granted I could have found an easier one but where is the fun in that) is the pokemondb website which lists the evolutions of all pokemon in this form:
# [![Pika.png](https://i.postimg.cc/qR8yX0qW/Pika.png)](https://postimg.cc/CBxRS3j4)
# 

# In[ ]:


poke_url = 'https://pokemondb.net/evolution#evo-g2'
response = get(poke_url) 
html_soup = BeautifulSoup(response.text, 'html.parser') ## Getting the html into python


# *Step 2: Examining the HTML of the page and extracting the data*
# 
# Now that we got the HTML we have to examine where exactly the information that we want is located. We can use the developer tools in Google Chrome to see which part of the HTML tree is of interest to us.
# 
# [![Pika-HTML.png](https://i.postimg.cc/qRf8WwFf/Pika-HTML.png)](https://postimg.cc/d7Bky2vn)
# 
# We can see that the information we want (the pokemon evolution tree) is in the Divs with class 'infocard-list-evo', so let's get that. This will also include cases like the one in the example where there are 'complications' in the evolution tree.

# In[ ]:


infocard = html_soup.find_all('div', {'class' : 'infocard-list-evo'})


# *Step 3: Locating and extracting all the useful information in the HTML* 
# 
# Now that we have selected all the divs we need to extract the information that is useful for us. I would like to get the following information:
# 1. The evolution is the pokemon (1st, 2nd, etc.)
# 2. The pokemon number (we are going to need that in the future for filtering)
# 3. The small text under the pokemon (again this will be used for filtering)
# 4. The pokemon name
# 
# All of the information that we want can be found under the span with class 'infocard-lg-data text-muted' as can be seen in the screenshot below:
# [![Info-Pic.png](https://i.postimg.cc/yxh1vgJp/Info-Pic.png)](https://postimg.cc/643xQQfC)
# 
# We just need to extract the first small (pokemon number), the second small (pokemon name) and the a tag with class 'ent-name' which has the small text under the pokemon. 
# 
# I also use enumerate in order to create an index for the evolution of the pokemon. So for example in the simple cases (Charmander -> Charmeleon -> Charizard) with enumerate I manage to give Charmander the number 1, Charmeleon the number 2 and Charizard the number 3. 
# 
# Unfortunately, this fails miserably for the complex cases (like the one of Pikatchu) but we will deal with this at a later step. 

# In[ ]:


raw = []
for chain in infocard:
    raw.append(([(i, x.find('small').getText(), x.find_all('small')[1].getText(), x.find('a', {'class':'ent-name'}).getText(), ) for i, x in enumerate(list(chain.find_all('span', {'class':'infocard-lg-data text-muted'})), 1)]))

raw[:4] ## Our list contains touple lists with the evolution branch of each pokemon. We keep that in mind for our future processing


# *Step 4: Dataset cleaning*
# 
# Now that we have extracted the data, we definitely need to do some filtering:
# * Remove all the pokemon that have 'alolan' in their name as these are from newer generations I assume
# * Remove all pokemon with a number higher than 251, as these are again from newer generations
# * Remove all remaining entries that have 1 or less entries from the list

# In[ ]:


#And what better way to do all this than list comprehensions! 
step_1 = [[elem for elem in branch_list if 'Alol' not in elem[2]] for branch_list in raw] ## Remove all touples that contain 'Alol' in the small text (removing the second Raichu in our example)
step_2 = [[elem for elem in branch_list if int(elem[1][1:]) <= 251] for branch_list in step_1] ## Remove all pokemon that are after Gen 2 (Number > 251) 
step_3 = [elem for elem in step_2 if len(elem) > 1] ##Remove entries that have a length of 1 or less (remove leftovers that are not actual branches in Gens 1 & 2)


# *Step 5: Enhancing our existing dataset*
# 
# For the analysis I would like to be able to tell which pokemon are the last evolution of each branch and which pokemon do not have an evolution (and therefore were not included in the list at all). 
# For example: Charmander is the first in the Charmander-Charmeleon-Charizard evolution tree and I would like to exclude him from the analysis. Snorlax on the other hand has no evolution and I would like to include him in the analysis
# 
# Therefore, I would like to create 2 lists, one with all the pokemon that belong to an evolution branch and one with pokemon that are the last of an evolution branch.

# In[ ]:


has_evolution = [[elem[3] for elem in branch_list] for branch_list in step_3] ## Extract all pokemon that in the raw list
has_evolution = sum(has_evolution, []) ## 'Un-tupple' them -- This is not the optimal way to do this but I find it really really cool and for such a small list it doesn't matter


# In[ ]:


last_evolutions  = [x[-1] for x in step_3] ## Take the last entry into each element list
last_evolutions.extend([x[-2] for x in step_3 if x[-1][0] == 4]) # The ones that have 2 end evolutions like Politoed and Poliwrath

## This is the eevee and hitmontop section as they have 5 and 3 final evolutions respecitvely
eevee_and_hitmon = [x[1:] for x in step_3 if (x[0][3] == 'Tyrogue') or (x[0][3] ==  'Eevee')]
eevee_and_hitmon = sum(eevee_and_hitmon, [])
last_evolutions.extend(eevee_and_hitmon)

last_evo_df = pd.DataFrame([[int(x[1][1:]), x[3]] for x in last_evolutions], columns=['Number', 'Pokemon'] )
last_evo_df = last_evo_df.drop_duplicates() ## Remove duplicates because some hitmontop and eevee evolutions have been added more than once in the process


# We are at the end of the process, we have a list of pokemon that belong to an evolution branch and we also have a list of pokemon that are the last evolutions. Let's go ahead and update our initial dataset

# In[ ]:


my_pokemon['Has_evol'] = my_pokemon.Name.isin(has_evolution)
my_pokemon['Last_evol'] = my_pokemon.Name.isin(last_evo_df.Pokemon)
my_evolved_pokemon = my_pokemon.loc[(my_pokemon.Last_evol == True) | (my_pokemon.Has_evol == False)].copy()


# ### Attributes per pokemon and type for only last evolution pokemon <a id="8"></a>

# In[ ]:


(f, axes) = attr_per_type(my_evolved_pokemon, 'Type 1', 'Legendary')
_ = f.suptitle('Attributes broken down by type for last evolution pokemon', va='center', y=0.92)


# This is much more representative! These are the actual pokemon that might be used in a pokemon battle as having an evolution means that your stats will increase and you will have access to a wider move pool.

# ### Addressing the elephant in the room and dealing with dual types <a id="9"></a>
# 
# While this chart started out nicely, we have narrowed the list of pokemon so much that now the boxplot does not make sense. In very few types is there a big enough number of pokemon to justify it. 
# 
# The logical follow-up would be to switch to a different type of chart. Before we do that though we should consider whether using the second pokemon type could help (a bit).
# From my online search I concluded that there is no discernible difference between a pokemon that is Type A/Type B compared to a pokemon that is Type B/Type A. 
# Which means that we could allocate pokemon of dual types to both of their types. For example, Charizard could belong to both Fire (his type 1) and Flying (his type 2)

# #### A brief overview of dual types in the dataset <a id="10"></a>

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
mpl.rcParams["font.size"] = 12

ax1 = my_evolved_pokemon.groupby('Type 2').size().sort_values().plot(kind='barh', ax=ax1)
start, end = ax1.get_xlim()
ax1.xaxis.set_ticks(np.arange(start, end, 5))
ax1.axes.get_yaxis().get_label().set_visible(False)
_ = ax1.set_title('Type 2 frequency')

pivot_dat = my_evolved_pokemon.pivot_table(index='Type 1', columns='Type 2', aggfunc = 'size')
pivot_dat = pivot_dat[pivot_dat.sum().sort_values(ascending = False).index] #Reordering
ax2 = sns.heatmap(pivot_dat, annot=True, square=True, cmap=sns.color_palette("Blues"), cbar=False, linewidths=0.3, ax=ax2)
_ = ax2.set_title('Type 1 - Type 2 pairs')
fig.tight_layout()


# We can observe the following from the two charts above:
# * Flying is the most popular type 2 by far
# * Normal-Flying is the most popular combination, with 4 pairs coming in second with 3 occurences Bug-Flying, Bug-Poison, Grass-Poison and Water-Ice
# * Water as Type 1 has the most pokemon with a second type and also the most variety

# ### Attributes per pokemon and type for last evolution pokemon taking into account their dual type <a id="11"></a>
# 
# Now what I would like to do is to create a new dataset where each dual type pokemon is represented in both types that it belongs to.

# In[ ]:


my_evolved_pokemon['New type'] = my_evolved_pokemon['Type 1']
my_evolved_pokemon['Dual type'] = ~pd.isnull(my_pokemon['Type 2'])
dual_evolved_pokemon = my_evolved_pokemon.append(my_evolved_pokemon[my_evolved_pokemon['Dual type']==True], ignore_index=True)

#Change the second instance of the pokemon to have the second type
dual_evolved_pokemon.loc[dual_evolved_pokemon.duplicated(), 'New type'] = dual_evolved_pokemon.loc[dual_evolved_pokemon.duplicated(), 'Type 2'] 


# In[ ]:


(f, axes) = attr_per_type(dual_evolved_pokemon, 'New type', 'Dual type')
_ = f.suptitle("Attributes broken down by type (Yellow indicates dual type)", va='center', y=0.92)


# This figure concludes my experimentation with this chart type. We can get some useful information from this chart:
# * A new type! Flying apparently only comes as a type 2!
# * Ice, Dragon, Steel and Flying come only in dual forms. The pokemon that have one of these types have a second one as well
# * Bug and ghost pokemon have low HP

# ## Attributes of the average pokemon of each type <a id="12"></a>
# While this has been nice, what I would like to see is a spider chart for the "average" pokemon of each type for last evolution pokemon. I think that would pronounce the differences more. Let's try it out!

# In[ ]:


def create_radar_chart(input_df, group_col, col_num):
    poke_stats = input_df.loc[:, 'HP':'Speed']
    norm_poke_stats = poke_stats / poke_stats.max() #Let's normalise the data 
    radar_dat = pd.concat([norm_poke_stats, input_df.loc[:, [group_col, 'Name']]], axis=1)
    radar_dat_short = radar_dat.groupby([group_col]).mean().loc[:,['HP', 'Defense', 'Sp. Def', 'Speed', 'Sp. Atk', 'Attack']]
    radar_dat_short.columns = ['HP', 'Def', 'Sp. Def', 'Speed', 'Sp. Atk', 'Atk']

    fig = make_subplots(rows=4, cols=col_num, vertical_spacing = 0.08, horizontal_spacing=0.04, specs=[[{"type": "polar"}]*col_num]*4,  subplot_titles=radar_dat_short.index.values)
    chart_counter = 0

    for type in radar_dat_short.index.values:

        fig.add_trace(go.Scatterpolar(
              r=radar_dat_short.loc[type].values,
              theta=list(radar_dat_short.loc[type].index),
              fill='toself',
              name=type),
              row=chart_counter // col_num + 1, col=chart_counter % col_num + 1
        )
        chart_counter += 1

    initial_elements = [elem*col_num+1 for elem in range(4)]    
    polar_initial_list = ['polar' + str(num) for num in initial_elements]
    polar_rest_list = ['polar' + str(num) for num in list(set(range(1,18)) - set(initial_elements))]

    polar_initial = {pol: dict(
                radialaxis=dict(
                  visible=False,
                  range=[0, 1]
                ),

                angularaxis = dict(
                    #categoryorder = ['Defense', 'Attack', 'Sp. Def', 'Sp. Atk', 'Speed', 'HP']
                     categoryorder = 'array',
                    categoryarray = ['HP', 'Def', 'Sp. Def', 'Speed', 'Sp. Atk', 'Atk'],
                    rotation = 90
                )
                  ) for pol in polar_initial_list}

    polar_rest = {pol: dict(
                radialaxis=dict(
                  visible=False,
                  range=[0, 1]
                ),

                angularaxis = dict(
                    #categoryorder = ['Defense', 'Attack', 'Sp. Def', 'Sp. Atk', 'Speed', 'HP']
                     categoryorder = 'array',
                    categoryarray = ['HP', 'Def', 'Sp. Def', 'Speed', 'Sp. Atk', 'Atk'],
                    showticklabels=False,
                    rotation = 90
                )
                  ) for pol in polar_rest_list}

    polar_dict = {**polar_initial, **polar_rest}
    fig.update_layout(
              polar_dict,
              height = 800,
              width = 800,
              font=dict(size=8),
              showlegend=False,
              title={'text': "Average stats per pokemon type",
                     'font_size':15,
                     'y':0.95,
                     'x':0.5,
                     'xanchor': 'center',
                     'yanchor': 'top'}
              )
    
    fig_counter = 1
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12)
        if fig_counter in initial_elements:
            i['height'] = 35
        else:
            i['height'] = 15
        fig_counter += 1
    return(fig)
    #fig['layout'].xaxis1[{'automargin' : 'False'}]
    #fig.show()


# In[ ]:


f = create_radar_chart(my_evolved_pokemon, 'Type 1', 4)
f.show()


# In this chart the differences between types are more pronounced. We can clearly see which are the types that are on average strong and in which attribute:
# * Fighting, Dragon (This is kinda cheating as the only last evolution Dragon pokemon is Dragonite but I will let it slide) and rock are high on Attack
# * Ghost and Electric are good on Sp. Atk and Speed
# * Steel are the ones with the highest defense 
# * No type is that great in Sp. Def and HP mainly because we have some distinct outliers (Blissey, Shuckle etc.)
# 
# Let's try to have the same chart but with the dual types included to see how and if it changes

# ### Last evolution pokemon taking into account their dual type <a id="13"></a>

# In[ ]:


f = create_radar_chart(dual_evolved_pokemon, 'New type', 5)
f.show()


# Overall, the types look more rounded in this view as we have included more pokemon into each type subset.
# We also managed to make the Dragon type look more down to earth now (Kingdra was added to the mix!).

# ## Next steps <a id="14"></a>
# 
# This was the first take with this dataset. Possible extensions of the analysis could include:
# * Which type is more rounded and which is more polarised (from the looks of it Fighting!)
# * Pokemon clustering based on their stats! Are there pokemon whose stats indicate that they probably should belong to a different type?
# * Type weaknesses, which types are the more versatile, which types have the most weaknesses etc.
# * Optimal selection, which pokemon to choose in a N pokemon team
# 
# ### Thats a wrap for now! Thank you for reading through this kernel and I hope you liked it!!
