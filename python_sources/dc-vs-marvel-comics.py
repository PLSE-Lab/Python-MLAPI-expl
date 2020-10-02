#!/usr/bin/env python
# coding: utf-8

# ![](https://res.cloudinary.com/jerrick/image/upload/f_auto,fl_progressive,q_auto,c_fit,w_1024/peisbhgtuumpivgz5zbs)

# # DC Comics, Inc. is an American comic book publisher. It is the publishing unit of DC Entertainment, a subsidiary of Warner Bros. since 1967. DC Comics is one of the largest and oldest American comic book companies, and produces material featuring numerous culturally iconic heroic characters including: Superman, Batman, Wonder Woman, The Flash, Green Lantern, Martian Manhunter, Nightwing, Green Arrow, Starfire, Aquaman, and Cyborg.
# 
# # Most of their material takes place in the fictional DC Universe, which also features teams such as the Justice League, the Justice Society of America, the Suicide Squad, and the Teen Titans, and well-known villains such as The Joker, Lex Luthor, Catwoman, Darkseid, Sinestro, Brainiac, Black Adam, Ra's al Ghul and Deathstroke. The company has also published non-DC Universe-related material, including Watchmen, V for Vendetta, and many titles under their alternative imprint Vertigo.
# 
# # The initials "DC" came from the company's popular series Detective Comics, which featured Batman's debut and subsequently became part of the company's name. Originally in Manhattan at 432 Fourth Avenue, the DC Comics offices have been located at 480 and later 575 Lexington Avenue; 909 Third Avenue; 75 Rockefeller Plaza; 666 Fifth Avenue; and 1325 Avenue of the Americas. DC had its headquarters at 1700 Broadway, Midtown Manhattan, New York City, but it was announced in October 2013 that DC Entertainment would relocate its headquarters from New York to Burbank, California in April 2015.
# 
# # Random House distributes DC Comics' books to the bookstore market,while Diamond Comic Distributors supplies the comics shop specialty market.DC Comics and its longtime major competitor Marvel Comics (acquired in 2009 by The Walt Disney Company, WarnerMedia's main competitor) together shared approximately 70% of the American comic book market in 2017.

# # Marvel Comics is the brand name and primary imprint of Marvel Worldwide Inc., formerly Marvel Publishing, Inc. and Marvel Comics Group, a publisher of American comic books and related media. In 2009, The Walt Disney Company acquired Marvel Entertainment, Marvel Worldwide's parent company.
# 
# # Marvel started in 1939 as Timely Publications, and by the early 1950s, had generally become known as Atlas Comics. The Marvel branding began in 1961, the year that the company launched The Fantastic Four and other superhero titles created by Steve Ditko, Stan Lee, Jack Kirby and many others.
# 
# # Marvel counts among its characters such well-known superheroes as Spider-Man, Iron Man, Captain America, Thor, the Hulk, Captain Marvel, Black Panther, Deadpool, Silver Surfer, Doctor Strange, Wolverine, Daredevil, Ghost Rider and the Punisher, such teams as the Avengers, the X-Men, the Fantastic Four, the Inhumans and the Guardians of the Galaxy, and supervillains including Thanos, Doctor Doom, Magneto, Red Skull, Green Goblin, Ultron, Doctor Octopus, Loki, Galactus, and Venom. Most of Marvel's fictional characters operate in a single reality known as the Marvel Universe, with most locations mirroring real-life places; many major characters are based in New York City.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)  
import plotly.figure_factory as ff
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dc = pd.read_csv('../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv')
dc.head()


# In[ ]:


marvel = pd.read_csv('../input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv')
marvel.head()


# # Comparisions of DC and Marvel Comics characters

# In[ ]:


sex_count = dc['SEX'].value_counts()
sex1_count = marvel['SEX'].value_counts()
trace1 = go.Bar(
    x=sex_count.index,
    y=sex_count.values,
    name='DC'
)
trace2 = go.Bar(
    x=sex1_count.index,
    y=sex1_count.values,
    name='Marvel'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title= 'Gender Comparisions in between DC and Marvel'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')


# In[ ]:


sex_count = dc['ID'].value_counts()
sex1_count = marvel['ID'].value_counts()
trace1 = go.Bar(
    x=sex_count.index,
    y=sex_count.values,
    name='DC'
)
trace2 = go.Bar(
    x=sex1_count.index,
    y=sex1_count.values,
    name='Marvel'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title= 'Identity comparisions in between DC and Marvel'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')


# In[ ]:


sex_count = dc['ALIGN'].value_counts()
sex1_count = marvel['ALIGN'].value_counts()
trace1 = go.Bar(
    x=sex_count.index,
    y=sex_count.values,
    name='DC'
)
trace2 = go.Bar(
    x=sex1_count.index,
    y=sex1_count.values,
    name='Marvel'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title= 'How many good and bad characters in between DC and Marvel?'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')


# In[ ]:


sex_count = dc['ALIVE'].value_counts()
sex1_count = marvel['ALIVE'].value_counts()
trace1 = go.Bar(
    x=sex_count.index,
    y=sex_count.values,
    name='DC'
)
trace2 = go.Bar(
    x=sex1_count.index,
    y=sex1_count.values,
    name='Marvel'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title= 'Alive or Dead ?'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')


# # Appearances with respect to Origin year of the characters 

# In[ ]:


trace0 = go.Bar(
    x= dc.YEAR,
    y= dc.APPEARANCES,
    name='DC Characters',
    text= dc.name,
    marker=dict(
        color='rgb(49,130,189)'
    )
)
trace1 = go.Bar(
    x= marvel.Year,
    y= marvel.APPEARANCES,
    name='Marvel Characters',
    text= marvel.name,
    marker=dict(
        color='rgb(204,204,204)',
    )
)

data = [trace0, trace1]
layout = go.Layout(
    xaxis=dict(tickangle=-45,
              title='Year'),
    yaxis=dict(title='Appearances'),
    title='Appearances with respect to Origin year Bar Plot',
    barmode='group',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='angled-text-bar')


# In[ ]:


trace_high = go.Scatter(
                x=marvel.Year,
                y=marvel.APPEARANCES,
                name = "Marvel Appearances",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

trace_low = go.Scatter(
                x=dc.YEAR,
                y=dc.APPEARANCES,
                name = "DC Appearances",
                line = dict(color = '#7F7F7F'),
                opacity = 0.8)

data = [trace_high,trace_low]

layout = dict(
    title='Appearances with respect to Origin year',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1Y',
                     step='year',
                     stepmode='backward'),
                dict(count=6,
                     label='6Y',
                     step='year',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "Time Series with Rangeslider")


# ![](http://hdqwalls.com/wallpapers/batman-vs-joker-a1.jpg)

# In[ ]:


dc_top = dc.iloc[dc.groupby(dc['ALIGN'])['APPEARANCES'].idxmax()][['name', 'ALIGN']]


# # Top appearances in alignment of the characters

# In[ ]:


dc_top


# **Batman's secret identity is Bruce Wayne, a wealthy American playboy, philanthropist, and owner of Wayne Enterprises. After witnessing the murder of his parents Dr. Thomas Wayne and Martha Wayne as a child, he swore vengeance against criminals, an oath tempered by a sense of justice. Bruce Wayne trains himself physically and intellectually and crafts a bat-inspired persona to fight crime.**
# 
# **Batman operates in the fictional Gotham City with assistance from various supporting characters, including his butler Alfred, police commissioner Gordon, and vigilante allies such as Robin. Unlike most superheroes, Batman does not possess any superpowers; rather, he relies on his genius intellect, physical prowess, martial arts abilities, detective skills, science and technology, vast wealth, intimidation, and indomitable will. A large assortment of villains make up Batman's rogues gallery, including his archenemy, the Joker.**
# 
# **Originally introduced as a mad scientist whose schemes Superman would routinely foil, Lex's portrayal has evolved over the years and his characterisation has deepened. In contemporary stories, Lex is portrayed as a wealthy, power-mad American business magnate, ingenious engineer, philanthropist to the city of Metropolis, and one of the most intelligent people in the world. A well-known public figure, he is the owner of a conglomerate called LexCorp. He is intent on ridding the world of the alien Superman, whom Lex Luthor views as an obstacle to his plans and as a threat to the very existence of humanity. Given his high status as a supervillain, he has often come into conflict with Batman and other superheroes in the DC Universe.**
# 
# **The character has traditionally lacked superpowers or a dual identity and typically appears with a bald head. He periodically wears his Warsuit, a high-tech battle suit giving him enhanced strength, flight, advanced weaponry, and other capabilities. The character was originally introduced as a diabolical recluse, but during the Modern Age, he was reimagined by writers as a devious, high-profile industrialist, who has crafted his public persona in order to avoid suspicion and arrest. He is well known for his philanthropy, donating vast sums of money to Metropolis over the years, funding parks, foundations, and charities.**

# ![](https://www.bleedingcool.com/wp-content/uploads/2017/06/BatLex.jpg)

# In[ ]:


dc_alive = dc.iloc[dc.groupby(dc['ALIVE'])['APPEARANCES'].idxmax()][['name', 'ALIVE']]


# # Top appearances depending on whether they continue to exist

# In[ ]:


dc_alive


# **Alan Scott was created after Nodell became inspired by the characters from Greek and Norse myths, seeking to create a popular entertainment character who fought evil with the aid of a magic ring which grants him a variety of supernatural powers. After debuting in All-American Comics, Alan Scott soon became popular enough to sustain his own comic book, Green Lantern. Around this time DC also began experimenting with fictional crossovers between its characters, leading towards a shared universe of characters. As one of the publisher's most popular heroes, Alan became a founding member of the Justice Society of America, one of the first such teams of "mystery men" or superheroes in comic books.**

# ![](https://static.comicvine.com/uploads/scale_small/10/100647/5516293-society3.jpg)

# In[ ]:


marvel_top = marvel.iloc[marvel.groupby(marvel['ALIGN'])['APPEARANCES'].idxmax()][['name', 'ALIGN']]


# In[ ]:


marvel_top


# **Doctor Victor Von Doom is a fictional supervillain appearing in American comic books published by Marvel Comics. Created by writer-editor Stan Lee and artist/co-plotter Jack Kirby, the character made his debut in The Fantastic Four #5 (July 1962). The Monarch of the fictional nation Latveria, Doom is usually depicted as the archenemy of Reed Richards and the Fantastic Four, though he has come into conflict with other superheroes as well, including Spider-Man, Iron Man, Black Panther, and the Avengers.**
# 
# **Spiderman first appeared in the anthology comic book Amazing Fantasy #15 (August 1962) in the Silver Age of Comic Books. He appears in American comic books published by Marvel Comics, as well as in a number of movies, television shows, and video game adaptations set in the Marvel Universe. In the stories, Spider-Man is the alias of Peter Parker, an orphan raised by his Aunt May and Uncle Ben in New York City after his parents Richard and Mary Parker were killed in a plane crash. Lee and Ditko had the character deal with the struggles of adolescence and financial issues, and accompanied him with many supporting characters, such as J. Jonah Jameson, Flash Thompson, Harry Osborn, romantic interests Gwen Stacy and Mary Jane Watson, and foes such as Doctor Octopus, Green Goblin and Venom. His origin story has him acquiring spider-related abilities after a bite from a radioactive spider; these include clinging to surfaces, shooting spider-webs from wrist-mounted devices, and detecting danger with his "spider-sense".**
# 
# **Wolverine is a mutant who possesses animal-keen senses, enhanced physical capabilities, powerful regenerative ability known as a healing factor, and three retractable claws in each hand. Wolverine has been depicted variously as a member of the X-Men, Alpha Flight, and the Avengers.**

# ![](https://static.comicvine.com/uploads/original/14/145389/3673655-2606956533-wolve.jpg)

# In[ ]:


marvel_alive = marvel.iloc[marvel.groupby(marvel['ALIVE'])['APPEARANCES'].idxmax()][['name', 'ALIVE']]


# In[ ]:


marvel_alive


# **Xavier is a member of a subspecies of humans known as mutants, who are born with superhuman abilities. The founder of the X-Men, Xavier is an exceptionally powerful telepath who can read and control the minds of others. To both shelter and train mutants from around the world, he runs a private school in the X-Mansion in Salem Center, located in Westchester County, New York. Xavier also strives to serve a greater good by promoting peaceful coexistence and equality between humans and mutants in a world where zealous anti-mutant bigotry is widespread.**
# 
# **Throughout much of the character's history in comics, Xavier is a paraplegic variously using either a wheelchair or a modified version of one. One of the world's most powerful mutant telepaths, Xavier is a scientific genius and a leading authority in genetics. Furthermore, he has shown noteworthy talents in devising equipment to greatly enhance psionic powers. Xavier is perhaps best known in this regard for the creation of a device called Cerebro, a technology that serves to detect and track those individuals possessing the mutant gene, at the same time greatly expanding the gifts of those with existing psionic abilities.**

# ![](https://www.writeups.org/wp-content/uploads/Professor-X-Marvel-Comics-X-Men-Charles-Xavier-g.jpg)

# In[ ]:


dc['name'] = dc['name'].replace({'Earth':' ', 'earth':' '})


# In[ ]:


from PIL import Image

d = np.array(Image.open('../input/comic-pict/images (17).jpeg'))


# In[ ]:


DC_DA = ' '.join(dc['name'].tolist())


# In[ ]:


DC_DAA = "".join(str(v) for v in DC_DA).lower()


# In[ ]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=d,background_color="white").generate(DC_DAA)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Popular Names of DC',size=24)
plt.show()


# In[ ]:


from PIL import Image

m = np.array(Image.open('../input/comic-pict/images (18).jpeg'))


# In[ ]:


M_DA = ' '.join(marvel['name'].tolist())


# In[ ]:


M_DAA = "".join(str(v) for v in M_DA).lower()


# In[ ]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
sns.set(rc={'figure.figsize':(11.7,8.27)})

wordcloud = WordCloud(mask=m,background_color="white").generate(M_DAA)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Popular Names of Marvel',size=24)
plt.show()


# In[ ]:


dc['comics']= 'DC'


# In[ ]:


dc= dc.truncate(before=-1, after=20)


# In[ ]:


import networkx as nx
FG = nx.from_pandas_edgelist(dc, source='comics', target='name', edge_attr=True,)


# # Top 20 characters of DC

# In[ ]:


nx.draw_networkx(FG, with_labels=True)


# In[ ]:


marvel['comics'] = 'Marvel'


# In[ ]:


marvel = marvel.truncate(before=-1, after=20)


# In[ ]:


import networkx as nx
FG1 = nx.from_pandas_edgelist(marvel, source='comics', target='name', edge_attr=True,)


# # Top 20 characters of Marvel

# In[ ]:


nx.draw_networkx(FG1, with_labels=True)


# # Upvote If You liked i'll update more
