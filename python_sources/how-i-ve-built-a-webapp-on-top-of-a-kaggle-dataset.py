#!/usr/bin/env python
# coding: utf-8

# # [**Fifafinder - Click here to try the app**](https://fifafinder.pythonanywhere.com)

# In this notebook I'll show you how I've built my own fifa player browser, called "Fifafinder" (what an original name!) using this [**kaggle FIFA 19 complete player dataset**](https://www.kaggle.com/karangadiya/fifa19) and the python package [**Flask**](http://flask.pocoo.org/docs/1.0/).
# 
# This is an app without any pretention, if you are familiar with Flask you'll find it very simple as well as the code. However, if you don't know the basics of web developpement, you may be surprised by how simple it is to build an app like this, by learning only some HTML, CSS and Javascript.

# ![](http://nightdeveloper.net/wp-content/uploads/2014/05/flask-1024x400.png)

# This website is a modest implementation of a fifa browser, obviously less complete than the leaders of the sector which are [Futbin](https://www.futbin.com/), [Sofifa](https://sofifa.com/[](http://) or others, but still functional and which combines harmoniously the main statistics about a player.

# ## **How the application works**

# The goal is to display essential informations about a player quickly, searching the player by his name. We will have two pages on our site : 
# - the homepage will display a searchbar with an advanced autocomplete function to allow the user to find as quickly as possible the player he is looking for
# - the playerpage will display the main statistics of the player

# The backend of the site is written in Python, as we are using the Flask framework. Basically, the user will use the search bar of the home page to type the name of the player he is looking for :

# ![](https://i.imgur.com/YtLtx69.jpg)

# Then, the homepage will pass the playername to the playerpage python backend function. This function will call the following "utils" function to get players attributes as a JSON object. Then, the JSON object is passed to the HTML frontend :

# In[ ]:


import pandas as pd

def get_player_attributes(playername):
    df = pd.read_csv('../input/data.csv')
    df = df[df.drop(
        ['Value', 'Wage', 'Special', 'Joined', 'Loaned From', 'Contract Valid Until', 'LS', 'ST', 'RS',
         'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',
         'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', ], 1).columns.tolist()[2:-1]]

    temp_wr = df['Work Rate'].str.split('/', expand=True)
    df['Work Rate'] = temp_wr[0] + ' / ' + temp_wr[1]

    temp_pic = df['Photo'].str.split('players/4', expand=True)
    df['Photo'] = temp_pic[0] + 'players/10' + temp_pic[1]

    temp_logo = df['Club Logo'].str.split('teams/2', expand=True)
    df['Club Logo'] = temp_logo[0] + 'teams/10' + temp_logo[1]

    return df[df['Name'] == playername].to_dict('r')[0]


# In[ ]:


get_player_attributes('P. Pogba')


# Finally, the HTML page will display the attributes in a stylised way, using CSS for styling and Javascript for animation. The radar chart is made using the great [**Chart.js**](https://www.chartjs.org/) library. Here is the rendering :

# ![](https://i.imgur.com/hOirmQM.png)

# ## **Conclusion**

# This notebook is just a quick introduction of how I built this app, I didn't went into the details of the code as this is not a datascience project. Moreover, most of the code is  written in Javascript, CSS and HTML. However, I think a data scientist must be also capable of building small functionnal applications for selling his models and justify of his business & marketing skills which are often neglicted if the field. To do so, having the basic knowledge and some beginner skills of web developpement seems to be quite an advantage. This is the main purpose of this kernel: showing that it is possible and not very complicated to build a small app like this, without a strong knowledge of web developpement.
# 
# Thanks for reading ! The code is available here on my [**github profile**](https://github.com/nicohlr), don't hesitate to explore it, to star it and to contribute if you liked the project.
