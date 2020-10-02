#!/usr/bin/env python
# coding: utf-8

# ### My First Project

# ![](https://i.ibb.co/m8ZdtqY/6d77a81a-324c-4732-810e-c83a2570b287-FIFA19-5.jpg)

# This is the first time I upload something to this page, I recently started studying python and data science so I would appreciate any comments that help me improve the code and performance.
# My idea for this project was to take the statistics of the players, unite them, and create new data to be able to show in a polar type graphic, once the graphic was done I could use my time to focus on the format and the visualization. Hope you like.

# ### Libraries that I will use

# In[29]:


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import (OffsetImage,AnnotationBbox)
import pandas as pd
import numpy as np
from math import pi
import requests
from time import time
from datetime import datetime, timedelta
import random

get_ipython().run_line_magic('matplotlib', 'inline')


# ### I read the csv

# In[30]:


players = pd.read_csv('../input/data.csv')


# ### I eliminate the rows where the players do not have the club and the position. Then I complete the fields that are string with No Data and those that are numeric with a zero, finally I reset the index.

# In[31]:


players.drop('Unnamed: 0', axis=1, inplace=True)
players = players.dropna(subset=['Club', 'Position'])
players[['Release Clause','Loaned From','Joined']] = players[['Release Clause',
                                                              'Loaned From',
                                                              'Joined']].fillna('No data')
players = players.fillna(0)
players = players.reset_index()


# ### I create functions that are going to be responsible for taking data from several columns and get the percentage among all, thus creating new information
# 
# ### For example, the percentage of Marking, StandingTackle and SlidingTackle creates the defending data

# In[32]:


def defending(players):
    return int(round((players[['Marking', 'StandingTackle', 
                               'SlidingTackle']].mean()).mean()))

def general(players):
    return int(round((players[['HeadingAccuracy', 'Dribbling', 'Curve', 
                               'BallControl']].mean()).mean()))

def mental(players):
    return int(round((players[['Aggression', 'Interceptions', 'Positioning', 
                               'Vision','Composure']].mean()).mean()))

def passing(players):
    return int(round((players[['Crossing', 'ShortPassing', 
                               'LongPassing']].mean()).mean()))

def mobility(players):
    return int(round((players[['Acceleration', 'SprintSpeed', 
                               'Agility','Reactions']].mean()).mean()))

def power(players):
    return int(round((players[['Balance', 'Jumping', 'Stamina', 
                               'Strength']].mean()).mean()))

def rating(players):
    return int(round((players[['Potential', 'Overall']].mean()).mean()))

def shooting(players):
    return int(round((players[['Finishing', 'Volleys', 'FKAccuracy', 
                               'ShotPower','LongShots', 'Penalties']].mean()).mean()))


# ### I call each of the functions associating them to the corresponding column. These columns and the information are new.

# ### NEW CATEGORIES
# 
# New categories are created by joining player statistics and obtaining the average. For example, to obtain the strength of the player join the balance, jump, stamina and strength and the average of each of them is obtained a general average that transforms into the Power column
# 
# ### DIVISION OF CURRENT FEATURES TO GENERATE THE NEW COLUMNS
# 
# - Defending = [Marking, StandingTackle, SlidingTackle]
# - General = [HeadingAccuracy, Dribbling, Curve, BallControl]
# - Mental = [Aggression, Interceptions, Positioning, Vision, Composure]
# - Passing = [Crossing, ShortPassing, LongPassing]
# - Mobility = [Acceleration, SprintSpeed, Agility, Reactions]
# - Power = [Balance, Jumping, Stamina, Strength]
# - Rating = [Potential, Overall]
# - Shooting = [Finishing, Volleys, FKAccuracy, ShotPower, LongShots, Penalties]

# In[33]:


timeFinifih = 0

start_time = time()

players['Defending'] = players.apply(defending, axis=1)
players['General'] = players.apply(general, axis=1)
players['Mental'] = players.apply(mental, axis=1)
players['Passing'] = players.apply(passing, axis=1)
players['Mobility'] = players.apply(mobility, axis=1)
players['Power'] = players.apply(power, axis=1)
players['Rating'] = players.apply(rating, axis=1)
players['Shooting'] = players.apply(shooting, axis=1)

timeFinifih += (time() - start_time)

print('Ending - time: ' + str(timedelta(seconds=timeFinifih)))


# ### Now I rename a column so I can use it later

# In[34]:


players.rename(columns={'Club Logo':'Club_Logo'}, inplace=True)


# ### I create a new set of data only with the columns that I'm going to use

# In[35]:


data = players[['Name','Defending','General','Mental','Passing',
                'Mobility','Power','Rating','Shooting','Flag','Age',
                'Nationality', 'Photo', 'Club_Logo', 'Club']]


# In[36]:


data.head()


# ### TYPE OF GRAPHIC
# 
# - Matplotlib Polar
# 
# The graphic contains the name of the player, his age, the team where he currently plays, the nationality, the photo, the image of the club logo, the flag of the country where he was born and the statistics that were created with the function.
# 
# To see the information just pass the id to the graphPolar function, this calls the detail function by passing everything necessary to create the graph and visualize it. This function can be put inside a loop to bring all the players that are needed.
# 
# The dataset has 17918 so you can pass an id from 0 to 17917.
# 
# ### CREATION, DOWNLOAD AND VISUALIZATION OF IMAGES
# 
# 3 images will be downloaded: the flag of the country where the player was born, the logo of the team where he plays and a photo of his face (flag_image , player_image, logo_image). These images will be downloaded only once in the project folder, then they will be modified as the function is called again.

# ### This is the function that creates the graph

# In[37]:


### I receive the information of the graphPolar function (explained below), with this information I will create the graph

def detalle(row, title, image, age, nationality, photo, logo, club):

### I create variables with the names that the images will have after downloading when using the request. When the image of the player's face is downloaded it will be saved as img_player.jpg.

    flag_image = "img_flag.jpg"
    player_image = "img_player.jpg"
    logo_image = "img_club_logo.jpg"
    fondo_image = "img_fondo.jpg"
    url_fondo = "https://i.ibb.co/TMcGQXg/fondo.jpg"

### This variable contains the url of the image that I am going to use in the background of the graphic (FIFA)

    fondo_grafico = requests.get(url_fondo).content
    with open(fondo_image, 'wb') as handler:
        handler.write(fondo_grafico)
    
    fondo = mpimg.imread(fondo_image)

### I download the images of the player's face, the club where he currently plays and the flag of the country where he was born.

    img_flag = requests.get(image).content
    with open(flag_image, 'wb') as handler:
        handler.write(img_flag)
    
    player_img = requests.get(photo).content
    with open(player_image, 'wb') as handler:
        handler.write(player_img)
     
    logo_img = requests.get(logo).content
    with open(logo_image, 'wb') as handler:
        handler.write(logo_img)
        
    r = lambda: random.randint(0,255)
    colorRandom = '#%02X%02X%02X' % (r(),r(),r())

### I create a function that generates random colors to fill the graphic with new colors as the function is executed.

    if colorRandom == '#ffffff':
        colorRandom = '#a5d6a7'

### I declare colors that will always be used, one for the name of the different statistics and the name of the player and the other for the background color of each of the notes that I put on the graph

    basic_color = '#37474f'
    color_annotate = '#01579b'

### This is the image that will be used in the background, the flag.
### The image file to read. This can be a filename, a URL or a Python file-like object opened in read-binary mode.

    img = mpimg.imread(flag_image)

### I delimit the size of the graphic and the categories

    plt.figure(figsize=(15,8))
    categories=list(data)[1:]

### My data set has columns with data string so I do not need them for the graph so I create an array 
### with the name of the columns that I do not need so that when creating the N (categories to be plotted) 
### the function do not select the categories that are part of the array. When I get to it apart from getting the 
### values I also do the same thing, I get all the values of the columns that are different from the ones that 
### I declare in this array.

    coulumnDontUseGraph = ['Flag', 'Age', 'Nationality', 'Photo', 'Logo', 'Club']
    N = len(categories) - len(coulumnDontUseGraph)

### Basic configuration of the graph
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color= 'black', size=17)
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75,100], ["25","50","75","100"], color= basic_color, size= 10)
    plt.ylim(0,100)

### Values that will fill in the graph
    
    values = data.loc[row].drop('Name').values.flatten().tolist() 
    valuesDontUseGraph = [image, age, nationality, photo, logo, club]
    values = [e for e in values if e not in (valuesDontUseGraph)]
    values += values[:1]
    
    ax.plot(angles, values, color= basic_color, linewidth=1, linestyle='solid')
    ax.fill(angles, values, color= colorRandom, alpha=0.5)

### I put the background image of the graphic plane (FLAG)

    axes_coords = [0, 0, 1, 1]
    ax_image = plt.gcf().add_axes(axes_coords,zorder= -1)
    ax_image.imshow(img,alpha=0.5)
    ax_image.axis('off')

### I put the background image of the polar graph (FIFA)
### With ax.get_children () you get the list of objects that make up the plot,
### among which are the polygonal line of the data, text labels, etc. One of these objects 
### is of type matplotlib.patches.Wedge, which is a circular sector (or also a full circle as in this case). 
### Just extract that patch and use it to make the clip

    ax_fondo = plt.gcf().add_axes(axes_coords, zorder=1)
    ax_fondo.axis('off')
    fondo = ax_fondo.imshow(fondo, alpha=0.2)
    clip = [c for c in ax.get_children() if type(c) == matplotlib.patches.Wedge][0]
    fondo.set_clip_path(clip)

### I create notes that I am going to fill in with the age of the player, the nationality 
### and the club where he currently plays. I set its color of letter, background and font size 
###and in the position of the x-axis and y-axis they have to be
    
    ax.annotate('Nacionality: ' + nationality.upper(), xy=(10,10), xytext=(103, 138),
                fontsize= 12,
                color = 'white',
                bbox={'facecolor': color_annotate, 'pad': 7})
        
    ax.annotate('Age: ' + str(age), xy=(10,10), xytext=(43, 180),
                fontsize= 15,
                color = 'white',
                bbox={'facecolor': color_annotate, 'pad': 7})
    
    ax.annotate('Team: ' + club.upper(), xy=(10,10), xytext=(92, 168),
                fontsize= 12,
                color = 'white',
                bbox={'facecolor': color_annotate, 'pad': 7})

### I add the images of the player's face and the logo of the club where he currently plays    

    arr_img_player = plt.imread(player_image, format='jpg')

    imagebox_player = OffsetImage(arr_img_player)
    imagebox_player.image.axes = ax
    abPlayer = AnnotationBbox(imagebox_player, (0.5, 0.7),
                        xybox=(313, 223),
                        xycoords='data',
                        boxcoords="offset points"
                        )

    arr_img_logo = plt.imread(logo_image, format='jpg')

    imagebox_logo = OffsetImage(arr_img_logo)
    imagebox_logo.image.axes = ax
    abLogo = AnnotationBbox(imagebox_logo, (0.5, 0.7),
                        xybox=(-320, -226),
                        xycoords='data',
                        boxcoords="offset points"
                        )

    ax.add_artist(abPlayer)
    ax.add_artist(abLogo)

### Finally, I declare the title of the graphic, the size of the letter and the color

    plt.title(title, size=50, color= basic_color)


# ### This function receives the id of the player that I want to graph and sends the necessary data to the function that is responsible for creating the graph

# In[38]:


def graphPolar(id = 0):
    if 0 <= id < len(players.ID):
        detalle(row = data.index[id], 
                title = data['Name'][id], 
                age = data['Age'][id], 
                photo = data['Photo'][id],
                nationality = data['Nationality'][id],
                image = data['Flag'][id], 
                logo = data['Club_Logo'][id], 
                club = data['Club'][id])
    else:
        print('The base has 17917 players. You can put positive numbers from 0 to 17917')       


# ### Finally, just call the function and send the id of the player you want to see

# ### DATA VISUALIZATION

# In[39]:


graphPolar(11)


# In[40]:


graphPolar(0)


# In[41]:


graphPolar(1)


# ### You can create an array with players id and call the function in a loop to show several cards at once
# 
#     timeFinifih = 0
#     start_time = time()
# 
#     for x in range(0,3):
#         graphPolar(x)
#     
#     timeFinifih += (time() - start_time)
#     print('Ending - time: ' + str(timedelta(seconds=timeFinifih)))

# ### BONUS:
# 
# ### On a small change you can make the clip be about the data polygon
# 
# `clip = [c for c in ax.get_children() if type(c) == matplotlib.patches.Polygon][0]`
# 
# `im.set_clip_path(clip)`
# 
# ### You can even create two ax_image, put a different image to each one as well as a different clip_path, and play with your z-order, one image can go to the polygon and another to the circle
# 
# # Circle background
# 
# `im2 = ax_image2.imshow(img2, alpha=0.5)
# clip = [c for c in ax.get_children() if type(c) == matplotlib.patches.Wedge][0]
# im2.set_clip_path(clip)`
# 
# # Polygon background
# 
# `im = ax_image.imshow(img, alpha=.5)
# clip = [c for c in ax.get_children() if type(c) == matplotlib.patches.Polygon][0]
# im.set_clip_path(clip)`

# In[42]:


df = pd.DataFrame({
'Cero': ['Uno'],
'Uno': [20],
'Dos': [30],
'Tres': [40],
'Cuatro': [50],
'Cinco': [60],
'Seis': [70],
'Siete': [80],
'Ocho': [90],
'Nueve': [100]
})

def detalle():

    color = '#80cbc4'
    fondo_img = "fondo_test.jpg"
    circle_img = "circle_test.jpg"
    polygon_img = "polygon_test.jpg"

    url_fondo_img = 'https://i.ibb.co/TMcGQXg/fondo.jpg'
    url_circle_img = 'https://i.ibb.co/QCGF8wv/circle.jpg'
    url_polygon_img = 'https://i.ibb.co/L8xYT5c/polygon.jpg'

    fondo_img_url = requests.get(url_fondo_img).content
    with open(fondo_img, 'wb') as handler:
        handler.write(fondo_img_url)
        
    circle_img_url = requests.get(url_circle_img).content
    with open(circle_img, 'wb') as handler:
        handler.write(circle_img_url)
        
    polygon_img_url = requests.get(url_polygon_img).content
    with open(polygon_img, 'wb') as handler:
        handler.write(polygon_img_url)

    fondo = mpimg.imread(fondo_img)
    circle = mpimg.imread(circle_img)
    polygon = mpimg.imread(polygon_img)

    plt.figure(figsize=(15,8))

    categories=list(df)[1:]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, projection='polar')

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color= "black", size=10)
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75,100], ["25","50","75","100"], color="grey", size=8)
    plt.ylim(0,100)

    values=df.loc[0].drop('Cero').values.flatten().tolist()   
    values += values[:1]

    ax.plot(angles, values, color= 'black', linewidth=1, linestyle='solid')
    ax.fill(angles, values, color= 'green', alpha=0.5)

    axes_coords = [0, 0, 1, 1]
    ax_image = plt.gcf().add_axes(axes_coords,zorder= -1)
    ax_image.imshow(fondo,alpha=0.5)
    ax_image.axis('off')

    ax_image2 = plt.gcf().add_axes(axes_coords, zorder=1)
    ax_image2.axis('off')
    ax_image1 = plt.gcf().add_axes(axes_coords, zorder=2)
    ax_image1.axis('off')

    im = ax_image1.imshow(polygon)
    clip = [c for c in ax.get_children() if type(c) == matplotlib.patches.Polygon][0]
    im.set_clip_path(clip)
    
    im2 = ax_image2.imshow(circle, alpha=0.6)
    clip = [c for c in ax.get_children() if type(c) == matplotlib.patches.Wedge][0]
    im2.set_clip_path(clip)

    my_palette = plt.cm.get_cmap("Set2", len(df.index))


# In[43]:


detalle()

