#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import os

from PIL import Image

from ast import literal_eval

plt.rcParams['figure.figsize'] = [20, 10]

map_img = Image.open('/kaggle/input/lec-spring-2020-g2-esports-level-one-positions/lec.png')
map_img = map_img.resize((700, 700))


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# # Using Logistic Regression to identify patterns in professional League of Legends level 1 positions
# 
# League of Legends (LoL) is a 5v5 multiplayer game with a very prominent professional scene. Every year hundreds of teams around the globe compete with the goal of winning the World Championship held every Autumn. The use of data analytics by professional teams is a rather recent development as the scene continues to grow every year. The youth of this part of the industry means there is still a significant amount of room for new technologies and research into how data can give teams an edge. 
# 
# The aim of this notebook is to showcase one potential use case of data analytics based on position data of a LoL team. Data was gathered using a custom version of [this program](https://github.com/Steeeephen/loltracker), so there may be inconsistencies in the data itself. Considering the fact that this is more a proof of concept and exploration of the possibilities than in-depth statistical analysis, I'm not too concerned about the results themselves.
# 
# ***
# ## Intro
# 
# ### The Premise: Level Ones
# 
# What I'll set out to do is to use predictive modelling (Multinomial Logistic Regression) to try and compare how a given team changes their formation at the start of the game, in an attempt to find patterns that can be exploited. 
# 
# Each game of LoL starts with all 5 members of both teams spawning in their respective base. Each team has 5 disinct roles; Toplaner, Jungler, Midlaner, ADC and Support. They leave the base, spread out and have until the in-game clock hits 90 seconds before the first set of neutral minions spawn. This is generally referred to as the 'level one', as it occurs before any player has the chance to level up. Some teams will get creative and try to invade enemy territory to gain an advantage, but oftentimes teams will play a defensive formation and wait for the game to properly begin.
# 
# <img src="https://i.imgur.com/m6HJ09B.png" width=700/>
# 
# The goal of this model will be to think of the scenario where a team invades in a specific place at a specific time and finds an enemy player. Will they be alone? Who is it likely to be? If we specifically want to find one player and set them back at the start of the game, is there a certain way we should go about it?
# 
# ***
# ### The Case Study: G2 Esports
# 
# For the purposes of this notebook I chose the LEC Spring Champions, G2 Esports. I gathered the data on all of their games and used that to build the models. G2 have been serial winners for the past year and a half, losing just a single tournament in that timespan (including winning Europe's first major tournament in 8 years). Most of their losses come from falling very far behind in the first ~8-10 minutes of the game, meaning any early advantages you can gain over them are especially crucial
# 
# ***
# ### The Data: Positional Data from the Spring Season
# 
# The data consists of (x,y) coordinates along with a 'Seconds' value showing the in-game timer the datapoint was taken at. The champion played by each character is also recorded, but is not currently used in the model due to the high cardinality (nearly 150 possible champions)
# 
# The (x,y) coordinates relate to the player's position on the in-game map:
# 
# <img src="https://i.imgur.com/2P0mkLt.png" width=500/>
# 
# 
# ***
# ### The Method: Probabilities of a Logistic Regression Model
# 
# The idea is to build a statistical model that takes the data in and predicts what champion is likely to be found at a given spot at a given time.
# 
# That is to say, we build a model such that the probability of a given point + time being attributed to a given role can be represented as:
# 
# $$\large Probability(Y_i = Role_{j} \mid time_{i}, x_{i}, y_{i}) = \frac{e^{\beta_{j,0} + \beta_{j,1}time_i + \beta_{j,2}x_i + \beta_{j,3}y_i}}{1 + \sum_{k=1}^{4} e^{\beta_{k,0} + \beta_{k,1}time_i + \beta_{k,2}x_i + \beta_{k,3}y_i}}$$
# 
# We'll then build a grid that spans the map and apply our model to every point in the grid, for a given time. This will give us a list of 5 probabilities for each point on our map, one for each role. I.e. "If someone is found at this time at this place, it is $p_{j}\%$ likely to be role $j$", with one probability for each role. High probability values will help us identify locations where it is almost certain to be this player, meaning the chances of finding them isolated at this time are very high
# 
# ***
# ## Data Cleaning
# 
# To start, we read in the data and drop unnecessary columns, saving our seconds column in the process

# In[ ]:


df = pd.read_csv("/kaggle/input/lec-spring-2020-g2-esports-level-one-positions/g2redside.csv")

secs = df['Seconds'].tolist()
df.drop(['Unnamed: 0','Seconds'],
         inplace=True, 
         axis=1)
cols = df.columns.tolist()


# Our input data will take the form (time, x, y), while our output will simply be the corresponding role (1-5 for top-support)

# In[ ]:


X = np.empty([len(secs)*df.shape[1],3])
y = np.empty(len(secs)*df.shape[1])

k = 0
lowest = min(secs)

for i in secs:
    for y_j, j in enumerate(df.loc[i-lowest]):
        j = literal_eval(j.replace('(nan, nan)', '(0,0)'))
        X[k] = [i, j[0],j[1]]
        y[k] = y_j % 10
        k+=1


# To validate our model we'll split it into a training and testing set, as is standard

# In[ ]:


y = (y[[0 not in i for i in X]])
X = (X[[0 not in i for i in X]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)


# ***
# ## The Model
# 
# Naturally there are 5 possible roles that the player could fit in, so we'll use a multinomial logistic regression. Again, I'm not too fussed with the model itself and won't do any tweaking to increase accuracy, I just want a quick solution that does the job.

# In[ ]:


clf = LogisticRegression(random_state=123, 
                         multi_class='multinomial', 
                         solver='newton-cg')
model = clf.fit(X_train, y_train)


# In[ ]:


clf.score(X_test,y_test)


# The model is only 70% accurate. Obviously not ideal but we can make do. We'll check the confusion matrix just to see where it falls down

# In[ ]:


metrics.plot_confusion_matrix(model, X_test,y_test,values_format='d')
plt.title("Confusion Matrix")
plt.xticks(np.arange(10),labels=['blu_top','blu_jgl','blu_mid','blu_adc','blu_sup','red_top','red_jgl','red_mid','red_adc','red_sup'])
plt.yticks(np.arange(10),labels=['blu_top','blu_jgl','blu_mid','blu_adc','blu_sup','red_top','red_jgl','red_mid','red_adc','red_sup'])
plt.xlabel("Predicted Role")
plt.ylabel("True Role")


plt.show()


# As we can see here, the model works really well with the exception of the botlane champions (ADC/support), who tend to stay in similar areas. This is ok as they are generally close enough for the model to recognise one or the other. It's also worth noting that the accuracy for red side is higher than blue side, which also makes sense as all of the red side data comes from the same team (G2) while the blueside data comes from the rest of the teams in the league. So there is much more variance in the blue side data

# ***
# ## Team Overview
# 
# First we'll use this model to build an overview of the whole map and find the regions that correspond to each player. This won't have the nuance of seeing the full probabilities, but what it will do is give a quicker overview of the formations the team tends to gravitate towards
# 
# First we build a mesh that will cover the entire map. We'll set our timer to 60 seconds, so we can see how they tend to lineup at the minute mark

# In[ ]:


x_mesh, y_mesh = np.meshgrid(np.linspace(0,150, 700),
                             np.linspace(0,149, 700))

timer = 60

time_mesh = np.zeros_like(x_mesh)
time_mesh[:,:] = timer

grid_predictor_vars = np.array([time_mesh.ravel(), 
                                x_mesh.ravel(), 
                                y_mesh.ravel()]).T


# We then run this model on the mesh to and graph the predictions for the whole map, showing the redside regions only.

# In[ ]:


preds = model.predict(grid_predictor_vars)


# In[ ]:


top = cm.get_cmap('Blues', 128)
bottom = cm.get_cmap('viridis', 128)

newcolors = np.vstack((top(np.linspace(0, 0, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')

plt.figure(figsize=(20,10))
plt.imshow(preds.reshape(x_mesh.shape), 
           cmap = newcmp, 
           resample = True)
plt.xticks([])
plt.yticks([])
plt.title("Player most likely to be the one found at each spot at %d seconds" % timer)
plt.imshow(map_img, 
           zorder=1, 
           alpha = 0.25, 
           interpolation= "nearest")

top_patch = mpatches.Patch(color=newcmp(0.6), label='Top', ec="black")
jgl_patch = mpatches.Patch(color=newcmp(0.7), label='Jungle', ec="black")
mid_patch = mpatches.Patch(color=newcmp(0.8), label='Mid', ec="black")
adc_patch = mpatches.Patch(color=newcmp(0.9), label='ADC', ec="black")
sup_patch = mpatches.Patch(color=newcmp(1.0), label='Support', ec="black")


plt.legend(loc="lower left",handles=[top_patch,jgl_patch, mid_patch, adc_patch, sup_patch], prop={'size': 16})#
plt.show()


# As we can see here, the toplaner tends to push past the halfway point often, while the jungler is most likely to be found isolated when in the top half of the map. This means that an invade at 60 seconds targeted there is more likely to set the jungler back, while invading at the 'pixel brush' (towards the bottom of the light green area) is more likely to catch the ADC out. This isn't revolutionary but it can be helpful to see how that changes over time

# In[ ]:


def red_levelonefull(timer):
    x_mesh, y_mesh = np.meshgrid(np.linspace(0,150, 700),
                                 np.linspace(0,149, 700))

    time_mesh = np.zeros_like(x_mesh)
    time_mesh[:,:] = timer

    grid_predictor_vars = np.array([time_mesh.ravel(), 
                                    x_mesh.ravel(), 
                                    y_mesh.ravel()]).T
    
    preds = model.predict(grid_predictor_vars)

    top = cm.get_cmap('Blues', 128)
    bottom = cm.get_cmap('viridis', 128)

    newcolors = np.vstack((top(np.linspace(0, 0, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    plt.figure(figsize=(20,10))
    plt.imshow(preds.reshape(x_mesh.shape), 
               cmap = newcmp, 
               resample = True)
    plt.xticks([])
    plt.yticks([])
    plt.title("Player most likely to be the one found at each spot at %d seconds" % timer)
    plt.imshow(map_img, 
               zorder=1, 
               alpha = 0.25, 
               interpolation= "nearest")

    top_patch = mpatches.Patch(color=newcmp(0.6), label='Top', ec="black")
    jgl_patch = mpatches.Patch(color=newcmp(0.7), label='Jungle', ec="black")
    mid_patch = mpatches.Patch(color=newcmp(0.8), label='Mid', ec="black")
    adc_patch = mpatches.Patch(color=newcmp(0.9), label='ADC', ec="black")
    sup_patch = mpatches.Patch(color=newcmp(1.0), label='Support', ec="black")


    plt.legend(loc="lower left",handles=[top_patch,jgl_patch, mid_patch, adc_patch, sup_patch], prop={'size': 16})


# In[ ]:


red_levelonefull(80)


# So between 60 and 80 seconds, the ADC tends to leave his position and move further towards his lane, while the jungler is also likely to move down towards the botlane. These are very small tendencies that are hard to spot, but can be crucial when planning a move. Catching them in this transition could be a good time to strike. Repeating this for different timings can give a good idea of what patterns to look for when analysing G2's level one.

# ## Individual Probabilities
# 
# While that is undeniably useful, it can also be useful to focus in on a particular player and see how the probability changes spatially. To do this, we'll run the model over the same grid, except this time we'll gather the probabilities from the model as opposed to the predictions. We'll then focus on a given position and graph their probabilities over the map.

# In[ ]:


def red_leveloneplayer(timer, position):
    positions = dict(zip(['top','jgl','mid','adc','sup'], [5,6,7,8,9]))
    x_mesh, y_mesh = np.meshgrid(np.linspace(0,150, 700),
                                 np.linspace(0,149, 700))

    time_mesh = np.zeros_like(x_mesh)
    time_mesh[:,:] = timer

    grid_predictor_vars = np.array([time_mesh.ravel(), 
                                    x_mesh.ravel(), 
                                    y_mesh.ravel()]).T
    preds = model.predict_proba(grid_predictor_vars)

    hp_mesh = preds.T[positions[position]].reshape(x_mesh.shape)
    
    hm = sns.heatmap(hp_mesh, cmap="Reds", 
                     yticklabels=False, 
                     xticklabels=False)
    plt.title("Probability that a player found at (x,y) at %d seconds is %s" % (timer,position))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.imshow(map_img, zorder=1, alpha = 0.3)
    
    fig = pl.figure()
    ax = Axes3D(fig)
    
    fig = ax.plot_surface(x_mesh, y_mesh, hp_mesh, 
        rstride=1, cstride=1, cmap='Reds',lw=0.1)
    ax.set_title("Probability that a player found at (x,y) at %d seconds is %s: 3D" % (timer,position))
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.set_zlabel("Probability")
    pl.show()


# In[ ]:


red_leveloneplayer(60, 'sup')


# So it seems that if you can't catch the support at the edge of the map (difficult place to invade), there's a very low chance of you finding them isolated. The sharp drop on the 3D map shows how the probability of him being there plummets as soon as you get further from the edge. This means that the model is unsure of who it would be, implying that there are multiple possible people who could be there
# 
# We can then do the same for blue side

# In[ ]:


blue_df = pd.read_csv("/kaggle/input/lec-spring-2020-g2-esports-level-one-positions/g2blueside.csv")

secs = blue_df['Seconds'].tolist()
blue_df.drop(['Unnamed: 0','Seconds'],
         inplace=True, 
         axis=1)

X = np.empty([len(secs)*blue_df.shape[1],3])
y = np.empty(len(secs)*blue_df.shape[1])

k = 0
lowest = min(secs)

for i in secs:
    for y_j, j in enumerate(blue_df.loc[i-lowest]):
        j = literal_eval(j.replace('(nan, nan)', '(0,0)'))
        X[k] = [i, j[0],j[1]]
        y[k] = y_j % 10
        k+=1

cols = blue_df.columns.tolist()

y = (y[[0 not in i for i in X]])
X = (X[[0 not in i for i in X]])

clf = LogisticRegression(random_state=0, 
                         multi_class='multinomial', 
                         solver='newton-cg')
blue_model = clf.fit(X, y)

def blue_levelonefull(timer):
    x_mesh, y_mesh = np.meshgrid(np.linspace(0,150, 700),
                                 np.linspace(0,149, 700))

    time_mesh = np.zeros_like(x_mesh)
    time_mesh[:,:] = timer

    grid_predictor_vars = np.array([time_mesh.ravel(), 
                                    x_mesh.ravel(), 
                                    y_mesh.ravel()]).T
    
    preds = blue_model.predict(grid_predictor_vars)

    top = cm.get_cmap('viridis', 128)
    bottom = cm.get_cmap('Reds', 256)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 0, 128))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    plt.figure(figsize=(20,10))
    plt.imshow(preds.reshape(x_mesh.shape), 
               cmap = newcmp, 
               resample = True)
    plt.xticks([])
    plt.yticks([])
    plt.title("Player most likely to be the one found at each spot at %d seconds" % timer)
    plt.imshow(map_img, 
               zorder=1, 
               alpha = 0.25, 
               interpolation= "nearest")

    top_patch = mpatches.Patch(color=newcmp(0), label='Top', ec="black")
    jgl_patch = mpatches.Patch(color=newcmp(0.1), label='Jungle', ec="black")
    mid_patch = mpatches.Patch(color=newcmp(0.2), label='Mid', ec="black")
    adc_patch = mpatches.Patch(color=newcmp(0.3), label='ADC', ec="black")
    sup_patch = mpatches.Patch(color=newcmp(0.4), label='Support', ec="black")


    plt.legend(loc="upper right",handles=[top_patch,jgl_patch, mid_patch, adc_patch, sup_patch], prop={'size': 16})

    
def blue_leveloneplayer(timer, position):
    positions = dict(zip(['top','jgl','mid','adc','sup'], [0,1,2,3,4]))
    x_mesh, y_mesh = np.meshgrid(np.linspace(0,150, 700),
                                 np.linspace(0,149, 700))

    time_mesh = np.zeros_like(x_mesh)
    time_mesh[:,:] = timer

    grid_predictor_vars = np.array([time_mesh.ravel(), 
                                    x_mesh.ravel(), 
                                    y_mesh.ravel()]).T
    preds = blue_model.predict_proba(grid_predictor_vars)

    hp_mesh = preds.T[positions[position]].reshape(x_mesh.shape)
    
    hm = sns.heatmap(hp_mesh, cmap="Blues", 
                     yticklabels=False, 
                     xticklabels=False)
    plt.title("Probability that a player found at (x,y) at %d seconds is %s" % (timer,position))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.imshow(map_img, zorder=1, alpha = 0.3)
    
    fig = pl.figure()
    ax = Axes3D(fig)
    
    fig = ax.plot_surface(x_mesh, y_mesh, hp_mesh, 
        rstride=1, cstride=1, cmap='Blues',lw=0.01)
    ax.set_title("Probability that a player found at (x,y) at %d seconds is %s: 3D" % (timer,position))
    ax.set_ylabel("x")
    ax.set_xlabel("y")
    ax.set_zlabel("Probability")
    
    pl.show()


# In[ ]:


blue_leveloneplayer(60,"jgl")


# In[ ]:


blue_levelonefull(60)


# In[ ]:


blue_levelonefull(80)


# ***
# ## Conclusion
# 
# To conclude, you can use this positional data to spot tendencies in the early-game formation of teams very quickly and effectively. Bringing these tendencies to the attention of an analyst can allow them to further explore the formations and how to work around them/force them. This is just a small taste of the sort of work that can be done applying analytics to LoL, hopefully you found it interesting! Please feel free to leave feedback as I'd love to discuss this idea further
