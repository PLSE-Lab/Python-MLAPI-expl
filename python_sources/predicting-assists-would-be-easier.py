#!/usr/bin/env python
# coding: utf-8

# **Understanding Kobe Bryant via python**
# 
# Kobe is a polarizing figure in sports. For some (one of) the greatest player(s) of all time, for others a talented but overrated ball hog who got great teams built around him. Whether either one of these assertions is true will probably remain unclear by the end of this notebook, but on the way there we will take a look into the process by which a great player chooses his shots, and how this evolves through time according to his coaches, teammates and physical capabilities.
# 
# The problem presented is a simple one: to predict whether each shot is going into the basket or not. Some of the variables in which this depends are going to be obvious - backcourt shots are less likely to go in than mid-range ones, and both less likely than dunks - but some are not. A full understanding of the correlation between all the factors that go into the probability of making a shot is virtually impossible without some sort of statistical analysis, so here we go!
# 
# In this kernel I will try to make a very simple introductory analysis of the data, which includes:
# * understanding each of the independent variables (binary, continuous, categorical) and their distribution, mainly whether they have many outliers and there are entries where they are not filled
# * cleaning the variables (e.g., combine correlated variables, process categorical variables into numerical distributions and deal with missing data) while studying the relationship between each of them and the variable we are trying to predict - whether a shot was made
# * After having a set of well treated and (hopefully) almost independent variables, we can predict the probability of having made the shot using some machine learning algorithms
# 
# So, let's load all the libraries we are going to need and read our input file

# In[ ]:


import numpy as np # linear algebra
import pandas as pd
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
import math

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn import mixture
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 50)


# In[ ]:


df_input = pd.read_csv("../input/data.csv")


# The first thing to do is studying what our dataset is made of:

# In[ ]:


print( "There are ", len(df_input.columns.values), " columns")
df_input.head()


# There are 25 columns, meaning a large (but manageable) number of 24 independent variables. A quick look at the first values shows us that they do not seem independent from each other. For example, action_type seems like a more specific description of combined_shot_type, game_id seems like a date and lat and long (presumably for latitude and longitude, more to come!) are probably just loc_x and loc_y on a different reference frame.
# 
# Our objective is transforming these 24 "raw" variables into treated ones, so I will created a string called untreated_variables which will be updated as we go through this kernel. When there are no more untreated variables, we can go into training!
# 
# But before we go into cleaning the data, let's look a bit further into the dataset:

# In[ ]:


untreated_variables = df_input.columns.values.tolist()
untreated_variables.remove("shot_made_flag")
untreated_variables.remove("shot_id")


# In[ ]:


cols_print = ('variable', 'type', 'unique values', 'missing values [%]')
print('{: <20} {: >20} {: >20} {: >20}'.format(*cols_print)); print()
for col in df_input.columns.values:
    type_col = str(type(df_input.at[0, str(col)])).split("'")[1]
    uniq_col = len(df_input[str(col)].unique())
    miss_col = str(100*round(df_input[str(col)].isnull().sum()/len(df_input.index), 2))+"%"
    cols_print = (str(col), type_col, str(uniq_col), miss_col)
    print('{: <20} {: >20} {: >20} {: >20}'.format(*cols_print))


# There are some good new and some bad news. First, the good ones: there are no missing values at all in our dataset! This is very rare, but certainly makes our life much easier. The only variable missing entries is the shot made flag, in which 5000 entries out of the total (~30000) were removed to be used as the test set. The second is that there are two variables with only one unique value, so we can go ahead and remove them:

# In[ ]:


print(df_input.at[0, "team_id"], df_input.at[0, "team_name"])
df_input = df_input.drop(["team_id", "team_name"], 1)
untreated_variables.remove("team_id") 
untreated_variables.remove("team_name")
print(len(untreated_variables), "variables to go!")


# As we already knew, Kobe played for the Lakers his entire career, so knowing the team he was playing for makes no impact in our analysis (team_id seems to be just a number identifying the Lakers). Now we only have 21 independent variables.
# 
# So, now for the bad news. There are a lot of categorical variables (10 type: str) and a lot of variables that have many unique values. Why is this bad? Because (assuming no correlations, which  surely exist) the number of possible combinations of variables really adds up:

# In[ ]:


upper_number_branches = 1
for col in df_input.columns.values:
  if col!="shot_id" and col!="shot_made_flag":
    upper_number_branches = upper_number_branches*len(df_input[str(col)].unique())
print("There are up to", "{:.0e}".format(upper_number_branches), "combinations")    


# A machine learning algorithm might be able to solve our problem if we just train it using the present 21 variables. But it might run into problems, at the very least taking longer and perhaps, worse, finding spurious features due to the large span it has to test. Plus, no insight is gained for us, which makes it harder to understand future results
# 
# So, let's make life easier for everyone and reduce the dimensionality of the problem!
# 
# **Spatial dimensions**
# 
# There are 9 variables related to the spatial distribution of the shots:
# * four categorical: shot_zone_area, shot_zone_basic, shot_zone_range, 
# * five numerical: loc_x, loc_y, lat, lon, shot_distance
# 
# First of all, let's figure out if the location and lon:lat variables are both necessary. First we will define few useful functions, like a draw_court function, taken from [Savvas Tjortjoglou](http://http://savvastjortjoglou.com/nba-shot-sharts.html)) but with an added function to rotate the court by an input angle and a color map. This makes us understand better the location of the shots with respect to the basket and the little "pockets" where Kobe prefers to shot. 
# 
# Then we draw loc_x:loc_y and lat:lon.

# In[ ]:


def draw_court(ax=None, color='black', lw=2, outer_lines=False, deg=0):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    
    for element in court_elements:
        r2 = mpl.transforms.Affine2D().rotate_deg(deg).translate(deg*422.5/90, 0) + ax.transData
        element.set_transform(r2)
        ax.add_patch(element)

    return ax

def color_map():
    cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.2, 0.2),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),
                 (0.1, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 1.0, 1.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}


    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)


# In[ ]:


fig = plt.figure(figsize=(18, 8))
fig.add_subplot(1,2,1)
plt.scatter(x=df_input['loc_x'],y=df_input['loc_y'])
plt.title("Location: y vs x")
fig.add_subplot(1,2,2)
plt.scatter(x=df_input['lon'],y=df_input['lat'])
plt.title("Latitude vs Longitude")
plt.show()


# It is clear they are both pairs of position variables, but the location ones seem to make more sense: the basket is at (0,0) and the y axis increases with the distance to the basket, so we will keep them and drop the other pair. The other numerical one, shot_distance, we will keep for now as it is strongly correlated with shot success, and will deal with it later..

# In[ ]:


df_input = df_input.drop(["lat", "lon"], 1)
untreated_variables.remove("lat"); untreated_variables.remove("lon");
fig = plt.figure(figsize=(12, 6))
sb.barplot('shot_distance', 'shot_made_flag', data=df_input[df_input["shot_distance"]<35])
plt.title("Efficiency as a function of shot distance")
plt.show()


# Now lets look at the categorical variables

# In[ ]:


def color_scatter_plot(var):
    gs = df_input.groupby(var)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c)

fig = plt.figure(figsize=(18, 6))
plt.subplot(1,4,1)
color_scatter_plot("shot_type")
plt.title("Shot type")
plt.subplot(1,4,2)
color_scatter_plot("shot_zone_range")
plt.title("Shot zone range")
plt.subplot(1,4,3)
color_scatter_plot("shot_zone_area")
plt.title("Shot zone area")
plt.subplot(1,4,4)
color_scatter_plot("shot_zone_basic")
plt.title("Shot zone basic")
plt.show()


# It is clear these are just four different ways of dividing the court, so:
# * they are all correlated with each other
# * they are just functions of the location (x and y)
# 
# Plus, some values were filled wrongly (there are 3 point shots inside the 3 point line and vice versa), so we can just drop all these categories and will make a better one ourselves, through the power of machine learning!
# 
# The first step is trying to learn a bit more about kobe shot selection, which we can do using density plots of his shots

# In[ ]:


cmap = mpl.cm.get_cmap('viridis')
jp = sb.jointplot(df_input["loc_x"], df_input["loc_y"], cmap=cmap, n_levels=50,
                  stat_func=None,kind='kde', space=0, color=cmap(0.1))
ax = jp.ax_joint
draw_court(ax=ax)
ax.set_xlim(-250,250)
ax.set_ylim(422.5, -47.5)
plt.show()


# Several things become clear:
# * Kobe really likes to shoot really close to the hoop (who knew!)
# * If he can't, the density does not decrease monotonously with distance - they are these little pockets where he can find his space to shoot
# * He prefers his right side - he is right handed after all
# * When shooting threes he prefers the wings than either the corner or the top of the key
# 
# So, there are clearly areas where shots aggregate, which we can exploit! Do these areas change as a function of time? First let's see how young Kobe (in the season where he won his first championship) plays against middle-age Kobe (when he won his last one)

# In[ ]:


def make_kobe_vs_kobe_plots(seasons, df_input):    

    df_input["eff_shot_made_flag"] = np.nan
    for i in range(0, len(df_input.index)):
        if math.isnan(df_input.at[i, "shot_made_flag"]):
            continue
        df_input.at[i, "eff_shot_made_flag"] = (float(df_input.at[i, "shot_made_flag"])*
                                                float(df_input.at[i, "shot_type"][0])/2)
 
    cmap = mpl.cm.get_cmap('viridis')
    df_input["loc_x_0"]=0; df_input["loc_x_1"]=0; df_input["loc_y_0"]=0; df_input["loc_y_1"]=0;
    for i in range(0, len(df_input.index)):
        df_input.at[i, "loc_y_0"] = df_input.at[i, "loc_y_1"] = df_input.at[i, "loc_x"]
        df_input.at[i, "loc_x_0"] = min(df_input.at[i, "loc_y"] - 422.5, -1)
        df_input.at[i, "loc_x_1"] = max(-df_input.at[i, "loc_y"] + 422.5, 1)

    js1 = sb.jointplot(df_input["loc_x_0"][df_input['season']==seasons[0]], 
                   df_input["loc_y_0"][df_input['season']==seasons[0]], 
                   stat_func=None,kind='kde', space=0, color=cmap(0.1),
                   cmap=cmap, n_levels=50)
    js1.fig.set_size_inches(12,8) 
    plt.close()
    
    js2 = sb.jointplot(df_input["loc_x_1"][df_input['season']==seasons[1]], 
                   df_input["loc_y_1"][df_input['season']==seasons[1]], 
                   stat_func=None,kind='kde', space=0, color=cmap(0.1),
                   cmap=cmap, n_levels=50)
    js2.fig.set_size_inches(12,8) 
    plt.close()
    
    f = plt.figure(figsize=(12,9))
    for J in [js1, js2]:
        for A in J.fig.axes:
            A.set_xlabel('')
            A.set_ylabel('')
            A.tick_params(labelbottom='off', labelleft='off', 
                      labelright='off', labeltop='off')   
            f._axstack.add(f._make_key(A), A)
        
    top_all = 0.75; bot_all = 0.05;

    f.axes[0].set_position([0.05, 0.05, 0.45, top_all-bot_all])
    f.axes[1].set_position([0.05, top_all,  0.45, 0.05])
    f.axes[2].set_position([0.0,  0.05, 0.05, top_all-bot_all])
    f.axes[3].set_position([0.5, 0.05, 0.45,  top_all-bot_all])
    f.axes[4].set_position([0.5, top_all, 0.45,  0.05])
    f.axes[5].set_position([0.95, 0.05, 0.05, top_all-bot_all])

    draw_court(ax=f.axes[0], outer_lines=True, deg=-90)
    draw_court(ax=f.axes[3], outer_lines=True, deg=90)
    f.axes[0].set_ylim(-275,275); f.axes[0].set_xlim(-455,0); 
    f.axes[3].set_ylim(-275,275); f.axes[3].set_xlim(0,455);     

    height = 0.68; x_k = [0.435, 0.49]; sign_k = [-1, 1];
    plt.gcf().text(x_k[0]+0.04, height - 0.02,"FG%", fontstyle="italic", 
                   fontsize=14, color="whitesmoke", family='sans-serif') 
    plt.gcf().text(x_k[0]+0.035, height - 0.09,"eFG%", fontstyle="italic", 
                   fontsize=14, color="whitesmoke", family='sans-serif')
       
    plt.gcf().text(x_k[0]-0.10, height + 0.06, seasons[0], fontstyle="italic", 
                   fontsize=20, color="darkblue", family='sans-serif') 
    plt.gcf().text(x_k[1]+0.06, height + 0.06, seasons[1], fontstyle="italic", 
                   fontsize=20, color="darkblue", family='sans-serif')       
    for iage in range(0, len(seasons)):    
        
        shot_perc = 100*df_input['shot_made_flag'][df_input['season']==seasons[iage]].mean()
        effi_perc = 100*df_input['eff_shot_made_flag'][df_input['season']==seasons[iage]].mean()
        stri_perc=(str(float(int(10*shot_perc))/10)+"%", 
                   str(float(int(10*effi_perc))/10)+"%")
        plt.gcf().text(x_k[iage] + sign_k[iage]*0.01, height - 0.05, 
                       stri_perc[0], fontstyle="italic", fontsize=14, 
                       color="whitesmoke", family='sans-serif')        
        plt.gcf().text(x_k[iage] + sign_k[iage]*0.01, height - 0.12, 
                       stri_perc[1], fontstyle="italic", fontsize=14, 
                       color="whitesmoke", family='sans-serif')   
    
    df_input = df_input.drop(["loc_x_0", "loc_y_0", "loc_x_1", "loc_y_1", "eff_shot_made_flag"], 1)

    plt.axis('off')
    plt.show()   


# In[ ]:


seasons = ["1999-00", "2009-10"] #first and last championship
make_kobe_vs_kobe_plots(seasons, df_input)


# His field goal percentage went down and he lost the ability to go to the basket as much as before, but because he shot more three pointers his effective field goal percentage actually went up. Interesting! Now let's look at his second to last season, the last where he was fully healthy

# In[ ]:


seasons = ["2009-10", "2014-15"] #last championship and last healthy season
make_kobe_vs_kobe_plots(seasons, df_input)


# Ouch, that's rough! Kobe almost completely gave up on shot selection (less well-defined high density areas), and his field goal percentage went way down, which was not compensated this time but the increase in three pointers.
# 
# Now, we can define our own areas, assuming shot density is distributed as a sum of gaussians - a simple and sturdy model. How many areas should we define? We know from previously that distributions are different center left and right, and within left and right there is the mid range shot and the corner three, so this makes 5 dimensions in x. For now, let's assume 4 dimensions in y: basket, close to the basket (hooks, floaters), mid range and three pointers. This makes a total of 20 areas to give as input to our gaussian mixture model.

# In[ ]:


numGaussians = 20
gaussianMixtureModel = mixture.GaussianMixture(n_components=numGaussians, n_init=5, random_state=3)
gaussianMixtureModel.fit(df_input[['loc_x','loc_y']])
df_input['LocCluster'] = gaussianMixtureModel.predict(df_input[['loc_x','loc_y']])   
fig = plt.figure(figsize=(12, 6))
draw_court(outer_lines=True); 
plt.ylim(-60,440); plt.xlim(270,-270); 
plt.title('cluster assignment')
num_elem = len(df_input['LocCluster'].unique()) 
colors = cm.rainbow(np.linspace(0, 1, num_elem))
for i in range(0, num_elem):
    plt.scatter(x=df_input['loc_x'][df_input['LocCluster']==i],
               y=df_input['loc_y'][df_input['LocCluster']==i], 
               c=colors[i], edgecolors='none')
plt.show()     


# Looks more or less like a sensible separation of zones, although (at least visually) there seems to be some "spillover" of some mid-range classifier to beside the three point line. 
# 
# So, let's test whether these clusters make sense by our criteria: are they a good predictor of shot success? Or, to put it otherwise, is the shot success probability within a zone independent of the specific location (or the distance to the basket)? To do this we do a chi-square test on the independence of the shot_made_flag with loc_x, loc_y and distance within which of the clusters

# In[ ]:


def prob_independence(df, var_compar, var_ref):
    ct = pd.crosstab(df[var_compar], df[var_ref])
    chi2, p_val, ndf, exct = sp_stats.chi2_contingency(ct)
    if ndf==0:
        return 0
    else:
        return chi2/ndf

x_x=[-1]
y_x=[prob_independence(df_input, "loc_x", "shot_made_flag")]
y_y=[prob_independence(df_input, "loc_y", "shot_made_flag")]
y_d=[prob_independence(df_input, "shot_distance", "shot_made_flag")]
for i in range(0,len(df_input['LocCluster'].unique())):
    x_x.append(i)
    y_x.append(prob_independence(df_input[df_input["LocCluster"]==i], "loc_x", "shot_made_flag"))
    y_y.append(prob_independence(df_input[df_input["LocCluster"]==i], "loc_y", "shot_made_flag"))
    y_d.append(prob_independence(df_input[df_input["LocCluster"]==i], "shot_distance", "shot_made_flag"))

fig = plt.figure(figsize=(18, 6))
plt.subplot(1,3,1); plt.bar(x_x, y_x); plt.title("Chi2/ndf vs loc_x");
plt.subplot(1,3,2); plt.bar(x_x, y_y); plt.title("Chi2/ndf vs loc_y");
plt.subplot(1,3,3); plt.bar(x_x, y_d); plt.title("Chi2/ndf vs distance");
plt.show()


# The first column is the reduced chi-squared on the correlation between shots from everywhere and the loc_x, loc_y and distance, while subsequent bars are the same within each specific cluster. So, shot success is in general dependent of location and distance, but it is very independent of the position within all areas except in the case of the cluster 12. Let's look at it more carefully

# In[ ]:


fig = plt.figure(figsize=(18, 8))
plt.subplot(1,4,1)
draw_court(outer_lines=True); 
plt.ylim(-60,440); plt.xlim(270,-270);
plt.title("Bad clusters")
bad_clusters = [1, 2, 15]
for i in bad_clusters:
    plt.scatter(x=df_input['loc_x'][df_input['LocCluster']==i],
                y=df_input['loc_y'][df_input['LocCluster']==i], 
                edgecolors='none')
for i in range(0,len(bad_clusters)): 
    plt.subplot(1,4,2+i)
    sb.barplot('shot_distance', 'shot_made_flag', data=df_input[df_input["LocCluster"]==bad_clusters[i]])
    plt.title('Eff. in Cluster '+str(bad_clusters[i]))
plt.show()


# The first zone is the one near the basket, where of course it makes difference whetheer you are dunking or shooting 2 meters away from it. The orange one seems like a mixture of several areas: it starts at the three point line and goes up to scattered desperation shots farther from it. The green one has clearly two different regions. So, let's subdivide these two areas and keep the other ones where efficiency is already independent of position

# In[ ]:


bad_clusters = [1, 2, 15]
bins_clusters = [[1,2,3],[25,26,27,28],[24,25,26]]
bins_pos = [0, len(bins_clusters[0]), len(bins_clusters[0])+len(bins_clusters[1])]
bin_ref = len(df_input['LocCluster'].unique())
for ie in range(0, len(df_input.index)):
    for ic in range(0, len(bad_clusters)):
        if df_input.at[ie, 'LocCluster']==bad_clusters[ic]:
            for ib in range(0, len(bins_clusters[ic])-1):
                #print(bad_clusters[ic], "==", bins_clusters[ic][ib], "at", bin_ref + bins_pos[ic] + ib)
                if df_input.at[ie, 'shot_distance']==bins_clusters[ic][ib]:
                    df_input.at[ie, 'LocCluster'] = bin_ref + bins_pos[ic] + ib
            #print(bad_clusters[ic], ">=", bins_clusters[ic][len(bins_clusters[ic])-1], "at", bin_ref + bins_pos[ic] + len(bins_clusters[ic])-1)
            if df_input.at[ie, 'shot_distance']>=bins_clusters[ic][len(bins_clusters[ic])-1]:
                df_input.at[ie, 'LocCluster'] = bin_ref + bins_pos[ic] + len(bins_clusters[ic])-1


# And now, everything works well

# In[ ]:


x_x=[]; y_x=[]; y_y=[]; y_d=[]
for i in range(0, len(df_input['LocCluster'].unique())):
    x_x.append(i)
    y_x.append(prob_independence(df_input[df_input["LocCluster"]==i], "loc_x", "shot_made_flag"))
    y_y.append(prob_independence(df_input[df_input["LocCluster"]==i], "loc_y", "shot_made_flag"))
    y_d.append(prob_independence(df_input[df_input["LocCluster"]==i], "shot_distance", "shot_made_flag"))

fig = plt.figure(figsize=(18, 6))
plt.subplot(1,3,1); plt.bar(x_x, y_x); plt.title("Chi2/ndf vs loc_x");
plt.subplot(1,3,2); plt.bar(x_x, y_y); plt.title("Chi2/ndf vs loc_y");
plt.subplot(1,3,3); plt.bar(x_x, y_d); plt.title("Chi2/ndf vs distance");
plt.show()


# Now, within these 30 clusters, the likelihood of making a shot is independent of the other position variables. So, we can remove all those columns. Notice that we achieved a very impressive reduction in the dimensionality of the problem (defined as the number of possible combinations of all spatial variables)

# In[ ]:


df_input = df_input.drop(["shot_zone_basic", "shot_zone_range", "shot_zone_area", 
                          "shot_type", "loc_x", "loc_y", "shot_distance"], 1)
untreated_variables.remove("shot_zone_basic"); untreated_variables.remove("shot_zone_range");
untreated_variables.remove("shot_zone_area"); untreated_variables.remove("shot_type");
untreated_variables.remove("shot_distance"); untreated_variables.remove("loc_y"); untreated_variables.remove("loc_x");

print("Decreased from spatial dimensionality of", 457*457*489*489*74*2*6*7*5, "to", 30)


# So, what do we have left in our data?

# In[ ]:


print(untreated_variables)


# Let's deal now with the first two variables, which deal with the description of the shot type. Does the likelihood of making a shot depend on it?

# In[ ]:


fig = plt.figure(figsize=(15, 6))
plt.subplot(1,2,1)
pl1 = sb.barplot('action_type', 'shot_made_flag', data=df_input)
pl1.set(xticklabels=[])
plt.subplot(1,2,2)
pl2 = sb.barplot('combined_shot_type', 'shot_made_flag', data=df_input)
pl2.set(xticklabels=[])
plt.show()


# Clearly these are important variables. Since combined_shot_type is just a combination of information in action_type, we will drop the combined variable and again try to find a more sensible way of combining all shot types

# In[ ]:


df_input = df_input.drop("combined_shot_type", 1)
untreated_variables.remove("combined_shot_type");
print(df_input["action_type"].unique())


# These are a lot of unique descriptions! But there are also a lot of repeated words - all have shot, more than half has jump, a lot have dunk, so we make binary columns for the presence of each word, and then clean them up looking for outliers and correlations between them

# In[ ]:





# In[ ]:


df_shot = df_input.copy()

#make set of unique words
class_shot = set()
for i in range(0, len(df_shot.index)):
    stri = df_shot.at[i,'action_type'].split(' ')
    class_shot.update(stri)
class_shot = list(class_shot)

#initialize columns and fill them
for i in range(0, len(class_shot)):
    df_shot["has_"+str(class_shot[i]).capitalize()] = 0    
for i in range(0, len(df_shot.index)):
    stri = df_shot.at[i,'action_type'].split(' ')
    for ist in range(0, len(stri)):
        df_shot.at[i,"has_"+stri[ist].capitalize()] = 1
        
for i in df_shot.columns.values:
    if i[:4]!="has_" and i!="shot_made_flag":
        df_shot = df_shot.drop(i, 1)
    
fig = plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.title("Fraction of shots per label")
df_shot_order = df_shot.mean().sort_values()
df_shot_order.plot(kind='bar')
mean_shot = []
for i in df_shot_order.index:
    mean_shot.append(df_shot["shot_made_flag"][df_shot[i]==1].mean())

ax = plt.subplot(1,2,2)
list(range(len(df_shot_order.index)))
ax.bar(range(len(df_shot_order.index)), mean_shot, color='b')
plt.xticks(range(len(df_shot_order.index)), df_shot_order.index)
plt.xticks(rotation='vertical')
plt.title("Efficiency of shots per label")
plt.show()

def make_pretty_corr_plot(df_shot):
    corr = df_shot.corr()
    corr_mat = np.triu(corr, k=1)
    mask =np.zeros_like(corr, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True
    cmap = sb.diverging_palette(220, 10, as_cmap=True)
    f, ax = plt.subplots(figsize=(11, 9))
    sb.heatmap(corr_mat, mask=mask, cmap=cmap, xticklabels=corr.columns.values,
           yticklabels=corr.columns.values)
    plt.show()
    
make_pretty_corr_plot(df_shot)


# So, all shots are has_shot! Not much of a surprise here, but we can drop this column as it has no information. Also, it is interesting to see that the likelihood of a shot is much more likely when there is extra information (i.e., description is not just Jump Shot). This is probably just a human error, as NBA staff are more zealous if a shot is made, and also because shots closer to the basket have more specific nomenclature. Finally, we plot the correlation matrix. Note that some pairs of names are completely correlated (Follow-up, Alley-Oop, Step-Back, Finger-Roll) so we will combine them. Also that there is a Pullup and a Pull-up column which can be combined, and since all "Slam" are dunk, but not vice versa, we can create a less correlated variable "has_Slam-Dunk" without increased dimensionality

# In[ ]:


df_shot["has_Pullup"] = df_shot["has_Pullup"] + df_shot["has_Pull-up"]
df_shot = df_shot.drop(["has_Shot", "has_Pull-up"], 1)
shots_pre = ["Alley", "Finger", "Step", "Follow"]
shots_pos = ["Oop", "Roll", "Back", "Up"]
for i in range(0, len(shots_pre)):
    df_shot["has_"+shots_pre[i]+"-"+shots_pos[i]] = df_shot["has_"+shots_pre[i]]
    df_shot = df_shot.drop(["has_"+shots_pre[i], "has_"+shots_pos[i]], 1)

print(df_shot["has_Dunk"][df_shot["has_Slam"]==1].mean())
df_shot["has_Slam-Dunk"] = df_shot["has_Slam"]
df_shot["has_Other-Dunk"] = df_shot["has_Dunk"][df_shot["has_Slam"]==0]
df_shot["has_Other-Dunk"] = df_shot["has_Other-Dunk"].fillna(0)
df_shot = df_shot.drop(["has_Dunk", "has_Slam"],1)

make_pretty_corr_plot(df_shot)


# Things already look better, at least no variable is completely correlated. However, some still are a bit, most notably Jump-Layup (not much to do since of course no jumper is a layup and vice versa, but some shots are neither so it is a necessary flag). Finally, we remove outliers - in this case we choose the three falgs for which there are less than 30 events out of 30.000 and add our construct columns to the main dataset

# In[ ]:


df_shot.drop("shot_made_flag", 1)
for i in df_shot.columns.values:
    if df_shot[i].mean()*df_shot[i].count()<30:
        df_shot.drop(i, 1)
    else:
        df_input[i] = df_shot[i]

df_input = df_input.drop("action_type", 1)
untreated_variables.remove("action_type"); 
print(untreated_variables)


# Now we can deal with two easy variables: matchup and opponent. Matchup is always of the form LAL vs OPP or LAL @ OPP, so the only added information is whether the game is at home. Since the opponent name is not going to be used anymore we can turn it into numeric values. Also, since we know that when he plays the LAC he does not have to travel and that they have been notoriously a terrible team until the last few years, we test whether he shoots better or worse at LAC stadium than at home

# In[ ]:


df_input["Home_game"] = 1
for i in range(0, len(df_input.index)):
    if df_input.at[i, "matchup"].split(" ")[1]=="@":
        df_input.at[i,'Home_game']=0

fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(1, 3, 1)
sb.barplot("Home_game", "shot_made_flag", data=df_input)
plt.title("Efficiency vs Home_game")
ax = fig.add_subplot(1, 3, 2)
sb.barplot("opponent", "shot_made_flag", data=df_input)
plt.xticks(rotation='vertical')
ax = fig.add_subplot(1, 3, 3)
sb.barplot("Home_game", "shot_made_flag", data=df_input[df_input["opponent"]=="LAC"])
plt.title("Efficiency against the Clippers")
plt.show()

#he shoots as well as home as in the clippers court - sorry steve ballmer!
df_input.loc[df_input["opponent"]=="LAC", "Home_game"] = 1
#make opponent numeric
ishot=0
df_input["opponent_N"]=0
for team in df_input['opponent'].unique():
    df_input.loc[df_input["opponent"]==team, "opponent_N"] = ishot 
    ishot = ishot + 1
    
df_input = df_input.drop(["matchup", "opponent"], 1);
untreated_variables.remove("matchup"); untreated_variables.remove("opponent"); 


# In[ ]:


print(untreated_variables)


# Now there are only two types of time variables: date and in-game time. First, let's deal with the first kind ('game_id', 'playoffs', 'season', 'game_date'). The first, game_id, does not have any information w.r.t the date, so we can just drop it. Also, the season and month of the game fully describe the year, so we do not have to make that variable.

# In[ ]:


df_input = df_input.drop("game_id", 1) 
untreated_variables.remove("game_id")

df_input['game_date'] = pd.to_datetime(df_input['game_date'])
df_input = df_input.sort_values(by="game_date", ascending=1)
df_input['game_month'] = df_input['game_date'].dt.month
df_input['game_month_day'] = df_input['game_date'].dt.day
df_input['game_weekday'] = df_input['game_date'].dt.dayofweek

df_input.at[i, "season_N"] = 0
for i in range(0, len(df_input.index)):
    df_input.at[i, "season_N"] = int(df_input.at[i,'season'].split('-')[0])-1995
df_input = df_input.drop("season", 1)
df_input['season'] = df_input['season_N'].copy()
df_input = df_input.drop("season_N", 1)
    
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(1, 4, 1)
sb.barplot("game_month", "shot_made_flag", data=df_input)
plt.title("Efficiency vs Month")
ax = fig.add_subplot(1, 4, 2)
sb.barplot("game_month_day", "shot_made_flag", data=df_input)
plt.title("Efficiency vs Day of the month")
ax = fig.add_subplot(1, 4, 3)
sb.barplot("game_weekday", "shot_made_flag", data=df_input)
plt.title("Efficiency vs Day of the week")
ax = fig.add_subplot(1, 4, 4)
sb.barplot("season", "shot_made_flag", data=df_input)
plt.title("Efficiency vs season")
plt.show()


# All variables seem relevant (except maybe the day of the month): Kobe is clearly worse on mondays (why?) and shoots worse in June/July (end of playoffs) and March/April (post All Star weekend). Even more clearly, you can see well the different stages of his LA teams: the Shaq years were great (3rd to 6th season), as were the Gasol/Fisher years (~11th to 14th season). In other years he had to shoot much more, and the last years were very rough. Let's make some new variables: does the amount of rest influence efficiency? And when he plays back-to-back (2 game in a row) or back-to-back-to-back (3 games in a row)?

# In[ ]:


df_input["days_between_games"] = 7
str_days_between_games = [0]; str_game_id = [0];
    
days_in=7
for i in range(1, len(df_input.index)):
    if df_input.at[i, 'game_date'] != df_input.at[i-1, 'game_date']:
        days_in = min(abs((df_input.at[i, 'game_date'] - df_input.at[i-1, 'game_date']).days), 7) 
    df_input.at[i, "days_between_games"] = days_in

fig = plt.figure(figsize=(12, 6))
sb.barplot("days_between_games", "shot_made_flag", data=df_input)
plt.show()


# So, resting is not very useful. What about those (back-to-back)^N situations?

# In[ ]:


df_input["days_between_games"] = 7
str_days_between_games = [0]; str_game_id = [0];
    
days_in=7
for i in range(1, len(df_input.index)):
    if df_input.at[i, 'game_date'] != df_input.at[i-1, 'game_date']:
        days_in = min(abs((df_input.at[i, 'game_date'] - df_input.at[i-1, 'game_date']).days), 7) 
    df_input.at[i, "days_between_games"] = days_in

fig = plt.figure(figsize=(12, 6))
sb.barplot("days_between_games", "shot_made_flag", data=df_input)
plt.show()


# This is more promising: when making three games in three days there seems to be a decrease in efficiency (though statistics are low, but we will keep this variable). We already know that throughout the season there are some seasonal effects in efficiency seen in month. A factor we haven't looked at yet is whether being in the playoffs (and in each of the rounds) makes a difference.

# In[ ]:


for i in range(0, len(df_input.index)):
   df_input.at[i, 'game_date_N'] = int(df_input.at[i, 'game_date'].strftime("%d%m%Y"))

df_po = df_input[df_input["playoffs"]==1]
uniq_series_opp = []; uniq_series_key = [];

for season in df_po["season"].unique():
    data_season = df_po[df_po["season"]==season]
    uniq_season = data_season["opponent_N"].unique()
    for opp in range(0,len(uniq_season)):
        data_opp = data_season["game_date_N"][df_po["opponent_N"]==uniq_season[opp]]
        for game in range(0,len(data_opp.unique())):
            date=data_opp.unique()[game]
            uniq_series_opp.append([int(season), int(uniq_season[opp]), date])
            uniq_series_key.append([opp+1, game+1])
            
df_input["game_in_series"] = 0
df_input["playoff_series"] = 0
for i in range(0, len(df_input.index)):
    if df_input.at[i, "playoffs"]==0:
        continue
    for ik in range(0, len(uniq_series_key)):
        if uniq_series_opp[ik][0] == int(df_input.at[i, "season"])         and uniq_series_opp[ik][1] == df_input.at[i, "opponent_N"]         and uniq_series_opp[ik][2] == df_input.at[i, "game_date_N"]:
            df_input.at[i, "playoff_series"] = uniq_series_key[ik][0]
            df_input.at[i, "game_in_series"] = uniq_series_key[ik][1]
            
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(1, 2, 1)
sb.barplot("playoff_series", "shot_made_flag", data=df_input)
plt.title("Efficiency by playoff series (reg. season = 0)")
ax = fig.add_subplot(1, 2, 2)
sb.barplot("game_in_series", "shot_made_flag", data=df_input)
plt.title("Efficiency by game in series (reg. season = 0)")
plt.show()


# As expected, Kobe shoots less efficiently in the finals and in game 7, because the games slow down and teams defend more much aggressively. However, it is surprising that actually in the conference finals the efficiency is higher than normal, as is in the second games of series.

# In[ ]:


untreated_variables.remove("playoffs"); untreated_variables.remove("season"); untreated_variables.remove("game_date"); 
df_input = df_input.drop(["playoffs", "game_date", "game_date_N", "game_month_day"], 1)

print(untreated_variables)


# Finally, only in-game time variables left to treat! Game event id can be dropped, as it is just a counter of all shots on game. We will make a more significant counter on Kobe shots. Seconds and minutes remaining can also be combined.

# In[ ]:


df_input["seconds_remaining"] = df_input["seconds_remaining"] + 60*df_input["minutes_remaining"]
bin_sec = [120, 60, 30, 10, 5, 3]; bin_pos = [480, 240, 60, 20, 11, 3];
for i in range(0,len(df_input.index)):
    for bin in range(0,len(bin_sec)):
        if(df_input.at[i, "seconds_remaining"]>bin_pos[bin]):
            df_input.at[i, "seconds_remaining"] = bin_sec[bin]*int(df_input.at[i, "seconds_remaining"]/bin_sec[bin])    
df_input = df_input.drop(["minutes_remaining", "game_event_id"], 1)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
sb.barplot("seconds_remaining", "shot_made_flag", data=df_input)
plt.title("Efficiency vs seconds left")
ax = fig.add_subplot(1, 2, 2)
sb.barplot("period", "shot_made_flag", data=df_input)
plt.title("Efficiency vs period")
plt.show()


# Two main things are obvious from these plots: efficiency goes down in he last seconds of the same and on the last (fourth) period. In overtime there are not a lot of statistics. Finally, we introduce another engineered variable: whether the fact that he made the last shots is a plus (confidence) ou a minus (over-confidence) on his shot selection.

# In[ ]:


df_input["made_in_a_row"]=np.nan
df_input["made_last"]=np.nan
for i in range(2, len(df_input.index)):
    compar = df_input.at[i-1, "shot_made_flag"]
    counter = 1
        
    if np.isnan(compar):
        continue
        
    keep_loop=True
    while keep_loop==True:
        prev_val = df_input.at[i-counter-1, "shot_made_flag"]
        if np.isnan(prev_val) or prev_val!=compar:
            keep_loop=False
        else:
            counter=counter+1
    df_input.at[i, "made_last"] = (2*compar-1)*min(counter, 1) 
    df_input.at[i, "made_in_a_row"] = (2*compar-1)*min(counter, 5) 
    
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1,2,1)
sb.barplot("made_in_a_row", "shot_made_flag", data=df_input)
plt.title("Efficiency vs previous shots success")
plt.ylim(0.3, 0.6)
ax = fig.add_subplot(1,2,2)
sb.barplot("made_last", "shot_made_flag", data=df_input)
plt.title("Efficiency vs previous shot success")
plt.ylim(0.35, 0.5)
plt.show()

df_input = df_input.drop(["made_last", "made_in_a_row"], 1)


# It seems like Kobe gets over-confident: when he made his last shot (and especially when he made his last 3 or 4 shots he is more likely to miss the next. However, this is not a very strong effect, so we will not include the variables in our model. Finally, we can check the correlation plot in our new, treated database

# In[ ]:


df_input = df_input.drop(["loc_x_0", "loc_x_1", "loc_y_0", "loc_y_1", "eff_shot_made_flag"], 1)
upper_number_branches = 1
for col in df_input.columns.values:
  if col!="shot_id" and col!="shot_made_flag":
    upper_number_branches = upper_number_branches*len(df_input[str(col)].unique())
print("Possible combinations decreased from 4e+35 to", "{:.0e}".format(upper_number_branches))   
make_pretty_corr_plot(df_input)


# We managed to achieve a strong decrease in dimensionality (a factor of 10e20!!) which will certainly make the life of our classifier easier. Also, the correlation plot shows that apart from the playoff series and game in series there are not very correlated variables in the dataset. So, we can go ahead and predict the missing shots. This seems like a job for either a random forest or a boosted tree classifier. Since we have many events (~30.000) we can try to separate in a train and test and validate each of the models, as well as get a accuracy prediction for both and choose the best.

# In[ ]:


unknown_mask = df_input['shot_made_flag'].isnull()
data_cl = df_input.copy()
target = data_cl['shot_made_flag'].copy()

X = data_cl[~unknown_mask]
X = X.drop(["shot_id","shot_made_flag"], 1);
Y = target[~unknown_mask]


seed = 9
test_size = 0.25
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
# make predictions for test data
y_pred = model_xgb.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy XGB: %.2f%%" % (accuracy * 100.0))

model_rf = RandomForestClassifier(n_estimators=1000, max_features="sqrt", min_samples_leaf=50)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy Random Forest: %.2f%%" % (accuracy * 100.0))


# Submitting these results gives a logLoss score of around 0.60, which is pretty respectable. More importantly, we did so with few variables that we can understand, so if Kobe came around to ask what we think we might have some answers!
