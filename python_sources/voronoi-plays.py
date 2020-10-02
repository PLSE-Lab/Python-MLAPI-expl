#!/usr/bin/env python
# coding: utf-8

# # NFL Big Data Bowl 
# # Plays as colorized Voronoi Diagrams
# ### Hopefully will help to understand the relationship between the controlled areas by the teams and the yards gained
# 
# #### (All X and Y values and directions are standardized...)
# 
# Diagrams shows voronoi region for each player colored as:
# * red    : rusher
# * green  : offense players (with black dots)
# * yellow : defense players (with red dots)
#     
# Rusher direction and speed shown as black arrow.
# 
# Left red line is the scrimmage line. (Yards lost if Yards is negatif)
# 
# Right red line shows yards gained at the end of play.(scrimmage line if Yards is negatif)
# 
# Usage : voronoi_display(df, yards_filter=2, count_limit=10):
# * df : standardized dataframe
# * yards_filter : filter to select plays according to yards gained
# * count_limit : number of plays to draw

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from scipy.spatial import Voronoi, voronoi_plot_2d


# In[ ]:


#from https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
def get_dx_dy(radian_angle, dist):
    dx = dist * math.cos(radian_angle)
    dy = dist * math.sin(radian_angle)
    return dx, dy


# In[ ]:


#from https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


# In[ ]:


def visualize_play(rusher, ptype, vor):
        x = rusher.X_std.values[0]
        y = rusher.Y_std.values[0]
        dx, dy = get_dx_dy(rusher.Dir_rad.values[0], rusher.S.values[0])
        #fig = voronoi_plot_2d(vor)
        regions, vertices = voronoi_finite_polygons_2d(vor)
        # colorize
        for i, region in enumerate(regions):
            if ptype[i]:
                color = "g"
            else:
                color = "y"
            if i == 0:
                color = "r"
            polygon = vertices[region]
            plt.fill(*zip(*polygon), color= color, alpha=0.4)
        for i, point in enumerate(vor.points):
            if ptype[i]:
                plt.plot(point[0], point[1], "ko")
            else:
                plt.plot(point[0], point[1], "ro")
        plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
        plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
        ax = plt.gca()
        ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.1, color='black')
        scrimage_line = rusher.loc[:,["YardLine_std"]].values
        down_line = scrimage_line + rusher.loc[:,["Yards"]].values
        plt.axvline(x=scrimage_line, linewidth=1, color='r')
        plt.axvline(x=down_line, linewidth=1, color='r')
        plt.show()    


# In[ ]:


def voronoi_display(df, yards_filter=2, count_limit=10):
    count = 0
    for name, play in df[df.Yards == yards_filter].groupby("PlayId"):
        rusher = play[play.IsBallCarrier]
        #print(play[play.IsBallCarrier].Dir_std)
        carrier_loc = play[play.IsBallCarrier].loc[:,["X_std","Y_std"]].values
        ctype = play[play.IsBallCarrier].loc[:,["IsOnOffense"]].values
        #defense_loc = play[~play.IsOnOffense].loc[:,["X_std","Y_std"]].values
        notcarrier_loc = play[~play.IsBallCarrier].loc[:,["X_std","Y_std"]].values
        nctype = play[~play.IsBallCarrier].loc[:,["IsOnOffense"]].values
        ptype = np.append(ctype, nctype, axis=0)
        points = np.append(carrier_loc, notcarrier_loc, axis=0)
        vor = Voronoi(points)
        visualize_play(rusher, ptype, vor)
        count += 1
        if count == count_limit:
            break


# In[ ]:


def data_prep(df):
    # from https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
    df['ToLeft'] = df.PlayDirection == "left"
    df['IsBallCarrier'] = df.NflId == df.NflIdRusher
    
    df.loc[df.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    df.loc[df.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

    df.loc[df.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    df.loc[df.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

    df.loc[df.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    df.loc[df.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

    df.loc[df.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    df.loc[df.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

    
    # from https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python
    df['TeamOnOffense'] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    df['IsOnOffense'] = df.Team == df.TeamOnOffense  # Is player on offense?
    df['YardLine_std'] = 100 - df.YardLine
    df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,
                 'YardLine_std'
    ] = df.loc[df.FieldPosition.fillna('') == df.PossessionTeam,
                     'YardLine']
    df['X_std'] = df.X - 10
    df.loc[df.ToLeft, 'X_std'] = 120 - df.loc[df.ToLeft, 'X'] - 10
    df['Y_std'] = df.Y
    df.loc[df.ToLeft, 'Y_std'] = 160 / 3 - df.loc[df.ToLeft, 'Y']
    df['Orientation_std'] = df.Orientation
    df.loc[df.ToLeft, 'Orientation_std'] = np.mod(180 + df.loc[df.ToLeft, 'Orientation_std'],
                                                              360)
    df['Dir_std'] = df.Dir
    df.loc[df.ToLeft, 'Dir_std'] = np.mod(180 + df.loc[df.ToLeft, 'Dir_std'], 360)

    df['Dir_rad'] = np.mod(90 - df.Dir_std, 360) * math.pi/180.0

    df.DefendersInTheBox.fillna(7.0, inplace=True)
    df.Dir.fillna(0.0, inplace=True)
    df.FieldPosition.fillna(df.PossessionTeam, inplace=True)
    df.GameWeather.fillna("Sunny", inplace=True)
    df.Humidity.fillna(0, inplace=True)
    df.OffenseFormation.fillna("SINGLEBACK", inplace=True)
    df.StadiumType.fillna("Outdoor", inplace=True)
    df.StadiumType.replace("Outdoors", "Outdoor", inplace=True)
    df.Temperature.fillna(62, inplace=True)
    df.WindDirection.fillna("NE", inplace=True)
    df.WindSpeed.fillna(5, inplace=True)
    
    return df


# In[ ]:


dcols = {'GameId':np.int64, 'PlayId':np.int64, 'Team':str, 'X':np.float64, 'Y':np.float64, 
         'S':np.float64, 'A':np.float64, 'Dis':np.float64, 'Orientation':np.float64, 'Dir':np.float64, 
         'NflId':np.int64, 'DisplayName':str, 'JerseyNumber':np.int32, 'Season':np.int32, 'YardLine':np.int32, 
         'Quarter':np.int32, 'PossessionTeam':str, 'Down':np.int32, 'Distance':np.int32, 'FieldPosition':str,
         'HomeScoreBeforePlay':np.int32, 'VisitorScoreBeforePlay':np.int32, 'NflIdRusher':np.int64, 
         'OffenseFormation':str, 'OffensePersonnel':str, 'DefendersInTheBox':np.float32, 
         'DefensePersonnel':str, 'PlayDirection':str, 'Yards':np.int32, 'PlayerHeight':str, 
         'PlayerWeight':np.int32, 'PlayerCollegeName':str, 'Position':str, 'HomeTeamAbbr':str, 
         'VisitorTeamAbbr':str, 'Week':np.int32, 'Stadium':str, 'Location':str, 'StadiumType':str, 
         'Turf':str, 'GameWeather':str, 'Temperature':np.float32, 'Humidity':np.float32, 'WindSpeed':str, 
         'WindDirection':str}


# In[ ]:


seed = 42
np.random.seed(seed)
NFL_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype=dcols,
                       #parse_dates=date_spec,
                       low_memory=False)
NFL_df = data_prep(NFL_df)


# In[ ]:


voronoi_display(NFL_df, yards_filter=3, count_limit=5)


# In[ ]:




