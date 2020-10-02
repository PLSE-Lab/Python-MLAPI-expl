#!/usr/bin/env python
# coding: utf-8

# # **Predicting Player Position using Football Manager data**
# Using attributes from [Football Manager 2017 data](https://www.kaggle.com/ajinkyablaze/football-manager-data) we predict player position.
# 
# Only 1 position for each player was selected. This is a limitation, as some players are highly versatile and would have more than just 1 optimal position.
# 
# These positions were used:
# * GK
# * CB
# * WB (both right / left, both WB and FB)
# * WM (both right / left, both WM and AWM)
# * DM
# * CM
# * AM
# * ST
# 

# First, we import libraries and load the data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, InputLayer, Flatten

from sklearn.model_selection import cross_validate


# In[ ]:


md = pd.read_csv("../input/dataset.csv", index_col = 'UID')
md.head(3)


# We can see that there are way too many data here. Let's drop unnecessary columns

# In[ ]:


players = md.drop(['Name', 'NationID', 'Born', 'IntCaps', 'IntGoals', 'U21Caps', 'U21Goals', 'PositionsDesc',
                   'Consistency', 'Dirtiness', 'ImportantMatches', 'Versatility', 'Adaptability', 'Ambition',
                   'Loyalty', 'Pressure', 'Professional', 'Sportsmanship', 'Temperament', 'Controversy',
                   'Age', 'Weight', 'Height', 'InjuryProness'
            ], axis=1)
players.head(3)


# Ok, that's better. We kept most of the attributes and the positions. Some mental attributes were removed,because they are not that important for player position and are not even visible in the game itself
# 
# Next, we split the dataframe into **X** (attributes) and **y** (scores for each position out of 20). We also separate 'RightFoot' and 'LeftFoot' into a **X_foot** dataframe, which will be used later

# In[ ]:


X = players.loc[:,:'Strength'].drop(['RightFoot', 'LeftFoot'], axis=1)
X_foot = players.loc[:, ['RightFoot', 'LeftFoot']]
y = players.loc[:,'Goalkeeper':]


# In[ ]:


X.head(3)


# In[ ]:


X_foot.head(3)


# In[ ]:


y.head(3)


# There is no need to scale columns because all data are in the same scale (0-20). However, it might be a good idea to scale rows, because we are not interested in player overall level to be a factor in position prediction. For example, Eden Hazard might be a better striker than Emile Heskey, but we want Heskey's predicted ST level to be higher. Having said that, we need to exclude 'RightFoot' and 'LeftFoot' from this, as at least one of these values would always be 20

# In[ ]:


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
vectors = X.values
scaled_rows = scaler.fit_transform(vectors.T).T
X_normalized = pd.DataFrame(data = scaled_rows, columns = X.columns)


# Ok, now we have our variables, let's visualise them

# In[ ]:


fig, axes = plt.subplots(len(X_normalized.columns)//3, 3, figsize=(12, 48))

i = 0
for triaxis in axes:
    for axis in triaxis:
        X.hist(column = X_normalized.columns[i], bins = 100, ax=axis)
        i = i+1


# Some attributes are very goalkeeper-speciffic, but other attributes are relatively normally distributed

# In[ ]:


for col in y.columns.unique():
    print(col, y[col][y[col] == 20].count())


# Some Models

# In[ ]:


XGRegModel = MultiOutputRegressor(XGBRegressor(objective='reg:squarederror'))
cv_results_XGRegModel = cross_validate(XGRegModel, X, y, cv=5, verbose=1)
plt.plot(cv_results_XGRegModel['test_score'])


# In[ ]:


XGRegModel.fit(X, y)


# In[ ]:


NNetModel = Sequential()

NNetModel.add(Dense(len(X.columns), activation='relu', input_dim=len(X.columns)))
NNetModel.add(BatchNormalization())
NNetModel.add(Dropout(0.2))
NNetModel.add(Dense((len(X.columns) + len(y.columns)) // 2, activation='relu'))
NNetModel.add(Flatten())
NNetModel.add(Dense(len(y.columns), activation='relu'))

NNetModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

NNetModel.fit(X.values, y.values, batch_size=128, epochs=50, verbose=1, validation_split=0.2)


# Let's see some predictions

# In[ ]:


def predict_player_position(uid, model):
    player_id = uid

    positions = model.predict(X.loc[[player_id]])[0]

    R_scaler = preprocessing.MinMaxScaler(feature_range=(1, 20))
    vectors = pd.DataFrame(positions.reshape(1,15), columns=y.columns).values
    scaled_rows = R_scaler.fit_transform(vectors.T).T

    results = pd.DataFrame({'Position':y.columns, 'Real':y.loc[player_id], 'Predicted':scaled_rows[0], 'Difference':y.loc[player_id]-scaled_rows[0]})
    results.sort_values('Real', ascending=False)
    
    accuracy = abs(results['Difference']).median()
    
    return results, accuracy


# In[ ]:


def draw_positions(results, p_name=''):

    fig, ax = plt.subplots(1,2)

    x_coords = [10,10,10,10,-45,65,10,-45,65,10,10,-45,65,-45,65]
    y_coords = [-85,-55,95,65,65,65,-25,-25,-25,5,35,35,35,5,5]
    size = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    patches = []
    for x1,y1,r in zip(x_coords, y_coords, size):
        circle = Circle((x1,y1), r)
        patches.append(circle)

    colors1 = results['Predicted']
    colors2 = results['Real']
    newcmp = ListedColormap(['red', 'orange', 'yellow', 'green'])
    p1 = PatchCollection(patches, cmap=newcmp, alpha=1)
    p2 = PatchCollection(patches, cmap=newcmp, alpha=1)
    p1.set_array(colors1)
    p2.set_array(colors2)

    ax[0].set_ylim((-110,120))
    ax[0].set_xlim((-70,90))
    ax[0].add_collection(p1)
    ax[0].set_title('Predicted')

    ax[1].set_ylim((-110,120))
    ax[1].set_xlim((-70,90))
    ax[1].add_collection(p2)
    ax[1].set_title('Real')

    fig.suptitle(p_name)


# In[ ]:


# to see where mistakes were made:
def show_predictions(predicted_pos, true_pos):
    return pred_test_table[pred_test_table['True_position'] == true_pos][pred_test_table[pred_test_table['True_position'] == true_pos]['Pred_position'] == predicted_pos]

# to get specific attributes using player UID:
def get_attribute(UID, attribute):
    return md[md.index == UID][attribute].tolist()[0]

# to get data using names:
def search_by_name(name):
    return md[md['Name'].str.contains(name)]


# In[ ]:


# to get attributes for FM-like shapes for players:
def get_shape_attributes(UID):
    shape_attributes = {}
    shape_attributes['Speed'] = (get_attribute(UID, 'Acceleration') + get_attribute(UID, 'Pace')) / 2
    shape_attributes['Physical'] = (get_attribute(UID, 'Agility') + get_attribute(UID, 'Balance')
                    + get_attribute(UID, 'Stamina') + get_attribute(UID, 'Strength')) / 4
    shape_attributes['Defence'] = (get_attribute(UID, 'Tackling') + get_attribute(UID, 'Marking') + get_attribute(UID, 'Positioning')) / 3
    shape_attributes['Mental'] = (get_attribute(UID, 'Anticipation') + get_attribute(UID, 'Bravery')
                + get_attribute(UID, 'Concentration') + get_attribute(UID, 'Decisions') + get_attribute(UID, 'Determination') + get_attribute(UID, 'Teamwork')) / 6
    shape_attributes['Aerial'] = (get_attribute(UID, 'Heading') + get_attribute(UID, 'Jumping')) / 2
    shape_attributes['Technique'] = (get_attribute(UID, 'Dribbling') + get_attribute(UID, 'FirstTouch') + get_attribute(UID, 'Technique')) / 3
    shape_attributes['Attack'] = (get_attribute(UID, 'Finishing') + get_attribute(UID, 'Composure') + get_attribute(UID, 'OffTheBall')) / 3
    shape_attributes['Vision'] = (get_attribute(UID, 'Passing') + get_attribute(UID, 'Flair') + get_attribute(UID, 'Vision')) / 3
      
    return shape_attributes

# same for keepers:
def get_shape_attributes_GK(UID):
    shape_attributes = {}
    shape_attributes['Speed'] = (get_attribute(UID, 'Acceleration') + get_attribute(UID, 'Pace')) / 2
    shape_attributes['Physical'] = (get_attribute(UID, 'Agility') + get_attribute(UID, 'Balance')
                + get_attribute(UID, 'Stamina') + get_attribute(UID, 'Strength')) / 4
    shape_attributes['ShotStopping'] = (get_attribute(UID, 'Handling') + get_attribute(UID, 'OneOnOnes')
                + get_attribute(UID, 'Reflexes') + get_attribute(UID, 'Positioning')) / 4
    shape_attributes['Distribution'] = (get_attribute(UID, 'FirstTouch') + get_attribute(UID, 'Throwing')) / 2
    shape_attributes['Aerial_GK'] = (get_attribute(UID, 'Jumping') + get_attribute(UID, 'AerialAbility')) / 2
    shape_attributes['Eccentricity'] = (get_attribute(UID, 'Eccentricity'))
    shape_attributes['Communication'] = (get_attribute(UID, 'Communication') + get_attribute(UID, 'RushingOut') + get_attribute(UID, 'CommandOfArea')) / 3
    shape_attributes['Mental'] = (get_attribute(UID, 'Anticipation') + get_attribute(UID, 'Bravery')
                + get_attribute(UID, 'Concentration') + get_attribute(UID, 'Decisions') + get_attribute(UID, 'Determination') + get_attribute(UID, 'Teamwork')) / 6
    
    return shape_attributes


# Attributes were used without combining for training and predicting, however for visualization purposes they were combined into groups:
# 
# **Speed group:**
# * Acceleration
# * Pace
# 
# **Defence group:**
# * Tackling
# * Marking
# * Positioning
# 
# **Mental group:**
# * Anticipation
# * Bravery
# * Concentration
# * Decisions
# * Determination
# * Teamwork
# 
# **Aerial group:**
# * Heading
# * Jumping
# 
# **Technique group:**
# * Dribbling
# * FirstTouch
# * Technique
# 
# **Attack group:**
# * Finishing
# * Composure
# * OffTheBall
# 
# **Vision group:**
# * Passing
# * Flair
# * Vision
# 
# This was also done for Goalkeepers, but I am not sure if correctly

# In[ ]:


# to draw FM-like shapes for players:
def draw_shape(UID):
    
    values_dict = get_shape_attributes(UID)

    labels = np.array(list(values_dict.keys()))
    stats = list(values_dict.values())
    name = get_attribute(UID, 'Name')
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats.append(stats[0])
    angles = np.concatenate((angles,[angles[0]]))

    fig= plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_title(name)
    ax.set_yticks(np.arange(0,25,10))

    plt.show()

# Same for keepers:
def draw_shape_GK(UID):
    
    values_dict = get_shape_attributes_GK(UID)

    labels = np.array(list(values_dict.keys()))
    stats = list(values_dict.values())
    name = get_attribute(UID, 'Name') + ' - GK'
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats.append(stats[0])
    angles = np.concatenate((angles,[angles[0]]))

    fig= plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_title(name)
    ax.set_yticks(np.arange(0,25,10))

    plt.show()


# In[ ]:


# to make a prediction for a player using UID:
def predict_and_show(uid, model, p_name=''):
    results = predict_player_position(uid, model)
    print('MAD =', results[1])
    draw_positions(results[0], p_name)
    draw_shape(uid)
    draw_shape_GK(uid)


# In[ ]:


search_by_name('Mohamed Salah')


# In[ ]:


predict_and_show(98028755, NNetModel, 'Mohamed Salah')


# In[ ]:


predict_and_show(98028755, XGRegModel, 'Mohamed Salah')


# In[ ]:




