#!/usr/bin/env python
# coding: utf-8

# 2019 Data Science Bowl
# ===
# Damien Park  
# 2019.11.14

# In[ ]:


import pandas as pd
import numpy as np

import json

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import keras

# import pprint
import gc
import matplotlib.pyplot as plt

# pandas display option
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_row', 1500)
pd.set_option('max_colwidth', 150)
pd.set_option('display.float_format', '{:.2f}'.format)


# In[ ]:


train = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")
# test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")
label = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")
# sample = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")
specs = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")


# In[ ]:


df = train.merge(label, how="left", on=["installation_id", "game_session", "title"])
df.timestamp = pd.to_datetime(df.timestamp)
df.sort_values("timestamp", ascending=False, inplace=True)
del train, label
gc.collect()


# In[ ]:


df.head()


# ---
# # Feature Engineering

# ## World

# In[ ]:


world = df.groupby(["installation_id", "world"]).size().unstack().reset_index().fillna(0)
world.columns.name = None


# In[ ]:


len(world), len(pd.unique(world.installation_id))


# In[ ]:


plt.figure(figsize=(30, 10))
plt.subplot(2, 4, 1)
plt.hist(world.CRYSTALCAVES)
plt.title("CRYSTALCAVES")
plt.subplot(2, 4, 2)
plt.hist(np.log(world.CRYSTALCAVES+1))
plt.title("CRYSTALCAVES(log)")

plt.subplot(2, 4, 3)
plt.hist(world.MAGMAPEAK)
plt.title("MAGMAPEAK")
plt.subplot(2, 4, 4)
plt.hist(np.log(world.MAGMAPEAK+1))
plt.title("MAGMAPEAK(log)")

plt.subplot(2, 4, 5)
plt.hist(world.NONE)
plt.title("NONE")
plt.subplot(2, 4, 6)
plt.hist(np.log(world.NONE+1))
plt.title("NONE(log)")

plt.subplot(2, 4, 7)
plt.hist(world.TREETOPCITY)
plt.title("TREETOPCITY")
plt.subplot(2, 4, 8)
plt.hist(np.log(world.TREETOPCITY+1))
plt.title("TREETOPCITY(log)")

plt.show()


# In[ ]:


world.iloc[:, 1:] = np.log(world.iloc[:, 1:]+1)


# In[ ]:


# display(len(df[df.world=="NONE"])), 
# display(len(df[df.world=="NONE"].installation_id), len(pd.unique(df[df.world=="NONE"].installation_id)))
# display(len(df[df.world=="NONE"].game_session), len(pd.unique(df[df.world=="NONE"].game_session)))


# ## Event_count
# Event count is not always monotonic

# In[ ]:


event_count = df.groupby(["installation_id", "game_session"]).event_count.is_monotonic_decreasing


# In[ ]:


event_count


# ## Event_code

# In[ ]:


event_code = df.groupby(["installation_id", "event_code"]).size().unstack().reset_index().fillna(0)
event_code.columns.name = None
event_code = event_code.add_prefix("x_")
event_code.rename(columns={"x_installation_id":"installation_id"}, inplace=True)


# In[ ]:


len(event_code), len(pd.unique(event_code.installation_id))


# In[ ]:


plt.figure(figsize=(30, 20))
for idx, val in enumerate(event_code.columns[1:]):
    plt.subplot(6, 7, idx+1)
    plt.hist(event_code[val])
    plt.title(val)


# In[ ]:


plt.figure(figsize=(30, 20))
for idx, val in enumerate(event_code.columns[1:]):
    plt.subplot(6, 7, idx+1)
    plt.hist(np.log(event_code[val]+1))
    plt.title(val)


# In[ ]:


event_code.iloc[:, 1:] = np.log(event_code.iloc[:, 1:]+1)


# ## Game time

# In[ ]:


game_time = pd.DataFrame(df.groupby(["installation_id", "game_session"]).game_time.max()).reset_index()
game_time = game_time.groupby("installation_id").mean().reset_index()


# In[ ]:


len(game_time), len(pd.unique(game_time.installation_id))


# In[ ]:


plt.figure(figsize=(40, 5))
plt.subplot(1, 4, 1)
plt.hist(game_time.game_time, bins=100)
plt.subplot(1, 4, 2)
plt.hist(np.log(game_time.game_time+1), bins=100)
plt.subplot(1, 4, 3)
plt.hist(np.log(np.cbrt(game_time.game_time+1)), bins=100)
plt.subplot(1, 4, 4)
plt.hist(np.log(np.cbrt(game_time.game_time+100)+1), bins=100)
plt.show()


# In[ ]:


# game_time.iloc[:, 1:] = np.log(game_time.iloc[:, 1:]+1)
game_time.iloc[:, 1:] = np.log(np.cbrt(game_time.iloc[:, 1:]+100)+1)


# ## Session length

# In[ ]:


session = pd.DataFrame(df.groupby(["installation_id"]).game_session.nunique()).reset_index()
session.columns.name = None


# In[ ]:


len(session), len(pd.unique(session.installation_id))


# In[ ]:


plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.hist(session.game_session, bins=50)
plt.subplot(1, 3, 2)
plt.hist(np.log(session.game_session+10), bins=50)
plt.subplot(1, 3, 3)
plt.hist(np.cbrt(session.game_session+10), bins=50)
plt.show()


# In[ ]:


session.iloc[:, 1:] = np.cbrt(session.iloc[:, 1:]+10)


# ## Type

# In[ ]:


types = df.groupby(["installation_id", "type"]).type.size().unstack().reset_index().fillna(0)
types.columns.name = None


# In[ ]:


plt.figure(figsize=(40, 7))
for idx, val in enumerate(types.columns[1:]):
    plt.subplot(1, 4, idx+1)
    plt.hist(types[val], bins=50)
    plt.title(val)


# In[ ]:


plt.figure(figsize=(40, 7))
for idx, val in enumerate(types.columns[1:]):
    plt.subplot(1, 4, idx+1)
    plt.hist(np.log(types[val]+1), bins=50)
    plt.title(val)


# In[ ]:


types.iloc[:, 1:] = np.log(types.iloc[:, 1:]+1)


# ## Title

# In[ ]:


title = df.groupby(["installation_id", "title"]).size().unstack().reset_index().fillna(0)
title.columns.name = None


# In[ ]:


len(title), len(pd.unique(title.installation_id))


# In[ ]:


plt.figure(figsize=(30, 20))
for idx, val in enumerate(title.columns[1:]):
    plt.subplot(7, 7, idx+1)
    plt.hist(title[val])
    plt.title(val)


# In[ ]:


plt.figure(figsize=(30, 20))
for idx, val in enumerate(title.columns[1:]):
    plt.subplot(7, 7, idx+1)
    plt.hist(np.log2(title[val]+1))
    plt.title(val)


# In[ ]:


plt.figure(figsize=(30, 20))
for idx, val in enumerate(title.columns[1:]):
    plt.subplot(7, 7, idx+1)
    plt.hist(np.log(title[val]+1), bins=50)
    plt.title(val)


# ## Last Assessment

# In[ ]:


assessment = df[df.type=="Assessment"].groupby(["installation_id"]).title.last().reset_index()

assessment['Assessment_1'] = 0
assessment['Assessment_2'] = 0
assessment['Assessment_3'] = 0
assessment['Assessment_4'] = 0
assessment['Assessment_5'] = 0

assessment.loc[assessment.title=='Mushroom Sorter (Assessment)', 'Assessment_1'] = 1
assessment.loc[assessment.title=='Cauldron Filler (Assessment)', 'Assessment_2'] = 1
assessment.loc[assessment.title=='Chest Sorter (Assessment)', 'Assessment_3'] = 1
assessment.loc[assessment.title=='Cart Balancer (Assessment)', 'Assessment_4'] = 1
assessment.loc[assessment.title=='Bird Measurer (Assessment)', 'Assessment_5'] = 1

del assessment["title"]


# In[ ]:


len(assessment), len(pd.unique(assessment.installation_id))


# ## Label
# 3: the assessment was solved on the first attempt  
# 2: the assessment was solved on the second attempt  
# 1: the assessment was solved after 3 or more attempts  
# 0: the assessment was never solved

# In[ ]:


# label = pd.DataFrame(df.groupby(["installation_id"]).accuracy_group.last()).reset_index()


# In[ ]:


accuracy = df.query("((event_code==2000) or (event_code==4100 and title!='Bird Measurer (Assessment)') or                      (event_code==4110 and title=='Bird Measurer (Assessment)')) and (type=='Assessment')").reset_index(drop=True)

accuracy["event_data_json"] = accuracy["event_data"].apply(lambda x: json.loads(x))

accuracy["num_incorrect"] = accuracy["event_data_json"].apply(lambda x: (0 if x["correct"] else 1) if "correct" in x  else 0)
accuracy["num_correct"] = accuracy["event_data_json"].apply(lambda x: (1 if x["correct"] else 0)  if "correct" in x  else 0)

accuracy = accuracy.groupby(["installation_id", "game_session"]).agg(num_correct_pred = ("num_correct", "max"), num_incorrect_pred = ("num_incorrect", "sum"), ).reset_index()
accuracy["accuracy_group_pred"] = accuracy["num_incorrect_pred"].apply(lambda x: 3 if x == 0 else (2 if x == 1 else 1)) * accuracy["num_correct_pred"]

accuracy = accuracy.groupby(["installation_id"]).last().reset_index()
accuracy.drop("game_session", axis=1, inplace=True)


# In[ ]:


# _ = label[label.accuracy_group.isnull()].merge(accuracy, on=["installation_id"], how="left")
label = accuracy.rename(columns={"accuracy_group_pred":"accuracy_group"})


# ---
# # Merge all data set

# In[ ]:


_ = world.merge(event_code, how="left", on=["installation_id"])
_ = _.merge(game_time, how="left", on=["installation_id"])
_ = _.merge(session, how="left", on=["installation_id"])
_ = _.merge(types, how="left", on=["installation_id"])
_ = _.merge(title, how="left", on=["installation_id"])
_ = _.merge(assessment, how="left", on=["installation_id"])
_ = _.merge(label, how="left", on=["installation_id"])

train_x_col = ['installation_id', 
               'CRYSTALCAVES', 'MAGMAPEAK', 'NONE', 'TREETOPCITY',
               'x_2000', 'x_2010', 'x_2020', 'x_2025', 'x_2030', 'x_2035', 'x_2040',
               'x_2050', 'x_2060', 'x_2070', 'x_2075', 'x_2080', 'x_2081', 'x_2083',
               'x_3010', 'x_3020', 'x_3021', 'x_3110', 'x_3120', 'x_3121', 'x_4010',
               'x_4020', 'x_4021', 'x_4022', 'x_4025', 'x_4030', 'x_4031', 'x_4035',
               'x_4040', 'x_4045', 'x_4050', 'x_4070', 'x_4080', 'x_4090', 'x_4095',
               'x_4100', 'x_4110', 'x_4220', 'x_4230', 'x_4235', 'x_5000', 'x_5010',
               'game_session', 'game_time', 
               'Activity', 'Assessment', 'Clip', 'Game', 
               '12 Monkeys', 'Air Show', 'All Star Sorting', 'Balancing Act', 
               'Bird Measurer (Assessment)', 'Bottle Filler (Activity)', 'Bubble Bath', 
               'Bug Measurer (Activity)', 'Cart Balancer (Assessment)', 'Cauldron Filler (Assessment)',
               'Chest Sorter (Assessment)', 'Chicken Balancer (Activity)', 'Chow Time',
               'Costume Box', 'Crystal Caves - Level 1', 'Crystal Caves - Level 2',
               'Crystal Caves - Level 3', 'Crystals Rule', 'Dino Dive', 'Dino Drink',
               'Egg Dropper (Activity)', 'Fireworks (Activity)', 'Flower Waterer (Activity)', 
               'Happy Camel', 'Heavy, Heavier, Heaviest', 'Honey Cake', 'Leaf Leader', 
               'Lifting Heavy Things', 'Magma Peak - Level 1', 'Magma Peak - Level 2',
               'Mushroom Sorter (Assessment)', 'Ordering Spheres', 'Pan Balance',
               "Pirate's Tale", 'Rulers', 'Sandcastle Builder (Activity)', 'Scrub-A-Dub', 
               'Slop Problem', 'Treasure Map', 'Tree Top City - Level 1', 
               'Tree Top City - Level 2', 'Tree Top City - Level 3', 
               'Watering Hole (Activity)', 'Welcome to Lost Lagoon!', 
               'Assessment_1', 'Assessment_2', 'Assessment_3', 'Assessment_4', 'Assessment_5']
train_y_col = ["y_0", "y_1", "y_2", "y_3"]


# ---
# # One-Hot encoding / Scaling

# In[ ]:


_["y_0"] = 0
_["y_1"] = 0
_["y_2"] = 0
_["y_3"] = 0

_.loc[_.accuracy_group==0, "y_0"] = 1
_.loc[_.accuracy_group==1, "y_1"] = 1
_.loc[_.accuracy_group==2, "y_2"] = 1
_.loc[_.accuracy_group==3, "y_3"] = 1
_.dropna(inplace=True)
_ = _.reset_index(drop=True)
_.head()


# In[ ]:


scaler = StandardScaler()
scaler = MinMaxScaler()
train_x = scaler.fit_transform(_.loc[:, train_x_col[1:]])
train_y = _.loc[:, train_y_col]
train_x[0]


# ---
# # Modeling

# In[ ]:


model = keras.models.Sequential()

model.add(keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dropout(.3))

model.add(keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dropout(.3))

model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dropout(.3))

model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dropout(.3))

model.add(keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dropout(.3))

model.add(keras.layers.Dense(16, activation="relu", kernel_initializer="he_normal"))

model.add(keras.layers.Dense(4, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=['categorical_accuracy'])


# In[ ]:


# keras.backend.reset_uids()


# In[ ]:


model.fit(train_x, train_y.values, epochs=50, verbose=1, validation_split=.1, batch_size=10, shuffle=True)


# In[ ]:


plt.figure(figsize=(40, 20))
plt.subplot(2, 1, 1)
plt.plot(model.history.history["loss"], "o-", alpha=.4, label="loss")
plt.plot(model.history.history["val_loss"], "o-", alpha=.4, label="val_loss")
plt.axhline(1, linestyle="--", c="C2")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(model.history.history["categorical_accuracy"], "o-", alpha=.4, label="categorical_accuracy")
plt.plot(model.history.history["val_categorical_accuracy"], "o-", alpha=.4, label="val_categorical_accuracy")
plt.axhline(.7, linestyle="--", c="C2")
plt.legend()
plt.show()


# In[ ]:


def quadratic_kappa(actuals, preds, N=4):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 
    of adoption rating."""
    w = np.zeros((N,N))
    O = confusion_matrix(actuals, preds)
    for i in range(len(w)): 
        for j in range(len(w)):
            w[i][j] = float(((i-j)**2)/(N-1)**2)
    
    act_hist=np.zeros([N])
    for item in actuals: 
        act_hist[item]+=1
    
    pred_hist=np.zeros([N])
    for item in preds: 
        pred_hist[item]+=1
                         
    E = np.outer(act_hist, pred_hist);
    E = E/E.sum();
    O = O/O.sum();
    
    num=0
    den=0
    for i in range(len(w)):
        for j in range(len(w)):
            num+=w[i][j]*O[i][j]
            den+=w[i][j]*E[i][j]
    return (1 - (num/den))


# In[ ]:


_.accuracy_group = _.accuracy_group.astype("int")


# In[ ]:


result = model.predict(train_x)
quadratic_kappa(_.accuracy_group, result.argmax(axis=1))


# ---
# # Predict

# In[ ]:


# test
df = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")

# world
world = df.groupby(["installation_id", "world"]).size().unstack().reset_index().fillna(0)
world.columns.name = None
world.iloc[:, 1:] = np.log(world.iloc[:, 1:]+1)

# event_code
event_code = df.groupby(["installation_id", "event_code"]).size().unstack().reset_index().fillna(0)
event_code.columns.name = None
event_code = event_code.add_prefix("x_")
event_code.rename(columns={"x_installation_id":"installation_id"}, inplace=True)
event_code.iloc[:, 1:] = np.log(event_code.iloc[:, 1:]+1)

# game_time
game_time = pd.DataFrame(df.groupby(["installation_id", "game_session"]).game_time.max()).reset_index()
game_time = game_time.groupby("installation_id").mean().reset_index()
# game_time.iloc[:, 1:] = np.log(game_time.iloc[:, 1:]+1)
game_time.iloc[:, 1:] = np.log(np.cbrt(game_time.iloc[:, 1:]+100)+1)

# session length
session = pd.DataFrame(df.groupby(["installation_id"]).game_session.nunique()).reset_index()
session.columns.name = None
session.iloc[:, 1:] = np.cbrt(session.iloc[:, 1:]+10)

# type
types = df.groupby(["installation_id", "type"]).type.size().unstack().reset_index().fillna(0)
types.columns.name = None
types.iloc[:, 1:] = np.log(types.iloc[:, 1:]+1)

# title
title = df.groupby(["installation_id", "title"]).size().unstack().reset_index().fillna(0)
title.columns.name = None

# last assessment
assessment = df[df.type=="Assessment"].groupby(["installation_id"]).title.last().reset_index()

assessment['Assessment_1'] = 0
assessment['Assessment_2'] = 0
assessment['Assessment_3'] = 0
assessment['Assessment_4'] = 0
assessment['Assessment_5'] = 0

assessment.loc[assessment.title=='Mushroom Sorter (Assessment)', 'Assessment_1'] = 1
assessment.loc[assessment.title=='Cauldron Filler (Assessment)', 'Assessment_2'] = 1
assessment.loc[assessment.title=='Chest Sorter (Assessment)', 'Assessment_3'] = 1
assessment.loc[assessment.title=='Cart Balancer (Assessment)', 'Assessment_4'] = 1
assessment.loc[assessment.title=='Bird Measurer (Assessment)', 'Assessment_5'] = 1

del assessment["title"]


# In[ ]:


_ = world.merge(event_code, how="left", on=["installation_id"])
_ = _.merge(game_time, how="left", on=["installation_id"])
_ = _.merge(session, how="left", on=["installation_id"])
_ = _.merge(types, how="left", on=["installation_id"])
_ = _.merge(title, how="left", on=["installation_id"])
_ = _.merge(assessment, how="left", on=["installation_id"])

test_x = scaler.transform(_.loc[:, train_x_col[1:]])


# In[ ]:


result = model.predict(test_x)


# In[ ]:


submission = pd.DataFrame({"installation_id":_.installation_id, "accuracy_group":result.argmax(axis=1)})
submission.to_csv("submission.csv", index=False)


# In[ ]:


plt.hist(submission.accuracy_group)
plt.show()


# The end of notebook
