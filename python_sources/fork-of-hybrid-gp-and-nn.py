#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)


# In[ ]:


from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', usecols=['GameId', 'PlayId', 'Season','Team', 'X', 'Y', 'S', 'A', 'Dis',
                                                                               'Orientation', 'Dir', 'NflId', 'DisplayName', 'YardLine',
                                                                               'Quarter', 'GameClock', 'PossessionTeam', 'Down', 'Distance',
                                                                               'FieldPosition', 'NflIdRusher', 'OffenseFormation', 
                                                                               'OffensePersonnel', 'DefendersInTheBox', 'DefensePersonnel', 
                                                                               'PlayDirection', 'TimeHandoff', 'TimeSnap', 'Yards'])


# In[ ]:


train = train[train.Season>2017].copy().reset_index(drop=True)
del train['Season']


# In[ ]:


outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()


# ## Functions for anchoring offense moving left from {0,0}

# In[ ]:


def create_features(df, deploy=False):
    
    def new_X(x_coordinate, play_direction):
        if play_direction == 'left':
            return 120.0 - x_coordinate
        else:
            return x_coordinate

    def new_line(rush_team, field_position, yardline):
        if rush_team == field_position:
            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
            return 10.0 + yardline
        else:
            # half the field plus the yards between midfield and the line of scrimmage
            return 60.0 + (50 - yardline)

    def new_orientation(angle, play_direction):
        if play_direction == 'left':
            new_angle = 360.0 - angle
            if new_angle == 360.0:
                new_angle = 0.0
            return new_angle
        else:
            return angle

    def euclidean_distance(x1,y1,x2,y2):
        x_diff = (x1-x2)**2
        y_diff = (y1-y2)**2

        return np.sqrt(x_diff + y_diff)

    def back_direction(orientation):
        if orientation > 180.0:
            return 1
        else:
            return 0

    def update_yardline(df):
        new_yardline = df[df['NflId'] == df['NflIdRusher']]
        new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)
        new_yardline = new_yardline[['GameId','PlayId','YardLine']]

        return new_yardline

    def update_orientation(df, yardline):
        df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)
        df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)
        
        df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

        df = df.drop('YardLine', axis=1)
        df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

        return df

    def back_features(df):
        carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','NflIdRusher','X','Y','Orientation','Dir','YardLine']]
        carriers['back_from_scrimmage'] = carriers['YardLine'] - carriers['X']
        carriers['back_oriented_down_field'] = carriers['Orientation'].apply(lambda x: back_direction(x))
        carriers['back_moving_down_field'] = carriers['Dir'].apply(lambda x: back_direction(x))
        carriers = carriers.rename(columns={'X':'back_X',
                                            'Y':'back_Y'})
        carriers = carriers[['GameId','PlayId','NflIdRusher','back_X','back_Y','back_from_scrimmage','back_oriented_down_field','back_moving_down_field']]

        return carriers

    def features_relative_to_back(df, carriers):
        player_distance = df[['GameId','PlayId','NflId','X','Y']]
        player_distance = pd.merge(player_distance, carriers, on=['GameId','PlayId'], how='inner')
        player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
        player_distance['dist_to_back'] = player_distance[['X','Y','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        player_distance = player_distance.groupby(['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field'])                                         .agg({'dist_to_back':['min','max','mean','std']})                                         .reset_index()
        player_distance.columns = ['GameId','PlayId','back_from_scrimmage','back_oriented_down_field','back_moving_down_field',
                                   'min_dist','max_dist','mean_dist','std_dist']

        return player_distance

    def defense_features(df):
        rusher = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','Team','X','Y']]
        rusher.columns = ['GameId','PlayId','RusherTeam','RusherX','RusherY']

        defense = pd.merge(df,rusher,on=['GameId','PlayId'],how='inner')
        defense = defense[defense['Team'] != defense['RusherTeam']][['GameId','PlayId','X','Y','RusherX','RusherY']]
        defense['def_dist_to_back'] = defense[['X','Y','RusherX','RusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

        defense = defense.groupby(['GameId','PlayId'])                         .agg({'def_dist_to_back':['min','max','mean','std']})                         .reset_index()
        defense.columns = ['GameId','PlayId','def_min_dist','def_max_dist','def_mean_dist','def_std_dist']

        return defense

    def static_features(df):
        static_features = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y','S','A','Dis','Orientation','Dir',
                                                            'YardLine','Quarter','Down','Distance','DefendersInTheBox']].drop_duplicates()
        static_features['DefendersInTheBox'] = static_features['DefendersInTheBox'].fillna(np.mean(static_features['DefendersInTheBox']))

        return static_features


    def combine_features(relative_to_back, defense, static, deploy=deploy):
        df = pd.merge(relative_to_back,defense,on=['GameId','PlayId'],how='inner')
        df = pd.merge(df,static,on=['GameId','PlayId'],how='inner')

        if not deploy:
            df = pd.merge(df, outcomes, on=['GameId','PlayId'], how='inner')

        return df
    
    yardline = update_yardline(df)
    df = update_orientation(df, yardline)
    back_feats = back_features(df)
    rel_back = features_relative_to_back(df, back_feats)
    def_feats = defense_features(df)
    static_feats = static_features(df)
    basetable = combine_features(rel_back, def_feats, static_feats, deploy=deploy)
    return basetable


# In[ ]:


get_ipython().run_line_magic('time', 'train_basetable = create_features(train, False)')


# # Let's split our data into train/val

# In[ ]:


X = train_basetable.copy()
yards = X.Yards

y = np.zeros((yards.shape[0], 199))
for idx, target in enumerate(list(yards)):
    y[idx][99 + target] = 1

X.drop(['GameId','PlayId','Yards'], axis=1, inplace=True)


# In[ ]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=12345)


# In[ ]:


print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)


# # Let's build NN

# Below class Metric based entirely on: https://www.kaggle.com/kingychiu/keras-nn-starter-crps-early-stopping
# <br></br>
# Below early stopping entirely based on: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112868#latest-656533

# In[ ]:


class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred = self.model.predict(X_train)
        y_true = np.clip(np.cumsum(y_train, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        tr_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_train.shape[0])
        tr_s = np.round(tr_s, 6)
        logs['tr_CRPS'] = tr_s

        X_valid, y_valid = self.data[1][0], self.data[1][1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
        val_s = np.round(val_s, 6)
        logs['val_CRPS'] = val_s
        print('tr CRPS', tr_s, 'val CRPS', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


# In[ ]:


model = Sequential()
model.add(Dense(512, input_dim=X.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(199, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[])


# In[ ]:


es = EarlyStopping(monitor='val_CRPS', 
                   mode='min',
                   restore_best_weights=True, 
                   verbose=2, 
                   patience=5)
es.set_model(model)


# In[ ]:


metric = Metric(model, [es], [(X_train,y_train), (X_val,y_val)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(X_train, y_train, callbacks=[metric], epochs=100, batch_size=1024)')


# In[ ]:


class GP:
    def __init__(self):
        self.classes = 20
        self.class_names = [ 'class_0',
                             'class_1',
                             'class_2',
                             'class_3',
                             'class_4',
                             'class_5',
                             'class_6',
                             'class_7',
                             'class_8',
                             'class_9',
                             'class_10',
                             'class_11',
                             'class_12',
                             'class_13',
                            'class_14',
                            'class_15',
                            'class_16',
                            'class_17',
                            'class_18',
                            'class_19',
                           ]


    def GrabPredictions(self, data):
        oof_preds = np.zeros((len(data), len(self.class_names)))
        oof_preds[:,0] = self.GP_class_0(data)
        oof_preds[:,1] = self.GP_class_1(data)
        oof_preds[:,2] = self.GP_class_2(data)
        oof_preds[:,3] = self.GP_class_3(data)
        oof_preds[:,4] = self.GP_class_4(data)
        oof_preds[:,5] = self.GP_class_5(data)
        oof_preds[:,6] = self.GP_class_6(data)
        oof_preds[:,7] = self.GP_class_7(data)
        oof_preds[:,8] = self.GP_class_8(data)
        oof_preds[:,9] = self.GP_class_9(data)
        oof_preds[:,10] = self.GP_class_10(data)
        oof_preds[:,11] = self.GP_class_11(data)
        oof_preds[:,12] = self.GP_class_12(data)
        oof_preds[:,13] = self.GP_class_13(data)
        oof_preds[:,14] = self.GP_class_14(data)
        oof_preds[:,15] = self.GP_class_15(data)
        oof_preds[:,16] = self.GP_class_16(data)
        oof_preds[:,17] = self.GP_class_17(data)
        oof_preds[:,18] = self.GP_class_18(data)
        oof_preds[:,19] = self.GP_class_19(data)
        oof_df = pd.DataFrame(np.exp(oof_preds), columns=self.class_names)
        oof_df =oof_df.div(oof_df.sum(axis=1), axis=0)
        
        return oof_df.values


    def GP_class_0(self,data):
        return (-4.248615 +
                0.250000*np.tanh(((((((data["back_from_scrimmage"]) - (data["def_min_dist"]))) * 2.0)) * 2.0)) +
                0.250000*np.tanh(((((((data["back_from_scrimmage"]) + (data["back_moving_down_field"]))) + (((((((((data["back_from_scrimmage"]) + (data["back_from_scrimmage"]))) + (((((data["back_from_scrimmage"]) - (data["A"]))) + (data["back_from_scrimmage"]))))) - (data["def_min_dist"]))) + (((data["YardLine"]) - (data["Orientation"]))))))) + (((np.tanh((data["YardLine"]))) * 2.0)))) +
                0.250000*np.tanh(((((data["Orientation"]) + (((((((data["Orientation"]) * 2.0)) + (((data["back_from_scrimmage"]) - (((data["back_moving_down_field"]) + (((((data["Orientation"]) + (data["back_moving_down_field"]))) + (data["Orientation"]))))))))) - (((data["A"]) * 2.0)))))) + (((data["Orientation"]) - (data["def_min_dist"]))))) +
                0.250000*np.tanh(((((data["back_from_scrimmage"]) + (((data["back_from_scrimmage"]) + (((((((data["back_oriented_down_field"]) * 2.0)) - (data["def_min_dist"]))) * 2.0)))))) + ((((((data["back_moving_down_field"]) / 2.0)) + (((((data["back_moving_down_field"]) + (((((((data["def_mean_dist"]) + (data["Dis"]))/2.0)) + (data["YardLine"]))/2.0)))) * 2.0)))/2.0)))) +
                0.250000*np.tanh(((((data["Down"]) + (((((((data["Dir"]) + (data["back_oriented_down_field"]))) + (((data["back_from_scrimmage"]) + (data["Down"]))))) + (((data["back_from_scrimmage"]) + (data["YardLine"]))))))) + (((data["Orientation"]) + (data["min_dist"]))))) +
                0.246874*np.tanh(((((((data["back_moving_down_field"]) - (data["Dis"]))) + ((((data["back_moving_down_field"]) + (((((data["back_moving_down_field"]) - (data["A"]))) + (((data["back_moving_down_field"]) - (((((((data["Y"]) + (((data["min_dist"]) * ((((data["Dir"]) + (data["Dis"]))/2.0)))))) / 2.0)) + (data["back_moving_down_field"]))))))))/2.0)))) - (data["back_moving_down_field"]))) +
                0.249902*np.tanh(((((((((data["Dir"]) + (((data["Orientation"]) + (((data["Dir"]) * (data["Y"]))))))) * (data["Orientation"]))) + (data["Y"]))) + (((((data["Orientation"]) + (((((data["Dir"]) * (data["back_moving_down_field"]))) + (data["Orientation"]))))) + (data["Orientation"]))))) +
                0.250000*np.tanh(((((((data["back_moving_down_field"]) * 2.0)) + (((data["Orientation"]) * (((data["Orientation"]) + (data["Orientation"]))))))) + (((data["back_moving_down_field"]) * 2.0)))) +
                0.250000*np.tanh(((((((data["max_dist"]) - (((data["S"]) * 2.0)))) + (data["mean_dist"]))) - (((((((data["std_dist"]) + (((data["std_dist"]) + (((((data["Dir"]) + (np.tanh((data["min_dist"]))))) * 2.0)))))) * 2.0)) * (data["min_dist"]))))) +
                0.249805*np.tanh(((((((((((data["A"]) + (data["A"]))/2.0)) + ((((data["A"]) + ((-1.0*((data["A"])))))/2.0)))/2.0)) - (((data["A"]) - (data["back_oriented_down_field"]))))) - (data["A"]))))
    
    def GP_class_1(self,data):
        return (-3.574230 +
                0.250000*np.tanh(((((((((((((((data["back_from_scrimmage"]) - (data["def_min_dist"]))) * 2.0)) - (data["A"]))) + (((((data["back_from_scrimmage"]) - (data["def_min_dist"]))) - (data["A"]))))) + (data["def_min_dist"]))) + ((-1.0*((data["def_min_dist"])))))) + ((((data["Orientation"]) + (((data["def_std_dist"]) - (data["DefendersInTheBox"]))))/2.0)))) +
                0.250000*np.tanh((((((((((((((((data["back_from_scrimmage"]) - (data["Dis"]))) + (((((data["back_from_scrimmage"]) - (data["def_min_dist"]))) + (data["back_from_scrimmage"]))))/2.0)) - (data["def_min_dist"]))) - (data["A"]))) * 2.0)) + (((((((data["back_from_scrimmage"]) - (data["DefendersInTheBox"]))) - (data["def_min_dist"]))) - (data["A"]))))) + (data["back_from_scrimmage"]))) +
                0.250000*np.tanh(np.tanh((((data["Quarter"]) + ((((-1.0*(((((((data["min_dist"]) + (data["Down"]))/2.0)) + (data["Quarter"])))))) + (((((data["back_from_scrimmage"]) + (((data["S"]) + (((((data["DefendersInTheBox"]) + (data["back_from_scrimmage"]))) + (data["DefendersInTheBox"]))))))) + (data["Quarter"]))))))))) +
                0.250000*np.tanh((((((((((data["DefendersInTheBox"]) + (((data["DefendersInTheBox"]) / 2.0)))/2.0)) + (data["DefendersInTheBox"]))/2.0)) + (data["DefendersInTheBox"]))/2.0)) +
                0.250000*np.tanh((((((((((((data["A"]) + ((((((data["min_dist"]) + (data["min_dist"]))/2.0)) * 2.0)))/2.0)) + (((((data["min_dist"]) + (((data["min_dist"]) * 2.0)))) - (((((data["A"]) * 2.0)) * 2.0)))))) * 2.0)) - (data["min_dist"]))) - (((data["A"]) * 2.0)))) +
                0.246874*np.tanh(((((data["Y"]) + (((((((((((((data["Y"]) + (data["back_moving_down_field"]))) + ((((data["Dis"]) + (data["X"]))/2.0)))) + (data["Y"]))/2.0)) + ((((((data["Y"]) + (data["Y"]))/2.0)) + (data["back_from_scrimmage"]))))/2.0)) - (data["S"]))))) - (data["S"]))) +
                0.249902*np.tanh(((((((data["back_moving_down_field"]) - (((data["back_moving_down_field"]) - ((((data["back_from_scrimmage"]) + ((((((((data["back_moving_down_field"]) + (data["Dir"]))) + (data["X"]))/2.0)) - (data["def_min_dist"]))))/2.0)))))) - (data["def_min_dist"]))) + (data["back_oriented_down_field"]))) +
                0.250000*np.tanh(np.tanh((data["back_moving_down_field"]))) +
                0.250000*np.tanh((((data["back_from_scrimmage"]) + (((data["S"]) * (((((data["def_max_dist"]) * (np.tanh((((data["S"]) * (data["S"]))))))) + ((((data["def_mean_dist"]) + (data["Y"]))/2.0)))))))/2.0)) +
                0.249805*np.tanh(((data["Down"]) * ((((((((np.tanh((data["def_max_dist"]))) - (data["Down"]))) + ((((data["DefendersInTheBox"]) + (((data["DefendersInTheBox"]) - (data["def_max_dist"]))))/2.0)))/2.0)) - (data["def_max_dist"]))))))
    
    def GP_class_2(self,data):
        return (-3.173018 +
                0.250000*np.tanh(((data["back_from_scrimmage"]) + (((((data["back_moving_down_field"]) * 2.0)) + (((((((data["back_from_scrimmage"]) + (data["back_from_scrimmage"]))) - (((data["def_min_dist"]) * 2.0)))) + (((((data["back_from_scrimmage"]) + (data["back_from_scrimmage"]))) - (((data["def_min_dist"]) * 2.0)))))))))) +
                0.250000*np.tanh(((((((((data["back_from_scrimmage"]) - (data["A"]))) - (data["back_from_scrimmage"]))) + (data["back_from_scrimmage"]))) + (((((data["def_mean_dist"]) - (((data["def_mean_dist"]) + (data["S"]))))) + (((((data["back_from_scrimmage"]) - (((data["def_mean_dist"]) - (((data["back_from_scrimmage"]) - (data["def_min_dist"]))))))) - (data["A"]))))))) +
                0.250000*np.tanh((((data["DefendersInTheBox"]) + ((((data["DefendersInTheBox"]) + ((((((data["DefendersInTheBox"]) / 2.0)) + (data["DefendersInTheBox"]))/2.0)))/2.0)))/2.0)) +
                0.250000*np.tanh(((((((((((((data["YardLine"]) + ((((data["back_from_scrimmage"]) + (((data["back_from_scrimmage"]) * (data["back_from_scrimmage"]))))/2.0)))) + ((((data["YardLine"]) + (data["X"]))/2.0)))/2.0)) / 2.0)) + (data["back_from_scrimmage"]))) + (data["YardLine"]))/2.0)) +
                0.250000*np.tanh(((((np.tanh(((((((-1.0*((data["DefendersInTheBox"])))) * (((((((np.tanh(((((0.41962990164756775)) / 2.0)))) / 2.0)) / 2.0)) / 2.0)))) / 2.0)))) / 2.0)) / 2.0)) +
                0.246874*np.tanh((((((((((((data["Dis"]) / 2.0)) + ((((-1.0*((data["min_dist"])))) / 2.0)))/2.0)) / 2.0)) / 2.0)) / 2.0)) +
                0.249902*np.tanh(((data["X"]) * (((((data["Quarter"]) - (((np.tanh((data["max_dist"]))) * (data["Quarter"]))))) * (data["Dis"]))))) +
                0.250000*np.tanh(((((((data["back_from_scrimmage"]) - (data["def_min_dist"]))) + (((data["def_min_dist"]) * (data["min_dist"]))))) + (((((data["def_min_dist"]) + (((((((data["Dir"]) - (data["A"]))) * (data["min_dist"]))) + (data["min_dist"]))))) - (data["A"]))))) +
                0.250000*np.tanh(((data["def_min_dist"]) * (((data["X"]) * (((data["max_dist"]) - ((((data["A"]) + (((data["X"]) + (((data["A"]) - ((((-1.0*((((data["X"]) + (((data["X"]) * (data["max_dist"])))))))) / 2.0)))))))/2.0)))))))) +
                0.249805*np.tanh((((((((data["Distance"]) * (data["def_max_dist"]))) + (data["max_dist"]))/2.0)) + (((data["Distance"]) * (data["max_dist"]))))))
    
    def GP_class_3(self,data):
        return (-2.389767 +
                0.250000*np.tanh(((((((((data["back_from_scrimmage"]) - (data["def_mean_dist"]))) + (data["back_from_scrimmage"]))) - (data["Distance"]))) - (data["def_min_dist"]))) +
                0.250000*np.tanh(((data["Quarter"]) + ((((((((data["Dir"]) - (data["Dis"]))) + (((data["DefendersInTheBox"]) + ((((((data["DefendersInTheBox"]) + (data["DefendersInTheBox"]))) + (((((((data["Down"]) + (data["Quarter"]))/2.0)) + (data["Down"]))/2.0)))/2.0)))))) + (data["DefendersInTheBox"]))/2.0)))) +
                0.250000*np.tanh(((((((((data["DefendersInTheBox"]) - (data["A"]))) + ((((data["X"]) + ((((data["A"]) + (((data["A"]) * 2.0)))/2.0)))/2.0)))/2.0)) + ((((((data["Down"]) + ((((data["X"]) + (data["DefendersInTheBox"]))/2.0)))/2.0)) - (data["A"]))))/2.0)) +
                0.250000*np.tanh(((data["def_max_dist"]) + (((data["def_mean_dist"]) * (((data["Dis"]) - (((data["def_max_dist"]) * (((((((((((((data["def_mean_dist"]) + (data["Dis"]))/2.0)) + (data["std_dist"]))/2.0)) + ((((data["std_dist"]) + (data["S"]))/2.0)))/2.0)) + (((((((data["Dis"]) + (data["def_std_dist"]))/2.0)) + (data["def_mean_dist"]))/2.0)))/2.0)))))))))) +
                0.250000*np.tanh(((data["min_dist"]) * (((((data["A"]) - (data["back_oriented_down_field"]))) - ((((data["A"]) + (np.tanh((((data["back_moving_down_field"]) * (np.tanh((data["min_dist"]))))))))/2.0)))))) +
                0.246874*np.tanh(((((((data["back_moving_down_field"]) + (((data["Distance"]) + (((data["std_dist"]) * (data["DefendersInTheBox"]))))))/2.0)) + (((data["std_dist"]) + (((data["Distance"]) * (((((data["Distance"]) * (((data["DefendersInTheBox"]) * (data["def_mean_dist"]))))) * (data["def_mean_dist"]))))))))/2.0)) +
                0.249902*np.tanh(((((data["back_from_scrimmage"]) - (data["def_mean_dist"]))) + (((((data["S"]) * (data["back_from_scrimmage"]))) - (np.tanh((data["DefendersInTheBox"]))))))) +
                0.250000*np.tanh(((((((((data["max_dist"]) * (np.tanh((np.tanh(((((data["std_dist"]) + (((data["back_oriented_down_field"]) + (data["std_dist"]))))/2.0)))))))) + (np.tanh((data["std_dist"]))))/2.0)) + (np.tanh((((data["back_oriented_down_field"]) + (data["std_dist"]))))))/2.0)) +
                0.250000*np.tanh((((data["back_moving_down_field"]) + (((((data["Dis"]) * (data["Y"]))) - ((((((data["back_moving_down_field"]) * ((((data["Dis"]) + (data["Dis"]))/2.0)))) + (((((((np.tanh((data["Y"]))) + (data["Dis"]))/2.0)) + ((((data["Dis"]) + (((((-1.0*((data["Y"])))) + (data["Dis"]))/2.0)))/2.0)))/2.0)))/2.0)))))/2.0)) +
                0.249805*np.tanh(((((((data["back_oriented_down_field"]) + ((((((data["back_from_scrimmage"]) + ((((((data["std_dist"]) + (data["Dir"]))) + (data["back_from_scrimmage"]))/2.0)))/2.0)) + ((((data["Dir"]) + ((((data["back_from_scrimmage"]) + (((data["Dir"]) * (((data["Dir"]) / 2.0)))))/2.0)))/2.0)))))/2.0)) + (((data["Dir"]) * ((((data["back_moving_down_field"]) + (data["Dir"]))/2.0)))))/2.0)))
    
    def GP_class_4(self,data):
        return (-2.130314 +
                0.250000*np.tanh((((((np.tanh((data["DefendersInTheBox"]))) + (data["YardLine"]))) + (((((((data["YardLine"]) - (data["Dir"]))) - (data["def_mean_dist"]))) - (np.tanh((((((((data["Dir"]) + (data["YardLine"]))/2.0)) + (data["Dir"]))/2.0)))))))/2.0)) +
                0.250000*np.tanh((-1.0*(((-1.0*(((((((((((data["def_min_dist"]) + ((((data["DefendersInTheBox"]) + (data["Orientation"]))/2.0)))/2.0)) + ((((((data["YardLine"]) * (data["YardLine"]))) + (data["YardLine"]))/2.0)))/2.0)) + (data["DefendersInTheBox"]))/2.0)))))))) +
                0.250000*np.tanh((-1.0*(((((((data["Distance"]) * (((data["Distance"]) - (((data["Distance"]) * (data["X"]))))))) + ((((data["X"]) + ((((data["back_moving_down_field"]) + (data["Distance"]))/2.0)))/2.0)))/2.0))))) +
                0.250000*np.tanh(((((((((((((((((data["def_mean_dist"]) + (np.tanh((data["def_mean_dist"]))))/2.0)) + (data["def_mean_dist"]))/2.0)) - (((data["def_mean_dist"]) * (data["def_mean_dist"]))))) * (data["def_mean_dist"]))) + ((-1.0*((data["back_moving_down_field"])))))/2.0)) + (data["mean_dist"]))/2.0)) +
                0.250000*np.tanh((((data["back_from_scrimmage"]) + ((((((data["X"]) + (((((((((((data["back_from_scrimmage"]) + (data["X"]))/2.0)) - ((((np.tanh((data["back_from_scrimmage"]))) + (data["DefendersInTheBox"]))/2.0)))) + (data["DefendersInTheBox"]))/2.0)) * (data["X"]))))/2.0)) * ((((data["X"]) + (((data["X"]) * (data["DefendersInTheBox"]))))/2.0)))))/2.0)) +
                0.246874*np.tanh(((np.tanh((data["std_dist"]))) - ((((((((((np.tanh((((np.tanh((np.tanh((np.tanh((data["std_dist"]))))))) - (data["def_mean_dist"]))))) + (np.tanh((data["std_dist"]))))/2.0)) + (np.tanh((np.tanh((np.tanh((np.tanh((np.tanh((data["std_dist"]))))))))))))/2.0)) + (data["def_mean_dist"]))/2.0)))) +
                0.249902*np.tanh((((((data["Distance"]) * (((np.tanh((data["def_min_dist"]))) / 2.0)))) + (((data["def_min_dist"]) / 2.0)))/2.0)) +
                0.250000*np.tanh((((((((data["back_oriented_down_field"]) + (np.tanh((np.tanh(((((((((-1.0*((data["back_from_scrimmage"])))) / 2.0)) / 2.0)) / 2.0)))))))/2.0)) / 2.0)) * 2.0)) +
                0.250000*np.tanh(((((data["back_from_scrimmage"]) * (((((((data["min_dist"]) * (((((0.0)) + ((((1.0)) / 2.0)))/2.0)))) / 2.0)) * ((((-1.0*((data["back_moving_down_field"])))) / 2.0)))))) * ((0.0)))) +
                0.249805*np.tanh((((((((((data["def_std_dist"]) + (data["def_std_dist"]))/2.0)) + ((-1.0*(((((((data["def_std_dist"]) + ((((data["Dis"]) + ((((((data["def_std_dist"]) + (data["Dis"]))) + ((-1.0*(((1.03885555267333984))))))/2.0)))/2.0)))/2.0)) * (((data["def_std_dist"]) * (data["def_max_dist"])))))))))/2.0)) + (data["def_max_dist"]))/2.0)))
    
    def GP_class_5(self,data):
        return (-2.049984 +
                0.250000*np.tanh((((((((((data["DefendersInTheBox"]) + (data["DefendersInTheBox"]))/2.0)) + ((((((((np.tanh((data["back_oriented_down_field"]))) - (data["def_min_dist"]))) - (data["def_mean_dist"]))) + (data["def_min_dist"]))/2.0)))/2.0)) + (((data["def_min_dist"]) - (data["back_from_scrimmage"]))))/2.0)) +
                0.250000*np.tanh((((((((((((data["Down"]) + (np.tanh((data["Distance"]))))) + (data["def_min_dist"]))/2.0)) + (((data["def_min_dist"]) - (data["Distance"]))))/2.0)) + (((((((data["def_min_dist"]) + ((((data["def_min_dist"]) + (data["min_dist"]))/2.0)))/2.0)) + (data["def_min_dist"]))/2.0)))/2.0)) +
                0.250000*np.tanh((((((np.tanh((data["Quarter"]))) + (((((data["back_oriented_down_field"]) + (data["DefendersInTheBox"]))) - ((((data["back_moving_down_field"]) + ((((((data["back_moving_down_field"]) + (data["back_moving_down_field"]))/2.0)) * (data["Down"]))))/2.0)))))/2.0)) - (data["back_moving_down_field"]))) +
                0.250000*np.tanh(((((((((data["std_dist"]) / 2.0)) - (((data["def_mean_dist"]) - (((data["std_dist"]) - (((((data["std_dist"]) / 2.0)) * (data["def_mean_dist"]))))))))) / 2.0)) / 2.0)) +
                0.250000*np.tanh(((data["def_mean_dist"]) * (((data["max_dist"]) + (((((((((data["max_dist"]) + (data["back_from_scrimmage"]))/2.0)) - (data["max_dist"]))) + ((((((data["max_dist"]) * (data["max_dist"]))) + (np.tanh((((((((data["max_dist"]) + (np.tanh((np.tanh((data["back_from_scrimmage"]))))))/2.0)) + (np.tanh((data["def_mean_dist"]))))/2.0)))))/2.0)))/2.0)))))) +
                0.246874*np.tanh(((data["YardLine"]) * ((((data["A"]) + ((((np.tanh(((((data["std_dist"]) + (data["back_oriented_down_field"]))/2.0)))) + (np.tanh((data["Quarter"]))))/2.0)))/2.0)))) +
                0.249902*np.tanh((((((((((data["back_oriented_down_field"]) / 2.0)) / 2.0)) + ((((((((np.tanh((((((data["def_min_dist"]) / 2.0)) / 2.0)))) + (((data["back_oriented_down_field"]) / 2.0)))) + ((((data["def_min_dist"]) + (((((data["def_min_dist"]) / 2.0)) / 2.0)))/2.0)))/2.0)) / 2.0)))/2.0)) / 2.0)) +
                0.250000*np.tanh(((data["DefendersInTheBox"]) * (((((data["def_mean_dist"]) - (((data["back_moving_down_field"]) * (((data["A"]) * ((((data["back_moving_down_field"]) + (((data["A"]) / 2.0)))/2.0)))))))) * (data["A"]))))) +
                0.250000*np.tanh(np.tanh((((data["back_moving_down_field"]) * ((((-1.0*(((((0.0)) * ((0.0))))))) / 2.0)))))) +
                0.249805*np.tanh((((data["back_oriented_down_field"]) + (((((((((np.tanh(((((((data["def_max_dist"]) + (np.tanh((((((((data["back_oriented_down_field"]) + (data["back_moving_down_field"]))/2.0)) + (data["back_oriented_down_field"]))/2.0)))))/2.0)) / 2.0)))) / 2.0)) + (data["back_oriented_down_field"]))/2.0)) + (data["back_moving_down_field"]))/2.0)))/2.0)))
    
    def GP_class_6(self,data):
        return (-2.205513 +
                0.250000*np.tanh(((((((((data["def_min_dist"]) - (data["back_from_scrimmage"]))) - (((data["back_from_scrimmage"]) - (data["back_from_scrimmage"]))))) + (((data["def_min_dist"]) - (((data["back_from_scrimmage"]) - (data["def_min_dist"]))))))) - (data["def_min_dist"]))) +
                0.250000*np.tanh((((((((((((((((((((data["A"]) + (data["Distance"]))/2.0)) / 2.0)) + (np.tanh((data["A"]))))) + (data["Distance"]))/2.0)) / 2.0)) + ((-1.0*((data["Distance"])))))) + (((data["DefendersInTheBox"]) * ((((((data["Distance"]) * (data["DefendersInTheBox"]))) + (((data["Distance"]) * 2.0)))/2.0)))))) + (data["DefendersInTheBox"]))/2.0)) +
                0.250000*np.tanh((((((data["Distance"]) + ((((((((data["X"]) - ((((data["mean_dist"]) + ((-1.0*(((((((data["back_oriented_down_field"]) + ((((data["back_moving_down_field"]) + (data["back_moving_down_field"]))/2.0)))/2.0)) / 2.0))))))/2.0)))) + (data["Dir"]))/2.0)) - (data["back_moving_down_field"]))))/2.0)) - (data["back_moving_down_field"]))) +
                0.250000*np.tanh(((data["def_mean_dist"]) * ((((((data["def_max_dist"]) + (data["max_dist"]))/2.0)) * (data["def_min_dist"]))))) +
                0.250000*np.tanh((((((((((((((np.tanh((((((((((((data["def_std_dist"]) * 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)))) / 2.0)) + (np.tanh((((data["def_max_dist"]) / 2.0)))))/2.0)) / 2.0)) * ((((0.0)) / 2.0)))) / 2.0)) / 2.0)) +
                0.246874*np.tanh((((((data["max_dist"]) + (data["std_dist"]))/2.0)) / 2.0)) +
                0.249902*np.tanh(((((((((((data["def_min_dist"]) + (np.tanh(((-1.0*((((data["X"]) + (((data["Dir"]) + (((data["back_moving_down_field"]) / 2.0))))))))))))) / 2.0)) * (data["mean_dist"]))) * (np.tanh((data["back_oriented_down_field"]))))) / 2.0)) +
                0.250000*np.tanh(np.tanh((((((((((np.tanh(((((((0.0)) * ((((0.0)) / 2.0)))) * 2.0)))) / 2.0)) / 2.0)) * ((((((0.0)) / 2.0)) / 2.0)))) / 2.0)))) +
                0.250000*np.tanh(((((((((np.tanh((np.tanh(((((((((((((((((((((0.0)) / 2.0)) / 2.0)) * (data["def_min_dist"]))) / 2.0)) - (np.tanh((((((np.tanh(((0.0)))) / 2.0)) / 2.0)))))) / 2.0)) / 2.0)) / 2.0)) / 2.0)))))) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +
                0.249805*np.tanh(((((data["DefendersInTheBox"]) * ((0.0)))) + (((((data["A"]) / 2.0)) * (data["DefendersInTheBox"]))))))
    
    def GP_class_7(self,data):
        return (-2.292080 +
                0.250000*np.tanh(((((((data["def_min_dist"]) + (data["def_min_dist"]))) - (data["back_from_scrimmage"]))) + (((((data["def_min_dist"]) - (((data["def_min_dist"]) - (((data["def_min_dist"]) - (data["YardLine"]))))))) - ((((data["def_min_dist"]) + (data["back_from_scrimmage"]))/2.0)))))) +
                0.250000*np.tanh((((((data["min_dist"]) - (data["min_dist"]))) + ((((data["def_mean_dist"]) + (((((((data["Distance"]) - (data["back_from_scrimmage"]))) + ((((((((data["A"]) + (data["back_from_scrimmage"]))/2.0)) - (data["back_moving_down_field"]))) / 2.0)))) - (data["back_from_scrimmage"]))))/2.0)))/2.0)) +
                0.250000*np.tanh(((((((((((((0.0)) * ((0.0)))) * ((((((-1.0*((((np.tanh(((0.0)))) / 2.0))))) * ((0.0)))) / 2.0)))) + (((((data["back_oriented_down_field"]) / 2.0)) / 2.0)))/2.0)) / 2.0)) / 2.0)) +
                0.250000*np.tanh(((((((((-1.0*((data["back_oriented_down_field"])))) / 2.0)) + (np.tanh((data["def_mean_dist"]))))/2.0)) * ((((0.0)) / 2.0)))) +
                0.250000*np.tanh((0.0)) +
                0.246874*np.tanh(((((((0.58118116855621338)) / 2.0)) + (np.tanh((data["def_min_dist"]))))/2.0)) +
                0.249902*np.tanh(((((((((np.tanh((data["max_dist"]))) / 2.0)) + (((np.tanh((data["back_oriented_down_field"]))) / 2.0)))/2.0)) + ((((((((((((((data["Dir"]) + (data["back_oriented_down_field"]))/2.0)) / 2.0)) + (data["back_oriented_down_field"]))/2.0)) / 2.0)) + (data["back_oriented_down_field"]))/2.0)))/2.0)) +
                0.250000*np.tanh((((((((((data["back_moving_down_field"]) * (data["back_moving_down_field"]))) + (data["back_moving_down_field"]))/2.0)) * (data["back_from_scrimmage"]))) * ((((data["back_moving_down_field"]) + ((((((data["back_moving_down_field"]) * (data["back_from_scrimmage"]))) + (((data["back_moving_down_field"]) * (data["back_moving_down_field"]))))/2.0)))/2.0)))) +
                0.250000*np.tanh(((((data["Dis"]) / 2.0)) / 2.0)) +
                0.249805*np.tanh(((((0.0)) + ((((((((((((0.0)) / 2.0)) + (data["def_max_dist"]))/2.0)) / 2.0)) + ((((data["def_max_dist"]) + (((((((data["DefendersInTheBox"]) + (((((((data["def_max_dist"]) / 2.0)) / 2.0)) / 2.0)))/2.0)) + (data["def_mean_dist"]))/2.0)))/2.0)))/2.0)))/2.0)))
    
    def GP_class_8(self,data):
        return (-2.566108 +
                0.250000*np.tanh(((((((((((data["A"]) + (data["def_mean_dist"]))/2.0)) + ((((data["def_min_dist"]) + (data["def_mean_dist"]))/2.0)))/2.0)) * 2.0)) + (data["def_min_dist"]))) +
                0.250000*np.tanh((((data["Distance"]) + ((((data["Distance"]) + (data["Distance"]))/2.0)))/2.0)) +
                0.250000*np.tanh(((((((data["Dis"]) + ((((((np.tanh((np.tanh((data["Dis"]))))) + (data["A"]))/2.0)) * 2.0)))/2.0)) + ((((((data["Dis"]) + ((((((np.tanh((np.tanh((((np.tanh(((((data["S"]) + (data["A"]))/2.0)))) / 2.0)))))) / 2.0)) + (((data["A"]) / 2.0)))/2.0)))/2.0)) / 2.0)))/2.0)) +
                0.250000*np.tanh(((data["YardLine"]) * (((data["back_from_scrimmage"]) + ((((((data["back_from_scrimmage"]) + (((((data["back_from_scrimmage"]) + ((((data["back_from_scrimmage"]) + (((data["YardLine"]) + (data["max_dist"]))))/2.0)))) + (((data["YardLine"]) + (data["back_from_scrimmage"]))))))/2.0)) + (((((data["back_from_scrimmage"]) + (data["back_from_scrimmage"]))) + (data["YardLine"]))))))))) +
                0.250000*np.tanh((((((data["DefendersInTheBox"]) * (data["def_mean_dist"]))) + (np.tanh(((-1.0*((data["back_from_scrimmage"])))))))/2.0)) +
                0.246874*np.tanh(np.tanh(((((data["def_min_dist"]) + ((((data["def_mean_dist"]) + (data["max_dist"]))/2.0)))/2.0)))) +
                0.249902*np.tanh(((data["Quarter"]) * ((((((np.tanh((((data["Quarter"]) / 2.0)))) / 2.0)) + ((((((((data["Quarter"]) * (data["Down"]))) + (((data["def_std_dist"]) - (data["Distance"]))))/2.0)) - (data["Distance"]))))/2.0)))) +
                0.250000*np.tanh(((((((((((0.0)) * (data["DefendersInTheBox"]))) / 2.0)) + (((data["back_oriented_down_field"]) * 2.0)))/2.0)) * (data["Distance"]))) +
                0.250000*np.tanh(((np.tanh((data["max_dist"]))) / 2.0)) +
                0.249805*np.tanh((((((((((0.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)))
    
    def GP_class_9(self,data):
        return (-2.941433 +
                0.250000*np.tanh(data["def_mean_dist"]) +
                0.250000*np.tanh(((((data["def_min_dist"]) + (((((((((((data["def_min_dist"]) - (data["def_min_dist"]))) + (data["Distance"]))) + (((data["def_min_dist"]) - (data["back_from_scrimmage"]))))/2.0)) + (np.tanh((data["Distance"]))))/2.0)))) - (data["back_from_scrimmage"]))) +
                0.250000*np.tanh((((((((((data["Orientation"]) + (((data["back_oriented_down_field"]) / 2.0)))/2.0)) * (((((((((data["Y"]) / 2.0)) / 2.0)) / 2.0)) / 2.0)))) / 2.0)) / 2.0)) +
                0.250000*np.tanh((((data["def_mean_dist"]) + ((((((((((((np.tanh((data["def_mean_dist"]))) * (data["back_oriented_down_field"]))) / 2.0)) + (data["def_min_dist"]))/2.0)) * (data["back_oriented_down_field"]))) * (((data["def_mean_dist"]) / 2.0)))))/2.0)) +
                0.250000*np.tanh(((((data["def_min_dist"]) / 2.0)) / 2.0)) +
                0.246874*np.tanh((((-1.0*((data["X"])))) * ((((((((data["Dis"]) + (((data["Dis"]) + (((data["back_oriented_down_field"]) + (data["Quarter"]))))))) + (data["Quarter"]))) + ((((((((((data["mean_dist"]) + (data["X"]))/2.0)) + (((data["S"]) * 2.0)))) / 2.0)) + (data["S"]))))/2.0)))) +
                0.249902*np.tanh((((((-1.0*((((data["back_oriented_down_field"]) / 2.0))))) / 2.0)) / 2.0)) +
                0.250000*np.tanh(((data["back_from_scrimmage"]) * (((((data["back_moving_down_field"]) / 2.0)) / 2.0)))) +
                0.250000*np.tanh(((((((((((((((((0.0)) + ((((-1.0*((((((((data["back_moving_down_field"]) / 2.0)) / 2.0)) / 2.0))))) / 2.0)))/2.0)) * ((0.0)))) / 2.0)) * ((((0.0)) * (((((data["back_moving_down_field"]) / 2.0)) / 2.0)))))) + (data["Distance"]))/2.0)) + ((((0.0)) / 2.0)))/2.0)) +
                0.249805*np.tanh((((((data["back_moving_down_field"]) + (((((((data["back_oriented_down_field"]) * (((((np.tanh(((((data["min_dist"]) + (((data["back_moving_down_field"]) + (((data["Dis"]) / 2.0)))))/2.0)))) * (np.tanh((data["back_moving_down_field"]))))) * (data["back_oriented_down_field"]))))) * (data["back_oriented_down_field"]))) * (data["Dis"]))))/2.0)) / 2.0)))
    
    def GP_class_10(self,data):
        return (-3.222959 +
                0.250000*np.tanh((((((data["Dis"]) + (data["DefendersInTheBox"]))/2.0)) + ((((((data["Distance"]) + (data["def_mean_dist"]))/2.0)) + (data["Distance"]))))) +
                0.250000*np.tanh(((((((data["def_mean_dist"]) + (data["def_mean_dist"]))/2.0)) + (((((data["A"]) / 2.0)) / 2.0)))/2.0)) +
                0.250000*np.tanh((((-1.0*((data["back_oriented_down_field"])))) / 2.0)) +
                0.250000*np.tanh(((((data["back_moving_down_field"]) * (data["Distance"]))) * (np.tanh((data["back_oriented_down_field"]))))) +
                0.250000*np.tanh(((((((data["Distance"]) * ((((data["back_moving_down_field"]) + (((data["back_moving_down_field"]) - (((((-1.0*(((-1.0*((data["def_min_dist"]))))))) + ((((data["back_oriented_down_field"]) + (((data["def_min_dist"]) / 2.0)))/2.0)))/2.0)))))/2.0)))) - (data["def_std_dist"]))) + (data["back_moving_down_field"]))) +
                0.246874*np.tanh(((((((data["def_mean_dist"]) - (data["def_mean_dist"]))) + (data["def_mean_dist"]))) + ((((((((data["def_mean_dist"]) + (((((data["back_oriented_down_field"]) - (data["back_from_scrimmage"]))) - (data["Dir"]))))) - (data["Dir"]))) + (np.tanh((((data["def_min_dist"]) + (data["def_min_dist"]))))))/2.0)))) +
                0.249902*np.tanh((((np.tanh((data["Distance"]))) + (((((np.tanh(((((((data["Dis"]) + (data["Distance"]))/2.0)) + (((np.tanh((np.tanh((((data["Distance"]) / 2.0)))))) / 2.0)))))) / 2.0)) / 2.0)))/2.0)) +
                0.250000*np.tanh(data["back_oriented_down_field"]) +
                0.250000*np.tanh(((((np.tanh(((((((0.84626573324203491)) / 2.0)) / 2.0)))) / 2.0)) / 2.0)) +
                0.249805*np.tanh(((((((np.tanh((((data["back_oriented_down_field"]) / 2.0)))) / 2.0)) / 2.0)) * (((((data["back_oriented_down_field"]) / 2.0)) / 2.0)))))
    
    def GP_class_11(self,data):
        return (-3.530885 +
                0.250000*np.tanh(((((((data["def_mean_dist"]) * 2.0)) + ((((((data["def_mean_dist"]) + (((data["back_moving_down_field"]) - (data["def_mean_dist"]))))) + (data["A"]))/2.0)))) + (data["def_mean_dist"]))) +
                0.250000*np.tanh(((data["Distance"]) + (data["A"]))) +
                0.250000*np.tanh(((((((((((((data["mean_dist"]) + (((((data["min_dist"]) + (data["min_dist"]))) + (((data["mean_dist"]) + ((((data["Dis"]) + (data["S"]))/2.0)))))))) / 2.0)) + (((data["S"]) + ((((data["S"]) + (((data["S"]) + (data["mean_dist"]))))/2.0)))))/2.0)) + (data["S"]))/2.0)) + (data["S"]))) +
                0.250000*np.tanh((((((data["def_mean_dist"]) + (data["def_mean_dist"]))) + ((((data["min_dist"]) + (data["min_dist"]))/2.0)))/2.0)) +
                0.250000*np.tanh(((((((((data["back_moving_down_field"]) + (data["X"]))) + ((((data["std_dist"]) + (np.tanh(((((((data["YardLine"]) / 2.0)) + (data["back_moving_down_field"]))/2.0)))))/2.0)))/2.0)) + (np.tanh((data["min_dist"]))))/2.0)) +
                0.246874*np.tanh(data["def_mean_dist"]) +
                0.249902*np.tanh((((-1.0*((data["mean_dist"])))) - (((data["mean_dist"]) + ((((((((data["mean_dist"]) - (data["mean_dist"]))) + (((np.tanh((data["YardLine"]))) / 2.0)))) + (data["Dir"]))/2.0)))))) +
                0.250000*np.tanh((((((data["mean_dist"]) + (((((((((((data["YardLine"]) / 2.0)) + (data["mean_dist"]))/2.0)) + (data["max_dist"]))/2.0)) / 2.0)))/2.0)) * (data["X"]))) +
                0.250000*np.tanh((((data["def_mean_dist"]) + (data["def_mean_dist"]))/2.0)) +
                0.249805*np.tanh(((data["back_moving_down_field"]) * (((data["back_moving_down_field"]) * ((((((data["DefendersInTheBox"]) + (((((((data["Distance"]) + (data["def_std_dist"]))/2.0)) + ((-1.0*(((-1.0*(((((((data["back_moving_down_field"]) - (data["back_oriented_down_field"]))) + (np.tanh(((((data["X"]) + (data["back_moving_down_field"]))/2.0)))))/2.0)))))))))/2.0)))/2.0)) * ((((data["back_moving_down_field"]) + (data["Down"]))/2.0)))))))))
    
    def GP_class_12(self,data):
        return (-3.527884 +
                0.250000*np.tanh(((((data["Distance"]) + (data["Distance"]))) + (data["def_mean_dist"]))) +
                0.250000*np.tanh((((data["back_moving_down_field"]) + (((((((data["def_mean_dist"]) + (((((((data["back_moving_down_field"]) + (data["back_moving_down_field"]))) + (data["back_moving_down_field"]))) / 2.0)))) + (data["def_mean_dist"]))) + (data["def_mean_dist"]))))/2.0)) +
                0.250000*np.tanh(((((((data["back_moving_down_field"]) + ((((((((((data["Distance"]) + (data["Distance"]))/2.0)) + ((((data["Distance"]) + ((((data["min_dist"]) + (data["back_moving_down_field"]))/2.0)))/2.0)))/2.0)) + ((((data["back_moving_down_field"]) + (data["Distance"]))/2.0)))/2.0)))/2.0)) + (((data["back_moving_down_field"]) - (((data["back_moving_down_field"]) / 2.0)))))/2.0)) +
                0.250000*np.tanh(((((((((((((((((((data["def_mean_dist"]) + (data["back_oriented_down_field"]))/2.0)) / 2.0)) + (data["Dis"]))/2.0)) / 2.0)) + (data["def_max_dist"]))/2.0)) / 2.0)) + ((((data["def_mean_dist"]) + (((data["Distance"]) / 2.0)))/2.0)))/2.0)) +
                0.250000*np.tanh((((np.tanh((data["Y"]))) + ((((((((data["back_oriented_down_field"]) * (((data["back_moving_down_field"]) / 2.0)))) / 2.0)) + (data["Y"]))/2.0)))/2.0)) +
                0.246874*np.tanh(((((((((data["back_moving_down_field"]) + (((((((data["def_mean_dist"]) + (data["back_moving_down_field"]))/2.0)) + ((((data["def_mean_dist"]) + (data["def_min_dist"]))/2.0)))/2.0)))/2.0)) + (data["min_dist"]))/2.0)) / 2.0)) +
                0.249902*np.tanh(((((data["def_mean_dist"]) / 2.0)) / 2.0)) +
                0.250000*np.tanh(((((np.tanh((((data["back_moving_down_field"]) + ((((((((((-1.0*((data["S"])))) / 2.0)) / 2.0)) / 2.0)) + ((((((data["back_oriented_down_field"]) / 2.0)) + (((((data["back_oriented_down_field"]) / 2.0)) / 2.0)))/2.0)))))))) / 2.0)) * (((((np.tanh((data["back_moving_down_field"]))) / 2.0)) * (((((data["back_moving_down_field"]) / 2.0)) / 2.0)))))) +
                0.250000*np.tanh(((((data["back_moving_down_field"]) / 2.0)) * (((data["S"]) / 2.0)))) +
                0.249805*np.tanh(((data["def_max_dist"]) / 2.0)))
    
    def GP_class_13(self,data):
        return (-4.280120 +
                0.250000*np.tanh((((((data["back_oriented_down_field"]) + (data["def_mean_dist"]))/2.0)) * 2.0)) +
                0.250000*np.tanh((((((data["Down"]) + (((data["back_moving_down_field"]) + (data["Down"]))))) + (((((data["mean_dist"]) + (((data["Down"]) + (data["def_mean_dist"]))))) + ((-1.0*((np.tanh(((((((((data["Down"]) - (data["A"]))) * 2.0)) + (((data["def_mean_dist"]) + (data["Down"]))))/2.0))))))))))/2.0)) +
                0.250000*np.tanh(((((((((data["Dis"]) + (((((data["Down"]) - (data["DefendersInTheBox"]))) + ((((data["Down"]) + (data["Dis"]))/2.0)))))) - (((data["DefendersInTheBox"]) - (((data["min_dist"]) + (((((data["S"]) + (((data["min_dist"]) / 2.0)))) / 2.0)))))))) - (data["DefendersInTheBox"]))) + (data["Down"]))) +
                0.250000*np.tanh(data["back_moving_down_field"]) +
                0.250000*np.tanh((((((data["min_dist"]) + ((((data["def_mean_dist"]) + (data["Distance"]))/2.0)))/2.0)) / 2.0)) +
                0.246874*np.tanh(((((data["min_dist"]) + ((((data["Distance"]) + ((((data["Distance"]) + (((np.tanh((data["Distance"]))) / 2.0)))/2.0)))/2.0)))) - ((((((((data["Y"]) + (data["A"]))/2.0)) * (data["back_oriented_down_field"]))) + ((((data["YardLine"]) + (((data["S"]) * ((-1.0*((((data["min_dist"]) / 2.0))))))))/2.0)))))) +
                0.249902*np.tanh((((np.tanh((data["mean_dist"]))) + (data["Down"]))/2.0)) +
                0.250000*np.tanh(np.tanh((((data["S"]) * ((3.0)))))) +
                0.250000*np.tanh(((((((data["A"]) + ((((data["A"]) + (data["A"]))/2.0)))/2.0)) + (((np.tanh(((((data["A"]) + ((((((data["def_mean_dist"]) + (data["A"]))/2.0)) - (data["A"]))))/2.0)))) / 2.0)))/2.0)) +
                0.249805*np.tanh(((data["back_from_scrimmage"]) * (((data["DefendersInTheBox"]) - (((((((data["back_from_scrimmage"]) + (data["def_mean_dist"]))/2.0)) + ((((((data["back_oriented_down_field"]) + (data["def_max_dist"]))/2.0)) * (((data["DefendersInTheBox"]) - (((data["Y"]) / 2.0)))))))/2.0)))))))
    
    def GP_class_14(self,data):
        return (-4.056991 +
                0.250000*np.tanh(((((data["def_std_dist"]) + (((((data["Distance"]) + (((((data["back_moving_down_field"]) + (data["Dir"]))) + (data["def_mean_dist"]))))) / 2.0)))) / 2.0)) +
                0.250000*np.tanh((((data["def_mean_dist"]) + (data["def_mean_dist"]))/2.0)) +
                0.250000*np.tanh((((data["Distance"]) + (data["min_dist"]))/2.0)) +
                0.250000*np.tanh(data["def_mean_dist"]) +
                0.250000*np.tanh(((((data["back_oriented_down_field"]) + (np.tanh(((-1.0*(((((((data["X"]) * 2.0)) + ((6.83017873764038086)))/2.0))))))))) + ((((((((-1.0*((data["back_oriented_down_field"])))) + (data["max_dist"]))/2.0)) + ((((((data["Down"]) + (np.tanh((data["back_oriented_down_field"]))))/2.0)) * (data["back_oriented_down_field"]))))/2.0)))) +
                0.246874*np.tanh((((2.0)) + ((((data["S"]) + ((1.0)))/2.0)))) +
                0.249902*np.tanh(np.tanh((((((((((data["back_moving_down_field"]) + (np.tanh((np.tanh((data["Down"]))))))/2.0)) + ((((((((np.tanh(((-1.0*(((((((data["back_moving_down_field"]) - (data["def_std_dist"]))) + (((data["DefendersInTheBox"]) / 2.0)))/2.0))))))) / 2.0)) / 2.0)) + (data["back_moving_down_field"]))/2.0)))/2.0)) / 2.0)))) +
                0.250000*np.tanh((((-1.0*((np.tanh((data["back_oriented_down_field"])))))) + (data["back_moving_down_field"]))) +
                0.250000*np.tanh(((((data["Distance"]) * (data["min_dist"]))) + ((((((data["Down"]) + (data["DefendersInTheBox"]))) + (data["Distance"]))/2.0)))) +
                0.249805*np.tanh(((data["Dis"]) * (((((data["Down"]) * ((((((data["Down"]) * (data["Down"]))) + (data["Dis"]))/2.0)))) + (((((((data["max_dist"]) + (data["max_dist"]))) + (np.tanh((np.tanh((data["max_dist"]))))))) * (data["Down"]))))))))
    
    def GP_class_15(self,data):
        return (-4.525990 +
                0.250000*np.tanh(((np.tanh((((data["std_dist"]) + (data["std_dist"]))))) + (((data["S"]) + (((((data["S"]) / 2.0)) * 2.0)))))) +
                0.250000*np.tanh((((((((((data["def_std_dist"]) + ((((data["def_mean_dist"]) + (data["back_moving_down_field"]))/2.0)))) * 2.0)) + ((((data["back_moving_down_field"]) + (data["YardLine"]))/2.0)))) + (((data["def_mean_dist"]) + (data["S"]))))/2.0)) +
                0.250000*np.tanh(((((data["Distance"]) + ((0.0)))) + ((((((data["min_dist"]) + (np.tanh((data["back_moving_down_field"]))))/2.0)) + (((((data["min_dist"]) + ((((((((data["min_dist"]) * 2.0)) * 2.0)) + (data["Dis"]))/2.0)))) - (data["back_oriented_down_field"]))))))) +
                0.250000*np.tanh((((((((np.tanh((((((((data["back_moving_down_field"]) / 2.0)) + (data["def_std_dist"]))) + (np.tanh(((((data["std_dist"]) + ((((data["std_dist"]) + (data["back_moving_down_field"]))/2.0)))/2.0)))))))) + (((data["back_moving_down_field"]) / 2.0)))/2.0)) / 2.0)) / 2.0)) +
                0.250000*np.tanh((((data["back_moving_down_field"]) + ((((((np.tanh((data["YardLine"]))) / 2.0)) + (((np.tanh((np.tanh(((((((((data["def_min_dist"]) + ((((data["A"]) + (data["Down"]))/2.0)))) + (data["def_min_dist"]))/2.0)) / 2.0)))))) + (((data["def_mean_dist"]) / 2.0)))))/2.0)))/2.0)) +
                0.246874*np.tanh(np.tanh((np.tanh((((((((data["def_min_dist"]) * (data["back_moving_down_field"]))) * 2.0)) * (np.tanh((data["def_std_dist"]))))))))) +
                0.249902*np.tanh(data["X"]) +
                0.250000*np.tanh(((((data["std_dist"]) + (data["back_oriented_down_field"]))) / 2.0)) +
                0.250000*np.tanh((((data["Dis"]) + (data["Down"]))/2.0)) +
                0.249805*np.tanh((((((data["A"]) / 2.0)) + ((((data["def_mean_dist"]) + (data["min_dist"]))/2.0)))/2.0)))
    
    def GP_class_16(self,data):
        return (-4.576388 +
                0.250000*np.tanh(np.tanh((((data["def_mean_dist"]) + (data["def_mean_dist"]))))) +
                0.250000*np.tanh(((((((data["A"]) + (data["Dis"]))) / 2.0)) + ((((((((((data["Distance"]) + (((data["A"]) + (data["Dir"]))))) + (data["Dis"]))) + (data["min_dist"]))) + (data["Dir"]))/2.0)))) +
                0.250000*np.tanh((((np.tanh((((np.tanh((data["def_std_dist"]))) / 2.0)))) + (((data["std_dist"]) + (((data["S"]) + ((((data["S"]) + (((data["def_std_dist"]) + (data["def_max_dist"]))))/2.0)))))))/2.0)) +
                0.250000*np.tanh((((((((np.tanh((data["S"]))) * 2.0)) + (data["Dis"]))) + (data["back_from_scrimmage"]))/2.0)) +
                0.250000*np.tanh(np.tanh((((((((data["max_dist"]) + ((((((data["Distance"]) + (((data["Dir"]) + (data["def_mean_dist"]))))/2.0)) / 2.0)))/2.0)) + (((((data["Distance"]) + ((((data["Dir"]) + ((((data["std_dist"]) + (data["Dir"]))/2.0)))/2.0)))) / 2.0)))/2.0)))) +
                0.246874*np.tanh(((((data["back_oriented_down_field"]) / 2.0)) / 2.0)) +
                0.249902*np.tanh(((((((((((data["std_dist"]) + (((np.tanh(((((data["back_from_scrimmage"]) + (data["back_from_scrimmage"]))/2.0)))) + ((-1.0*((data["std_dist"])))))))/2.0)) + (data["mean_dist"]))) + (data["Dir"]))/2.0)) * (((((((data["max_dist"]) * ((-1.0*((((((data["Quarter"]) * (np.tanh((data["Dir"]))))) / 2.0))))))) / 2.0)) / 2.0)))) +
                0.250000*np.tanh(data["Distance"]) +
                0.250000*np.tanh(((((((((data["min_dist"]) + (data["DefendersInTheBox"]))/2.0)) - (data["max_dist"]))) + (data["back_from_scrimmage"]))/2.0)) +
                0.249805*np.tanh((((((data["def_std_dist"]) * 2.0)) + (((((-1.0*((((data["mean_dist"]) / 2.0))))) + (data["back_oriented_down_field"]))/2.0)))/2.0)))
    
    def GP_class_17(self,data):
        return (-4.786692 +
                0.250000*np.tanh((((data["min_dist"]) + ((((data["max_dist"]) + (data["Distance"]))/2.0)))/2.0)) +
                0.250000*np.tanh(data["Y"]) +
                0.250000*np.tanh(((data["def_max_dist"]) + ((((((((data["X"]) + (data["Distance"]))) + (data["S"]))/2.0)) + (data["S"]))))) +
                0.250000*np.tanh(data["back_from_scrimmage"]) +
                0.250000*np.tanh((((data["def_std_dist"]) + (((((((((((data["back_from_scrimmage"]) + (data["Dir"]))) + (data["def_mean_dist"]))) / 2.0)) * 2.0)) + ((((((0.0)) + ((((((data["def_std_dist"]) + (data["def_mean_dist"]))/2.0)) + (data["Dir"]))))) + (data["def_std_dist"]))))))/2.0)) +
                0.246874*np.tanh(data["YardLine"]) +
                0.249902*np.tanh((((((data["Distance"]) + (data["Distance"]))) + ((((data["Down"]) + ((((data["def_max_dist"]) + (data["Distance"]))/2.0)))/2.0)))/2.0)) +
                0.250000*np.tanh((((data["back_moving_down_field"]) + (((((((((data["back_moving_down_field"]) + (data["min_dist"]))/2.0)) + (data["Dir"]))/2.0)) - (data["back_moving_down_field"]))))/2.0)) +
                0.250000*np.tanh((((data["A"]) + (((data["Distance"]) + ((((data["min_dist"]) + ((((-1.0*((data["back_from_scrimmage"])))) / 2.0)))/2.0)))))/2.0)) +
                0.249805*np.tanh((((np.tanh(((((data["def_mean_dist"]) + (((data["Dis"]) * (np.tanh(((((data["back_from_scrimmage"]) + (data["std_dist"]))/2.0)))))))/2.0)))) + (((data["Dis"]) * 2.0)))/2.0)))
    
    def GP_class_18(self,data):
        return (-4.808144 +
                0.250000*np.tanh(((data["def_std_dist"]) + (((data["Distance"]) + (data["A"]))))) +
                0.250000*np.tanh(((((data["def_mean_dist"]) * 2.0)) + (((data["Dir"]) + ((((((data["Dis"]) + (((np.tanh((data["Dis"]))) / 2.0)))/2.0)) * 2.0)))))) +
                0.250000*np.tanh(((np.tanh((data["Dis"]))) + (((np.tanh((np.tanh(((((-1.0*(((((data["def_std_dist"]) + (((((((((data["S"]) * (data["back_moving_down_field"]))) + (data["Dis"]))) * 2.0)) / 2.0)))/2.0))))) / 2.0)))))) / 2.0)))) +
                0.250000*np.tanh((((((data["def_mean_dist"]) + (data["Dir"]))) + (data["def_std_dist"]))/2.0)) +
                0.250000*np.tanh((((np.tanh(((-1.0*((((((((data["def_min_dist"]) / 2.0)) / 2.0)) * (((np.tanh((((data["back_moving_down_field"]) / 2.0)))) * 2.0))))))))) + (((data["Y"]) / 2.0)))/2.0)) +
                0.246874*np.tanh(((((data["Orientation"]) * (((np.tanh((np.tanh((data["Dir"]))))) / 2.0)))) / 2.0)) +
                0.249902*np.tanh(data["Orientation"]) +
                0.250000*np.tanh(((((data["X"]) / 2.0)) * ((((data["back_moving_down_field"]) + (np.tanh((data["Dis"]))))/2.0)))) +
                0.250000*np.tanh((((0.0)) * (((data["back_from_scrimmage"]) * ((((0.0)) * (((((((((data["back_moving_down_field"]) * (data["Quarter"]))) * (np.tanh((data["mean_dist"]))))) * (((np.tanh((data["mean_dist"]))) / 2.0)))) / 2.0)))))))) +
                0.249805*np.tanh(data["mean_dist"]))
    
    def GP_class_19(self,data):
        return (-5.053334 +
                0.250000*np.tanh(((((data["def_mean_dist"]) + (((data["max_dist"]) - (((((data["min_dist"]) + (((data["Dir"]) / 2.0)))) + ((((((((data["max_dist"]) + (data["max_dist"]))) + ((((data["min_dist"]) + (data["back_from_scrimmage"]))/2.0)))/2.0)) * 2.0)))))))) + (data["max_dist"]))) +
                0.250000*np.tanh(((data["def_mean_dist"]) + (((((((data["max_dist"]) * 2.0)) + (data["max_dist"]))) + (data["Orientation"]))))) +
                0.250000*np.tanh(((((data["Dir"]) + (data["max_dist"]))) + (((data["def_max_dist"]) + (data["max_dist"]))))) +
                0.250000*np.tanh(data["back_moving_down_field"]) +
                0.250000*np.tanh(np.tanh(((((((data["Orientation"]) + (data["back_moving_down_field"]))/2.0)) + (data["Distance"]))))) +
                0.246874*np.tanh(((data["min_dist"]) + (((data["min_dist"]) + (((data["max_dist"]) + (((data["back_moving_down_field"]) + ((((data["Quarter"]) + (data["min_dist"]))/2.0)))))))))) +
                0.249902*np.tanh(((((np.tanh((data["max_dist"]))) / 2.0)) + ((((data["Orientation"]) + (data["def_mean_dist"]))/2.0)))) +
                0.250000*np.tanh((((((data["YardLine"]) + ((((((((data["min_dist"]) * ((((data["min_dist"]) + (data["YardLine"]))/2.0)))) * 2.0)) + (((data["max_dist"]) * (data["max_dist"]))))/2.0)))/2.0)) / 2.0)) +
                0.250000*np.tanh(((data["min_dist"]) / 2.0)) +
                0.249805*np.tanh(((data["A"]) + (np.tanh((np.tanh((((((data["back_moving_down_field"]) / 2.0)) / 2.0)))))))))
    


# # Time for the actual submission

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from kaggle.competitions import nflrush
env = nflrush.make_env()
iter_test = env.iter_test()
gp = GP()
for (test_df, sample_prediction_df) in iter_test:
    basetable = create_features(test_df, deploy=True)
    basetable.drop(['GameId','PlayId'], axis=1, inplace=True)
    scaled_basetable = scaler.transform(basetable)
    
    y_pred_nn = model.predict(scaled_basetable)

#     y_pred_gp = np.zeros((test_df.shape[0],199))
#     ans = gp.GrabPredictions(pd.DataFrame(data=scaled_basetable,columns=basetable.columns))
#     y_pred_gp[:,96:96+20] = ans
    
    y_pred = y_pred_nn#(.6*y_pred_nn+.4*y_pred_gp)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]
    
    preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
    env.predict(preds_df)
    
env.write_submission_file()

