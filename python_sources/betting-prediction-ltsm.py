#!/usr/bin/env python
# coding: utf-8

# # Tennis Match Betting Prediction
# 
# I'm trying to predict the winner of a match based on the current and past matches' information.
# 
# I'm using a LSTM layer to encode each player's past matches and results, concatenate with the features of the current match, then pass through a fully connected layer.
# 
# I've found the data a bit tricky to handle. The data is complex, without strong correlations and in 2 different formats.
# 
# At first I had lot of dataleak, giving me 99% accuracy. So I selected only a few columns with less NAs and did some normalisation.
# 
# Another issue I had was how to handle the past data of each player. It tourned out to be quite heavy.   
# I tried to process some of it on the fly with a generator, but that slowed even more the training.
# 
# I've managed to get around 70% * accuracy after pading and masking the input. So I think I'm getting somewhere.
# 
# I selected all features from only the recent rows, but not much improvement.   
# Fine-tunning the model and the hyperparameters might improve the results.
# 
# ** accuracy varied between 66 and 72 % trying different hyperparameters.*

# # Import and clean data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (15.0, 8.0)


# In[ ]:


def load_atp_data(filename='ATP') :
    if filename != 'ATP' :
        if str(filename)[-1] == '0' :
            filename = f'atp_rankings_{filename[-2:]}s.csv'
        else :
            filename = f'atp_rankings_current.csv'
    data_folder = "../input/atpdata/"
    data = pd.read_csv(data_folder + filename + ".csv", parse_dates=['tourney_date'])
    return data

def clean_atp_data(data) :
    data_clean = data.copy()
    data_clean.score = data_clean.score.apply(lambda x: x if str(x)[0].isnumeric() else np.nan)
    data_clean = data_clean.dropna(axis=0, subset=["score"])
    data_clean = data_clean[data_clean["score"] != 'W/O']
    data_clean = data_clean[data_clean["score"] != ' W/O']
    data_clean = data_clean[data_clean["score"] != 'DEF']
    data_clean = data_clean[data_clean["score"] != 'In Progress']
    data_clean = data_clean[data_clean["score"] != 'Walkover']
    
    data_clean["surface"] = data_clean["surface"].replace('None', np.nan)
    data_clean['match_id'] = data_clean.apply(lambda x: str(x['tourney_date']) + 
                                              str(x['match_num']).zfill(3) + 
                                              str(x['tourney_id']).split('-')[-1], axis=1)
    data_clean = data_clean.drop_duplicates(['tourney_id', 'match_num'])
    
    return data_clean


# In[ ]:


def scores_to_list(scores) :
    scores = scores.split()
    scores_list = []
    for i, score in enumerate(scores) :
#         if score == 'RET' :
#             score == '0-0'
        score_list = score.split('-')
        score_list = [s.split('(')[0] for s in score_list]
        score_list = [int(s) if s.isnumeric() else 0 for s in score_list]
        if max(score_list) > 7 :
            if not ((i + 1) == len(scores) and bool(len(scores) % 2)) :
                if score_list[0] > score_list[1] :
                    score_list = [7,6]
                else :
                    score_list = [6,7]
        scores_list.append(score_list)
    return scores_list


# In[ ]:


data = load_atp_data()
data = clean_atp_data(data)
data['score'] = data['score'].apply(scores_to_list)


# In[ ]:


data.describe()


# In[ ]:


data.dtypes


# In[ ]:


data


# In[ ]:


data.score[:20]


# # Feature selection
# 
# - 1st run. I selected the columns with less then ~ 25% null values.
# 
# I've created 2 features :
#  - match_num_norm - Mesure how far in the tournement the match occur.
#  - season - A player could perform better durring summer or winter.

# In[ ]:


g = sns.barplot(data.columns, data.isna().sum(axis=0))
g.set_xticklabels(data.columns, rotation=80)
plt.show()


# In[ ]:


# def get_player_hist(row, data, i=0) :
#     try :
#         player_id = row['id']
#     except :
#         player_id = row['id2']
#     match_id = row['match_id']
#     if match_id[8:11] == '069' :
#         print(match_id)
# #     print(match_id[8:11])
# #     match_date = row['match_date']
#     sel = data[data['id'] == player_id]
# #     data.sort_values('match_id')
#     past_matches = sel[sel['match_id'] < match_id]
#     return past_matches.sort_values('match_date', ascending=True).values

def select_features(data) :
    x = pd.DataFrame()
    x['surface'] = data['surface'].astype('category')
    x['match_id'] = data['match_id'].astype('str')
    x['best_of'] = data['best_of'].astype('category')
    x['round'] = data['round'].astype('category')
    x['tourney_level'] = data['tourney_level'].astype('category')
    x['season'] = data['tourney_date'].dt.strftime('%j').astype('float')
    d = data[['tourney_date','tourney_id','match_num']].copy()
    d['match_num_min'] = d[['tourney_id','match_num']].groupby('tourney_id', sort=False).transform(min)
    d['match_num_max'] = d[['tourney_id','match_num']].groupby('tourney_id', sort=False).transform(max)
    match_num_scale = lambda x: max(x['match_num'] - x['match_num_min'], 1) / max(x['match_num_max'] - x['match_num_min'], 1)
    x['match_num_norm'] = d.apply(match_num_scale, axis=1)
    x['match_date'] = d.apply(lambda x: x['tourney_date'] + pd.DateOffset(days=x['match_num'] - x['match_num_min']), axis=1)
    
    x2 = x.copy()
    max_rank = data['loser_rank'].max()
    
    x['p1_id'] = data['winner_id'].astype('int')
    x2['p1_id'] = data['loser_id'].astype('int')
    x['p1_rank'] = data['winner_rank'].fillna( max_rank ).astype('float')
    x2['p1_rank'] = data['loser_rank'].fillna( max_rank ).astype('float')
    x['p1_age']  = data['winner_age'].astype('float')
    x2['p1_age']  = data['loser_age'].astype('float')
    x['p1_ht']   = data['winner_ht'].astype('float')
    x2['p1_ht']   = data['loser_ht'].astype('float')
    x['p1_hand'] = data['winner_hand'].fillna('U').astype('category')
    x2['p1_hand'] = data['loser_hand'].fillna('U').astype('category')
    ioc = pd.concat((data.winner_ioc, data.loser_ioc))
    to_remove = ioc.value_counts().iloc[20:].index.tolist()
    x['p1_ioc'] = data['winner_ioc'].replace(to_remove, 'other').astype('category')
    x2['p1_ioc'] = data['loser_ioc'].replace(to_remove, 'other').astype('category')
    
    
    x2['p2_id'] = data['winner_id'].astype('int')
    x['p2_id'] = data['loser_id'].astype('int')
    x2['p2_rank'] = data['winner_rank'].fillna( max_rank ).astype('float')
    x['p2_rank'] = data['loser_rank'].fillna( max_rank ).astype('float')
    x2['p2_age']  = data['winner_age'].astype('float')
    x['p2_age']  = data['loser_age'].astype('float')
    x2['p2_ht']   = data['winner_ht'].astype('float')
    x['p2_ht']   = data['loser_ht'].astype('float')
    x2['p2_hand'] = data['winner_hand'].fillna('U').astype('category')
    x['p2_hand'] = data['loser_hand'].fillna('U').astype('category')
    ioc = pd.concat((data.winner_ioc, data.loser_ioc))
    to_remove = ioc.value_counts().iloc[20:].index.tolist()
    x2['p2_ioc'] = data['winner_ioc'].replace(to_remove, 'other').astype('category')
    x['p2_ioc'] = data['loser_ioc'].replace(to_remove, 'other').astype('category')
    
    x['label'] = 1
    x2['label'] = 0
    x['label'] = x['label'].astype('uint8')
    x2['label'] = x2['label'].astype('uint8')

    
    x_duplicated = pd.concat([x,x2]).sort_index()
    x_duplicated = x_duplicated.drop(['p1_ioc','p2_ioc','p1_hand','p2_hand'], axis=1)
    
    idx = np.random.randint(2, size=len(x)).astype('bool')
    x = x[idx]
    x2 = x2[~idx]

    x = pd.concat([x,x2]).sort_index()
    
    return x, x['label'].values, x_duplicated


# In[ ]:


# d = data.iloc[:500]
d = data.copy()
X, y, X_dup = select_features(d)
cols = X.columns


# # Some data exploration

# In[ ]:


sns.barplot(X.columns, X.isna().sum(axis=0))
plt.show()


# In[ ]:


def plot_feat(data, col, kind='bar', bins=10) :
    if kind == 'hist' :
        x = data[[col, 'label']].copy()
        x[col] = pd.cut(data[col], bins)
        x.groupby([col, 'label']).size().reset_index()         .pivot(columns='label', index=col, values=0).fillna(0)        .plot(kind='bar', stacked=True)
    else :
            data[[col, 'label']].groupby([col, 'label']).size().reset_index()             .pivot(columns='label', index=col, values=0).plot(kind=kind, stacked=True)
    plt.show()


# In[ ]:


plot_feat(X,'p1_ht')


# In[ ]:


plot_feat(X,'p1_ioc')


# In[ ]:


plot_feat(X,'p1_age', 'hist', 15)


# In[ ]:


plot_feat(X,'p1_hand')


# In[ ]:


plot_feat(X,'match_num_norm', 'hist', 10)


# In[ ]:


sns.lineplot(list(range(len(X.match_num_norm.unique()))),np.sort(X.match_num_norm.unique()))
plt.show()


# In[ ]:


idx = 5000
sns.boxplot(data['tourney_id'].iloc[-idx:], X['match_num_norm'].iloc[-idx:])
plt.show()


# In[ ]:


X.head()


# In[ ]:


X.tail()


# In[ ]:


sns.scatterplot(X['match_num_norm'], X['p1_age'], hue=X['label'])
plt.show()


# In[ ]:


X.columns


# In[ ]:


data.columns


# In[ ]:


cor = X.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


cor_target = abs(cor["label"])
relevant_features = cor_target[cor_target>0.01]
relevant_features


# # Normalize data

# In[ ]:


from sklearn import preprocessing


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler_dup = preprocessing.MinMaxScaler()

cols = X.select_dtypes('float').columns
x_scaled = min_max_scaler.fit_transform(X[cols])
X[cols] = x_scaled

cols_dup = X_dup.select_dtypes('float').columns
cols_dup =  list(set(cols_dup) - set(['match_id','p1_id','p2_id','match_date']))
x_scaled_dup = min_max_scaler_dup.fit_transform(X_dup[cols_dup])
X_dup[cols_dup] = x_scaled_dup


# In[ ]:


X = pd.get_dummies(X, columns=X.select_dtypes('category').columns)
X = X.fillna(0)

X_dup = pd.get_dummies(X_dup, columns=X_dup.select_dtypes('category').columns)
X_dup = X_dup.fillna(0)


# In[ ]:


X_dup = X_dup.drop(['match_id','p2_id'], axis=1)


# In[ ]:


test_size = .1
train_size = .8

total_len = X.shape[0]
X_split, X_test = X[:-int(total_len*test_size)], X[-int(total_len*test_size):]

total_len = X_split.shape[0]
idx_random = np.random.permutation(total_len)
idx_cut = int(total_len * train_size)
idx_train, idx_valid = idx_random[:idx_cut], idx_random[idx_cut:]
X_train, X_valid = X_split.iloc[idx_train,:], X_split.iloc[idx_valid,:]

len_train = len(X_train)


# In[ ]:


# from sklearn.model_selection import train_test_split


# # Checking again the correlation

# In[ ]:


from sklearn.linear_model import LassoCV


# In[ ]:


reg = LassoCV(random_state=42)
x = X.drop(['match_id','match_date','label'], axis=1)
reg.fit(x, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(x,y))
coef = pd.Series(reg.coef_, index = x.columns)


# In[ ]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[ ]:


imp_coef = coef.sort_values()
imp_coef = imp_coef[abs(imp_coef) > 0]
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(x, y)
clf.feature_importances_  

m = SelectFromModel(clf, prefit=True)
x_new = m.transform(x)
x_new.shape


# In[ ]:


g = sns.barplot(list(range(len(clf.feature_importances_))), clf.feature_importances_)
g = g.set_xticklabels(x.columns, rotation=80)


# # LSTM Model

# In[ ]:


from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import RepeatVector
# from tensorflow.keras.layers import TimeDistributed
# from tensorflow.keras.layers import Lambda
# from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Masking
from tensorflow.keras.utils import plot_model
# from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import Callback
import datetime


# In[ ]:


n_features = X_dup.shape[-1] - 2
# generator will drop ['match_date','p1_id']
n_match_features = X.shape[-1] - 5 
# generator will drop ['match_id','match_date','p1_id','p2_id','label']


# In[ ]:


class My_Callback(Callback):
    def __init__(self,batch_interval, validation_data):
        metrics = ['loss', 'val_loss', 'acc', 'val_acc']
        self.metrics = metrics
        self.batch_interval = batch_interval
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        self.test_data = validation_data

    def on_train_begin(self,log={}):
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        self.batch_number = 0
        self.epoch_number = 0

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_batch_end(self,batch,logs={}):
        self.batch_number +=1
        
        if self.batch_number % self.batch_interval == 0:
            self.epoch_number +=1
            self.loss.append(logs['loss'])
            self.acc.append(logs['accuracy'])
        return
    
    def on_epoch_end(self,batch,logs={}):
        loss, acc = self.model.evaluate_generator(self.test_data, steps=int(1000*.33), verbose=0)
        self.loss.append(loss)
        self.val_acc.append(acc)
        return


# In[ ]:


def create_model(n_feat, n_match_feat) :
    K.clear_session()

    input_p1 = Input(shape=(None, n_feat))
    input_p2 = Input(shape=(None, n_feat))
    input_match = Input(shape=([n_match_feat]))
    
    masking = Masking(mask_value=0.0)
    encoder = LSTM(32, activation='sigmoid', return_sequences=False)

    hidden_p1 = Dropout(.4)(encoder(masking(input_p1)))
    hidden_p2 = Dropout(.4)(encoder(masking(input_p2)))
    
    merge_layer = Concatenate()([input_match, hidden_p1, hidden_p2])
    hidden_merged = Dense(32, activation='relu')(merge_layer)
    predictions = Dense(1, activation='sigmoid')(hidden_merged)
    
    model = Model(inputs=[input_match, input_p1, input_p2], outputs=predictions)
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    return model


# In[ ]:


model = create_model(n_features, n_match_features)


# In[ ]:


batch_size = 64
t_delta = datetime.timedelta(days=600)

def train_generator(x_matches, x_players, n_feat, n_match_feat, batch_size=1):
    p1_input_batch = []
    p1_mask_batch = []
    p2_input_batch = []
    p2_mask_batch = []
    p1_seq_len_batch = []
    p2_seq_len_batch = []
    match_input_batch = []
    label_batch = []
    count_batch_i = 0
    p1_max_len = 0
    p2_max_len = 0
    while True :
        for i, x in x_matches.iterrows() :
            count_batch_i += 1
            p1_id = x['p1_id']
            p2_id = x['p2_id']
            match_date = x['match_date']
            p1_input = x_players[x_players['p1_id'] == p1_id]
            p1_input = p1_input[(p1_input['match_date'] < match_date) & 
                            (p1_input['match_date'] > match_date - t_delta)] \
                            .drop(['match_date','p1_id'], axis=1).values.tolist()
            if p1_input == [] :
                p1_input = [[0]*n_feat]
            p1_seq_len = len(p1_input)
            p1_max_len = max(p1_max_len, p1_seq_len)
            p1_input_batch.append(p1_input)
            p1_seq_len_batch.append(p1_seq_len)
            
            p2_input = x_players[x_players['p1_id'] == p2_id]
            p2_input = p2_input[(p2_input['match_date'] < match_date) & 
                            (p2_input['match_date'] > match_date - t_delta)] \
                            .drop(['match_date','p1_id'], axis=1).values.tolist()
            if p2_input == [] :
                p2_input = [[0]*n_feat]
            p2_seq_len = len(p2_input)
            p2_max_len = max(p2_max_len, p2_seq_len)
            p2_input_batch.append(p2_input)
            p2_seq_len_batch.append(p2_seq_len)
            
            label = x['label']
            match_input = x.drop(['match_id','match_date','p1_id','p2_id','label'])
            match_input_batch.append(match_input.values)
            label_batch.append(label)
            
            if count_batch_i == batch_size :
                match_input_batch = np.array(match_input_batch).reshape(-1,n_match_feat)
#                 p1_mask_batch = np.zeros((batch_size, p1_max_len))
#                 p2_mask_batch = np.zeros((batch_size, p2_max_len))
                for s_idx, s_len in enumerate(p1_seq_len_batch) :
#                     p1_mask_batch[s_idx,:s_len] = True
                    p1_input_batch[s_idx] = p1_input_batch[s_idx] + [[0]*n_feat] * (p1_max_len - s_len)
                for s_idx, s_len in enumerate(p2_seq_len_batch) :
#                     p2_mask_batch[s_idx,:s_len] = True
                    p2_input_batch[s_idx] = p2_input_batch[s_idx] + [[0]*n_feat] * (p2_max_len - s_len)
                input_batch = [match_input_batch, 
                               np.array(p1_input_batch).reshape(batch_size,-1,n_feat), 
#                                np.array(p1_mask_batch ).reshape(batch_size,-1,1), 
                               np.array(p2_input_batch).reshape(batch_size,-1,n_feat)
#                                np.array(p2_mask_batch ).reshape(batch_size,-1,1)
                              ]
                return_labels = np.array(label_batch).reshape(batch_size,1)
    
                p1_input_batch = []
                p1_mask_batch = []
                p2_input_batch = []
                p2_mask_batch = []
                match_input_batch = []
                label_batch = []
                count_batch_i = 0
                p1_max_len = 0
                p2_max_len = 0
                p1_seq_len_batch = []
                p2_seq_len_batch = []
                
                yield input_batch, return_labels


# In[ ]:


# history = My_Callback(batch_interval=10, validation_data=train_generator(X_valid)) # 100*len(X_train)/16
history = model.fit_generator(train_generator(X_train, X_dup, n_features, n_match_features, batch_size), 
                              steps_per_epoch=int(len(X_train)/batch_size), 
                              validation_data=train_generator(X_valid, X_dup, n_features, n_match_features, batch_size), 
                              validation_steps=int(len(X_valid)/batch_size), 
                              epochs=2, verbose=1)  # , callbacks=[history]


# In[ ]:


for key in history.history:
    plt.plot(history.history[key], label=key)
    plt.legend()
plt.show()


# In[ ]:


test_score = model.evaluate_generator(train_generator(X_test, X_dup, n_features, n_match_features, batch_size), 
                                      steps=len(X_test), verbose=1)
test_score


# # Simple dense model without past matches' info

# In[ ]:


def create_model_dense() :
    K.clear_session()
    
    model = Sequential()
    model.add(Dense(32, input_dim=n_match_features, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
model_dense = create_model_dense()


# In[ ]:


y_train_dense = X_train['label'].values.reshape(-1,1)
y_valid_dense = X_valid['label'].values.reshape(-1,1)
y_test_dense = X_test['label'].values.reshape(-1,1)
X_train_dense = X_train.drop(['match_id','match_date','p1_id','p2_id','label'], axis=1).values.reshape(-1,n_match_features)
X_valid_dense = X_valid.drop(['match_id','match_date','p1_id','p2_id','label'], axis=1).values.reshape(-1,n_match_features)
X_test_dense = X_test.drop(['match_id','match_date','p1_id','p2_id','label'], axis=1).values.reshape(-1,n_match_features)
history_dense = model_dense.fit(X_train_dense, y_train_dense, batch_size=batch_size,
                                validation_data=[X_valid_dense, y_valid_dense], 
                                epochs=4, verbose=1)


# In[ ]:


for key in history_dense.history:
    plt.plot(history_dense.history[key], label=key)
    plt.legend()
plt.show()


# In[ ]:


test_score_dense = model_dense.evaluate(X_test_dense, y_test_dense, batch_size=batch_size, verbose=1)
test_score_dense


# # Select features
# - Now instead I'll try using the most recent data that has more information about each match.
# 
# I've only used the extra features to encode each player's past data, since those features (first serve, ace, etc..) are not known prior to the match.

# In[ ]:


d = data[data['tourney_date'] < '1991-01-01'].copy()
g = sns.barplot(d.columns, d.isna().sum(axis=0))
g.set_xticklabels(d.columns, rotation=90)
print(f'NA count prior to 1991 : {d["l_1stIn"].isna().sum()}')
plt.show()


# In[ ]:


d = data[data['tourney_date'] > '1991-01-01'].copy()
g = sns.barplot(d.columns, d.isna().sum(axis=0))
g.set_xticklabels(d.columns, rotation=90)
print(f'NA count after 1991 : {d["l_1stIn"].isna().sum()}')
plt.show()


# In[ ]:


d = data[['tourney_date','l_1stIn']].copy()
d['l_1stIn'] = d['l_1stIn'].isna()
d['tourney_date'] = pd.cut(d['tourney_date'],60)
d = d.groupby(['tourney_date']).sum()
g = sns.barplot(d.index.values.astype('str'), d.l_1stIn.values)
g.set_xticklabels(d.index.values.astype('str'), rotation=90)
plt.show()


# In[ ]:


def select_all_features(data) :
    x = pd.DataFrame()
    x['surface'] = data['surface'].astype('category')
    x['match_id'] = data['match_id'].astype('str')
    x['best_of'] = data['best_of'].astype('category')
    x['round'] = data['round'].astype('category')
    x['tourney_level'] = data['tourney_level'].astype('category')
    d = data[['tourney_date','tourney_id','match_num']].copy()
    date = pd.to_datetime(data['tourney_date'], format='%Y%m%d').copy()
    x['season'] = date.dt.strftime('%j').astype('float')    
    d['tourney_date'] = date
    d['match_num_min'] = d[['tourney_id','match_num']].groupby('tourney_id', sort=False).transform(min)
    d['match_num_max'] = d[['tourney_id','match_num']].groupby('tourney_id', sort=False).transform(max)
    match_num_scale = lambda x: max(x['match_num'] - x['match_num_min'], 1) / max(x['match_num_max'] - x['match_num_min'], 1)
    x['match_num_norm'] = d.apply(match_num_scale, axis=1)
    x['match_date'] = d.apply(lambda x: x['tourney_date'] + pd.DateOffset(days=x['match_num'] - x['match_num_min']), axis=1)
    
    x2 = x.copy()
    max_rank = data['loser_rank'].max()

    x['p1_id'] = data['winner_id'].astype('int')
    x2['p1_id'] = data['loser_id'].astype('int')
    x['p1_rank'] = data['winner_rank'].fillna(max_rank).astype('float')
    x2['p1_rank'] = data['loser_rank'].fillna(max_rank).astype('float')
    x['p1_age']  = data['winner_age'].astype('float')
    x2['p1_age']  = data['loser_age'].astype('float')
    x['p1_ht']   = data['winner_ht'].astype('float')
    x2['p1_ht']   = data['loser_ht'].astype('float')
    x['p1_hand'] = data['winner_hand'].fillna('U').astype('category')
    x2['p1_hand'] = data['loser_hand'].fillna('U').astype('category')
    ioc = pd.concat((data.winner_ioc, data.loser_ioc))
    to_remove = ioc.value_counts().iloc[20:].index.tolist()
    x['p1_ioc'] = data['winner_ioc'].replace(to_remove, 'other').astype('category')
    x2['p1_ioc'] = data['loser_ioc'].replace(to_remove, 'other').astype('category')
    
    x2['p2_id'] = data['winner_id'].astype('int')
    x['p2_id'] = data['loser_id'].astype('int')
    x2['p2_rank'] = data['winner_rank'].fillna(max_rank).astype('float')
    x['p2_rank'] = data['loser_rank'].fillna(max_rank).astype('float')
    x2['p2_age']  = data['winner_age'].astype('float')
    x['p2_age']  = data['loser_age'].astype('float')
    x2['p2_ht']   = data['winner_ht'].astype('float')
    x['p2_ht']   = data['loser_ht'].astype('float')
    x2['p2_hand'] = data['winner_hand'].fillna('U').astype('category')
    x['p2_hand'] = data['loser_hand'].fillna('U').astype('category')
    ioc = pd.concat((data.winner_ioc, data.loser_ioc))
    to_remove = ioc.value_counts().iloc[20:].index.tolist()
    x2['p2_ioc'] = data['winner_ioc'].replace(to_remove, 'other').astype('category')
    x['p2_ioc'] = data['loser_ioc'].replace(to_remove, 'other').astype('category')
    
    cols_to_add = ['_1stIn', '_1stWon', '_2ndWon', '_SvGms', '_ace', '_bpFaced', '_bpSaved', '_df', '_svpt']
    for c in cols_to_add :
        x['p1'+c] = data['w'+c].astype('float')
        x2['p1'+c] = data['l'+c].astype('float')
        x2['p2'+c] = data['w'+c].astype('float')
        x['p2'+c] = data['l'+c].astype('float')
    
    x['label'] = 1
    x2['label'] = 0
    x['label'] = x['label'].astype('uint8')
    x2['label'] = x2['label'].astype('uint8')

    
    x_duplicated = pd.concat([x,x2]).sort_index()
    x_duplicated = x_duplicated.drop(['p1_ioc','p2_ioc','p1_hand','p2_hand'], axis=1)
    
    idx = np.random.randint(2, size=len(x)).astype('bool')
    x = x[idx]
    x2 = x2[~idx]
    

    x = pd.concat([x,x2]).sort_index()
    drop_cols = [p+c for c in cols_to_add for p in ['p1','p2']]
    x = x.drop(drop_cols, axis=1)
    
    return x, x['label'].values, x_duplicated


# In[ ]:


d = data[data['tourney_date'] > datetime.datetime(1991,1,1)].copy()
X_1991, y_1991, X_dup_1991 = select_all_features(d)
cols_1991 = X_1991.columns


# In[ ]:


min_max_scaler_1991 = preprocessing.MinMaxScaler()
min_max_scaler_dup_1991 = preprocessing.MinMaxScaler()

cols = X_1991.select_dtypes('float').columns
x_scaled_1991 = min_max_scaler_1991.fit_transform(X_1991[cols])
X_1991[cols] = x_scaled_1991

cols_dup = X_dup_1991.select_dtypes('float').columns
cols_dup =  list(set(cols_dup) - set(['match_id','p1_id','p2_id','match_date']))
x_scaled_dup_1991 = min_max_scaler_dup_1991.fit_transform(X_dup_1991[cols_dup])
X_dup_1991[cols_dup] = x_scaled_dup_1991

X_1991 = pd.get_dummies(X_1991, columns=X_1991.select_dtypes('category').columns)
X_1991 = X_1991.fillna(0)

X_dup_1991 = pd.get_dummies(X_dup_1991, columns=X_dup_1991.select_dtypes('category').columns)
X_dup_1991 = X_dup_1991.fillna(0)

X_dup_1991 = X_dup_1991.drop(['match_id','p2_id'], axis=1)

test_size = .1
train_size = .8
total_len = X_1991.shape[0]
X_split_1991, X_test_1991 = X_1991[:-int(total_len*test_size)], X_1991[-int(total_len*test_size):]
total_len = X_split_1991.shape[0]
idx_random = np.random.permutation(total_len)
idx_cut = int(total_len * train_size)
idx_train, idx_valid = idx_random[:idx_cut], idx_random[idx_cut:]
X_train_1991, X_valid_1991 = X_split_1991.iloc[idx_train,:], X_split_1991.iloc[idx_valid,:]


# # 2nd LSTM model
# Same model architecture as before, but different input shape.

# In[ ]:


n_features_1991 = X_dup_1991.shape[-1] - 2
n_match_features_1991 = X_1991.shape[-1] - 5

model_1991 = create_model(n_features_1991, n_match_features_1991)

history_1991 = model_1991.fit_generator(train_generator(X_train_1991, X_dup_1991, n_features_1991, n_match_features_1991, batch_size), 
                                        steps_per_epoch=int(len(X_train_1991)/batch_size), 
                                        validation_steps=int(len(X_valid_1991)/batch_size), 
                                        validation_data=train_generator(X_valid_1991, X_dup_1991, 
                                                                        n_features_1991, n_match_features_1991, batch_size), 
                                        epochs=2, verbose=1)  # , callbacks=[history]


# In[ ]:


for key in history_1991.history:
    plt.plot(history_1991.history[key], label=key)
    plt.legend()
plt.show()


# In[ ]:


test_score_1991 = model_1991.evaluate_generator(train_generator(X_test_1991, X_dup_1991, 
                                                                n_features_1991, n_match_features_1991, batch_size), 
                                                steps=len(X_test_1991), verbose=1)
test_score_1991

