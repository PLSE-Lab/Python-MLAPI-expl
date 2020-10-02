# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import os.path as path
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
plt.rcParams['figure.figsize'] = 16, 12

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
#from sklearn import cross_validation #Additional scklearn functions
# import xgboost as xgb
# from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 17
from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

path2data = '/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw'
import os
for dirname, _, filenames in os.walk(path2data):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%% Loading data
df_target_train = pd.read_csv(path.join(path2data,'df_target_train.csv'))
print('df_target_train:', df_target_train.shape)

df_sample_submit = pd.read_csv(path.join(path2data,'df_sample_submit.csv'))
print('df_sample_submit:', df_sample_submit.shape)

df_tracks = pd.read_csv(path.join(path2data,'df_tracks.csv'))
print('df_tracks:', df_tracks.shape)

df_genres = pd.read_csv(path.join(path2data,'df_genres.csv'))
print('df_genres:', df_genres.shape)

df_features = pd.read_csv(path.join(path2data,'df_features.csv'))
print('df_features:', df_features.shape)

#%% Data Prepration
from tqdm import tqdm
from collections import defaultdict

# extract tracks for each genre
genre2tracks = defaultdict(list)
for _, row in tqdm(df_target_train.iterrows(), total=df_target_train.shape[0]):
    for g_id in row['track:genres'].split(' '):
        genre2tracks[int(g_id)].append(row['track_id'])

## How many songs per genre
df_tmp = pd.DataFrame(
    [(k, len(v)) for (k, v) in genre2tracks.items()],
    columns=['genre', 'n_tracks']
).sort_values(
    ['n_tracks'],
    ascending=False
)

sns.barplot(
    data=df_tmp,
    x='genre',
    y='n_tracks',
    order=df_tmp['genre']
)
plt.title('Disribution of tracks per genre')
plt.show()

df_features.set_index(['track_id'], inplace=True)
### Applying PCA transformation
scaler = StandardScaler()
df_features_scaled = scaler.fit_transform(df_features)

pca = PCA(n_components=0.90, random_state=RANDOM_STATE).fit(df_features_scaled)
df_features_scaled_pca = pca.transform(df_features_scaled)

df_features_scaled_pca = pd.DataFrame(df_features_scaled_pca, index = df_features.index)



df_target_train.set_index(['track_id'], inplace=True)

from collections import defaultdict
r = defaultdict(int)
for _, row in df_target_train.iterrows():
    for x in row['track:genres'].split(' '):
        r[int(x)] += 1

labels = list(sorted(r.keys()))

#%% Builing Linear model for Binary classification

# for each class we build a linear model for binary classification
# we skip all classes if it has less then 1000 positive samples

val_ratio = 0.2
models = {}

for g_id, positive_samples in tqdm(genre2tracks.items()):
    if len(positive_samples) < 1000:
        continue
    # construct train/val using negative sampling
    negative_samples = list(set(reduce(lambda a, b: a + b, [v for (k, v) in genre2tracks.items() if k != g_id])).difference(positive_samples))
    negative_samples = np.random.choice(
        negative_samples,
        size=len(positive_samples),
        replace=len(negative_samples) < len(positive_samples)
    )
    train_positive_samples = np.random.choice(
        positive_samples,
        size=int((1 - val_ratio)*len(positive_samples)),
        replace=False
    )
    val_positive_samples = list(set(positive_samples).difference(train_positive_samples))

    train_negative_samples = np.random.choice(
        negative_samples,
        size=int((1 - val_ratio)*len(negative_samples)),
        replace=False
    )
    val_negative_samples = list(set(negative_samples).difference(train_negative_samples))
    # train a models and pick one
    models[g_id] = {
        'acc': -1
        }
    X = df_features.loc[np.hstack([train_positive_samples, train_negative_samples])]
    Y = [1.0]*len(train_positive_samples) + [0.0]*len(train_negative_samples)
###################################################################################################
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 10)]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [5,8,10,12,15]

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'bootstrap': bootstrap}
    RFC = RandomForestClassifier()
    folds = 3
    param_comb = 5
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = RANDOM_STATE)
    random_search = RandomizedSearchCV(RFC,
                                       param_distributions=random_grid,
                                       n_iter = param_comb,
                                       scoring='f1',
                                       n_jobs=-1,
                                       cv=skf.split(X,Y),
                                       verbose=3,
                                       random_state=RANDOM_STATE)
    # Here we go
    random_search.fit(X, Y)

    model = random_search.best_estimator_
    p_val = model.predict_proba(df_features.loc[np.hstack([val_positive_samples, val_negative_samples])])[:, 1]
    y_val = [1.0]*len(val_positive_samples) + [0.0]*len(val_negative_samples)

    # choose threshold
    best_t = -1
    best_acc = -1
    for t in np.linspace(0.01, 0.99, 99):
        acc = f1_score(y_val, (p_val >= t).astype(np.float))
        if acc > best_acc:
            best_acc = acc
            best_t = t
    models[g_id]['acc'] = best_acc
    models[g_id]['t'] = best_t
    models[g_id]['model'] = model
#    models[g_id]['c'] = c

#%%

with open('./models_RF.pkl', 'wb') as f:
    pickle.dump(models, f)

# correct test preditions to equalize median number of genres per track
def get_test(k=1.0):
    g_prediction = {}
    for g_id, d in tqdm(models.items()):
        p = d['model'].predict_proba(df_features.loc[df_sample_submit['track_id'].values])[:, 1]
        g_prediction[g_id] = df_sample_submit['track_id'].values[p > k*d['t']]

    track2genres = defaultdict(list)
    for g_id, tracks in g_prediction.items():
        for t_id in tracks:
            track2genres[t_id].append(g_id)

    return track2genres

track2genres = get_test(k=1.0)

# median number of genres per track in test
np.median([len(v) for v in track2genres.values()])

# median number of genres per track in train
z = df_target_train['track:genres'].apply(lambda s: len([int(x) for x in s.split(' ')])).median()

z
#%%
for k in np.linspace(1, 2, 11):
    track2genres = get_test(k=k)
    print(k, np.median([len(v) for v in track2genres.values()]))


track2genres = get_test(k=1.55)
#%%
df_sample_submit['track:genres'] = df_sample_submit.apply(lambda r: ' '.join([str(x) for x in track2genres[r['track_id']]]), axis=1)

def genre_check(string):
    #string = list(string.values)
    if len(string)==0:
        string = '15 38'
    if '15' in string:
        string = string
    else:
        string = string + ' 15'
    if '38' in string:
        string = string
    else:
        string = string + ' 38'
    return string

df_sample_submit['track:genres'] = df_sample_submit['track:genres'].apply(genre_check)

df_sample_submit.to_csv(path.join('/kaggle/working','submit_RF_2_after.csv'), index=False)
