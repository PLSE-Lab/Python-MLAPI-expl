#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import tarfile
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.decomposition
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def make_int_json_index(data):\n    return {int(k):v for k,v in data.items()}    \nbase_dir = "../input/data"\nfiles = {f:open(os.path.join(base_dir, f)) for f in os.listdir(base_dir)}\nmeta_info = {\n    part: make_int_json_index(json.load(files[part]))\n    for part in ["track_names", "album_names", "artist_names", "genre_names"]\n}\ntrack_links = make_int_json_index(json.load(files[\'track_links\']))\ntrain_set, test_set, user_events = [pd.read_csv(files[part]) for part in ["train_set", "test_set", "user_events"]]')


# In[ ]:


user_events.head()


# In[ ]:


train_set.head()


# In[ ]:


print("users", user_events.userId.nunique())


# In[ ]:


get_ipython().run_cell_magic('time', '', "user_events.groupby(['itemType', 'event'])[['userId', 'itemId']].count()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "user_events.groupby(['itemType', 'event'])[['userId', 'itemId']].nunique()")


# In[ ]:


def get_track_info(_id):
    return u"{}[{}] - {} - {}".format(meta_info["track_names"].get(_id, u"##"), int(_id),
        u",".join([meta_info["genre_names"].get(g, u"{}".format(g)) for g in track_links.get(_id, {}).get("genres", [])]),
        u",".join([meta_info["artist_names"].get(a, u"##") for a in track_links.get(_id, {}).get("artists", [])]),
    )


# In[ ]:


get_track_info(414578)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'event_scores = pd.DataFrame({"event": ["dislike", "skip", "play", "like"],\n                             "score":[-10, -1, 1, 10]})\ntrack_events = user_events[user_events.itemType == "track"][[\'userId\', \'itemId\', \'event\']].copy()\nscore_matrix = track_events.merge(event_scores, on="event").groupby([\'userId\', \'itemId\']).score.sum().reset_index()\ntrack_events = None')


# In[ ]:


score_matrix.head(2)


# In[ ]:


track_scores = score_matrix.groupby('itemId').score.sum().reset_index()
track_scores.sort_values('score', ascending=False, inplace=True)


# In[ ]:


plt.plot(track_scores.score.values)
plt.yscale('log')


# In[ ]:


for t in track_scores[:20].itemId.values:
    print(get_track_info(t))


# In[ ]:


get_ipython().run_cell_magic('time', '', "track_popularity = track_scores.set_index('itemId').score.to_dict()")


# In[ ]:


def train_accuracy(metric):
    total_lines, right_answers = train_set.shape[0], 0
    for _, line in train_set.iterrows():
        track1_score, track2_score = metric(line.user, line.track1),  metric(line.user, line.track2)
        is_track1_best = 1 if track1_score > track2_score else 0
        right_answers += 1 if is_track1_best == line.Category else 0
    return right_answers / float(total_lines)

def save_test_prediction(metric, output):
    test_set_predictions = []
    for _, line in test_set.iterrows():
        track1_score, track2_score = metric(line.user, line.track1),  metric(line.user, line.track2)
        is_track1_best = 1 if track1_score > track2_score else 0
        test_set_predictions.append({"Id": line.Id, "Category": is_track1_best})
    pd.DataFrame(test_set_predictions)[["Id", "Category"]].to_csv(output, index=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'save_test_prediction(lambda user, track: track_popularity.get(track, 0), "popularity")\nprint("train accuracy", train_accuracy(lambda user, track: track_popularity.get(track, 0)))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'using_tracks = track_scores[:100000].itemId.values\nmatrix = score_matrix[score_matrix.itemId.isin(using_tracks)]\n\nuser_index, user_ids = pd.factorize(matrix.userId)\ntrack_index, track_ids = pd.factorize(matrix.itemId)\nmatrix = sp.coo_matrix((matrix.score, (user_index, track_index)))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'svd = sklearn.decomposition.TruncatedSVD(n_components=40, n_iter=10)\nsvd.fit(matrix)\n\nuser_profiles = svd.transform(matrix)\nitem_profiles = svd.components_')


# In[ ]:


svd.explained_variance_ratio_.sum()


# In[ ]:


track_to_p = {_id:p for p,_id in enumerate(track_ids)}
uid_to_p = {_id:p for p,_id in enumerate(user_ids)}
item_normed_profiles = item_profiles / np.linalg.norm(item_profiles, axis=0)
def print_nearest(trackId, n=20):
    query = item_normed_profiles[:, track_to_p[trackId]]
    for t in track_ids[np.argsort(-item_normed_profiles.T.dot(query))[:n]]:
        print(get_track_info(t))


# In[ ]:


print_nearest(1695498)


# In[ ]:


def predict_by_svd(userId, trackId):
    user_p = uid_to_p.get(userId, None)
    track_p = track_to_p.get(trackId, None)
    if user_p is None or track_p is None:
        return 0
    return item_profiles[:, track_p].dot(user_profiles[user_p])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'save_test_prediction(predict_by_svd, "simplest_svd")\nprint("train accuracy", train_accuracy(predict_by_svd))')


# In[ ]:


def build_playlist(user_scores):
    sorted_tracks = sorted([(t, user_scores(t)) for t in track_ids], key=lambda x: -x[1])
    res = []
    taken_artists = set()
    for t, _ in sorted_tracks:
        artists = set(track_links[t]['artists'])
        if len(artists.intersection(taken_artists)) != 0:
            continue
        taken_artists.update(artists)
        res.append(t)
    return res[:30]


# In[ ]:


prob_user_events = pd.DataFrame([
    (0, "track", 1695498, "play", 1550318400),
], columns=["userId", "itemType", "itemId", "event", "unixtime"])

prob_user_track_events = prob_user_events[prob_user_events.itemType == "track"][['userId', 'itemId', 'event']].copy()
prob_score_matrix = prob_user_track_events.merge(event_scores, on="event").groupby(['userId', 'itemId']).score.sum().reset_index()
prob_score_matrix = prob_score_matrix[prob_score_matrix.itemId.isin(track_ids)]
prob_matrix = sp.coo_matrix((prob_score_matrix.score, (prob_score_matrix.userId, prob_score_matrix.itemId.map(track_to_p))),
                            (1, item_profiles.shape[1]))
prob_user_profile = svd.transform(prob_matrix)[0]

def prob_user_score(trackId):
    track_p = track_to_p.get(trackId, None)
    if track_p is None:
        return 0
    return item_profiles[:, track_p].dot(prob_user_profile)

for t in build_playlist(prob_user_score):
    print(get_track_info(t))


# In[ ]:




