import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd

import rfe_walk_forward_release

REPOSITORY = 'rvm/rvm1-ansible'

data = pd.read_csv('../input/ansibledefectsprediction/ansible.csv')
data = data[data.repo == REPOSITORY].fillna(0)

# Create column to group files belonging to the same release (identified by the commit hash)
groups = data.commit.astype('category').cat.rename_categories(range(1, data.commit.nunique()+1)).tolist()

# Select only releases with defective data
releases = data.commit.unique()
to_remove = []
for r in releases:
    df = data[data.commit == r]
    if not df.defective.tolist().count(0) or not df.defective.tolist().count(1):
        to_remove.append(r)
        
data = data[~data.commit.isin(to_remove)]

# Make sure the data is sorted by commit time (ascending)
data.sort_values(by=['committed_at'], ascending=True)
data = data.reset_index(drop=True)
data = data.drop(['commit', 'committed_at', 'filepath', 'repo', 'path_when_added', 'tokens'], axis=1)

# Train
X, y = data.drop(['defective'], axis=1), data.defective.values.ravel()

selected, ranking, optimal, scores = rfe_walk_forward_release.rfecv(X, y, groups, 'pr_auc', step=1)

selected_features = X.columns[selected]

ranked_features = sorted(zip(map(lambda x: round(x, 4), ranking), X.columns))
serializable_ranked = []
for tup in ranked_features:
    serializable_ranked.append((int(tup[0]), tup[1]))

result = {
    'optimal_n': int(optimal),
    'ranked': serializable_ranked,
    'selected': selected_features.tolist(),
    'scores': scores.tolist()
}

dest = 'pr_auc.json'
with open(dest, 'w') as outfile:
    json.dump(result, outfile)