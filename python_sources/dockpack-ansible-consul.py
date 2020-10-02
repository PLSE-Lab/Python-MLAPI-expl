import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd

import walk_forward_release

REPOSITORY = 'dockpack/ansible-consul'

data = pd.read_csv('../input/ansibledefectsprediction/ansible.csv')
data = data[data.repo == REPOSITORY].fillna(0)

# Create column to group files belonging to the same release (identified by the commit hash)
data['group'] = data.commit.astype('category').cat.rename_categories(range(1, data.commit.nunique()+1))

# Make sure the data is sorted by commit time (ascending)
data.sort_values(by=['committed_at'], ascending=True)
data = data.reset_index(drop=True)
data = data.drop(['commit', 'committed_at', 'filepath', 'repo', 'path_when_added', 'tokens'], axis=1)

# Train
X, y = data.drop(['defective'], axis=1), data.defective.values.ravel()

most_recurrent_features = [['text_entropy', 'num_tokens','group'],
                            ['num_keys', 'num_tokens','group'],
                            ['text_entropy', 'num_keys','group'],
                            ['text_entropy', 'num_keys', 'num_tokens','group'],
                            ['num_tokens', 'lines_code','group'],
                            ['text_entropy', 'num_tokens', 'lines_code','group'],
                            ['num_keys', 'lines_code','group'],
                            ['num_keys', 'num_tokens', 'lines_code','group'],
                            ['text_entropy', 'avg_task_size','group'],
                            ['avg_task_size', 'num_tokens','group']]


for i in range(0, len(most_recurrent_features)):
    
    X_local = X[most_recurrent_features[i]]
    
    # Train
    cv_results, best_index = walk_forward_release.learning(X_local, y, walk_forward_release.models['random_forest'], walk_forward_release.search_params['random_forest'])

    cv_results = pd.DataFrame(cv_results).iloc[[best_index]] # Take only the scores at the best index
    cv_results['n_features'] = X.shape[1]
    cv_results['y_0'] = y.tolist().count(0)
    cv_results['y_1'] = y.tolist().count(1)
    cv_results['features'] = str(most_recurrent_features[i])

    with open(f'./rf_{i}.json', 'w') as outfile:
        cv_results.to_json(outfile, orient='table', index=False)