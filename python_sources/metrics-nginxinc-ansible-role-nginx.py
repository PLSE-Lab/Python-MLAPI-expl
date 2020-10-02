import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd

import walk_forward_release

REPOSITORY = 'nginxinc/ansible-role-nginx'


def iac_metrics_only(df):
    return df[['avg_play_size','avg_task_size','lines_blank','lines_code','lines_comment','num_authorized_key','num_block_error_handling','num_blocks','num_commands',
               'num_conditions','num_decisions','num_deprecated_keywords','num_deprecated_modules','num_distinct_modules','num_external_modules','num_fact_modules',
               'num_file_exists','num_file_mode','num_file_modules','num_filters','num_ignore_errors','num_import_playbook','num_import_role','num_import_tasks',
               'num_include','num_include_role','num_include_tasks','num_include_vars','num_keys','num_lookups','num_loops','num_math_operations','num_names_with_vars',
               'num_parameters','num_paths','num_plays','num_regex','num_roles','num_suspicious_comments','num_tasks','num_tokens','num_unique_names','num_user_interaction',
               'num_vars','text_entropy']]

def delta_metrics_only(df):
    return df[['delta_avg_play_size','delta_avg_task_size','delta_lines_blank','delta_lines_code','delta_lines_comment','delta_num_authorized_key','delta_num_block_error_handling','delta_num_blocks','delta_num_commands',
               'delta_num_conditions','delta_num_decisions','delta_num_deprecated_keywords','delta_num_deprecated_modules','delta_num_distinct_modules','delta_num_external_modules','delta_num_fact_modules',
               'delta_num_file_exists','delta_num_file_mode','delta_num_file_modules','delta_num_filters','delta_num_ignore_errors','delta_num_import_playbook','delta_num_import_role','delta_num_import_tasks',
               'delta_num_include','delta_num_include_role','delta_num_include_tasks','delta_num_include_vars','delta_num_keys','delta_num_lookups','delta_num_loops','delta_num_math_operations','delta_num_names_with_vars',
               'delta_num_parameters','delta_num_paths','delta_num_plays','delta_num_regex','delta_num_roles','delta_num_suspicious_comments','delta_num_tasks','delta_num_tokens','delta_num_unique_names','delta_num_user_interaction',
               'delta_num_vars','delta_text_entropy']]

def process_metrics_only(df):
    return df[['change_set_avg', 'change_set_max', 'code_churn', 'code_churn_avg', 'code_churn_max', 'commits_count', 'contributors', 'highest_experience',
               'loc_added', 'loc_added_avg', 'loc_added_max', 'loc_removed', 'loc_removed_avg', 'loc_removed_max', 'median_hunks_count', 'minor_contributors']]


data = pd.read_csv('../input/ansibledefectsprediction/ansible.csv')
data = data[data.repo == REPOSITORY].fillna(0)

# Create column to group files belonging to the same release (identified by the commit hash)
data['group'] = data.commit.astype('category').cat.rename_categories(range(1, data.commit.nunique()+1))

# Make sure the data is sorted by commit time (ascending)
data.sort_values(by=['committed_at'], ascending=True)
data = data.reset_index(drop=True)

# Train
X, y = data.drop(['defective'], axis=1), data.defective.values.ravel()

for metrics in ['iac', 'delta', 'process', 'iac-process', 'iac-delta', 'delta-process']:
        
    print(f'====================== {metrics.upper()} ======================')
        
    if metrics == 'iac':
        X_local = iac_metrics_only(X)
        
    elif metrics == 'delta':
        X_local = delta_metrics_only(X)
    
    elif metrics == 'process':
        X_local = process_metrics_only(X)
            
    elif metrics == 'iac-delta':
        X_local = pd.concat([iac_metrics_only(X), delta_metrics_only(X)], axis=1)

    elif metrics == 'iac-process':
        X_local = pd.concat([iac_metrics_only(X), process_metrics_only(X)], axis=1)
    
    elif metrics == 'delta-process':
        X_local = pd.concat([delta_metrics_only(X), process_metrics_only(X)], axis=1)
    
    X_local['group'] = X.group

    # Train
    cv_results, best_index = walk_forward_release.learning(X_local, y, walk_forward_release.models['random_forest'], walk_forward_release.search_params['random_forest'])

    cv_results = pd.DataFrame(cv_results).iloc[[best_index]] # Take only the scores at the best index
    cv_results['n_features'] = X.shape[1]
    cv_results['y_0'] = y.tolist().count(0)
    cv_results['y_1'] = y.tolist().count(1)
    
    with open(f'./{metrics}.json', 'w') as outfile:
        cv_results.to_json(outfile, orient='table', index=False)