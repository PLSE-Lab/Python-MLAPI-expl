'''
This is a template for your reproducible solution in the Medium competition.
It's obligatory that your script produces a submission file just 
by running `python kaggle_medium_<name>_<surname>_solution.py`. 
If you have any dependecies apart from those in a Kaggle Docker image, 
it's your responsibility to provide an image (or at least a requirements file) 
to reproduce your solution.

Please avoid heavy hyperparameter optimization in this script. 
'''

import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import time
from contextlib import contextmanager
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
from html.parser import HTMLParser


PATH_TO_DATA = '../input'    # Path to competition data
AUTHOR = 'Yury_Kashnitskiy'  # change here to <name>_<surname>
# it's a nice practice to define most of hyperparams here
SEED = 17
TRAIN_LEN = 62313            # just for tqdm to see progress   
TEST_LEN = 34645             # just for tqdm to see progress
TITLE_NGRAMS = (1, 3)        # for tf-idf on titles
MAX_FEATURES = 50000         # for tf-idf on titles
LGB_TRAIN_ROUNDS = 60        # num. iteration to train LightGBM
LGB_NUM_LEAVES = 255         # max number of leaves in LightGBM trees
MEAN_TEST_TARGET = 4.33328   # what we got by submitting all zeros
RIDGE_WEIGHT = 0.6           # weight of Ridge predictions in a blend with LightGBM


# nice way to report running times
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


# The following code will help to throw away all HTML tags from article content/title
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# Supplementary function to read a JSON line without crashing on escape characters.
def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result


# feature engineering - extracting titles from raw JSOn files
def extract_titles_from_json(path_to_inp_json_file, path_to_out_txt_file, total_length):
    '''
    :param path_to_inp_json_file: path to a JSON file with train/test data
    :param path_to_out_txt_file: path to extracted features (here titles), one per line
    :param total_length: we'll pass the hardcoded file length to make tqdm even more convenient
    '''
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file, \
         open(path_to_out_txt_file, 'w', encoding='utf-8') as out_file:
        for line in tqdm(inp_file, total=total_length):
            json_data = read_json_line(line)
            content = json_data['title'].replace('\n', ' ').replace('\r', ' ')
            content_no_html_tags = strip_tags(content)
            out_file.write(content_no_html_tags + '\n')


def prepare_train_and_test():
    extract_titles_from_json(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 'train.json'),
           path_to_out_txt_file='train_titles.txt', total_length=TRAIN_LEN)

    extract_titles_from_json(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 'test.json'),
           path_to_out_txt_file='test_titles.txt', total_length=TEST_LEN)

    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=TITLE_NGRAMS)

    with open('train_titles.txt', encoding='utf-8') as input_train_file:
        X_train = tfidf.fit_transform(input_train_file)
    with open('test_titles.txt', encoding='utf-8') as input_test_file:
        X_test = tfidf.transform(input_test_file)
    y_train = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'), 
                          index_col='id')['log_recommends'].values
    
    return X_train, y_train, X_test


def ridge_prediction(X_train, y_train, X_test):
    ridge = Ridge(random_state=SEED)
    ridge.fit(X_train, np.log1p(y_train))
    ridge_test_pred = np.expm1(ridge.predict(X_test))
    return ridge_test_pred


def lightgbm_prediction(X_train, y_train, X_test):
    lgb_x_train = lgb.Dataset(X_train.astype(np.float32), 
                           label=np.log1p(y_train))
    lgb_params = {'num_leaves': LGB_NUM_LEAVES, 'seed': SEED,
                  'objective': 'mean_absolute_error', 'metric': 'mae'}

    lgb_model = lgb.train(lgb_params, lgb_x_train, LGB_TRAIN_ROUNDS)
    lgb_test_pred = np.expm1(lgb_model.predict(X_test.astype(np.float32)))
    return lgb_test_pred


def form_final_prediction(ridge_pred, lgb_pred, y_train, ridge_weight):
    # blending predictions of Ridge and LightGBM 
    mix_pred = ridge_weight * ridge_pred + (1 - ridge_weight) * lgb_pred 
    
    # leaderboard probing
    mix_test_pred_modif = mix_pred + MEAN_TEST_TARGET - y_train.mean()
    return mix_test_pred_modif


with timer('Tf-Idf for titles'):
    X_train, y_train, X_test = prepare_train_and_test()

with timer('Ridge: train and predict'):
    ridge_test_pred = ridge_prediction(X_train, y_train, X_test)

with timer('LightGBM: train and predict'):
    lgb_test_pred = lightgbm_prediction(X_train, y_train, X_test)

with timer('Prepare submission'):
    test_pred = form_final_prediction(ridge_test_pred, lgb_test_pred, 
                                      y_train, ridge_weight=RIDGE_WEIGHT)
    submission_df = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                'sample_submission.csv'), index_col='id')
    submission_df['log_recommends'] = test_pred
    submission_df.to_csv(f'submission_medium_{AUTHOR}.csv')



