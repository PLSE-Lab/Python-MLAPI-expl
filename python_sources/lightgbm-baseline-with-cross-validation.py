import gc
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from tqdm import tqdm
import lightgbm as lgb

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Prepare data
def clean_word(x):
    x_clean = ''.join(e for e in x if e.isalnum())
    return x_clean

def clean(df):
    df['word1_clean'] = df['word1'].apply(lambda x: clean_word(x))
    df['word2_clean'] = df['word2'].apply(lambda x: clean_word(x))

clean(train)
clean(test)

# Ectract features
def extract_features(df):
    df['n_symbols_1'] = df['word1'].apply(lambda x: len(x))
    df['n_symbols_2'] = df['word2'].apply(lambda x: len(x))
    
    df['n_symbols_clean_1'] = df['word1_clean'].apply(lambda x: len(x))
    df['n_symbols_clean_2'] = df['word2_clean'].apply(lambda x: len(x))

    df['is_word1_in_word2'] = df.apply(lambda row: int(row['word1'] in row['word2']), axis=1)

extract_features(train)
extract_features(test)

def le_encode(train, test):
    pairs = [
        ('word1', 'word2'),
        ('word1_clean', 'word2_clean'),
    ]
                     
    for p in tqdm(pairs):
        train_words = np.unique(np.concatenate([train[p[0]].values, train[p[1]].values], axis=0))
        test_words = np.unique(np.concatenate([test[p[0]].values, test[p[1]].values], axis=0))
        all_words = np.unique(np.concatenate((train_words, test_words)))
        le = LabelEncoder()
        le.fit(all_words)
        
        train['{}'.format(p[0])] = le.transform(train[p[0]].values)
        train['{}'.format(p[1])] = le.transform(train[p[1]].values)
        test['{}'.format(p[0])] = le.transform(test[p[0]].values)
        test['{}'.format(p[1])] = le.transform(test[p[1]].values)

le_encode(train, test)

# Prepare dataset for training
cols_to_drop = [
    'id',
    'similarity',
]

X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train.similarity.values

X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test.id.values

print('train.shape = {}, test.shape = {}'.format(train.shape, test.shape))

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 5,
    'learning_rate': 0.01, 
    'verbose': -1,
    'num_threads': 2,
}

# Build the model
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = KFold(
    n_splits=n_splits, 
    random_state=0)
err_buf = []   

n_features = X.shape[1]

for train_index, valid_index in kf.split(X, y):
    print('Fold {}/{}*{}'.format(cnt + 1, n_splits, n_repeats))
    params = lgb_params.copy() 
    
    lgb_train = lgb.Dataset(
        X.iloc[train_index], 
        y[train_index], 
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X.iloc[valid_index], 
        y[valid_index],
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=150, 
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(60):
            if i < len(tuples):
                print(tuples[i])
            else:
                break

        del importance, model_fnames, tuples

    p = model.predict(X.iloc[valid_index], num_iteration=model.best_iteration)
    err = log_loss(y[valid_index], p)

    print('{} LogLoss: {}'.format(cnt + 1, err))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    err_buf.append(err)


    cnt += 1
    # if cnt > 0: # Comment this to run several folds
    #     break

    del model, lgb_train, lgb_valid, p
    gc.collect

err_mean = np.mean(err_buf)
err_std = np.std(err_buf)
print('LogLoss = {:.6f} +/- {:.6f}'.format(err_mean, err_std))

preds = p_buf/cnt

# Prepare submission
subm = pd.DataFrame()
subm['id'] = id_test
subm['similarity'] = preds
subm.to_csv('submission_baseline.csv', index=False)
