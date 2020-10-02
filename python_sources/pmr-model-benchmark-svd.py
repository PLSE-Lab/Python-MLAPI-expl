import pandas as pd
import numpy as np
import surprise as sp
import time
import warnings; warnings.simplefilter('ignore')

train_ratings_df = pd.read_csv('../input/predict-movie-ratings-v22/train.csv')[['user', 'movie', 'rating']]
dataset_reader = sp.Reader()
bench_trainset = sp.Dataset.load_from_df(train_ratings_df, dataset_reader)

bench_testset = pd.read_csv('../input/predict-movie-ratings-v22/test.csv')[['user', 'movie']]
submission = pd.read_csv('../input/predict-movie-ratings-v22/sampleSubmission.csv')
submission['rating'] = submission['rating'].astype(float)

training_data = bench_trainset.build_full_trainset()

models = {
    'SVD': {'clf': sp.SVD(random_state=0)},
    'NMF': {'clf': sp.NMF(random_state=0)},
    'CoClustering': {'clf': sp.CoClustering(random_state=0)},
    'SlopeOne': {'clf': sp.SlopeOne()}
}

def predict(row):
    return bench_model.predict(row['user'], row['movie']).est

for name, model in models.items():
    print('Fitting', name, 'model...', end='')
    
    bench_model = model['clf']
    time_start = time.time()
    bench_model.fit(training_data)
    model['fit_time'] = time.time() - time_start

    print(' completed in', model['fit_time'], 'seconds.')
    print('Predicting ratings using', name, 'model...', end='')
    
    time_start = time.time()
    submission['rating'] = bench_testset.apply(predict, axis=1)
    model['predict_time'] = time.time() - time_start
    print(submission.head())

    print(' completed in', model['predict_time'], 'seconds.')
    submission.to_csv('submission_{}.csv'.format(name), index=False)