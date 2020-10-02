# This is a Kernel to replace the standard train_k from beluga's validation and training split. This kernel can be used as a 
#  direct replacement, since it creates same the train_k csvs along with additional "valid.csv.gz".
# Advantages:
#   1. unique valid set without leakage of data to training set, e.g. select (--valid_samples 100) so we have unique 
#      34000 samples and we can use rest of the data in the training set - a clear distinction :)
#   2. using the complete image set for training 
#         i) if valid_samples == 34000 (100 per class)
#         ii) train samples == 49673579 
#   3. In the keras code by beluga, the following code can be used to run the full image data on the network:
#         total_samples = 49673579
#         batchsize = 340
#         steps = 800
#         epochs = math.ceil(total_samples/ (batchsize * steps))
#   4. Also, this kernel uses multi-processing, reduces data creation by n_processes, mine reduced by 8 times, instead of 
#           beluga's kernel --- full samples ~ 3hr
#           current kernel --- full samples ~ 25 mins

import os
import pandas as pd
import numpy as np
import argparse
import tqdm
import random
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='create a unique validation set and training set')
parser.add_argument('--valid_samples', help='# of samples for validation per class', type=int,default=100)
parser.add_argument('--create_csvs', help='create the train and valid csvs', action='store_true')
parser.add_argument('--ncsvs', help='number of csv splits to be used', type=int, default=100)
parser.add_argument('--shuffle', help='random shuffle and compress the train and validation files', action='store_true')
parser.add_argument('--num_threads', help='# of threads for shuffle task', type=int, default=8)
parser.add_argument('--debug', help='debug command', action='store_true')
args = parser.parse_args()

args.root_dir = os.path.join(os.getcwd(), '..')
train_dir = os.path.join(args.root_dir, 'input', 'train')
traink_dir = os.path.join(args.root_dir, 'input', 'train_k')
if not os.path.exists(traink_dir):
    os.makedirs(traink_dir)


def read_training_csv(category: str, nrows: int=None):
    df = pd.read_csv(os.path.join(train_dir, '{}.csv'.format(category)), nrows=nrows)
    return df


def list_all_categories():
    files = os.listdir(train_dir)
    return sorted([f.split('.')[0] for f in files], key=str.lower)


def get_splits(df):
    indices = range(len(df))
    valid = random.sample(indices, k=args.valid_samples)
    train = [i for i in indices if i not in valid]
    return train, valid


doodles = os.listdir(train_dir)

NCSVS = args.ncsvs
categories = list_all_categories()

if args.debug:
    print(categories)

if args.create_csvs:
    for y, cat in tqdm.tqdm(enumerate(categories)):
        df = read_training_csv(cat)
        df['y'] = y
        train, valid = get_splits(df)
        val = df.loc[valid, :]
        df = df.loc[train, :]

        if args.debug:
            print('\ntrain shape: ({})'.format(df.shape))
            print('valid shape: ({})'.format(val.shape))

        df['cv'] = (df.key_id // 10 ** 7) % NCSVS
        for k in range(NCSVS):
            filename = 'train_k{}.csv'.format(k)
            chunk = df[df.cv == k]
            chunk = chunk.drop(['key_id'], axis=1)
            if y == 0:
                chunk.to_csv(os.path.join(traink_dir, filename), index=False)
            else:
                chunk.to_csv(os.path.join(traink_dir, filename), mode='a', header=False,
                             index=False)

        val.drop('key_id', inplace=True, axis=1)
        if y == 0:
            val.to_csv(os.path.join(args.root_dir, 'input', 'valid.csv'), index=False)
        else:
            val.to_csv(os.path.join(args.root_dir, 'input', 'valid.csv'), index=False, header=False, mode='a')
else:
    print('\n Skipping csv creation. Please use \"--create_csv\" option to run the create block.')


if args.shuffle:
    valid = pd.read_csv(os.path.join(args.root_dir, 'input', 'valid.csv'))
    if args.debug:
        print('\n valid dataset shape: ({})'.format(valid.shape))
        print('valid dataset columns: ({})'.format(valid.columns))
    valid['rnd'] = np.random.rand(len(valid))
    valid = valid.sort_values(by='rnd').drop('rnd', axis=1)
    valid.to_csv(os.path.join(args.root_dir, 'input', 'valid.csv.gz'), compression='gzip', index=False)

    def shuffle_csv(k):
        filename = 'train_k{}.csv'.format(k)
        if os.path.exists(os.path.join(traink_dir, filename)):
            df = pd.read_csv(os.path.join(traink_dir, filename))
            df['rnd'] = np.random.rand(len(df))
            df = df.sort_values(by='rnd').drop('rnd', axis=1)
            df.to_csv(os.path.join(args.root_dir, 'input', 'train_k', filename) + '.gz', compression='gzip',
                      index=False)
            os.remove(os.path.join(args.root_dir, 'input', 'train_k', filename))
            print('({}): train shuffle completed.'.format(k))

    with Pool(processes=args.num_threads) as pool:
        pool.map(shuffle_csv, range(NCSVS))

    print('\n==> random shuffling completed.')
else:
    print('\n Skipping random shuffle of csv. Please use \"--shuffle\" option to run the random shuffle the csvs.')




