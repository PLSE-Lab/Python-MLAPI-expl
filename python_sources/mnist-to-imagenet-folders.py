#! /usr/bin/env python

import sys
from math import ceil, log10
from pathlib import Path

from tqdm import tqdm

import numpy as np
import pandas as pd
from imageio import imwrite


def reindex_with_filename(df: pd.DataFrame):
    width = ceil(log10(len(df)))
    prefix = ('train/' + df['label'].astype(np.str)
              if 'label' in df.columns
              else 'test')
    df['name'] = prefix + df.index.map(f'/{{:0{width}}}.png'.format)
    df.set_index('name', inplace=True)

    if 'label' in df.columns:
        df['label'].to_csv(data_dir/'labels.csv', header=True)
        df.drop(columns='label', inplace=True)


def csv_to_imgs(csv_file: str):
    df = pd.read_csv(input_dir/csv_file, dtype=np.uint8)
    reindex_with_filename(df)

    imgs = df.values.reshape(-1, 28, 28)
    for i, (name, im) in tqdm(enumerate(zip(df.index, imgs)),
                              desc=csv_file, total=len(df)):
        imwrite(data_dir/name, im, 'PNG-PIL', optimize=True)


if __name__ == '__main__' and len(sys.argv) == 3:
    input_dir = Path(sys.argv[1])
    data_dir = Path(sys.argv[2])

    for i in range(10):
        (data_dir/'train'/f'{i}').mkdir(parents=True)
    (data_dir/'test').mkdir()

    for f in 'train.csv', 'test.csv':
        csv_to_imgs(f)
