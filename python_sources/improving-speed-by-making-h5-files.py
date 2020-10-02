from shutil import copyfile
import imageio
import numpy as np
import h5py
import os
import pandas as pd

## This part from https://www.kaggle.com/drn01z3/create-k-fold-splits-depths-stratified
## as an example of making folds
DATA_ROOT = '../input/'
n_fold = 8

def main():
    depths = pd.read_csv(os.path.join(DATA_ROOT, 'depths.csv'))
    train = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv')) ## We will save train files only
    depths = depths[depths['id'].isin(train['id'].tolist())]
    depths.sort_values('z', inplace=True)
    depths.drop('z', axis=1, inplace=True)
    depths['fold'] = (list(range(n_fold))*depths.shape[0])[:depths.shape[0]]
    
    ## Lets make h5 files for each fold. This should improve speed a bit
    for fold in range(n_fold):
        hf = h5py.File(DATA_ROOT+'fold_{}.h5'.format(fold), 'w')
        files = depths[depths.fold==fold]['id'].values
        for file in files:
            im = imageio.imread(DATA_ROOT+'train_images/{}.png'.format(file))
            hf.create_dataset(file, data=im)
            mask = imageio.imread(DATA_ROOT+'masks/{}.png'.format(file))
            hf.create_dataset('mask_'+file, data=mask)
        hf.close()
    
def show_file():
    ## If you want to read a file just use script example below
    hf = h5py.File(DATA_ROOT+'fold_0.h5', 'r')
    file_id_example = '0c089f7c1b'
    import matplotlib.pyplot as plt
    f, plot = plt.subplots(1,2)
    plot[0].imshow(hf.get(file_id_example))
    plot[1].imshow(hf.get('mask_'+file_id_example))
    hf.close()

if __name__ == '__main__':
    main()
    show_file()