import numpy as np

im_data = np.load('/kaggle/input/imdbwikiimagedataset/imdb-wiki-image-npy/imdb-wiki-image-data.npy')
im_data = np.array(im_data, 'uint8')
ages = np.load('/kaggle/input/imdbwikiimagedataset/imdb-wiki-image-npy/age.npy')
ages = np.array(ages, 'uint8')
genders = np.load('/kaggle/input/imdbwikiimagedataset/imdb-wiki-image-npy/gender.npy')
genders = np.array(genders, 'uint8')

np.savez_compressed('imdb_dataset.npz', images=im_data.reshape(-1, 100, 100), ages=ages, genders=genders)
