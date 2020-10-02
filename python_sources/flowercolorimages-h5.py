# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

images = []
for i in range(1, 211):
    fname = '../input/flower_images/' + str(i).zfill(4) + '.png'
    image = np.array(ndimage.imread(fname, flatten = False))
    image = scipy.misc.imresize(image, size = (128, 128))
    images.append(image)
    
images = np.asarray(images)    

labels = pd.read_csv('../input/flower_images/flower_labels.csv')
labels = labels['label']
labels = np.asarray(labels)
with h5py.File('../input/FlowerColorImages.h5', 'w') as f:
    f.create_dataset('images', data = images)
    f.create_dataset('labels', data = labels)