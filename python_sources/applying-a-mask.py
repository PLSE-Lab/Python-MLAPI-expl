# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

h5f = h5py.File('../input/overlapping_chromosomes_examples.h5','r')
pairs = h5f['dataset_1'][:]
h5f.close()

def mask_plot(n,th=5):
    grey = pairs[n,:,:,0]
    mask= grey>th
    plt.subplot(221)
    plt.imshow(grey)
    plt.title('max='+str(grey.max()))
    plt.subplot(222)
    plt.imshow(mask)
    tmask = pairs[n,:,:,1]
    plt.subplot(223)
    plt.imshow(tmask)
    plt.savefig("figure.png")
    
mask_plot(6)