
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np 
import pandas as pand 
import matplotlib.pyplot as matplt
import seaborn as sb
import os
import glob

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

os.listdir('../input')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
for d in os.listdir('../input/sample_images'):
    print("Patient '{}' has {} scans".format(d, len(os.listdir('../input/sample_images/' + d))))
print('----')
print('Total patients {} Total DCM files {}'.format(len(os.listdir('../input/sample_images')), len(glob.glob('../input/sample_images/*/*.dcm'))))
sizes = [os.path.getsize(dcm)/1000000 for dcm in glob.glob('../input/sample_images/*/*.dcm')]
print('DCM file sizes: min {:.3}MB max {:.3}MB avg {:.3}MB std {:.3}MB'.format(np.min(sizes),np.max(sizes), np.mean(sizes), np.std(sizes)))

df_train = pand.read_csv('../input/stage1_labels.csv')
df_train.head()

print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))

from sklearn.metrics import log_loss
logloss = log_loss(df_train.cancer, np.zeros_like(df_train.cancer) + df_train.cancer.mean())
print('Training logloss is {}'.format(logloss))

sample = pand.read_csv('../input/stage1_sample_submission.csv')
sample['cancer'] = df_train.cancer.mean()
sample.to_csv('output_submision.csv')