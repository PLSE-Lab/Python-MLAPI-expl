import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

os.listdir('../input')

for d in os.listdir('../input/sample_images'):
    print("Patient '{}' has {} scans".format(d, len(os.listdir('../input/sample_images/' + d))))
print('----')
print('Total patients {} Total DCM files {}'.format(len(os.listdir('../input/sample_images')), 
                                                      len(glob.glob('../input/sample_images/*/*.dcm'))))
#Total patients 20 Total DCM files 3604

df_train = pd.read_csv('../input/stage1_labels.csv')
df_train.head()

print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
#Number of training patients: 1397
#Cancer rate: 25.91%

from sklearn.metrics import log_loss
logloss = log_loss(df_train.cancer, np.zeros_like(df_train.cancer) + df_train.cancer.mean())
print('Training logloss is {}'.format(logloss)) 
#0.5721414894789518

sample = pd.read_csv('../input/stage1_sample_submission.csv')
#sample['cancer'] = df_train.cancer.mean()+0.001
#sample.to_csv('naive_submission.csv', index=False) # LB 0.60235 
sample['cancer'] = df_train.cancer.mean()+0.004
sample.to_csv('HDKIM_submission.csv', index=False) # LB 