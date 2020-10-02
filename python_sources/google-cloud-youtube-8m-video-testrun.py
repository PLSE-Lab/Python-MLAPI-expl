#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


labels_df = pd.read_csv('../input/label_names.csv')
filenames = ["../input/video_level/train-{}.tfrecord".format(i) for i in range(10)]
print("Unique labels = {}" . format(len(labels_df['label_name'].unique())))
print("Total labels = {}".format(len(labels_df['label_name'])))

