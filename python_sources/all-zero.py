# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


train = pd.read_csv("../input/test.csv")
print(train.shape)

sub = np.zeros((97320),dtype=float)
print(sub.shape)
print(train.values.shape)

submission = pd.DataFrame.from_dict({
    'id': train.id,
    'prediction':sub
})
submission.to_csv('submission.csv', index=False)

print(submission.values[0])