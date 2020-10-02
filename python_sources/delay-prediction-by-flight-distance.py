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

# Read the test data
test = pd.read_csv("../input/test.csv")
print(test.iloc[0])

# We posit that the longer the flight distance, the more likely it is to be delayed
test["is_delayed"] = test["distance"] / np.max(test["distance"])

# Make submission and save to file
submission = pd.DataFrame()
submission["id"] = test["id"]
submission["is_delayed"] = test["is_delayed"]
submission.to_csv("submission_by_distance.csv", index=False)