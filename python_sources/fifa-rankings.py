# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
print("Setup Complete")
# Path of the file to read
fifa_filepath = "/kaggle/input/data-for-datavis/fifa.csv"

# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)
# Print the first 5 rows of the data
fifa_data.head()
# Set the width and height of the figure
plt.figure(figsize=(10,5))

# Line chart showing how FIFA rankings evolved over time 
sns.lineplot(data=fifa_data)