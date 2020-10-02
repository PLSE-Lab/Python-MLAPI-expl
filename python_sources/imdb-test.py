# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Print one line of the CSV, which is viewable in the Log view pane
with open("../input/movie_metadata.csv") as movie_file:
    reader = csv.reader(movie_file)
    print(next(reader))

# Use pandas to read and print only 2 lines
data_frame = pd.read_csv("../input/movie_metadata.csv", nrows=2)
print(data_frame)
