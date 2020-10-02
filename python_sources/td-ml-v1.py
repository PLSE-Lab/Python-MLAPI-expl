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
import numpy as np # linear algebra
import pandas as pd
app_events = pd.read_csv("../input/app_events.csv", nrows=100)


print(app_events.head())
app_events.info()

app_labels = pd.read_csv('../input/app_labels.csv', nrows=100)
print(app_labels.head())

events = pd.read_csv("../input/events.csv", nrows=100)

print(events.head())
#print(type(events["timestamp"][1]))
#print(type(events["timestamp"]))

gender_age_train = pd.read_csv('../input/gender_age_train.csv', nrows=100)
print(gender_age_train.head())


