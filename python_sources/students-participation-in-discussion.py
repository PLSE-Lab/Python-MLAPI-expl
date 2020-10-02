# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data_train= pd.read_csv('../input/xAPI-Edu-Data.csv')
data_train.info()
data_train.sample(5)

# Any results you write to the current directory are saved as output.
sns.factorplot("Topic","Discussion",data=data_train,hue="Semester",kind='bar',ci=None, aspect=1.6)
plt.title("Students Participation in the Discussion ")
plt.ylabel("Participation Count")
plt.xlabel("Course Topic");
plt.subplots_adjust(top=0.8)
plt.xticks(rotation=45) 
