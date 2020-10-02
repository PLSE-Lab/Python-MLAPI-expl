# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


df  = pd.read_csv("../input/Speed Dating Data.csv", encoding="ISO-8859-1")

fig = plt.figure(figsize=(10,10))
plt.suptitle("What do YOU look for in a partner?")
ax1 = plt.subplot2grid((6,2), (0,0), rowspan =2, colspan=1)
ax2 = plt.subplot2grid((6,2), (0,1), rowspan =2, colspan=2)
ax3 = plt.subplot2grid((6,2), (2,0), rowspan =2, colspan=1)
ax4 = plt.subplot2grid((6,2), (2,1), rowspan =2, colspan=2)
ax5 = plt.subplot2grid((6,2), (4,0), rowspan =2, colspan=1)
ax6 = plt.subplot2grid((6,2), (4,1), rowspan =2, colspan=2)


ax1.hist(df['attr1_1'].dropna(), 10,alpha=0.5, color = 'violet', range=(0,100))
ax1.set_title("Attractive")
ax1.axes.get_xaxis().set_visible(False)

ax2.hist(df['sinc1_1'].dropna(), alpha=0.5, color = 'indigo',  range=(0,100))
ax2.set_title("Sincere")
ax2.axes.get_xaxis().set_visible(False)

ax3.hist(df['intel1_1'].dropna(), alpha=0.5, color = 'blue',  range=(0,100))
ax3.set_title("Intelligent")
ax3.axes.get_xaxis().set_visible(False)

ax4.hist(df['fun1_1'].dropna(), alpha=0.5, color = 'green',  range=(0,100))
ax4.set_title("Fun")
ax4.axes.get_xaxis().set_visible(False)

ax5.hist(df['amb1_1'].dropna(), alpha=0.5, color = 'yellow',  range=(0,100))
ax5.set_title("Ambitious")

ax6.hist(df['shar1_1'].dropna(), alpha=0.5, color = 'pink',  range=(0,100))
ax6.set_title("Shared Interests")

#Comments:
# Intelligent, sincere and fun categories are winning


# sd_train = pd.read_csv('../input/train.csv')
# sd_test = pd.read_csv('../input/test.csv')
