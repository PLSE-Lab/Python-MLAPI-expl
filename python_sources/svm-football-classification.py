# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

Primaryskills = pd.read_csv('../input/footballplayers/RonaldovsMessiAvgskills - Sheet1 (1).csv')
Primaryskills.head()

Secondaryskills = pd.read_csv('../input/ronaldo-vs-messi-secondary-skills/RonaldovsMessiSecondaryskills - Sheet1.csv')
Secondaryskills.head()


##sns.lmplot('BallSkills', 'Shooting', data=Primaryskills, palette='Set1', hue="PlayerName", fit_reg=False, scatter_kws={"s":70})
##sns.lmplot('Physical', 'Passing', data=Primaryskills, palette='Set1', hue="PlayerName", fit_reg=False, scatter_kws={"s":70})


RonaldovsMessi = Primaryskills[['BallSkills', 'Shooting']].as_matrix()
type_label= np.where(Primaryskills['PlayerName']=='Lionel Messi',0, 1)

Messi_features = Primaryskills.columns.values[1:].tolist()
Messi_features

model = svm.SVC(kernel='linear')
model.fit(RonaldovsMessi, type_label)

# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 100)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


sns.lmplot('Mental', 'Shooting', data=Primaryskills, palette='Set1', hue="PlayerName", fit_reg=False, scatter_kws={"s":70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down,'k--')
plt.plot(xx, yy_up,'k--')
plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=80, facecolors='none')
plt.plot(xx, yy, linewidth=2, color='black')

sns.lmplot('Physical', 'Passing', data=Primaryskills, palette='Set1', hue="PlayerName", fit_reg=False, scatter_kws={"s":70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down,'k--')
plt.plot(xx, yy_up,'k--')
plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=80, facecolors='none')
plt.plot(xx, yy, linewidth=2, color='black')
