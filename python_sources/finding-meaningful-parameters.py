# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df = pd.read_csv('../input/train.csv')

df = df[['Pclass', 'Survived','Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']]
df['Sex'] = df['Sex'].map({'female': 1, 'male' : 0})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df = df.fillna(0)

surv_yes = df[df.Survived == 1]
surv_no = df[df.Survived == 0]

fig, axes = plt.subplots(4, 2, figsize=(15,30))
ax = axes.ravel()

for i in range(7):

    _, par = np.histogram(df.values[:, i], bins=30)
    ax[i].hist(surv_yes.values[:, i], bins=par, color='blue', alpha=1)
    ax[i].hist(surv_no.values[:, i], bins=par, color='red', alpha=0.5)
    ax[i].set_title(df.columns[i])
ax[i].set_yticks(())
ax[0].legend(["Survived", "Dead"])
plt.show()
