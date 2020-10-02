import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
combine = [train, test]

# print(train.columns.values)


#Print to standard output, and see the results in the "log" section below after running your script
# print("\n\nTop of the training data:")
# print(train.head())
# train.info()

# print("\n\nSummary statistics of training data")
# print(train.describe(include='all'))

#Print grouped averages to identify feature correlations
# print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=True))
# print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=True))
# print(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=True))
# print(train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=True))
# print(train[['SibSp', 'Parch', 'Survived']].groupby(['Parch', 'SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=True))

g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)



#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)