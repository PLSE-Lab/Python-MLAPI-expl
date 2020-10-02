import numpy as np
import pandas as pd
#Visualization libraries
import seaborn as sb
import matplotlib.pyplot as plt
import pylab as pl

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())
print(train.describe(include=['O']))
print(train.columns.values)

#Print the data types and the number of values for each variable in the test and train datasets
train.info()
print('_'*40)
test.info()

#Check if there is any correlation between the features and the survival
print('='*50)
disp = train[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean().sort_values(by='Survived', ascending=False)
print(disp)
disp = train[['Sex', 'Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Survived', ascending=False)
print(disp)
disp = train[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean().sort_values(by='Survived', ascending=False)
print(disp)
disp = train[['Parch', 'Survived']].groupby('Parch', as_index=False).mean().sort_values(by='Survived', ascending=False)
print(disp)

# Visualizations - for better understanding
g = sb.FacetGrid(train, col='Survived')
g.map(plt.hist,'Age', bins=20)


# Make an array of x values
x = [1, 2, 3, 4, 5]
# Make an array of y values for each x value
y = [1, 4, 9, 16, 25]
# use pylab to plot x and y
pl.plot(x, y)
# show the plot on the screen
pl.show()

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)