import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import savefig
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Convert Sex into workable feature, 1 for female and 0 for male passenger
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})
# Impute missing age values with median age
print(train['Age'].head)
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
print(train['Age'].head)
# Create a variable for if passenger was a minor (ie<18)
train['Minor?'] = 0
train['Minor?'][train['Age'] < 15] = 1
test['Minor?'] = 0
test['Minor?'][test['Age'] < 15] = 1

# View correlation matrix with survival
corr = train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.title('Correlation Matrix for Titanic Training Data')
savefig("CorrelationMatrix.png")

# Select relevant features, in this case only two
# Highest correlation is with Sex, then Pclass, but Age is surprisingly poorly correlated
trainX = train[['Survived','Pclass','Sex']]
testX = test[['Pclass','Sex']]

# Select output variable
trainY = trainX['Survived'].copy()
del trainX['Survived']

# Flatten into a 1-D array
trainY = np.ravel(trainY)

# Create multilayer perceptron model
clf = MLPClassifier()
clf.fit(trainX,trainY)

# Create prediction file
predictions = clf.predict(testX)
submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId'].copy()
submission['Survived'] = predictions

#Any files you save will be available in the output tab below
submission.to_csv('submission.csv', index=False)