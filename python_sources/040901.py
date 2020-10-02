import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

print("loading train data")

x_train=train.values[:,1:].astype(float)
y_train=train.values[:,0]

scores=list()
scores_std=list()

print("starting learning")

n_trees=[10,15,20,25,30,40,50,70,100,150]

for n_tree in n_trees:
    print(n_tree)
    recongnizer=RandomForestClassifier(n_tree)
    score=cross_val_score(recongnizer,x_train,y_train)
    scores.append(np.mean(score))
    scores_std.append(np.std(score))
    
sc_array=np.array(scores)
std_array=np.array(scores_std)

print('Score:',sc_array)
print('score_std',std_array)

plt.plot(n_trees,scores)
plt.plot(n_trees,sc_array+std_array,'b--')
plt.plot(n_trees,sc_array-std_array,'b--')
plt.ylabel('CV scores')
plt.xlabel('#of trees')
plt.savefig('cv_trees.png')
plt.show()

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs