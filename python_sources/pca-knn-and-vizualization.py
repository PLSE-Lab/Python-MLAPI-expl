import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import math as m

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

print ('Loading files...\n')
Xtr = pd.read_csv('../input/train.csv')
Xts = pd.read_csv('../input/test.csv')
ytr = Xtr['label']
del Xtr['label']

print("Lest's start with a class histogram\n")

plt.hist(ytr, normed = False, bins = 10)
plt.xlabel('Digits')
plt.ylabel(' Number of apearences in training')
plt.title('Class Histogram')
plt.xticks(range(10), ['%d' % i for i in range(10)])
plt.savefig('histogram.png')

print('Classes are well balanced\n')

print("Time to see how this digits look like\n")

#We take the first rows and reshape them  to get (28, 28) images
#Let's make a funcion to do that. We might need it again

def see_digits(X, labels, n_digits, figure_name):
    X_ = X[: n_digits].as_matrix()
    X_ = X_.reshape(X_.shape[0], 28, 28)
    fig = plt.figure()
    fig.suptitle(figure_name)
    for i in range (n_digits):
        fig.add_subplot(int(m.sqrt(n_digits))+1, int(m.sqrt(n_digits))+1, i+1)
        plt.imshow(X_[i])
        plt.axis('off')
        plt.title(str(labels[i]))
    fig.tight_layout()
    plt.show()

#now lets see the first 10 digits
see_digits(Xtr, ytr, 10, 'Digits and true labels')

#Divide by 255 to normalize data
Xtr = Xtr/255
Xts = Xts/255
 
pipe = Pipeline([
    ('pca', PCA()),
    ('clf', KNeighborsClassifier())
])

#Best parameters are: N_COMPONENTS = 30, NEIGHBORS = 4 , WEIGHTS =' distance'
#add more params when not running in kaggle
N_COMPONENTS = [20, 50]
NEIGHBORS = [3, 4]
WEIGHTS = ['distance']

param_grid = [{
        'pca__n_components': N_COMPONENTS,
        'clf__n_neighbors': NEIGHBORS,
        'clf__weights': WEIGHTS
    }]

print('Training all possible combinations...\n')
print('*This could take one hour or two*\n')
grid = GridSearchCV(pipe, cv=3, n_jobs=-1, param_grid=param_grid)
grid.fit(Xtr, ytr)
print(pd.DataFrame(grid.cv_results_), '\n')

print('Using best classifier to predict...\n')
yp = grid.best_estimator_.predict(Xts)

#Now lets check some digits and their predicted labels
see_digits(Xts, yp, 10, 'Digits and predicted labels')

#create submission
submission = pd.DataFrame({
    "ImageId": np.arange(1, yp.shape[0] + 1),
    "Label": yp
})

print('Creating submission file...\n')
submission.to_csv("submission.csv", index=False)
print('Done!\n')