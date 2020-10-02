import numpy as np
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.model_selection import GridSearchCV
import itertools
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import warnings
warnings.simplefilter('ignore')

iris = datasets.load_iris()
X, y = iris.data[:,1:3], iris.target

clf1 = KNeighborsClassifier(n_neighbors = 1)
clf2 = GaussianNB()
clf3 = RandomForestClassifier(random_state = 42)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers = [clf1,clf2,clf3], meta_classifier = lr)
print("3 fold cross validtion results:")
for clf, lbl in zip([clf1, clf2, clf3,sclf],
                   ['KNeigbors','Naivebayes','RandomForest','StackingClassifier']):
    score = model_selection.cross_val_score(clf, X, y, cv = 3, scoring = 'accuracy')
    print("Accuracy : %0.2f +/- %0.2f , %s" %(score.mean(), score.std(), lbl))
    
##plotting
gs = gridspec.GridSpec(2,2)

fig = plt.figure(figsize = (10,8))

for clf, lbl, grd in zip([clf1, clf2, clf3, sclf],
                        ['KNeighbors','NaiveBayes','RandomForest','StackingClassifier'],
                        itertools.product([0,1], repeat = 2)):
    clf.fit(X,y)
    ax = plt.subplot(gs[grd[0],grd[1]])
    fig = plot_decision_regions(X = X, y = y, clf = clf)
    plt.title(lbl)

## Using probabilities as Meta features
# use_probas = True will enable probailities, we can either average the scores from multiple classifiers or stack them(recommended)


clf1 = KNeighborsClassifier(n_neighbors = 1)
clf2 = GaussianNB()
clf3 = RandomForestClassifier(random_state = 42)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers = [clf1,clf2,clf3], meta_classifier = lr,
                         use_probas = True, average_probas = False)
print("Using probabilities results:")
for clf, lbl in zip([clf1, clf2, clf3,sclf],
                   ['KNeigbors','Naivebayes','RandomForest','StackingClassifier']):
    score = model_selection.cross_val_score(clf, X, y, cv = 3, scoring = 'accuracy')
    print("Accuracy : %0.2f +/- %0.2f , %s" %(score.mean(), score.std(), lbl))

    
## Using Grid search parameters in stacking classifiers

print("Using grid search for classifiers and metaclassifiers -results:")
params = {'kneighborsclassifier__n_neighbors' :[1,5],
        'randomforestclassifier__n_estimators':[10,50],
        'meta_classifier__C': [0.1,10.0]}

grid = GridSearchCV(estimator = sclf, param_grid = params, cv = 5, refit = True)

grid.fit(X,y)

cv_keys = ('mean_test_score','std_test_score','params')

for r,_ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.3f - %r"
         %(grid.cv_results_[cv_keys[0]][r],
          grid.cv_results_[cv_keys[1]][r],
          grid.cv_results_[cv_keys[2]][r])
         )
    
print("Best parameters : ", grid.best_params_)
print("Accuracy : %0.2f " %(grid.best_score_))


## Stacking classifiers that operaate on different feature sabsubsets
X, y = iris.data, iris.target

pipe1 = make_pipeline(ColumnSelector(cols=(0,2)),
                     LogisticRegression())
pipe2 = make_pipeline(ColumnSelector(cols=(1,2,3)),
                     LogisticRegression())

sclf = StackingClassifier(classifiers = [pipe1, pipe2], meta_classifier= LogisticRegression())

sclf.fit(X,y)
print("Classifiers with different feature set -results: ")
for clf, lbl in zip([clf1, clf2, clf3, sclf],
                   ['Kneighbors','Naivebayes','Randomforest','Stackingclassifiers']):
    scores = model_selection.cross_val_score(clf,  X, y,cv = 3, scoring = 'accuracy')
    print("Accuracy : %0.3f +/- %0.3f - %r" %(scores.mean(), scores.std(), lbl))