'''
Train and fine-tune a Decision Tree for the moons dataset by following these steps:
Use make_moons(n_samples=10000, noise=0.4) to generate a moons dataset.
Use train_test_split() to split the dataset into a training set and a test set.
Use grid search with cross-validation (with the help of the GridSearchCV class) 
to find good hyperparameter values for a DecisionTreeClassifier.  
Hint: try various values for max_leaf_nodes.Train it on the full training set using these hyperparameters, 
and measure your model’s performance on the test set. You should get roughly 85% to 87% accuracy.
'''
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

X_moons, y_moons = make_moons(n_samples=100000, noise=0.4, random_state=42)
#X is a 2d matrix - represent scatter plots of each column's value
plt.scatter(X_moons[:, 0], X_moons[:, 1])
plt.show()


# %% [code]
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size=0.2)
clf = DecisionTreeClassifier()
param_grid = [{'max_leaf_nodes':[2, 4, 10, 20, 30], 'random_state':[42]
    
}]

grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

y_pred = grid_search.predict(X_test)

print(classification_report(y_pred, y_test))