import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn import model_selection
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('../input/data.csv', encoding='ISO-8859-1')
# print(data.head())

y = data.diagnosis
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
# print(le.classes_)

x = data.drop(['Unnamed: 32','id','diagnosis'],axis = 1 )
# print(x.head())

# correlation map
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
# plt.show()

corr_columns_to_drop = ['perimeter_mean', 'radius_mean', 'compactness_mean', 'concave points_mean', 'radius_se',
                        'perimeter_se', 'radius_worst', 'perimeter_worst', 'compactness_worst', 'concave points_worst',
                        'compactness_se', 'concave points_se', 'texture_worst', 'area_worst']
x_drop_corr = x.drop(corr_columns_to_drop, axis = 1)        # do not modify x, we will use it later
# print(x_drop_corr.head())

f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x_drop_corr.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
# plt.show()


def func(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1)
    MLA = [
        # Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        # Gaussian Processes
        gaussian_process.GaussianProcessClassifier(),

        # GLM
        linear_model.LogisticRegressionCV(),
        linear_model.PassiveAggressiveClassifier(),
        linear_model.RidgeClassifierCV(),
        linear_model.SGDClassifier(),
        linear_model.Perceptron(),

        # Navies Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),

        # Nearest Neighbor
        neighbors.KNeighborsClassifier(),

        # SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),

        # Trees
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),
    ]

    MLA_compare = pd.DataFrame()
    for row_index, alg in enumerate(MLA):
        alg.fit(x_train, y_train)
        MLA_compare.loc[row_index, 'MLA Name'] = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(x_train, y_train), 4)
        MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(x_test, y_test), 4)

    MLA_compare.sort_values(by=['MLA Test Accuracy'], ascending=False, inplace=True)
    # print(MLA_compare)


    #base model
    tunealg = ensemble.GradientBoostingClassifier()
    tunealg.fit(x_train, y_train)

    # print('Before tuning, parameters: ', tunealg.get_params(), sep="\n\n")
    # print("Before tuning, training set score: {:.5f}". format(tunealg.score(x_train, y_train)))
    print("Before tuning, test     set score: {:.5f}". format(tunealg.score(x_test, y_test)))

    # tune parameters. you can set the value parameter as much as you want as long as within it's requirement.
    param_grid = {
        # 'criterion': 'friedman_mse',
        # 'init': None,
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        # 'loss': 'deviance',
        'max_depth': [1, 2, 3, 4],
        # 'max_features': None,
        # 'max_leaf_nodes': None,
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        # 'min_samples_leaf': 1,
        # 'min_samples_split': 2,
        # 'min_weight_fraction_leaf': 0.0,
        'n_estimators': [10, 15, 25, 35, 45, 100],
        # 'presort': 'auto',
        # 'random_state': None,
        # 'subsample': 1.0,
        # 'verbose': 0,
        'warm_start': [True, False]
    }

    tune_model = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid=param_grid,
                                              scoring='roc_auc', cv=5)
    tune_model.fit(x_train, y_train)

    # print('After tuning, parameters: ', tune_model.best_params_, sep="\n\n")
    # print("After tuning, training set score: {:.5f}".format(tune_model.score(x_train, y_train)))
    print("After tuning, test     set score: {:.5f}".format(tune_model.score(x_test, y_test)))
    print('-'*10)


for x_ in [x, x_drop_corr]:
    func(x_, y)

# OUTPUT:
# Before tuning, test     set score: 0.96503
# After tuning, test     set score: 0.99184
# ----------
# Before tuning, test     set score: 0.95804
# After tuning, test     set score: 0.99029
# ----------

# My thoughts:
# As you can see, if we do not drop correlation columns, 
# we can still achieve the same result.
# So, before jump to drop correlation columns, 
# you should try not drop the columns.
