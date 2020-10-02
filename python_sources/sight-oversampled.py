import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import ensemble, tree
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
import warnings

# silence warnings
warnings.filterwarnings("ignore")

# get data
df = pd.read_csv("../input/sight.csv")

# get predictors
df_X = df.iloc[:,3:]
cols = df_X.columns.tolist()

# get targets
df_T_binary = df['total'].apply(lambda x: 1 if x > 0 else x).values

# scaling
scalar = MinMaxScaler(feature_range=(0, 1))
df_X_scaled = pd.DataFrame(scalar.fit_transform(df_X))
df_X_scaled.columns = cols

# add new features
df_X_scaled['wspd*wvht'] = df_X_scaled['wspd'] * df_X_scaled['wvht']
df_X_scaled['wspd*wvht*mwd'] = df_X_scaled['wspd'] * df_X_scaled['wvht'] * df_X_scaled['mwd']
df_X_scaled['inday_slot*day_of_week'] = df_X_scaled['inday_slot'] * df_X_scaled['day_of_week']
cols = df_X_scaled.columns.tolist()

# function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# function of grid search
def gridSearch(X, Y, model, param_grid):
    grid_search = GridSearchCV(model, param_grid=param_grid)
    grid_search.fit(X, Y)
    print("GridSearchCV invastigated %d candidate parameter settings."
          % (len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)
    return grid_search

# random forest grid paramenters and grid search
def random_forest_classifier(X, Y):
    model = ensemble.RandomForestClassifier()
    param_grid = {'n_estimators': [5, 50, 100, 250],
                  'max_features': [5, 8, 14],
                  'max_depth': [3, 8, None],
                  'min_samples_split': [0.01, 0.05],
                  'min_samples_leaf': [0.01, 0.05],
                  "criterion": ["gini", "entropy"]}
    return gridSearch(X, Y, model, param_grid)

def take_second(alist):
    return alist[1]
    
# get variable importance with random forest
def variable_importance_with_random_forest(X, Y, model, maxFeature):
    model.fit(X, Y)
    importance_list = []
    max_importance = 0.0
    for name, importance in zip(cols, model.feature_importances_):
        importance_list.append([name, importance])
        if importance > max_importance: max_importance = importance
    for i in importance_list:
        i[1] = np.round((i[1]/max_importance)* 100, 2)
    importance_list.sort(key=take_second, reverse=True)
    importance_list = importance_list[:maxFeature]
    return importance_list

# simulation
def simulate(X, Y, model, number_of_run, oversample):
    c00 = 0.0; c01 = 0.0; c10 = 0.0; c11 = 0.0
    for i in range(number_of_run):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=i)
        # oversampling minority class
        if oversample:
            sm = SMOTE(sampling_strategy='minority', random_state=123456)
            x_train, y_train = sm.fit_sample(x_train, y_train)
        model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        # smooter = lambda x: 1 if x >= threshold else 0
        # prediction = np.array([smooter(xi) for xi in y_pred])
        prediction = np.array(model.predict(x_test))
        conf_matrix = confusion_matrix(y_test, prediction)
        c00 += conf_matrix[0][0]
        c01 += conf_matrix[0][1]
        c10 += conf_matrix[1][0]
        c11 += conf_matrix[1][1]
    c00 /= number_of_run   
    c01 /= number_of_run  
    c10 /= number_of_run   
    c11 /= number_of_run 
    return np.array([[c00, c01] / (c00+c01), [c10, c11] / (c10+c11)])

def drawTree(X, Y, tree, col):
    export_graphviz(tree, out_file='tree.dot', 
                feature_names = col,
                class_names = ['0', '1'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    Image(filename = 'tree.png')

####################################################################################################################

# RANDOM FOREST
# random_forest_classifier(df_X_scaled, df_T_binary)
# GridSearchCV invastigated 288 candidate parameter settings. 
# Model with rank: 1 Mean validation score: 0.016 (std: 0.012) 
# Parameters: {'criterion': 'gini', 'max_depth': None, 'max_features': 5, 
# 'min_samples_leaf': 0.05, 'min_samples_split': 0.01, 'n_estimators': 5} 

model = ensemble.RandomForestClassifier(criterion='gini', max_depth=None, max_features=3, min_samples_leaf=0.05,
                                min_samples_split=0.01, n_estimators=50)
conf_matrix = simulate(df_X_scaled, df_T_binary, model, 100, oversample=True)
var_importance = variable_importance_with_random_forest(df_X_scaled, df_T_binary, model, 16)

# draw a tree
sub_cols = ["wspd", "wvht", "inday_slot"]
sm = SMOTE(sampling_strategy='minority', random_state=123456)
x_train, y_train = sm.fit_sample(df[sub_cols], df_T_binary)
model.fit(x_train, y_train)
drawTree(x_train, y_train, model.estimators_[25], sub_cols)

# plot variable importances and confusion matrix
plt.figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1,2,1)
sns.barplot(x=[ x[1] for x in var_importance], y=[ x[0] for x in var_importance]).set_title("Relative Importance of Variables")
plt.subplot(1,2,2)
sns.heatmap(conf_matrix, annot=True).set_title("Confusion Matrix")


# naive bayes
model = GaussianNB()
conf_matrix_original = simulate(df_X_scaled, df_T_binary, model, 100, oversample=False)
conf_matrix_oversampled = simulate(df_X_scaled, df_T_binary, model, 100, oversample=True)

# plot variable importances and confusion matrix
plt.figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1,2,1)
sns.heatmap(conf_matrix_original, annot=True).set_title("GaussianNB() (normal)")
plt.subplot(1,2,2)
sns.heatmap(conf_matrix_oversampled, annot=True).set_title("GaussianNB() (over-sampled)")
