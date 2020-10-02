# Author: Bruno Abreu Calfa <bacalfa@gmail.com>

#####################################
# Utility Script for Classification #
#####################################

import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time

class ClassificationAlgorithms:
    '''
    Emulates an 'enum' structure with the supported classification machine learning algorithms.
    '''
    SVC = "SVC"
    """
    C-Support Vector Classification
    """

    NuSVC = "NuSVC"
    """
    Nu-Support Vector Classification
    """

    LinearSVC = "LinearSVC"
    """
    Linear Support Vector Classification
    """

    SGD = "SGD"
    """
    Stochastic Gradient Descent
    """

    kNN = "kNN"
    """
    k-Nearest Neighbors Classification
    """

    RNN = "RNN"
    """
    Radius Nearest Neighbors Classification
    """

    GPC = "GPC"
    """
    Gaussian Process Classification (GPC) based on Laplace approximation
    """

    DTC = "DTC"
    """
    Decision Tree Classifier
    """

    GTB = "GTB"
    """
    Gradient Tree Boosting
    """

    AdaB = "AdaB"
    """
    AdaBoost Classifier
    """

    MLPC = "MLPC"
    """
    Multi-Layer Perceptron Classifier
    """

class MLClassification:
    """
    Main class for classification machine learning algorithms.
    """

    def __init__(self, X_train, Y_train, X_test, Y_test=None):
        """
        Constructor with train and test data.
        :param X_train: Train descriptor data
        :param Y_train: Train observed data
        :param X_test: Test descriptor data
        :param Y_test: Test observed data
        """
        self.X_train = np.matrix(X_train)
        self.Y_train = np.array(Y_train).ravel()
        self.X_test = np.matrix(X_test)
        self.Y_test = np.array(Y_test).ravel() if Y_test != None else None
        self.allmlalgs = {}
        self.allclfs = {}
        self.allpredictions = {}

    def setmlalg(self, mlalg):
        """
        Sets a the machine learning algorithm.
        :param mlalg: Dictionary of the machine learning algorithm. The key is the name of the algorithm, and the value
        is a dictionary (or a list of dictionaries) with keys as the name of the hyperparameters of the algorithm and
        values as their numeric values or other (e.g., distribution information for randomized search or list for grid
        search in hyperparameter optimization).
        """
        for k, v in mlalg.items():
            if k in ClassificationAlgorithms.__dict__.values():
                self.addmlalg(mlalg={k: v})
            else:
                print("Machine learning algorithm '{0}' not supported.".format(k))

    def setallmlalgs(self, allmlalgs):
        """
        Sets all the machine learning algorithms.
        :param allmlalgs: Dictionary of machine learning algorithms. Each key is the name of the algorithm (see list of
        available algorithms below). The value of each top-level key is a dictionary (or a list of dictionaries) with
        keys as the name of the hyperparameters of the algorithm and values as their numeric values or other (e.g.,
        distribution information for randomized search or list for grid search in hyperparameter optimization).
        """
        for mlalg, v in allmlalgs.items():
            self.addmlalg(mlalg={mlalg: v})

    def addmlalg(self, mlalg):
        """
        Adds an algorithm to the dictionary of algorithms. See method setallmlalgs for description.
        :param mlalg: Dictionary of machine learning algorithm.
        """
        self.allmlalgs.update(mlalg)

    def deleteallmlalgs(self):
        """
        Deletes all machine learning algorithms to be fitted.
        """
        for mlalg in self.allmlalgs:
            self.deletemlalg(mlalg=mlalg)
        self.allmlalgs = {}

    def deletemlalg(self, mlalg):
        """
        Deletes a machine learning algorithm to be fitted.
        :param mlalg: Machine learning algorithm
        """
        if self.allmlalgs and mlalg in self.allmlalgs:
            del(self.allmlalgs[mlalg])

    def fitmlalg(self, mlalg):
        """
        Fits a machine learning algorithm.
        :param mlalg: Machine learning algorithm
        """
        if mlalg not in self.allmlalgs:
            raise NameError("Machine learning algorithm '{0}' not supported.".format(mlalg))
        else:
            if self.allclfs and mlalg in self.allclfs:
                for eachrun in self.allclfs[mlalg]:
                    if eachrun["isfitted"]:
                        raise Warning("Algorithm '{0}' already fitted. Call methods 'refitmlalg' or 'refitallmlalgs' instead.")
            self.allclfs[mlalg] = []
            for eachrun in self.allmlalgs[mlalg]:
                # Check if hyperparameter optimization is requested
                if eachrun["dohyperopt"] == False:
                    # Build dict of parameters for classifier
                    params = {param: eachrun[param] for param in eachrun.keys() if param != "dohyperopt"}

                    tstart = time.perf_counter()
                    if mlalg == ClassificationAlgorithms.SVC:
                        # C-Support Vector Classification
                        clf = svm.SVC(**params).fit(self.X_train, self.Y_train)
                    elif mlalg == ClassificationAlgorithms.NuSVC:
                        # Nu-Support Vector Classification
                        clf = svm.NuSVC(**params).fit(self.X_train, self.Y_train)
                    elif mlalg == ClassificationAlgorithms.LinearSVC:
                        # Linear Support Vector Classification
                        clf = svm.LinearSVC(**params).fit(self.X_train, self.Y_train)
                    elif mlalg == ClassificationAlgorithms.SGD:
                        # Stochastic Gradient Descent
                        clf = SGDClassifier(**params).fit(self.X_train, self.Y_train)
                    elif mlalg == ClassificationAlgorithms.kNN:
                        # k-Nearest Neighbors Classification
                        clf = KNeighborsClassifier(**params).fit(self.X_train, self.Y_train)
                    elif mlalg == ClassificationAlgorithms.RNN:
                        # Radius Nearest Neighbors Classification
                        clf = RadiusNeighborsClassifier(**params).fit(self.X_train, self.Y_train)
                    elif mlalg == ClassificationAlgorithms.GPC:
                        # Gaussian Process Classification (GPC) based on Laplace approximation
                        clf = GaussianProcessClassifier(**params).fit(self.X_train, self.Y_train)
                    elif mlalg == ClassificationAlgorithms.DTC:
                        # Decision Tree Classifier
                        clf = DecisionTreeClassifier(**params).fit(self.X_train, self.Y_train)
                    elif mlalg == ClassificationAlgorithms.GTB:
                        # Gradient Tree Boosting
                        clf = GradientBoostingClassifier(**params).fit(self.X_train, self.Y_train)
                    elif mlalg == ClassificationAlgorithms.AdaB:
                        # AdaBoost Classifier
                        clf = AdaBoostClassifier(**params).fit(self.X_train, self.Y_train)
                    elif mlalg == ClassificationAlgorithms.MLPC:
                        # Multi-Layer Perceptron Classifier
                        clf = MLPClassifier(**params).fit(self.X_train, self.Y_train)
                    self.allclfs[mlalg].append(
                        {"dohyperopt": False, "clf": clf, "params": params, "time": time.perf_counter() - tstart, "isfitted": True})
                else:
                    # Build dict of fixed and varying parameters for classifier
                    params = {param: eachrun["fixed_params"][param] for param in eachrun["fixed_params"].keys()} if "fixed_params" in eachrun else {}
                    vparams = {param: eachrun[param] for param in eachrun.keys() if param != "dohyperopt" and param != "fixed_params"}

                    if mlalg == ClassificationAlgorithms.SVC:
                        # C-Support Vector Classification
                        clf = svm.SVC(**params)
                    elif mlalg == ClassificationAlgorithms.NuSVC:
                        # Nu-Support Vector Classification
                        clf = svm.NuSVC(**params)
                    elif mlalg == ClassificationAlgorithms.LinearSVC:
                        # Linear Support Vector Classification
                        clf = svm.LinearSVC(**params)
                    elif mlalg == ClassificationAlgorithms.SGD:
                        # Stochastic Gradient Descent
                        clf = SGDClassifier(**params)
                    elif mlalg == ClassificationAlgorithms.kNN:
                        # k-Nearest Neighbors Classification
                        clf = KNeighborsClassifier(**params)
                    elif mlalg == ClassificationAlgorithms.RNN:
                        # Radius Nearest Neighbors Classification
                        clf = RadiusNeighborsClassifier(**params)
                    elif mlalg == ClassificationAlgorithms.GPC:
                        # Gaussian Process Classification (GPC) based on Laplace approximation
                        clf = GaussianProcessClassifier(**params)
                    elif mlalg == ClassificationAlgorithms.DTC:
                        # Decision Tree Classifier
                        clf = DecisionTreeClassifier(**params)
                    elif mlalg == ClassificationAlgorithms.GTB:
                        # Gradient Tree Boosting
                        clf = GradientBoostingClassifier(**params)
                    elif mlalg == ClassificationAlgorithms.AdaB:
                        # AdaBoost Classifier
                        clf = AdaBoostClassifier(**params)
                    elif mlalg == ClassificationAlgorithms.MLPC:
                        # Multi-Layer Perceptron Classifier
                        clf = MLPClassifier(**params)
                    hyperoptparams = {param: eachrun["dohyperopt"][param] for param in eachrun["dohyperopt"].keys()
                                      if param != "type"}
                    if eachrun["dohyperopt"]["type"] == "RandomizedSearchCV":
                        random_search = RandomizedSearchCV(clf, **hyperoptparams, param_distributions=vparams)
                    else:
                        random_search = GridSearchCV(clf, **hyperoptparams, param_grid=vparams)
                    tstart = time.perf_counter()
                    random_search.fit(self.X_train, self.Y_train)
                    self.allclfs[mlalg].append(
                        {"dohyperopt": eachrun["dohyperopt"]["type"], "clf": random_search.best_estimator_,
                         "params": random_search.best_params_, "time": time.perf_counter() - tstart, "isfitted": True})

    def fitallmlalgs(self):
        """
        Fits all machine learning algorithms that have been set.
        """
        if not self.allmlalgs:
            print("No algorithms to fit.")
        for mlalg in self.allmlalgs:
            try:
                self.fitmlalg(mlalg)
            except NameError as ne:
                print(ne)
            except Warning as wa:
                print(wa)
            except Exception as e:
                print(e)

    def refitmlalg(self, mlalg):
        """
        Refits a machine learning algorithm.
        :param mlalg: Machine learning algorithm
        :return:
        """
        if mlalg not in self.allmlalgs:
            raise Warning("Machine learning algorithm '{0}' not fitted. Will fit it now".format(mlalg))
        else:
            for eachrun in self.allclfs[mlalg]:
                eachrun["isfitted"] = False
        self.fitmlalg(mlalg=mlalg)

    def refitallmlalgs(self):
        """
        Refits all machine learning algorithms that have been set.
        """
        for mlalg in self.allmlalgs:
            self.refitmlalg(mlalg)

    def predictmlalg(self, mlalg):
        """
        Predicts a machine learning algorithm that has been fit.
        :param mlalg: Machine learning algorithm
        """
        if mlalg not in self.allclfs:
            print("Algorithm '{0}' has not been fitted. Skipping prediction.".format(mlalg))
        else:
            self.allpredictions[mlalg] = []
            for eachrun in self.allclfs[mlalg]:
                self.allpredictions[mlalg].append(
                    {"dohyperopt": eachrun["dohyperopt"], "Y_pred": eachrun["clf"].predict(self.X_test)})

    def predictallmlalgs(self):
        """
        Predicts all machine learning algorithms that have been fit.
        """
        for mlalg in self.allmlalgs:
            self.predictmlalg(mlalg)

    def displayallfits(self):
        """
        Displays information of all fits.
        """
        for mlalg in self.allclfs:
            self.displayfit(mlalg)

    def displayfit(self, mlalg):
        """
        Displays information of fit.
        :param mlalg: Machine learning algorithm
        """
        print()
        print(mlalg)
        print("=======")
        for eachrun in self.allclfs[mlalg]:
            for k, v in eachrun.items():
                print(k + ": " + str(v))
            print()

    def displayallpredictions(self):
        """
        Displays information of all predictions.
        """
        if not self.allpredictions:
            print("No algorithms have been fitted. Nothing to predict.")
        else:
            for mlalg in self.allpredictions:
                self.displaypredictions(mlalg)

    def displaypredictions(self, mlalg):
        """
        Displays information of predictions of a machine learning algorithm.
        :param mlalg: Machine learning algorithm
        """
        if not self.allpredictions:
            print("No algorithms have been fitted. Nothing to predict.")
        elif not self.allpredictions[mlalg]:
            print("Algorithm '{0}' has not been fitted. Nothing to predict.".format(mlalg))
        else:
            print()
            print(mlalg)
            print("=======")
            for eachrun in self.allpredictions[mlalg]:
                for k, v in eachrun.items():
                    print(k + ": " + str(v))
                print()

############################################################################

###############
# Main Script #
###############

### First, read data and create plots

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Train and test data files
train_fname = "../input/train.csv"
test_fname = "../input/test.csv"

# Read train data
train_df = pd.read_csv(train_fname)
print()
print("Train Data")
print()
print(train_df.info())

# Read test data
test_df = pd.read_csv(test_fname)
print()
print("Test Data")
print()
print(test_df.info())

# Sexes
male_df = train_df[train_df["Sex"] == "male"]
female_df = train_df[train_df["Sex"] == "female"]

# Totals
male_tot = len(male_df)
female_tot = len(female_df)

# Unique Pclass entries
pclass_entries = sorted(train_df["Pclass"].unique())

fig = plt.figure(facecolor="white")
plt.subplots_adjust(
    left=0.125,  # the left side of the subplots of the figure
    right=0.9,  # the right side of the subplots of the figure
    bottom=0.1,  # the bottom of the subplots of the figure
    top=0.9,  # the top of the subplots of the figure
    wspace=0.5,  # the amount of width reserved for blank space between subplots
    hspace=0.2  # the amount of height reserved for white space between subplots)
)
bar_width = 0.5

# Sex x Survived
ax_sex = fig.add_subplot(2, 1, 1)
bar_l = np.arange(1, 3)
tick_pos = [i + (bar_width / 2) for i in bar_l]
male_surv = len(male_df[male_df["Survived"] == 1])
male_data = [male_surv, male_tot - male_surv]
female_surv = len(female_df[female_df["Survived"] == 1])
ax_sex_male = ax_sex.bar(bar_l, [male_surv, male_tot - male_surv], width=bar_width, label="Male", color="green")
ax_sex_female = ax_sex.bar(bar_l, [female_surv, female_tot - female_surv], bottom=male_data, width=bar_width,
                           label="Female",
                           color="blue")
ax_sex.set_ylabel("Count", fontsize=18)
ax_sex.set_xlabel("Sex (Totals)", fontsize=18)
ax_sex.legend(loc="best")
plt.xticks(tick_pos, ["Survived (" + str(male_surv + female_surv) + ")",
                      "Not Survived (" + str(male_tot - male_surv + female_tot - female_surv) + ")"], fontsize=16)
plt.yticks(fontsize=16)

for r1, r2 in zip(ax_sex_male, ax_sex_female):
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="center", color="white", fontsize=16,
             fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="center", color="white",
             fontsize=16, fontweight="bold")

# Pclass x Survived
ax_pclass = fig.add_subplot(2, 1, 2)
bar_width = 0.5
bar_l = np.arange(1, len(pclass_entries) + 1)
tick_pos = [i + (bar_width / 2) for i in bar_l]
male_pclass = {}
female_pclass = {}
all_pclass = {}
pclass_legend = []
for pclass in pclass_entries:
    male_pclass[pclass] = len(male_df[(male_df["Pclass"] == pclass) & (male_df["Survived"] == 1)])
    female_pclass[pclass] = len(female_df[(female_df["Pclass"] == pclass) & (female_df["Survived"] == 1)])
    all_pclass[pclass] = len(train_df[(train_df["Pclass"] == pclass)])
    pclass_legend.append(str(pclass) + " (" + str(len(train_df[train_df["Pclass"] == pclass])) + ")")
ax_pclass_male = ax_pclass.bar(bar_l, male_pclass.values(), width=bar_width, label="Male", color="green")
ax_pclass_female = ax_pclass.bar(bar_l, female_pclass.values(), bottom=male_pclass.values(), width=bar_width,
                                 label="Female",
                                 color="blue")
ax_pclass.set_ylabel("Count", fontsize=18)
ax_pclass.set_xlabel("Pclass Survived (Totals)", fontsize=18)
ax_pclass.legend(loc="best")
plt.xticks(tick_pos, pclass_legend, fontsize=16)
plt.yticks(fontsize=16)

for r1, r2 in zip(ax_pclass_male, ax_pclass_female):
    h1 = r1.get_height()
    h2 = r2.get_height()
    plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="center", color="white", fontsize=16,
             fontweight="bold")
    plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="center", color="white",
             fontsize=16, fontweight="bold")

plt.show()


### Second, train and predict from all models

descriptors = ["Sex", "Age", "Pclass", "SibSp", "Parch"]
response = ["Survived"]
train_nona = train_df[descriptors + response].dropna()
X_train = train_nona[descriptors]
Y_train = train_nona[response]
X_test = test_df[descriptors].dropna()
print()
print("After NaN Removal:")
print("Train Data")
print()
print(pd.concat([X_train, Y_train]).info())

print()
print("Test Data")
print()
print(pd.concat([X_test]).info())

# Store PassengerId entries for dropped rows in the test data set
pid_all = set(test_df["PassengerId"])
pid_nona = set(test_df[["PassengerId"] + descriptors].dropna()["PassengerId"])
pid_na = pid_all - pid_nona

## Data encoding
male_code = 1
female_code = 0
X_train = X_train.replace(["male", "female"], [male_code, female_code])
X_test = X_test.replace(["male", "female"], [male_code, female_code])

## Serialization and deserialization
import pickle

deserialization = False

## Create classifier object and set the algorithms
if deserialization:
    with open("clf.pickle", "rb") as f:
        clf = pickle.load(f)
else:
    clf = MLClassification(X_train=X_train, Y_train=Y_train, X_test=X_test)

    ## Set up ML algorithms
    from scipy.stats import uniform as sp_uniform
    from scipy.stats import randint as sp_randint
    from sklearn.gaussian_process.kernels import *

    # Parameters for MLPClassifier
    sumInOutSize = len(X_train) + 1
    minNeutron = int(sumInOutSize / 3)
    maxNeutron = int(sumInOutSize * 2 / 3) + 1

    # The following dictionary controls all ML algorithms to be trained and fitted
    allmlalgs = {
        # ClassificationAlgorithms.SVC: [
        #     {
        #         "dohyperopt": False,
        #         "C": 0.5,
        #         "kernel": "poly"
        #     }]
        # ClassificationAlgorithms.SVC: [
        #     {
        #         "dohyperopt": {
        #             "type": "RandomizedSearchCV",
        #             "n_iter": 500,
        #             "n_jobs": -1
        #         },
        #         "C": sp_uniform(),
        #         "kernel": ["rbf", "poly", "linear"]
        #     }
        # ],
        # ClassificationAlgorithms.GPC: [
        #     {
        #         "dohyperopt": {
        #             "type": "GridSearchCV",
        #             "n_jobs": -1
        #         },
        #         "fixed_params": {
        #             "random_state": 123,
        #             "warm_start": True
        #         },
        #         "kernel": [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
        #                    1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
        #                    1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
        #                                         length_scale_bounds=(0.1, 10.0),
        #                                         periodicity_bounds=(1.0, 10.0)),
        #                    ConstantKernel(0.1, (0.01, 10.0))
        #                    * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.0, 10.0)) ** 2),
        #                    1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
        #                                 nu=1.5)]
        #     }
        # ],
        # ClassificationAlgorithms.DTC: [
        #     {
        #         "dohyperopt": {
        #             "type": "RandomizedSearchCV",
        #             "n_iter": 500,
        #             "n_jobs": -1
        #         },
        #         "fixed_params": {
        #             "random_state": 123
        #         },
        #         "criterion": ["gini", "entropy"],
        #         "max_depth": sp_randint(low=1, high=20)
        #     }
        # ],
        # ClassificationAlgorithms.AdaB: [
        #     {
        #         "dohyperopt": {
        #             "type": "RandomizedSearchCV",
        #             "n_iter": 500,
        #             "n_jobs": -1
        #         },
        #         "fixed_params": {
        #             "random_state": 123
        #         },
        #         "n_estimators": sp_randint(low=50, high=300),
        #         "learning_rate": sp_uniform(),
        #     }
        # ],
        ClassificationAlgorithms.GTB: [
            {
                "dohyperopt": {
                    "type": "RandomizedSearchCV",
                    "n_iter": 500,
                    "n_jobs": -1
                },
                "fixed_params": {
                    "random_state": 123
                },
                "n_estimators": sp_randint(low=50, high=300),
                "learning_rate": sp_uniform(),
                "max_depth": sp_randint(low=1, high=20)
            }
        ],
        # ClassificationAlgorithms.MLPC: [
        #     {
        #         "dohyperopt": {
        #             "type": "RandomizedSearchCV",
        #             "n_iter": 500,
        #             "n_jobs": -1
        #         },
        #         "fixed_params": {
        #             "random_state": 123,
        #             "max_iter": 500,
        #             "warm_start": True,
        #         },
        #         "hidden_layer_sizes": [(x,) for x in range(minNeutron,maxNeutron)],
        #         "solver": ["lbfgs", "sgd", "adam"],
        #         "learning_rate": ["constant", "invscaling", "adaptive"],
        #         "activation": ["identity", "logistic", "tanh", "relu"]
        #     }
        # ]
    }

    ## Fit all algorithms and display info
    clf.setallmlalgs(allmlalgs)
    clf.fitallmlalgs()
    clf.displayallfits()

    clf.predictallmlalgs()
    clf.displayallpredictions()

    # Serialize classifier objects
    with open("clf.pickle", "wb") as f:
        pickle.dump(clf, f)

## Write predictions to .csv file
results_df = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"].values, "Survived": np.zeros(shape=(len(test_df),), dtype=int)})

for algname, alg in clf.allpredictions.items():
    for eachrun in alg:
        surv_pred = eachrun["Y_pred"]
        results_df_tmp = results_df
        results_df_tmp.loc[results_df_tmp["PassengerId"].isin(
            pid_nona), "Survived"] = surv_pred  # Write Y_pred for non-NaN PassengerId entries
        results_df_tmp.to_csv(path_or_buf=algname + "_hyperopt_" + str(eachrun["dohyperopt"]) + ".csv", index=False)

print("Done!")
