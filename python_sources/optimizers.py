import catboost
from optimizer_utils import *


# loop dataframe columns and return indices of "category" type features for catboost cat_features
def categorical_indices(df):
    cat_indices = []
    index = 0
    dtypes = df.dtypes
    for col in df.columns:
        dtype = dtypes[col]
        if dtype.name == "category":
            cat_indices.append(index)
        index += 1

    return cat_indices


class CatboostOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # max number of trials hyperopt runs
    n_trials = 200
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    cb_verbosity = 0
    verbosity = 0
    use_gpu = False
    n_classes = 2
    classifier = catboost.CatBoostClassifier
    use_calibration = False
    cat_features = None
    scale_pos_weight = None

    all_accuracies = []
    all_losses = []
    all_params = []
    all_times = []

    def objective_sklearn(self, params):
        int_types = ["depth", "iterations", "early_stopping_rounds"]
        params = convert_int_params(int_types, params)
        if params['bootstrap_type'].lower() != "bayesian":
            # catboost gives error if bootstrap option defined with bootstrap disabled
            del params['bagging_temperature']

        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
        space = {
            # 'shrinkage': hp.loguniform('shrinkage', -7, 0),
            'depth': hp.quniform('depth', 2, 10, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'border_count': hp.qloguniform('border_count', np.log(32), np.log(255), 1),
            # 'ctr_border_count': hp.qloguniform('ctr_border_count', np.log(32), np.log(255), 1),
            'l2_leaf_reg': hp.quniform('l2_leaf_reg', 0, 5, 1),
            'leaf_estimation_method': hp.choice('leaf_estimation_method', ['Newton', 'Gradient']),
            'bagging_temperature': hp.loguniform('bagging_temperature', np.log(1), np.log(3)),
            'use_best_model': True,
            'early_stopping_rounds': 10,
            'iterations': 1000,
            'feature_border_type': hp.choice('feature_border_type',
                                             ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum']),
            # 'gradient_iterations': hp.quniform('gradient_iterations', 1, 100, 1),
        }
        if self.use_gpu:
            space['task_type'] = "GPU"
            space['bootstrap_type'] = hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'Poisson', 'No'])
        else:
            space['task_type'] = "CPU"
            space['rsm'] = hp.uniform('rsm', 0.5, 1)
            space['bootstrap_type'] = hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS', 'No'])

        if self.n_classes > 2:
            space['objective'] = "multiclass"
            space["eval_metric"] = "multi_logloss"
        else:
            space['objective'] = "Logloss"
        if self.scale_pos_weight is not None:
            space["scale_pos_weight"] = self.scale_pos_weight
        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = {'verbose': self.cb_verbosity,
                           'use_eval_set': True}
        if self.cat_features is not None:
            self.fit_params["cat_features"] = self.cat_features

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)


import lightgbm as lgbm
from optimizer_utils import *


class LGBMOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    # max number of trials hyperopt runs
    n_trials = 200
    # verbosity in LGBM is how often progress is printed. with 100=print progress every 100 rounds. 0 is quiet?
    lgbm_verbosity = 0
    verbosity = 0
    n_classes = 2
    classifier = lgbm.LGBMClassifier
    use_calibration = False
    scale_pos_weight = None

    all_accuracies = []
    all_losses = []
    all_params = []
    all_times = []

    def create_fit_params(self, params):
        using_dart = params['boosting_type'] == "dart"
        if params["objective"] == "binary":
            # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
            fit_params = {"eval_metric": ["binary_logloss", "auc"]}
        else:
            fit_params = {"eval_metric": "multi_logloss"}
        if using_dart:
            n_estimators = 2000
        else:
            n_estimators = 15000
            fit_params["early_stopping_rounds"] = 100
        params["n_estimators"] = n_estimators
        fit_params['use_eval_set'] = True
        fit_params['verbose'] = self.lgbm_verbosity
        return fit_params

    # this is the objective function the hyperopt aims to minimize
    # i call it objective_sklearn because the lgbm functions called use sklearn API
    def objective_sklearn(self, params):
        int_types = ["num_leaves", "min_child_samples", "subsample_for_bin", "min_data_in_leaf"]
        params = convert_int_params(int_types, params)

        # Extract the boosting type
        params['boosting_type'] = params['boosting_type']['boosting_type']
        #    print("running with params:"+str(params))

        self.fit_params = self.create_fit_params(params)

        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
        # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
        # https://indico.cern.ch/event/617754/contributions/2590694/attachments/1459648/2254154/catboost_for_CMS.pdf
        space = {
            # this is just piling on most of the possible parameter values for LGBM
            # some of them apparently don't make sense together, but works for now.. :)
            'class_weight': hp.choice('class_weight', [None, 'balanced']),
            'boosting_type': hp.choice('boosting_type',
                                       [{'boosting_type': 'gbdt',
                                         #                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)
                                         },
                                        # NOTE: DART IS COMMENTED DUE TO SLOW SPEED. HAVE TO MAKE IT AN OPTION..
                                        #                                    {'boosting_type': 'dart',
                                        #                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)
                                        #                                     },
                                        {'boosting_type': 'goss'}]),
            'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),  # alias "subsample"
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
            'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
            'verbose': -1,
            # the LGBM parameters docs list various aliases, and the LGBM implementation seems to complain about
            # the following not being used due to other params, so trying to silence the complaints by setting to None
            'subsample': None,  # overridden by bagging_fraction
            'reg_alpha': None,  # overridden by lambda_l1
            'reg_lambda': None,  # overridden by lambda_l2
            'min_sum_hessian_in_leaf': None,  # overrides min_child_weight
            'min_child_samples': None,  # overridden by min_data_in_leaf
            'colsample_bytree': None,  # overridden by feature_fraction
            #        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),  # also aliases to min_sum_hessian
            #        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            #        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            #        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
        }
        if self.n_classes > 2:
            space['objective'] = "multiclass"
        else:
            space['objective'] = "binary"
        if self.scale_pos_weight is not None:
            space["scale_pos_weight"] = self.scale_pos_weight
        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = None

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)


from sklearn.linear_model import LogisticRegression
from optimizer_utils import *


class LogRegOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    # max number of trials hyperopt runs
    n_trials = 200
    n_classes = 2
    lr_verbosity = 0
    verbosity = 0
    classifier = LogisticRegression
    use_calibration = False

    all_accuracies = []
    all_losses = []
    all_params = []
    all_times = []

    def objective_sklearn(self, params):
        # print(params)
        params.update(params["solver_params"])  # pop nested dict to top level
        del params["solver_params"]  # delete the original nested dict after pop (could pop() above too..)
        if params["penalty"] == "none":
            del params["C"]
            del params["l1_ratio"]
        elif params["penalty"] != "elasticnet":
            del params["l1_ratio"]
        if params["solver"] == "liblinear":
            params["n_jobs"] = 1
        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        space = {
            'solver_params': hp.choice('solver_params', [
                #this was very slow in some dataset, so commented out for now
                #                {'solver': 'newton-cg',
                #                 'penalty': hp.choice('penalty-ncg', ["l2", 'none'])}, #also multiclass loss supported
                {'solver': 'lbfgs',
                 'penalty': hp.choice('penalty-lbfgs', ["l2", 'none'])},
                {'solver': 'liblinear',
                 'penalty': hp.choice('penalty-liblin', ["l1", "l2"])},
                {'solver': 'sag',
                 'penalty': hp.choice('penalty-sag', ["l2", 'none'])},
                {'solver': 'saga',
                 'penalty': hp.choice('penalty-saga', ["elasticnet", "l1", "l2", 'none'])},
            ]),
            'C': hp.uniform('C', 1e-5, 10),
            'tol': hp.uniform('tol', 1e-5, 10),
            'fit_intercept': hp.choice("fit_intercept", [True, False]),
            'class_weight': hp.choice("class_weight", ["balanced", None]),
            'l1_ratio': hp.uniform('l1_ratio', 0.00001, 0.99999),  # only elasticnet penalty?
            'n_jobs': -1,
            'verbose': self.lr_verbosity
        }
        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = {'use_eval_set': False}

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)


from sklearn.ensemble import RandomForestClassifier
from optimizer_utils import *


class RFOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    n_classes = 2
    # max number of trials hyperopt runs
    n_trials = 200
    # verbosity 0 in RF is quiet, 1 = print epoch, 2 = print within epoch
    # https://stackoverflow.com/questions/31952991/what-does-the-verbosity-parameter-of-a-random-forest-mean-sklearn
    rf_verbosity = 0
    verbosity = 0

    classifier = RandomForestClassifier
    use_calibration = False

    all_accuracies = []
    all_losses = []
    all_params = []
    all_times = []

    def objective_sklearn(self, params):
        int_types = ["n_estimators", "min_samples_leaf", "min_samples_split", "max_features"]
        params = convert_int_params(int_types, params)
        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        space = {
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            # 'scale': hp.choice('scale', [0, 1]),
            # 'normalize': hp.choice('normalize', [0, 1]),
            'bootstrap': hp.choice('bootstrap', [True, False]),
            # nested choice: https://medium.com/vooban-ai/hyperopt-tutorial-for-optimizing-neural-networks-hyperparameters-e3102814b919
            'max_depth': hp.choice('max_depth', [None, hp.quniform('max_depth_num', 10, 100, 10)]),
            'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None, hp.quniform('max_features_num', 1, 5, 1)]),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 4, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 4),
            'class_weight': hp.choice('class_weight', ["balanced", None]),
            'n_estimators': hp.quniform('n_estimators', 200, 2000, 200),
            'n_jobs': -1,
            'verbose': self.rf_verbosity
        }
        # save and reload trials for hyperopt state: https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/blob/master/hyperopt_optimize.py

        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = {'use_eval_set': False}

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)


from sklearn.linear_model import SGDClassifier
from optimizer_utils import *


class SGDOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # max number of rows to use for X and y. to reduce time and compare options faster
    max_n = None
    # max number of trials hyperopt runs
    n_trials = 200
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    sgd_verbosity = 0
    verbosity = 0

    n_classes = 2
    classifier = SGDClassifier
    use_calibration = True

    all_accuracies = []
    all_losses = []
    all_params = []
    all_times = []

    def objective_sklearn(self, params):
        int_types = ["n_iter_no_change", "max_iter"]
        params = convert_int_params(int_types, params)
        if params["learning_rate"] == "optimal":
            del params["alpha"]  # alpha cannot be zero in optimal learning rate as it is used as a divider

        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
        space = {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
            # hinge = linear SVM, log = logistic regression,
            'loss': hp.choice('loss', ['hinge', 'modified_huber', 'squared_hinge', 'perceptron']),
            # https://github.com/scikit-learn/scikit-learn/issues/7278
            # only "log" gives probabilities so have use that. else have to rewrite to allow passing instance,
            # TODO see link above for CalibratedClassifierCV
            # 'loss': hp.choice('loss', ['log']),
            'penalty': hp.choice('penalty', ['none', 'l1', 'l2', 'elasticnet']),
            'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
            'l1_ratio': hp.choice('l1_ratio', [1, hp.quniform('l1_ratio_fraction', 0.05, 0.95, 0.05)]),
            'max_iter': 1000,
            'n_iter_no_change': 5,
            'early_stopping': True,
            'n_jobs': -1,
            'tol': 0.001,
            'shuffle': True,
            # 'epsilon': ?,
            'learning_rate': hp.choice('learning_rate', ['optimal', 'adaptive']),
            'eta0': 0.001,
            'validation_fraction': 0.1,
            'verbose': self.sgd_verbosity,
            'class_weight': hp.choice('class_weight', ['balanced', None]),
        }
        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = {'use_eval_set': False}

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)


import xgboost as xgb
from optimizer_utils import *


class XGBOptimizer:
    # how many CV folds to do on the data
    n_folds = 5
    # max number of trials hyperopt runs
    n_trials = 200
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    xgb_verbosity = 0
    verbosity = 0
    # if true, print summary accuracy/loss after each round
    print_summary = False
    n_classes = 2
    use_gpu = False
    classifier = xgb.XGBClassifier
    use_calibration = False
    scale_pos_weight = None

    all_accuracies = []
    all_losses = []
    all_params = []
    all_times = []

    def cleanup(self, clf):
        # print("cleaning up..")
        # TODO: run in different process.. : https://stackoverflow.com/questions/56298728/how-do-i-free-all-memory-on-gpu-in-xgboost
        clf._Booster.__del__()
        import gc
        gc.collect()

    def objective_sklearn(self, params):
        int_params = ['max_depth']
        params = convert_int_params(int_params, params)
        float_params = ['gamma', 'colsample_bytree']
        params = convert_float_params(float_params, params)

        return hyperopt_objective_run(self, params)

    def create_hyperspace(self):
        space = {
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
            # removed gblinear since it does not support early stopping and it was getting tricky
            'booster': hp.choice('booster', ['gbtree']),  # , 'dart']),
            # 'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            # nthread defaults to maximum so not setting it
            'subsample': hp.uniform('subsample', 0.75, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1.0),
            # 'gamma': hp.uniform('gamma', 0.0, 0.5),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
            'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
            'lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
            'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)]),
            'verbose': self.xgb_verbosity,
            'n_jobs': 4,
            # 'tree_method': 'gpu_hist',
            # 'n_estimators': 1000   #n_estimators = n_trees -> get error this only valid for gbtree
            # https://github.com/dmlc/xgboost/issues/3789
        }
        if self.use_gpu:
            # tree_method: 'gpu_hist' causes xgboost to use GPU.
            # https://xgboost.readthedocs.io/en/latest/gpu/
            space["tree_method"] = "gpu_hist"
        if self.scale_pos_weight is not None:
            space["scale_pos_weight"] = self.scale_pos_weight
        return space

    # run a search for binary classification
    def classify_binary(self, X_cols, df_train, df_test, y_param, train_pct = None, stratify_train = None):
        self.n_classes = 2
        self.fit_params = {'use_eval_set': False}

        return hyperopt_search_classify(self, X_cols, df_train, df_test, y_param, train_pct, stratify_train)


class CatboostRunner:
    # how many CV folds to do on the data
    n_folds = 5
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    cb_verbosity = 0
    verbosity = 0
    n_classes = 2
    use_calibration = False
    cat_features = None


class XGBRunner:
    # how many CV folds to do on the data
    n_folds = 5
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    xgb_verbosity = 0
    verbosity = 0
    n_classes = 2
    use_calibration = False


class LGBMRunner:
    # how many CV folds to do on the data
    n_folds = 5
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    # verbosity in LGBM is how often progress is printed. with 100=print progress every 100 rounds. 0 is quiet?
    lgbm_verbosity = 0
    verbosity = 0
    n_classes = 2
    use_calibration = False


class LogregRunner:
    # how many CV folds to do on the data
    n_folds = 5
    # rows in training data to use to train, subsetting allows training on smaller set if slow
    train_indices = None
    lr_verbosity = 0
    verbosity = 0
    n_classes = 2
    use_calibration = False    












