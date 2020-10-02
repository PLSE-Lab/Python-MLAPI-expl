import pandas as pd
import random as rand
import numpy as np

def create_bagged_predictor(train_csv_path, response_var, train_model_fn, p, verbose=False, seed=35901):
    """
    Returns predictor_fn(X), a bagged predictor function. Requires the following inputs:
        * train_csv_path: relative path to a csv file containing the training data.
                          First row should be feature names
        * response_var: the name of the response variable (as it appears in the training data csv file)
        * train_model_fn: a function for training the model. Should be created using create_train_model_fn
        * p : number of features bootstrapped samples should choose
    """
    #first row of csv should be feature names
    rand.seed(seed)
   
    # CONSTANTS
    n = train_data.shape[0]     # sample size 
    B = 50                      # number of bootstrap samples
    p = p                       # number of features per bootstrap sample

    # Preliminaries
    train_data = pd.read_csv(train_csv_path, header=0)
    features = [x for x in train_data if x != response_var]
    if(verbose): print("features: " + ", ".join(features))

    # Create Bootstrap Samples
    boot_data_sets = [rand.sample(train_data.index, n, with_replacement=True) for __ in range(B)]
    boot_feat_sets = [rand.sample(features, p, with_replacement=True) for __ in range(B)]

    # Train models on bootstrapped data/feature sets
    Models = []
    for i in range(B):
        data = boot_data_sets[i]
        feats = boot_featsets[i]
        Models.append(train_model_fn(data[feats], data[response_var])) 

    # Create and return predictor function which bags these models
    def predictor_fn(X):
        probabilities = []
        for i in range(B):
            try:
                probabilities.append(Models[i].predict_proba(X))
            except:
                print("ERROR: could not predict model")
                return -1

        return np.mean(probabilities)

    return predictor_fn



def create_train_model_fn(model_obj, params_dict):
    """
    Given a Scikit-learn model object and a dictionary of parameters + values,
    returns a train_model_fn which can be provided as input to create_bagged_predictor
    """

    def train_model_fn(X, y):
        model_obj.set_params(params_dict)
        model_obj.fit(X, y)
        return model_obj
    return train_model_fn


def evaluate_bagged_model(train_csv_path, test_csv_path, response_var, model, model_param, p, verbose=False, seed=35901):
    train_model_fn = create_train_model_fn(model, model_param)
    predictor_fn = create_bagged_predictor(train_csv_path, response_var, train_model_fn, p, verbose, seed)

    test_data = pd.read_csv(test_csv_path, header=0)
    features = [x for x in test_data.columns if x != response_var]
    classes = np.unique(pd.read_csv(train_csv_path, header=0)[response_var].values)
    n_t = test_data.shape[0]
    predictions = [np.argmax(predictor_fn(test_data.iloc[i])) for i in range(n_t)]
    num_correct = sum([1 if classes[predictions[i]] == test_data[response_var].iloc[i] else 0 for i in range(n_t)])
    return num_correct, n_t, predictions
    