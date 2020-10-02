import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm

class BayesianGLM(object):
    def __init__(self, X, y, prior_fn, *prior_args):
        self.X = X
        
        with pm.Model() as model:
            intercept = pm.Normal("intercept")
            coefs = [prior_fn(v, *prior_args) for v in X.columns]
        
            wx = sum([c * X.values[:, i] for c, i in zip(coefs, range(X.shape[1]))])
        
            likelihood = pm.Normal("y", mu=intercept + wx, observed=y)
        
        self.model = model
        
        self.sampled = False
        
    def fit(self, iterations=2000, **kwargs):
        assert not self.sampled
        
        with self.model:
            self.trace = pm.sample(iterations, **kwargs)
        
        self.coef_ = np.array([self.trace[name].mean() for name in self.X.columns])
        self.intercept_ = self.trace["intercept"].mean()
        self.sampled = True
        
    def predict(self, X):
        assert self.sampled
        
        return self.intercept_ + np.dot(X, self.coef_.T)
    
def plot_errors_and_coef_magnitudes(data, title, figsize=(12, 8), hyperparam_name="alpha", reverse_x=False):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    fig.suptitle(title)

    cols = [
        {"x": hyperparam_name, "y": "mean_error"}, 
        {"x": "log_" + hyperparam_name, "y": "mean_error"},
        {"x": hyperparam_name, "y": "mean_largest_coef_value"},
        {"x": "log_" + hyperparam_name, "y": "mean_largest_coef_value"}
    ]

    for ax, col in zip(axes.flatten(), cols):
        ax.plot(col["x"], col["y"], data=data)
        ax.set(xlabel=col["x"], ylabel=col["y"])
        
        if reverse_x:
            ax.set(xlim=(data[col["x"]].max(), data[col["x"]].min()))
            
def cross_validate_hyperparam_choices(hyperparams, X_train, y, cv_splitter, ModelObj, 
                                      is_bayesian=False, bayesian_prior_fn=None, progressbar=False):
    if is_bayesian:
        param_name = "sigma"
    else:
        param_name = "alpha"
    
    results = []
    for param in hyperparams:
        errors = []
        largest_coef_values = []

        for train_index, val_index in cv_splitter.split(X_train, y):
            X_t, X_v = X_train.loc[train_index], X_train.loc[val_index]
            y_t, y_v = y[train_index], y[val_index]
            
            if is_bayesian:
                model = ModelObj(X_t, y_t, bayesian_prior_fn, 0, param)
                model.fit(**{"progressbar": progressbar})            
            else:
                model = ModelObj(alpha=param)
                model.fit(X_t, y_t)

            largest_coef_values.append(model.coef_.max())

            y_pred = model.predict(X_v)
            errors.append(np.sqrt(np.mean((y_pred - y_v)**2)))

        results.append(
            {param_name: param, 
             "mean_error": np.mean(errors), 
             "mean_largest_coef_value": np.mean(largest_coef_values)}
        )
    
    results = pd.DataFrame(results)
    results["log_" + param_name] = np.log10(results[param_name])
    
    return results