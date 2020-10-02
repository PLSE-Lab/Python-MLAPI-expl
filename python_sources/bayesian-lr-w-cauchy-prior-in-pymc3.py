#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This competition asks us to build a model of k = 300 features with only N = 250 observations. This resembles the situation often faced by scientists in the real world: we have limited observations and many possible explanations. 
# 
# In a normal scientific context, we could use domain knowledge to help navigate this situation: we would likely know something about these variables' meanings, the accuracy with which they are measured, their temporal orderings, or best of all, their causal relationships.  But, here we have a purely _data science_ context. We have no domain knowledge about these features at all, so we have only our data-science tools to help sort out this mess.
# 
# First, I'll get my environment ready, load the data, and define some custom helper functions.

# In[ ]:


get_ipython().system('pip install scikit-misc')

import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pymc3 as pm
import random
import matplotlib
import plotnine
from plotnine import ggplot, aes, geom_point, geom_jitter, geom_smooth, geom_histogram, geom_line, geom_errorbar, stat_smooth, geom_ribbon
from plotnine.facets import facet_wrap
import seaborn as sns
import theano.tensor as tt
from scipy.special import expit
from scipy.special import logit
from scipy.stats import cauchy
from sklearn.metrics import roc_auc_score
from skmisc.loess import loess
import warnings
warnings.filterwarnings("ignore") # plotnine is causing all kinds of matplotlib warnings


## Define custom functions

invlogit = lambda x: 1/(1 + tt.exp(-x))

def trace_predict(trace, X):
    y_hat = np.apply_along_axis(np.mean, 1, expit(trace['alpha'] + np.dot(X, np.transpose(trace['beta']) )) )
    return(y_hat)


# Define prediction helper function
# for more help see: https://discourse.pymc.io/t/how-to-predict-new-values-on-hold-out-data/2568
def posterior_predict(trace, model, n=1000, progressbar=True):
    with model:
        ppc = pm.sample_posterior_predictive(trace,n, progressbar=progressbar)
    
    return(np.mean(np.array(ppc['y_obs']), axis=0))


## I much prefer the syntax of tidyr gather() and spread() to pandas' pivot() and melt()
def gather( df, key, value, cols ):
    id_vars = [ col for col in df.columns if col not in cols ]
    id_values = cols
    var_name = key
    value_name = value
    return pd.melt( df, id_vars, id_values, var_name, value_name )


def spread( df, index, columns, values ):
    return df.pivot(index, columns, values).reset_index(level=index).rename_axis(None,axis=1)


## define custom plotting functions

def fit_loess(df, transform_logit=False):
    l = loess(df["value"],df["target"])
    l.fit()
    pred_obj = l.predict(df["value"],stderror=True)
    conf = pred_obj.confidence()
    
    yhat = pred_obj.values
    ll = conf.lower
    ul = conf.upper
    
    df["loess"] = np.clip(yhat,.001,.999)
    df["ll"] = np.clip(ll,.001,.999)
    df["ul"] = np.clip(ul,.001,.999)
    
    if transform_logit:
        df["logit_loess"] = logit(df["loess"])
        df["logit_ll"] = logit(df["ll"])
        df["logit_ul"] = logit(df["ul"])
    
    return(df)


def plot_loess(df, features):
    
    z = gather(df[["id","target"]+features], "feature", "value", features)
    z = z.groupby("feature").apply(fit_loess, transform_logit=True)
    z["feature"] = pd.to_numeric(z["feature"])

    plot = (
        ggplot(z, aes("value","logit_loess",ymin="logit_ll",ymax="logit_ul")) + 
        geom_point(alpha=.5) + 
        geom_line(alpha=.5) + 
        geom_ribbon(alpha=.33) + 
        facet_wrap("~feature")
    )
    return(plot)


## Load data

df = pd.read_csv("../input/train.csv")
y = np.asarray(df.target)
X = np.array(df.ix[:, 2:302])
df2 = pd.read_csv('../input/test.csv')
df2.head()
X2 = np.array(df2.ix[:, 1:301])

print("training shape: ", X.shape)
print("test shape: ", X2.shape)


# # Exploratory data analysis
# 
# Before we begin modeling, let's do some exploratory data analysis. First, I'll pick 12 random features and make a pairs plot. 

# In[ ]:


random.seed(432532) # comment out for new random samples

rand_feats = [str(x) for x in random.sample(range(0,300), 12)]
dfp = gather(df[["id","target"]+rand_feats], "feature", "value", rand_feats)
sns.set(style="ticks")

sns.pairplot(spread(dfp, "id", "feature", "value").drop("id",1))


# Looking at the histograms, it appears all features are more-or-less normally-distributed and based on the scatterplots they appear largely uncorrelated with one another. That means we won't have much luck using any dimensionality-reduction techniques to combine or reduce redundant features--we're stuck dealing with all 300. (Indeed, another kaggler who tried using an autoencoder to reduce the dimensionality of the problem and found no joy).
# 
# ## Plotting features and target
# 
# Next, I'll poke around a bit and examine the possible relationships between features and the target using the same 12 randomly-selected features. For each, I fit a loess curve and plot the relationship transformed in the logit space (an idea I took from [this blog post](https://thestatsgeek.com/2014/09/13/checking-functional-form-in-logistic-regression-using-loess/)). This will let me inspect whether any of the features might exhibit a non-linear relationship with the target.

# In[ ]:


plotnine.options.figure_size = (12,9)

random.seed(432532) # comment out for new random samples

rand_feats = [str(x) for x in random.sample(range(0,300), 12)]
plot_loess(df,rand_feats)


# I've looked through a number of random feature plots like this and tried to get a sense of the relationships. Most features seem more-or-less unrelated to the target. Among those with potentially meaningful relationsips, some appeared linear and some appeared potentially non-linear. The original challenge was solvable by a hyperplane, so a non-linear answer seems like a logical next step for the challenge creators. I'll comment more on that below (in the "appendix"), but for now I will stick to looking for a linear solution.
# 

# # Modeling
# 
# ## Bayesian Logistic Regression with PyMC3
# 
# The crux of this challenge is managing the bias-variance trade-off. Because we have more predictor features than observations, even a simple classifier like logistic regression will have too much variance and will overfit (indeed, it will have no unique solutions). So to overcome this, we need to induce some bias.
# 
# I'm going to take a Bayesian approach, inducing bias with strong priors. Specifically, I'm going to use a Bayesian logistic regression classifier to predict the targets, implementing the Bayesian inference using [PyMC3](https://docs.pymc.io/). Because we have more features than observations, I'm going to place a strongly skeptical, or regularizing, prior over the coefficients of the model. Another example of this approach can be found in the [Bayesian Spike-and-Slab in PyMC3](https://www.kaggle.com/melondonkey/bayesian-spike-and-slab-in-pymc3) kernel. A spike-and-slab prior embodies the idea that only a small subset of the features are meaningful, so that most coefficients should be set to zero. To do this, it combines a "spike" at zero with a "slab" acros all other coefficient values. 
# 
# An issue with the spike-and-slab prior is that it relies on discrete parameter values (binary variable indicating presence/absence of each coefficient), which prevents us from using the superior NUTS sampler in PyMC3. So intead, I'm going to employ a continuous analogue to the spike-and-slab model by using a [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution) prior. Keeping things continuous will allow the use of the NUTS sampler in PyMC3 (and would also allow this to be implemented in Stan).
# 
# Below is the specification of the model in PyMC3, wrapped in a function to create the model.

# In[ ]:


def make_model(X, y, cauchy_scale):
    model = pm.Model()

    with model:

        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sd=3)
        beta = pm.Cauchy('beta', alpha=0, beta=cauchy_scale, shape=X.shape[1])
        mu = pm.math.dot(X, beta)

        # Likelihood (sampling distribution) of observations
        y_obs = pm.Bernoulli('y_obs', p=invlogit(alpha + mu),  observed=y)
    
    model.name = "linear_c_"+str(cauchy_scale)
    return(model)


# 
# ### Cauchy scale parameter ($\gamma$)
# 
# We have essentially one tuning parameter available in our model: the scale parameter for the cauchy distribution ($\gamma$). This is analogous to setting the size of the zero "spike" in the spike-and-slab model, essentially capturing how many of the features we think might actually be useful.
# 
# Reading around on the forums, most people seem to suspect that there are actually very few meaningful variables. Informally inspecting my loess plots above suggest a similar conclusion. So for starters, let's imagine that only 10% of variables are likely to be meaningful. We could approximate that with a cauchy scale parameter of .0175, which would give us a reasonable prior for our model.

# In[ ]:


shape_par = .0175

prior_df = pd.DataFrame({"value": np.arange(-2,2,.01)})
prior_df = prior_df.assign(dens = cauchy.pdf(prior_df["value"],0,shape_par))
cauchy_samples = cauchy.rvs(0, shape_par, 10000)

print("percent non-zero coefs :", 1-np.mean((cauchy_samples < .1) & (cauchy_samples > -.1)))

plotnine.options.figure_size = (8,6)
ggplot(prior_df, aes(x="value", y="dens")) + geom_line()


# We could stick with that, which would be a sort of "proper" Bayesian way to do things, where that prior would really represent our subjective beliefs about the likely coefficient values. But, to be fair, I formed these prior beliefs in a pretty causal way---I wouldn't necessarily go to the mat for them.
# 
# So instead, we could treat the prior as a tuneable parameter, and use a measure of model performance to select the "best" cauchy scale parameter value for our model. 
# 
# ## Model Comparison
# 
# Cross validation is a bit tricky with so little data, and mcmc methods are a bit slow to fit. So instead I'll use LOO, a Bayesian approximation of Leave-One-Out cross-validation, to assess different models' performance. I'll perform a search in the region of parameter values around our best initial guess of .0175. I'll look at parameters from .01 to .05 and I'll also throw in .10 to illustrate what a more extreme value does to things.
# 
# I should acknowledge and point interested readers to Richard McElreath and his amazing book, [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/), which greatly informs this approach.

# In[ ]:


cauchy_scale_pars = [.01,.015,.0175,.020,.0225,.025,.03,.04,.05, .1]

models = []
traces = []
model_dict = dict()

for scale_val in cauchy_scale_pars:
    model = make_model(X,y, scale_val)
    with model:
        trace = pm.sample(1000,
                          tune = 500,
                          init= "adapt_diag", 
                          cores = 1, 
                          progressbar = False, 
                          chains = 2, 
                          nuts_kwargs=dict(target_accept=.95),
                          random_seed = 12345
                         )
    
    traces.append(trace)
    models.append(model)
    model_dict[model] = trace
    


# In[ ]:


# compare models with LOO
comp = pm.stats.compare(model_dict, ic="LOO", method='BB-pseudo-BMA')

# generate posterior predictions for original data
for i in range(0,len(traces)):
    y_hat = trace_predict(traces[i], X)
    print("scale = ",cauchy_scale_pars[i],", training AUCROC:",roc_auc_score(y,y_hat))
    
# print comparisons
comp


# These LOO values need to be taken with a grain of salt: according to LOO the best model is one using a cauchy scale paramter of .10, a model that is completely overfit with an ROCAUC of 1.0. The `shape_warn` column indicates that LOO is having issues with all these models, which is all the more reason to be cautious. If we had more data and more time, we should use k-fold cross validation instead of LOO.
# 
# In addition to the LOO score for each model, we can see the effects of regularization in the pLOO value, our measure of model complexity. We can interpret this value as the effective number of parameters in our model. Consider the model with $\gamma$ = .025: although the model has 301 parameter values (300 betas + 1 alpha), our use of a strong prior makes the model only as flexible as if it had ~50 parameters.
# 
# Below I'm going to throw out the two clearly overfit models ($\gamma$ =.10 and .05) and compare the remaining plausible models.

# In[ ]:


# can throw out .1 and .05 as way out-of-bounds
comp_abridged = pm.stats.compare(dict(zip(models[0:-2], traces[0:-2])), ic="LOO", method='BB-pseudo-BMA')
comp_abridged


# Looking at the difference in LOO scores (dLOO) and the standard error of the difference (dSE), we see that the model with $\gamma$ = .03 model isn't appreciably worse than the model with scale parameter .04 (less than standard error difference) and the $\gamma$ = .025 model is borderline. But, the .0175 model we started with is coming up as worse than the $\gamma$ =.04 model. Being somewhat conservative, and considering that AUCROC indicates overfitting for all models, the $\gamma$ = .025 or .03 models might be reasonable solutions. 
# 
# Let's inspect the $\gamma$ = .025 model.

# ## Inspecting coefficients
# 
# Let's see which coefficients are in our model are largest and/or confidently non-zero. I'll pull out the coefficients and sort them by their posterior mean. I'll also identify the coefficients for which the highest posterior density does not include zero. I'm using a 90% HPDI, but that's arbitrary.
# 

# In[ ]:


model1 = models[5]
trace1 = traces[5]
model1.name


# In[ ]:


coefs = pm.summary(trace1, varnames=["beta"], alpha=.10)

top_coefs = (coefs
             .assign(abs_est = abs(coefs["mean"]), non_zero = np.sign(coefs["hpd_5"]) == np.sign( coefs["hpd_95"]))
             .sort_values("abs_est", ascending=False)
            ).head(20)

top_coefs


# Then, I'll plot the top 20 coefficients' loess curves in logit space for visual inspection. Features 33 and 65 are the strongest features (consistent with what others have found) and that's pretty evident from the plots.

# In[ ]:


plotnine.options.figure_size = (12,9)

regex = re.compile("__(.*)")
top_feats = [regex.search(x)[1] for x in list(top_coefs.index)]

plot_loess(df, top_feats)


# # Results
# 
# Finally, I'll generate predictions for the leaderboard for all these models.
# 

# In[ ]:


# # generate test predictions and create submission file
def generate_submission(trace, file_suffix=""):

    test_predictions = trace_predict(trace, X2)

    submission  = pd.DataFrame({'id':df2.id, 'target':test_predictions})
    submission.to_csv("submission_"+file_suffix+".csv", index = False)
    return(None)

for model in model_dict.keys():
    generate_submission(model_dict[model], model.name)


# To hedge bets further, we can also generate predictions using model averaging, averaging over the models that seemed reasonable.

# In[ ]:


# grab potentially reasonable models
comp_MA = pm.stats.compare(dict(zip(models[0:-3], traces[0:-3])), ic="LOO", method='BB-pseudo-BMA')

# do prediction from averaged model
ppc_w = pm.sample_posterior_predictive_w(traces[0:-3], 4000, [make_model(X2,np.zeros(19750),c) for c in cauchy_scale_pars[0:-3]],
                        weights=comp_MA.weight.sort_index(ascending=True),
                        progressbar=True)
                        
y_hatMA = np.mean(np.array(ppc_w['y_obs']), axis=0)
submission  = pd.DataFrame({'id':df2.id, 'target':y_hatMA})
submission.to_csv('submission_MA.csv', index = False)


# How did we do? I've tested some of these models on the public leaderboard. The winners are the $\gamma$ = .025 and the model averaged predictions, with a public leaderboard score of .860. Overall, the models' actual out-of-sample performance largely accords with LOO, with the exception of the overfit $\gamma$ = .05 model. Some of the differences are subtle, but largely so are the differences in LOO.
# 
# Here's a table manually reproducing the scores for a selection of models alongside public leaderboard scores.
# 
# |      model      |   LOO  |  pLOO | training AUC | LB AUC |
# |:---------------:|:------:|:-----:|:------------:|:------:|
# |  $\gamma$ = .01 | 250.75 | 38.35 |     .955     |  .847  |
# | $\gamma$ =.0175 | 238.78 | 47.69 |     .978     |  .859  |
# |  $\gamma$ =.025 | 236.63 | 55.65 |     .988     |__.860__|
# |  $\gamma$ =.04  | 229.64 | 65.64 |     .997     |  .857  |
# |  $\gamma$ =.05  |  220.5 | 68.08 |     .999     |  .856  |
# |     Averaged    |   n/a  |  n/a  |              |__.860__|
# 
# # Conclusion
# 
# This Bayesian logistic regression model with regularizing cauchy priors performs quite well on the public leaderboard, with a very respectable score of .860.
# 
# In addition:
# * We were able to use all predictors in our model, despite having more predictors than observations.
# * Using subjective intuitions and some visual diagnostics, it was relatively easy to come up with sensible starting values for our prior
# * Using LOO let us approximate cross validation in a Bayesian setting and with very little data
# * Using a continuous prior let us use the sophisticated NUTS sampler in PyMC3
# * Using PyMC3 let us do all of this at a high level of abstraction: creating, fitting, and predicting from a custom model in less than 20 lines of code
# 
# This was my first time using PyMC3 and I've come away pretty impressed!

# 
# ----------------------------------------
# 
# # Appendix
# 
# To me, the spirit of this competition is broken if you test too much based on submissions to the public leaderboard. Still, it's interesting to get feedback on how different approaches are working, and whether the methods deployed to compare models are translating properly into the real world.
# 
# ## Non-linearities?
# 
# I still have some suspicions there could be non-linear relationships in the dataset. I believe the last verion of this contest was solvable with a hyperplane, so a natural extension for this second version would be to add non-linear relationships. 
# 
# Below I've picked out 10 candidates I observed for non-linear relationships, plus 65 and 33, two feaures that others have found to have strong linear relationships. If they really are non-linear, I think most of these could be captured by a 3rd-order polynomial.

# In[ ]:


non_lin_feats = ["276","91","240","246","253","255","268","118","240","7","167","65","33"]

plot_loess(df, non_lin_feats)


# Despite these plots, I've played around with adding polynomial terms to the model but never had any success improving leaderboard scores. I just don't think there's enough data to fit these more complex models. 
# 
# Still, leaving the PyMC3 model here for posterity.

# In[ ]:


def make_polymodel(X, y):
    
    with pm.Model() as model:
        
        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sd=3)
        beta1 = pm.Cauchy('beta', alpha=0, beta=.07, shape=X.shape[1])
        beta2 = pm.Cauchy('beta^2', alpha=0, beta=.07, shape=X.shape[1])
        beta3 = pm.Cauchy('beta^3', alpha=0, beta=.07, shape=X.shape[1])
        
        mu1 = pm.math.dot(X, beta1)
        mu2 = pm.math.dot(np.power(X,2), beta2)
        mu3 = pm.math.dot(np.power(X,3), beta3)
        
        p = invlogit(alpha + mu1 + mu2 + mu3)
        
        # Likelihood (sampling distribution) of observations
        y_obs = pm.Bernoulli('y_obs', p=p,  observed=y)
        
    return(model)

