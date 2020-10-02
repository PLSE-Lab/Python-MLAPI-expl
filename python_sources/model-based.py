#!/usr/bin/env python
# coding: utf-8

# ### 0. Import libraries and model

# In[ ]:


# Import libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

get_ipython().system('pip install ema_workbench')


# In[ ]:


# Import EMA workbench and dike model function
from ema_workbench import (Model, CategoricalParameter,
                           ScalarOutcome, IntegerParameter, RealParameter, load_results, save_results)
from shutil import (copyfile, copytree)
copyfile(src = "../input/modules/dike_model_simulation.py", dst = "../working/dike_model_simulation.py")
copyfile(src = "../input/modules/dike_model_optimization.py", dst = "../working/dike_model_optimization.py")
copyfile(src = "../input/modules/dike_model_function.py", dst = "../working/dike_model_function.py")
copyfile(src = "../input/modules/funs_generate_network.py", dst = "../working/funs_generate_network.py")
copyfile(src = "../input/modules/funs_dikes.py", dst = "../working/funs_dikes.py")
copyfile(src = "../input/modules/funs_economy.py", dst = "../working/funs_economy.py")
copyfile(src = "../input/modules/funs_hydrostat.py", dst = "../working/funs_hydrostat.py")
copyfile(src = "../input/modules/problem_formulation.py", dst = "../working/problem_formulation.py")
copytree(src = "../input/modules/Data/", dst = "../working/data/")

from dike_model_function import DikeNetwork  # @UnresolvedImport


def sum_over(*args):
    return sum(args)


# In[ ]:


# Import lake model for problem formulation 3
from ema_workbench import (Model, MultiprocessingEvaluator, Policy, Scenario)
from ema_workbench.em_framework.evaluators import perform_experiments
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging
import time
from problem_formulation import get_model_for_problem_formulation

ema_logging.log_to_stderr(ema_logging.INFO)

dike_model, planning_steps = get_model_for_problem_formulation(3)


# # 1. Open exploration

# ### 1.1 Model initialization

# #### 1.1.1 Uncertainties

# In[ ]:


for unc in dike_model.uncertainties:
    print(repr(unc))
    
uncertainties = dike_model.uncertainties

import copy
uncertainties = copy.deepcopy(dike_model.uncertainties)


# #### 1.1.2 Levers

# In[ ]:


for policy in dike_model.levers:
    print(repr(policy))
    
levers = dike_model.levers 

import copy
levers = copy.deepcopy(dike_model.levers)


# #### 1.1.3 Outcomes

# In[ ]:


for outcome in dike_model.outcomes:
    print(repr(outcome))


# ### 1.2 Run model

# In[ ]:


from ema_workbench import (MultiprocessingEvaluator, ema_logging,
                           perform_experiments, SequentialEvaluator)

ema_logging.log_to_stderr(ema_logging.INFO)
 
with MultiprocessingEvaluator(dike_model) as evaluator:
    experiments, outcomes = evaluator.perform_experiments(scenarios=250, policies=4) # CHANGE SCENARIO N


# In[ ]:


# Save results from previous model run (scenarios = 250, policies = 4)
from ema_workbench import save_results

results = experiments, outcomes
save_results(results, 'Run 1.2 - 250 scenarios 4 policies.tar.gz')


# ### HIERONDER EVEN CONTROLEREN (DIT IS OM ALLEEN A.1 TE SELECTEREN)

# In[ ]:


results = load_results('Run 1.2 - 250 scenarios 4 policies.tar.gz')
experiments, outcomes = results


# In[ ]:


# Only consider A.1 for analysis. With all outcomes, the plot is too large.
A_list = ['A.1 Total Costs', 'A.1_Expected Number of Deaths', 'RfR Total Costs', 'Expected Evacuation Costs']
A_dict = dict((k, outcomes[k]) for k in A_list)


# ### 1.3 Visualize results in multi scatter plot

# Note: A.1 Total Costs is the sum of A.1 Annual Damage and A.1 Dike Investment Costs

# In[ ]:


from ema_workbench.analysis import pairs_plotting

fig, axes = pairs_plotting.pairs_scatter(experiments, A_dict, group_by='policy',
                                         legend=True)
fig.set_size_inches(14,14)
plt.show()


# ## 1.2 Advanced analysis

# ### 1.2.1 Scenario discovery

# Scenarios are points in the uncertainty space (Kwakkel, 2020). We will therefore drop the policy levers in the data.

# In[ ]:


from ema_workbench import load_results

results = load_results('Run 1.2 - 250 scenarios 4 policies.tar.gz')


# In[ ]:


# Drop the policy levers in the data. This will otherwise mess with the analysis.
cleaned_experiments = experiments.drop(labels=[l.name for l in dike_model.levers], axis=1)


# In[ ]:


# Make a dataframe of the data and add a new variable Total Expected Number of Deaths. This is the sum of all zones.
df_cleaned_experiments = pd.DataFrame(cleaned_experiments)
df_outcomes = pd.DataFrame(outcomes)

df_results = pd.concat([df_cleaned_experiments, df_outcomes], axis = 1, join = 'inner')

df_results['Total Expected Number of Deaths'] = df_results['A.1_Expected Number of Deaths'] +                                                 df_results['A.2_Expected Number of Deaths'] +                                                 df_results['A.3_Expected Number of Deaths'] +                                                 df_results['A.4_Expected Number of Deaths'] +                                                 df_results['A.5_Expected Number of Deaths']


# In[ ]:


# Run PRIM algorithm
from ema_workbench.analysis import prim

data = df_results['Total Expected Number of Deaths']

# Scenarios where total expected number of deaths are lower than 0.0001
y = data < 0.0001
x = cleaned_experiments

prim_alg = prim.Prim(x, y, threshold = 0.8)
box1 = prim_alg.find_box()

box1.show_tradeoff()
plt.show()


# In[ ]:


box1.inspect(style='graph')
plt.show()


# In[ ]:


from ema_workbench.analysis import dimensional_stacking

dimensional_stacking.create_pivot_plot(cleaned_experiments, y, nr_levels=3)
plt.show()


# Because there are a lot of white spaces in the dimensional stacking plot we will run the model many more times than we have done previously.

# In[ ]:


with MultiprocessingEvaluator(dike_model) as evaluator:
    results_1 = evaluator.perform_experiments(scenarios = 1000, policies=4) # Number of scenarios changed


# In[ ]:


save_results(results_1, 'Run 1.2 - 1000 scenarios 4 policies.tar.gz')


# ## Dimensional stacking 1000 scenarios met PRIM

# In[ ]:


#load 1000 scenarios
results_1 = load_results('Run 1.2 - 1000 scenarios 4 policies.tar.gz')
experiments_1, outcomes_1 = results_1


# In[ ]:


#combine Expected numer of deaths
data = outcomes_1['A.1_Expected Number of Deaths'] +         outcomes_1['A.2_Expected Number of Deaths'] +         outcomes_1['A.3_Expected Number of Deaths'] +         outcomes_1['A.4_Expected Number of Deaths'] +         outcomes_1['A.5_Expected Number of Deaths']


# In[ ]:


# Re-Run PRIM algorithm
from ema_workbench.analysis import prim

# Scenarios where total expected number of deaths are lower than 0.0001
y = data < 0.0001
x = experiments_1

prim_alg = prim.Prim(x, y, threshold = 0.8)
box1 = prim_alg.find_box()

box1.show_tradeoff()
plt.show()


# In[ ]:


box1.inspect(style='graph')
plt.show()


# In[ ]:


from ema_workbench.analysis import dimensional_stacking

dimensional_stacking.create_pivot_plot(x, y, nr_levels=3)
plt.show()


# ## Dimensional Stacking 1000 scenarios zonder PRIM

# In[ ]:


experiments_1, outcomes_1 = results_1

cleaned_experiments_1 = experiments_1.drop(labels=[l.name for l in dike_model.levers], axis=1)

df_cleaned_experiments_1 = pd.DataFrame(cleaned_experiments_1)
df_outcomes_1 = pd.DataFrame(outcomes_1)

df_results_1 = pd.concat([df_cleaned_experiments_1, df_outcomes_1], axis = 1, join = 'inner')

df_results_1['Total Expected Number of Deaths'] = df_results_1['A.1_Expected Number of Deaths'] +                                                 df_results_1['A.2_Expected Number of Deaths'] +                                                 df_results_1['A.3_Expected Number of Deaths'] +                                                 df_results_1['A.4_Expected Number of Deaths'] +                                                 df_results_1['A.5_Expected Number of Deaths']

data_1 = df_results_1['Total Expected Number of Deaths']

y_1 = data_1 < 0.0001

dimensional_stacking.create_pivot_plot(cleaned_experiments_1, y_1, nr_levels=3)
plt.show()


# ## Feauture scoring

# In[ ]:


from ema_workbench.analysis import feature_scoring

x = experiments_1
y = outcomes_1

fs = feature_scoring.get_feature_scores_all(x, y)
sns.heatmap(fs, cmap='viridis', annot=True)
plt.show()


# ## 1.3 Sensitivity Analysis

# In[ ]:


#Setting 1 policy for running sensitivity analysis
from ema_workbench import Policy
policies = [Policy('policy 1', **{'0_RfR 0':1,
                                  '0_RfR 1':1,
                                  '0_RfR 2':1,
                                  '1_RfR 0':0,
                                  '1_RfR 1':0,
                                  '1_RfR 2':0,
                                  '2_RfR 0':0,
                                  '2_RfR 1':0,
                                  '2_RfR 2':0,
                                  '3_RfR 0':0,
                                  '3_RfR 1':0,
                                  '3_RfR 2':0,
                                  '4_RfR 0':0,
                                  '4_RfR 1':0,
                                  '4_RfR 2':0,
                                  'EWS_DaysToThreat': 1,
                                  'A.1_DikeIncrease 0':5,
                                  'A.1_DikeIncrease 1':0,
                                  'A.1_DikeIncrease 2':0,
                                  'A.2_DikeIncrease 0':0,
                                  'A.2_DikeIncrease 1':0,
                                  'A.2_DikeIncrease 2':0,
                                  'A.3_DikeIncrease 0':0,
                                  'A.3_DikeIncrease 1':0,
                                  'A.3_DikeIncrease 2':0,
                                  'A.4_DikeIncrease 0':0,
                                  'A.4_DikeIncrease 1':0,
                                  'A.4_DikeIncrease 2':0,
                                  'A.5_DikeIncrease 0':0,
                                  'A.5_DikeIncrease 1':0,
                                  'A.5_DikeIncrease 2':0}),
           Policy('policy 2', **{'0_RfR 0':0,
                                  '0_RfR 1':0,
                                  '0_RfR 2':0,
                                  '1_RfR 0':0,
                                  '1_RfR 1':0,
                                  '1_RfR 2':0,
                                  '2_RfR 0':0,
                                  '2_RfR 1':0,
                                  '2_RfR 2':0,
                                  '3_RfR 0':0,
                                  '3_RfR 1':0,
                                  '3_RfR 2':0,
                                  '4_RfR 0':1,
                                  '4_RfR 1':1,
                                  '4_RfR 2':1,
                                  'EWS_DaysToThreat': 1,
                                  'A.1_DikeIncrease 0':0,
                                  'A.1_DikeIncrease 1':0,
                                  'A.1_DikeIncrease 2':0,
                                  'A.2_DikeIncrease 0':0,
                                  'A.2_DikeIncrease 1':0,
                                  'A.2_DikeIncrease 2':0,
                                  'A.3_DikeIncrease 0':0,
                                  'A.3_DikeIncrease 1':0,
                                  'A.3_DikeIncrease 2':0,
                                  'A.4_DikeIncrease 0':0,
                                  'A.4_DikeIncrease 1':0,
                                  'A.4_DikeIncrease 2':0,
                                  'A.5_DikeIncrease 0':5,
                                  'A.5_DikeIncrease 1':0,
                                  'A.5_DikeIncrease 2':0}),
           Policy('policy 3', **{'0_RfR 0':0,
                                  '0_RfR 1':0,
                                  '0_RfR 2':0,
                                  '1_RfR 0':1,
                                  '1_RfR 1':0,
                                  '1_RfR 2':0,
                                  '2_RfR 0':0,
                                  '2_RfR 1':1,
                                  '2_RfR 2':0,
                                  '3_RfR 0':0,
                                  '3_RfR 1':0,
                                  '3_RfR 2':1,
                                  '4_RfR 0':0,
                                  '4_RfR 1':0,
                                  '4_RfR 2':0,
                                  'EWS_DaysToThreat': 1,
                                  'A.1_DikeIncrease 0':0,
                                  'A.1_DikeIncrease 1':0,
                                  'A.1_DikeIncrease 2':0,
                                  'A.2_DikeIncrease 0':0,
                                  'A.2_DikeIncrease 1':0,
                                  'A.2_DikeIncrease 2':0,
                                  'A.3_DikeIncrease 0':5,
                                  'A.3_DikeIncrease 1':0,
                                  'A.3_DikeIncrease 2':0,
                                  'A.4_DikeIncrease 0':0,
                                  'A.4_DikeIncrease 1':0,
                                  'A.4_DikeIncrease 2':0,
                                  'A.5_DikeIncrease 0':0,
                                  'A.5_DikeIncrease 1':0,
                                  'A.5_DikeIncrease 2':0})]


# In[ ]:


#Running SOBOL evaluator
from ema_workbench import (MultiprocessingEvaluator, ema_logging,
                           perform_experiments, SequentialEvaluator)
from ema_workbench.em_framework.evaluators import SOBOL, LHS
from ema_workbench.em_framework import get_SALib_problem
from SALib.analyze import sobol

ema_logging.log_to_stderr(ema_logging.INFO)

n_exp = 1000

with MultiprocessingEvaluator(dike_model) as evaluator:
    results_SOBOL = evaluator.perform_experiments(n_exp, policies, uncertainty_sampling=SOBOL)


# In[ ]:


problem = get_SALib_problem(lake_model.uncertainties)

sobol_results = {}
for policy in experiments_SOBOL.policy.unique():
    logical = experiments_SOBOL.policy == policy
    y = results_SOBOL['reliability'][logical]
    indices = analyze(problem, y)
    sobol_results[policy] = indices


# In[ ]:




