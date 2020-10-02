# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [markdown]
# # Setup

# %% [markdown]
# ## Jupyter Setup 

# %% [code]
%reload_ext autoreload
%autoreload 2

# %% [code]
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# %% [markdown]
# ## Imports 

# %% [markdown]
# ### Basic Imports 

# %% [code]
import altair as alt
import pandas as pd
import sklearn as skl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# %% [markdown]
# ### Scikit Learn Imports 

# %% [code]
from sklearn.preprocessing import LabelEncoder, Normalizer, OrdinalEncoder, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# %% [markdown]
# ### Load Jax 

# %% [code]
!pip install jax jaxlib

# %% [markdown]
# ### Import Jax 

# %% [code]
from jax import grad, jit, vmap
from jax import lax
from jax import random
from jax.random import PRNGKey, split as splitkey
import jax
import jax.numpy as jnp