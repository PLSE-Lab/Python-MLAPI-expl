#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


mport numpy as np
from src.solver_api import *

if __name__ == "__main__":
    # variables lower/upper bound
    lb = np.array([0, 0], dtype=np.float)
    ub = np.array([np.inf, np.inf], dtype=np.float)

    # Objective function coefficient vector
    c = np.array([500, 450], dtype=np.float)

    # Coefficient matrix
    A = np.array([
        [6, 5],
        [10, 20],
        [0, 1]], dtype=np.float)

    # right hand side vector
    b = np.array([60, 150, 8], dtype=np.float)
    s = np.array(["L", "L", "L"])

    # CPLEX
    cplex_solver = CplexApi()
    cplex_solver.add_variables(lb, ub, c)
    cplex_solver.add_constraints(A, s, b)
    x = cplex_solver.solve_model()
    cplex_solver.print('cplex_model')

