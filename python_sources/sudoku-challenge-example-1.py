#!/usr/bin/env python
# coding: utf-8

# ![](http://)The project description is available on [Colab](https://colab.research.google.com/drive/1RNR3bkramHc2iVacBwHDR13IW78k7fWs).

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as scs # sparse matrix construction 
import scipy.linalg as scl # linear algebra algorithms
import scipy.optimize as sco # for minimization use
import matplotlib.pylab as plt # for visualization


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# First, let's have a look at the data itself. We take the smallest dataset in the ``input`` directory. 

# In[ ]:


# read the data in small1.csv, it only has 24 problems.
data = pd.read_csv("../input/small1.csv") 
print('data set legnth:', len(data))

# enumerate the first 5 data strings in quizzes
print(data["quizzes"][0:5])

# enumerate the first 5 corresponding solution strings.
print(data["solutions"][0:5])

# Each string is an 81-chartacter string of digits. '0' means empty slots in the problem.


# In[ ]:


# Take a close look at one string in quizzes

quiz = data["quizzes"][9]
print(quiz, '\ndata type is:', type(quiz))

# we can turn it into a numpy array by the following
np.reshape([int(c) for c in quiz], (9,9))


# In the following we consider the constrained optimization problem for Sudoku with the following objective function, such objective function is chosen as the so-called entropy. The entropy function $E(x) = -x\log x$ has the property that it reaches the minimum on $x=0$ and $x=1$.
# 
# $$\min_{0\le x_{ijk}\le 1}\quad \sum_{i,j,k} E(x_{ijk})$$
# 
# subject to the linear constraints $A X = B$, where $X$ represents the vector of $x_{ijk}$. 

# In[ ]:


# In the following, the fixed_constraints are constructed from the board directly. 
# This part only needs to be constructed once. The output has been returned as a sparse matrix for efficiency.

def fixed_constraints(N=9):
    rowC = np.zeros(N)
    rowC[0] =1
    rowR = np.zeros(N)
    rowR[0] =1
    row = scl.toeplitz(rowC, rowR)
    ROW = np.kron(row, np.kron(np.ones((1,N)), np.eye(N)))
    
    colR = np.kron(np.ones((1,N)), rowC)
    col  = scl.toeplitz(rowC, colR)
    COL  = np.kron(col, np.eye(N))
    
    M = int(np.sqrt(N))
    boxC = np.zeros(M)
    boxC[0]=1
    boxR = np.kron(np.ones((1, M)), boxC) 
    box = scl.toeplitz(boxC, boxR)
    box = np.kron(np.eye(M), box)
    BOX = np.kron(box, np.block([np.eye(N), np.eye(N) ,np.eye(N)]))
    
    cell = np.eye(N**2)
    CELL = np.kron(cell, np.ones((1,N)))
    
    return scs.csr_matrix(np.block([[ROW],[COL],[BOX],[CELL]]))


# In[ ]:


# Take a look of the constraint matrix A0 with adding the clues.
A0 = fixed_constraints()

# The spy visualization of A0
plt.spy(A0, markersize=0.2)


# In[ ]:


# For the constraint from clues, we extract the nonzeros from the quiz string.
def clue_constraint(input_quiz, N=9):
    m = np.reshape([int(c) for c in input_quiz], (N,N))
    r, c = np.where(m.T)
    v = np.array([m[c[d],r[d]] for d in range(len(r))])
    
    table = N * c + r
    table = np.block([[table],[v-1]])
    
    # it is faster to use lil_matrix when changing the sparse structure.
    CLUE = scs.lil_matrix((len(table.T), N**3))
    for i in range(len(table.T)):
        CLUE[i,table[0,i]*N + table[1,i]] = 1
    # change back to csr_matrix.
    CLUE = CLUE.tocsr() 
    
    return CLUE


# In[ ]:


# get the constraint matrix from clue.
A1 = clue_constraint(quiz)

# Formulate the matrix A and vector B (B is all ones).
A = scs.vstack((A0,A1))
B = np.ones((np.size(A, 0)))


# After we have formulated the linear constraint $AX = B$, now we just search the minimimum for this nonlinear minimization problem. 
