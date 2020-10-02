#!/usr/bin/env python
# coding: utf-8

# Small script to calculate exact class weights using probing results from :  https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194 

# In[ ]:


'''
Solve system of non-linear equations to determine exact weights after probing
'''

import numpy as np
from scipy.optimize import fsolve

# Class probing results and name for the weight var in fsolve below
cd = {
    '99' : (30.701, 'a'),
    '95' : (32.620, 'b'),
    '92' : (32.620, 'c'),
    '90' : (32.620, 'd'),
    '88' : (32.620, 'e'),
    '67' : (32.620, 'f'),
    '65' : (32.620, 'g'),
    '64' : (30.692, 'h'),
    '62' : (32.620, 'i'),
    '53' : (32.622, 'j'),
    '52' : (32.620, 'k'),
    '42' : (32.620, 'l'),
    '16' : (32.620, 'm'),
    '15' : (30.702, 'n'),
    '6' : (32.620, 'o'),
}

# Build non-linear system of equations expression
final_str = ''
for i, (v, var) in enumerate(cd.values()):
    sum_except = ''
    sum_all = ''
    for j, (v2, var2) in enumerate(cd.values()):
        sum_all += var2
        if j != i:
            sum_except += var2
        else:
            sum_except += '0'
        if j < len(cd.values()) - 1:
            sum_all += ' + '
            sum_except += ' + '
    final_str = final_str + f'{-np.log(1e-15)} * ({sum_except}) / ({sum_all}) - {str(v)},'

#print(final_str)

# Solve non-linear system of eqs
def eqs(aa):
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o = aa
    return (eval(final_str))

res = fsolve(eqs, np.ones(15))

# Normalize weights for convenience
res = res / np.min(res)

# Print exact weights
for _class, weight in zip(cd.keys(), res):
    print(f'Weight for class {_class} = {weight:.5f}')


# In[ ]:




