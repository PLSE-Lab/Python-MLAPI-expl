#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# https://www.minizinc.org/doc-2.3.1/en/jupyter.html

# In[ ]:


get_ipython().system('pip install iminizinc')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'rm MiniZincIDE-2.3.2-bundle-linux-x86_64.tgz\nrm -r MiniZincIDE-2.3.2-bundle-linux\nwget https://github.com/MiniZinc/MiniZincIDE/releases/download/2.3.2/MiniZincIDE-2.3.2-bundle-linux-x86_64.tgz')


# In[ ]:


get_ipython().run_cell_magic('bash', '', '\ntar xf MiniZincIDE-2.3.2-bundle-linux-x86_64.tgz')


# In[ ]:


get_ipython().run_line_magic('set_env', 'PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:MiniZincIDE-2.3.2-bundle-linux/bin')
get_ipython().run_line_magic('set_env', 'LD_LIBRARY_PATH=/opt/conda/lib:MiniZincIDE-2.3.2-bundle-linux/lib')
get_ipython().run_line_magic('set_env', 'QT_PLUGIN_PATH=MiniZincIDE-2.3.2-bundle-linux/plugins')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'iminizinc')


# In[ ]:


n=8


# In[ ]:


get_ipython().run_cell_magic('minizinc', '-m bind', '%-m bind, mapea las variables de minizinc a variables de python (ejemplo, queens)   \n\ninclude "globals.mzn";\nint: n=6;\narray[1..n] of var 1..n: queens;\nvar 1..4: prueba;\nconstraint all_different(queens);\nconstraint all_different([queens[i]+i | i in 1..n]);\nconstraint all_different([queens[i]-i | i in 1..n]);\nsolve satisfy;')


# In[ ]:


np_queens = np.asarray(queens, dtype=np.int)
np_queens


# In[ ]:


np_prueba = np.asarray(prueba, dtype=np.int)
np_prueba


# In[ ]:


get_ipython().run_line_magic('pinfo', '%%minizinc')


# In[ ]:


n=6


# In[ ]:


get_ipython().run_cell_magic('capture', 'out', '%%minizinc -a\n%-a devuelve todas las soluciones, el bind solo asigna una solucion, para obtenerlas todas hay que hacer un %%capture\n\ninclude "globals.mzn";\nint: n;\narray[1..n] of var 1..n: queens2;\nvar 0..1: prueba;\nconstraint all_different(queens2);\nconstraint all_different([queens2[i]+i | i in 1..n]);\nconstraint all_different([queens2[i]-i | i in 1..n]);\nsolve satisfy;')


# In[ ]:


import re

output = out.outputs[0].data['text/plain']
print(output)

substring = re.findall(r"\{.*\[(.*)\].*\}", output)

mat_queens = np.array(substring)
print(mat_queens[0])
print(mat_queens[1])

