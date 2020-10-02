# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import theano.tensor as T
from theano import function
from theano import pp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Adding scalars using theano
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

print (pp(z))
print (f(2,3))

# Adding matrices using theano

a = T.dmatrix('a')
b = T.dmatrix('b')
c = a + b
d = function([a,b], c)

print (pp(c))
print (d([[1, 2], [3, 4]], [[10, 20], [30, 40]]))
