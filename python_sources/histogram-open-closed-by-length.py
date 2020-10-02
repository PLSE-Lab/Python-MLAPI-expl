from pandas import *
import numpy as np

randn = np.random.randn
s = Series(randn(5), index=['a', 'b', 'c', 'd', 'e'])
print(s['a'])
d = range(1,10)
a = Series(d)
print(a.median())
print(np.exp(s['a']))

print(a.get('23',np.nan))