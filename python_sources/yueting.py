import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import sys
from subprocess import check_output

s=0
t=1
for i in range(1,51):
    t*=i
    s+=t
    
    print(s)


