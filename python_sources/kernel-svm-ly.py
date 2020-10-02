import numpy as np
import pandas as pd
sub=pd.read_csv('../input/output.csv',encoding = 'utf-8')
#print(sub)
sub1 = sub.to_csv('output.csv',index = False,header = False,encoding = 'utf-8')
