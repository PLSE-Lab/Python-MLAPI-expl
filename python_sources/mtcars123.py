import numpy as np
import pandas as pd

mtcars=pd.read_csv('../input/train.csv')

#print(mtcars['Cabin'][2:10])
mtcars['Pclass'] = mtcars.Pclass.astype( "category" )
mtcars.info()