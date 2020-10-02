# with all of the Elevator Rescues in Montgomery County, PA


import pandas as pd
import numpy as np
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)



# Read data 
d=pd.read_csv("../input/911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':datetime.datetime,'twp':str,'addr':str,'e':int}, 
     )




