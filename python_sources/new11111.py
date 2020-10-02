import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../input/train.csv')
df1 = pd.read_csv('../input/test.csv')                 

df.sample(5)