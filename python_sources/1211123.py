
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/tweets.csv')

summum_popularity = pd.DataFrame(data.groupby('username').followers.max())
summum_popularity.hist(bins=50)

