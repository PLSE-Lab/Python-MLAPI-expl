import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *
reviews= []
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

reviews.head()