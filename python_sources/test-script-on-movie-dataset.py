import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA # Principal Component Analysis module
from sklearn.cluster import KMeans # KMeans clustering 
import matplotlib.pyplot as plt # Python defacto plotting library
import seaborn as sns # More snazzy plotting library



movie = pd.read_csv('../input/movie_metadata.csv') # reads the csv and creates the dataframe called movie
print (movie.head())

# Any results you write to the current directory are saved as output.
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in movie.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = movie.columns.difference(str_list)   

movie_num = movie[num_list]
#del movie # Get rid of movie df as we won't need it now
print (movie_num.head())

movie_num = movie_num.fillna(value=0, axis=1)