# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# A study of the IMDB movie data downloaded from Kaggle
import pandas as pd
import numpy as np
movie = pd.read_csv('../input/movie_metadata.csv')


plot_keywords = movie['plot_keywords']
print(plot_keywords[0])
print(plot_keywords[3])
print(len(plot_keywords))
best_movies = movie.loc[movie['imdb_score'] >= 8 ]
the_best_keywords = best_movies['plot_keywords']

print()
my_list = []
for item in the_best_keywords:
    my_string = str(item)
    ky_wrds_list = my_string.split('|')
    for k_wrd in ky_wrds_list:
        my_list.append(k_wrd)

hit_plt_words = dict((x,my_list.count(x)) for x in set(my_list))
print(max(hit_plt_words.values()))
print(min(hit_plt_words.values()))
print()
print("The most common plot words for movies in IMDB with a rating greater than or equal to 8 are: ")
print()
for keys in hit_plt_words:
       
    if hit_plt_words[keys] >= 6 and keys != 'nan':
        print(str(keys) + ": " + str(hit_plt_words[keys]))
           
print()
